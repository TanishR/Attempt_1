"""
Step 7: LightGBM Model Training and Metric Evaluation.
Trains pairwise ranker on train-fold features, evaluates on val-fold,
computes feature importance, sanity decision macro F0.5, rule baseline,
and France proxy cross-country evaluation.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

import config
from metrics import build_gold_map, macro_f05


def parse_args():
    """
    Parses command-line arguments for model training.
    Returns: parsed ArgumentParser namespace.
    """
    parser = argparse.ArgumentParser(description="Step 7: Train LightGBM Model")
    parser.add_argument("--version", type=str, default="v1", help="Model version tag (e.g. v1)")
    parser.add_argument("--laptop-test", action="store_true", help="Use laptop test cache directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Explicit cache directory path")
    parser.add_argument("--proxy", action="store_true", help="Run France proxy cross-country experiment")
    return parser.parse_args()


def load_dataset_features(cache_dir: str) -> pd.DataFrame:
    """
    Loads all train candidate feature parquet files from cache.
    Returns: concatenated DataFrame of candidate pairs with features.
    """
    feat_files = []
    chunk_prefix = "feats_train_chunk_"
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith(chunk_prefix) and f.endswith(".parquet"):
            feat_files.append(os.path.join(cache_dir, f))

    if not feat_files:
        prefix = "feats_train_"
        for f in sorted(os.listdir(cache_dir)):
            if f.startswith(prefix) and f.endswith(".parquet"):
                feat_files.append(os.path.join(cache_dir, f))

    if not feat_files:
        single_path = os.path.join(cache_dir, "feats_train.parquet")
        if os.path.exists(single_path):
            feat_files = [single_path]
        else:
            raise FileNotFoundError(f"No feats_train_*.parquet files found in {cache_dir}")

    print(f"Loading features from {len(feat_files)} file(s)...")
    dfs = [pd.read_parquet(fp) for fp in feat_files]
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full_df):,} total feature pairs.")
    return full_df


def train_lgb_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str]
) -> lgb.Booster:
    """
    Trains LightGBM model with early stopping on validation logloss using exact Guidebook 7.2 params.
    Returns: trained LightGBM Booster object.
    """
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_jobs": min(8, os.cpu_count() or 8),
        "verbose": -1,
        "seed": 42,
    }

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    print("\n--- Training LightGBM Booster ---")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return booster


def compute_feature_importance(
    booster: lgb.Booster,
    feature_names: List[str],
    out_csv_path: str
) -> pd.DataFrame:
    """
    Computes split and gain feature importance, prints sorted table, and saves to CSV.
    Returns: DataFrame containing feature importances sorted by gain.
    """
    gain_imp = booster.feature_importance(importance_type="gain")
    split_imp = booster.feature_importance(importance_type="split")

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "gain": gain_imp,
        "split": split_imp,
    }).sort_values(by="gain", ascending=False).reset_index(drop=True)

    imp_df.to_csv(out_csv_path, index=False)
    print(f"\nFeature importance saved to: {out_csv_path}")
    print("\n" + "=" * 65)
    print(f"{'Feature':<25} | {'Gain':>16} | {'Split':>10}")
    print("-" * 65)
    for _, row in imp_df.iterrows():
        print(f"{row['feature']:<25} | {row['gain']:16.2f} | {int(row['split']):10d}")
    print("=" * 65)
    return imp_df


def predict_sanity_decisions(
    val_df: pd.DataFrame,
    val_probs: np.ndarray,
    t_top1: float = 0.5,
    t_extra: float = 0.8
) -> Dict[str, Set[str]]:
    """
    Applies quick sanity decision rule: top-1 if prob >= t_top1, extras if prob >= t_extra, no exclusivity.
    Returns: dict mapping s1_id -> set of predicted matching cand_ids.
    """
    df = val_df[["s1_id", "cand_id"]].copy()
    df["prob"] = val_probs

    # Sort candidates by prob descending
    df_sorted = df.sort_values(by=["s1_id", "prob"], ascending=[True, False])

    pred_map: Dict[str, Set[str]] = {}
    grouped = df_sorted.groupby("s1_id")

    for s1_id, group in grouped:
        probs = group["prob"].values
        cands = group["cand_id"].values

        if len(probs) > 0 and probs[0] >= t_top1:
            preds = {cands[0]}
            for i in range(1, len(probs)):
                if probs[i] >= t_extra:
                    preds.add(cands[i])
            pred_map[s1_id] = preds
        else:
            pred_map[s1_id] = set()

    return pred_map


def evaluate_rule_baseline(
    val_df: pd.DataFrame,
    gold_map: Dict[str, Set[str]],
    r_candidates: List[float] = None
) -> Tuple[float, float]:
    """
    Evaluates rule_score baseline with exclusivity across a range of thresholds R.
    Returns: tuple of (best_R, best_macro_f05).
    """
    print("\n--- Evaluating Simple Rule Baseline (rule_score with Exclusivity) ---")
    df = val_df[["s1_id", "cand_id", "rule_score"]].copy()

    # Exclusivity: For every cand_id that appears in >1 S1, keep only the pair where rule_score is highest
    df_sorted = df.sort_values(by="rule_score", ascending=False)
    df_exclusive = df_sorted.drop_duplicates(subset=["cand_id"], keep="first").copy()

    # Sort by s1_id and rule_score descending
    df_exclusive = df_exclusive.sort_values(by=["s1_id", "rule_score"], ascending=[True, False])

    all_val_s1 = list(gold_map.keys())

    if r_candidates is None:
        r_candidates = [float(x) for x in range(30, 95, 2)]

    best_r = 50.0
    best_f05 = -1.0

    for r_val in r_candidates:
        pred_map: Dict[str, Set[str]] = {s: set() for s in all_val_s1}
        # Filter pairs where rule_score >= r_val
        filtered = df_exclusive[df_exclusive["rule_score"] >= r_val]
        for s1, c in zip(filtered["s1_id"].values, filtered["cand_id"].values):
            pred_map[s1].add(c)

        score = macro_f05(pred_map, gold_map)
        if score > best_f05:
            best_f05 = score
            best_r = r_val

    print(f"Rule Baseline: Best R = {best_r:.1f} -> Val Macro F0.5 = {best_f05:.4f}")
    return best_r, best_f05


def run_france_proxy(
    feats_df: pd.DataFrame,
    split_df: pd.DataFrame,
    norm_s1_df: pd.DataFrame,
    gold_map_full: Dict[str, Set[str]],
    feature_names: List[str]
):
    """
    Executes France proxy evaluation: trains on country A train rows, evaluates on country B val rows, and reverse.
    Returns: None.
    """
    print("\n" + "=" * 70)
    print("FRANCE PROXY EVALUATION (--proxy flag)")
    print("=" * 70)

    # Map country to s1_id
    country_map = dict(zip(norm_s1_df["s1_id"].values, norm_s1_df["country"].values))
    feats_df = feats_df.copy()
    feats_df["country"] = feats_df["s1_id"].map(country_map)

    unique_countries = list(feats_df["country"].dropna().unique())
    print(f"Identified countries for proxy check: {unique_countries}")
    if len(unique_countries) < 2:
        print("Less than 2 countries available for proxy check. Skipping.")
        return

    # Prioritize US and India if present
    c1 = "US" if "US" in unique_countries else unique_countries[0]
    c2 = "India" if "India" in unique_countries else [c for c in unique_countries if c != c1][0]

    for source_c, target_c in [(c1, c2), (c2, c1)]:
        print(f"\n[Proxy Run] Train on {source_c} -> Evaluate on {target_c} Val:")
        train_sub = feats_df[(feats_df["is_competitor"] == 0) & (feats_df["fold"] == "train") & (feats_df["country"] == source_c)]
        val_sub = feats_df[(feats_df["is_competitor"] == 0) & (feats_df["fold"] == "val") & (feats_df["country"] == target_c)]

        if len(train_sub) == 0 or len(val_sub) == 0:
            print(f"Insufficient data for {source_c} -> {target_c}")
            continue

        X_tr = train_sub[feature_names].values.astype(np.float32)
        y_tr = train_sub["label"].values.astype(int)
        X_va = val_sub[feature_names].values.astype(np.float32)
        y_va = val_sub["label"].values.astype(int)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "n_jobs": min(8, os.cpu_count() or 8),
            "verbose": -1,
            "seed": 42,
        }

        dtr = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, free_raw_data=False)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, feature_name=feature_names, free_raw_data=False)

        proxy_booster = lgb.train(
            params,
            dtr,
            num_boost_round=500,
            valid_sets=[dtr, dva],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        va_preds = proxy_booster.predict(X_va)
        proxy_auc = roc_auc_score(y_va, va_preds)

        # Build target country gold map for ALL target country val entities
        target_val_s1 = set(norm_s1_df.loc[norm_s1_df["country"] == target_c, "s1_id"].values) & set(split_df.loc[split_df["fold"] == "val", "s1_id"].values)
        target_gold = {s: gold_map_full[s] for s in target_val_s1 if s in gold_map_full}
        proxy_preds = predict_sanity_decisions(val_sub, va_preds, t_top1=0.5, t_extra=0.8)
        proxy_f05 = macro_f05(proxy_preds, target_gold)

        print(f"  Target: {target_c} Val | AUC: {proxy_auc:.4f} | Sanity Macro F0.5: {proxy_f05:.4f}")
    print("=" * 70)


def append_experiment_log(
    version: str,
    what_changed: str,
    recall_cands: float,
    val_auc: float,
    val_macro_f05: float,
    thresholds: str = "0.5 / 0.8 (sanity)"
):
    """
    Appends experiment run results to experiments.md.
    Returns: None.
    """
    exp_path = "experiments.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"| {version} | {now_str} | {what_changed} | {recall_cands:.2f}% | {val_auc:.4f} | {val_macro_f05:.4f} | {thresholds} | - |\n"

    if os.path.exists(exp_path):
        with open(exp_path, "a") as f:
            f.write(line)
        print(f"Logged experiment run to {exp_path}")


def main():
    """
    Main orchestration routine for Step 7 model training and evaluation.
    Returns: None.
    """
    args = parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(config.CACHE_DIR, "laptop_test") if args.laptop_test else config.CACHE_DIR

    print("=" * 75)
    print(f"=== Step 7: LightGBM Training & Metric (Version: {args.version}) ===")
    print(f"Cache Directory: {cache_dir}")
    print(f"Model Features ({len(config.FEATURES)}): {config.FEATURES}")
    print("=" * 75)

    # 1. Load split.parquet
    split_path = os.path.join(cache_dir, "split.parquet")
    if not os.path.exists(split_path):
        print(f"Error: split.parquet not found in {cache_dir}")
        sys.exit(1)
    split_df = pd.read_parquet(split_path)
    split_map = dict(zip(split_df["s1_id"].values, split_df["fold"].values))
    print(f"Loaded split.parquet: {len(split_df):,} S1 (train: {(split_df['fold'] == 'train').sum():,}, val: {(split_df['fold'] == 'val').sum():,})")

    # 2. Load ground truth for full gold map evaluation
    gt_path = os.path.join(cache_dir, "gt_long.parquet")
    gt_df = pd.read_parquet(gt_path) if os.path.exists(gt_path) else None
    if gt_df is not None:
        print(f"Loaded ground truth gt_long.parquet: {len(gt_df):,} rows")

    val_s1_ids = split_df[split_df["fold"] == "val"]["s1_id"].unique()
    val_gold_map = build_gold_map(gt_df, val_s1_ids)
    print(f"Constructed full Val Gold Map: {len(val_gold_map):,} entities (including singletons & missed blocking).")

    # 3. Load feature dataset
    feats_df = load_dataset_features(cache_dir)
    feats_df["fold"] = feats_df["s1_id"].map(split_map)

    # Verify no competitor rows used in training
    train_mask = (feats_df["is_competitor"] == 0) & (feats_df["fold"] == "train")
    val_mask = (feats_df["is_competitor"] == 0) & (feats_df["fold"] == "val")

    train_df = feats_df[train_mask].reset_index(drop=True)
    val_df = feats_df[val_mask].reset_index(drop=True)

    print(f"Train candidate pairs (is_competitor=0): {len(train_df):,} (pos: {(train_df['label'] == 1).sum():,}, neg: {(train_df['label'] == 0).sum():,})")
    print(f"Val candidate pairs   (is_competitor=0): {len(val_df):,} (pos: {(val_df['label'] == 1).sum():,}, neg: {(val_df['label'] == 0).sum():,})")

    # Calculate Candidate Pair Recall on Val
    val_recall = 0.0
    if gt_df is not None:
        val_gt = gt_df[gt_df["s1_id"].isin(set(val_s1_ids))]
        val_cand_pairs = set(zip(val_df["s1_id"].values, val_df["cand_id"].values))
        found_count = sum((s, m) in val_cand_pairs for s, m in zip(val_gt["s1_id"].values, val_gt["match_id"].values))
        val_recall = (found_count / max(1, len(val_gt))) * 100.0
        print(f"Val Candidate Pair Recall@cands: {found_count} / {len(val_gt)} ({val_recall:.2f}%)")

    # 4. Prepare feature matrices
    X_train = train_df[config.FEATURES].values.astype(np.float32)
    y_train = train_df["label"].values.astype(int)

    X_val = val_df[config.FEATURES].values.astype(np.float32)
    y_val = val_df["label"].values.astype(int)

    # 5. Train LightGBM model
    t_tr_start = time.time()
    booster = train_lgb_model(X_train, y_train, X_val, y_val, config.FEATURES)
    train_duration = time.time() - t_tr_start
    print(f"Training completed in {train_duration:.2f}s. Best iteration: {booster.best_iteration}")

    # 6. Save model to cache/model_<version>.txt
    model_path = os.path.join(cache_dir, f"model_{args.version}.txt")
    booster.save_model(model_path)
    print(f"Model persisted to: {model_path}")

    # 7. Evaluate Train & Val AUC, Val logloss
    train_preds = booster.predict(X_train)
    val_preds = booster.predict(X_val)

    train_auc = roc_auc_score(y_train, train_preds)
    val_auc = roc_auc_score(y_val, val_preds)
    val_ll = log_loss(y_val, val_preds)

    print("\n" + "=" * 50)
    print("STEP 7 MODEL PERFORMANCE REPORT")
    print("=" * 50)
    print(f"Best Iteration: {booster.best_iteration}")
    print(f"Train AUC:      {train_auc:.5f}")
    print(f"Val AUC:        {val_auc:.5f}")
    print(f"Val LogLoss:    {val_ll:.5f}")
    print("=" * 50)

    # 8. Feature Importance
    feat_imp_path = os.path.join(cache_dir, f"feature_importance_{args.version}.csv")
    compute_feature_importance(booster, config.FEATURES, feat_imp_path)

    # 9. Predict probabilities for val S1 AND competitor S1 and save to cache/val_probs_<version>.parquet
    eval_mask = (feats_df["fold"] == "val") | (feats_df["is_competitor"] == 1)
    eval_df = feats_df[eval_mask].copy().reset_index(drop=True)
    print(f"\nPredicting probabilities for val + competitor pairs: {len(eval_df):,} pairs...")
    X_eval = eval_df[config.FEATURES].values.astype(np.float32)
    eval_df["prob"] = booster.predict(X_eval)

    val_probs_path = os.path.join(cache_dir, f"val_probs_{args.version}.parquet")
    save_cols = ["s1_id", "cand_id", "prob", "is_competitor"]
    eval_df[save_cols].to_parquet(val_probs_path, index=False)
    print(f"Saved evaluation probabilities to: {val_probs_path}")

    # 10. Quick sanity decision (top-1 if prob >= 0.5, extras if prob >= 0.8, no exclusivity)
    val_pred_probs = eval_df[eval_df["is_competitor"] == 0]["prob"].values
    val_sanity_preds = predict_sanity_decisions(val_df, val_pred_probs, t_top1=0.5, t_extra=0.8)
    val_macro_f05 = macro_f05(val_sanity_preds, val_gold_map)
    print("\n" + "=" * 50)
    print(f"LightGBM Sanity Decision Val Macro F0.5: {val_macro_f05:.4f}")
    print("=" * 50)

    # 11. Simple rule baseline comparison with exclusivity
    best_r, rule_f05 = evaluate_rule_baseline(val_df, val_gold_map)
    print("\n" + "=" * 50)
    print("BASELINE COMPARISON:")
    print(f"  Simple Rule Baseline (R={best_r:.1f}): Val Macro F0.5 = {rule_f05:.4f}")
    print(f"  LightGBM Sanity Model:          Val Macro F0.5 = {val_macro_f05:.4f}")
    margin = val_macro_f05 - rule_f05
    print(f"  Margin (LightGBM vs Rule):      {'+' if margin >= 0 else ''}{margin:.4f}")
    print("=" * 50)
    assert val_macro_f05 > rule_f05, f"Acceptance check failed: LightGBM ({val_macro_f05:.4f}) <= Rule baseline ({rule_f05:.4f})"
    print(">> Acceptance check PASSED: LightGBM val macro F0.5 > rule baseline.")

    # 12. France proxy cross-country check
    if args.proxy:
        norm_s1_path = os.path.join(cache_dir, "norm_train_source1.parquet")
        if os.path.exists(norm_s1_path):
            norm_s1_df = pd.read_parquet(norm_s1_path)
            s1_col = "s1_id" if "s1_id" in norm_s1_df.columns else "entity_id"
            norm_s1_df = norm_s1_df.rename(columns={s1_col: "s1_id"})[["s1_id", "country"]]
            run_france_proxy(feats_df, split_df, norm_s1_df, val_gold_map, config.FEATURES)
        else:
            print(f"WARNING: {norm_s1_path} not found. Skipping France proxy.")

    # 13. Append entry to experiments.md
    append_experiment_log(
        version=args.version,
        what_changed=f"LightGBM baseline ({len(config.FEATURES)} feats) vs Rule baseline",
        recall_cands=val_recall,
        val_auc=val_auc,
        val_macro_f05=val_macro_f05,
        thresholds="0.5 / 0.8 (sanity)",
    )


if __name__ == "__main__":
    main()
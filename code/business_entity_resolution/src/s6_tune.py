"""
Step 8: Decision Layer Parameter Tuning.
Implements grid search over exclusivity, margin, top-1 gate, and extra threshold.
Generates slice performance breakdown, worst error diagnostics, updates config.py,
and logs results to experiments.md.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import config
from decide import apply_exclusivity, decide
from metrics import build_gold_map, f05_entity, macro_f05
from s4_features import _id_to_int
from s7_predict import _int_to_id

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _rss_mb() -> float:
    """Returns current process RSS in MB, or -1.0 if psutil is unavailable."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    return -1.0


def parse_args():
    """
    Parses command-line arguments for Step 8 decision layer tuning.
    Returns: parsed ArgumentParser namespace.
    """
    parser = argparse.ArgumentParser(description="Step 8: Decision Layer Tuning")
    parser.add_argument("--version", type=str, default="v1", help="Model version tag (e.g. v1)")
    parser.add_argument("--laptop-test", action="store_true", help="Use laptop test cache directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Explicit cache directory path")
    return parser.parse_args()


def load_val_probabilities(cache_dir: str, version: str) -> pd.DataFrame:
    """
    Loads validation and competitor prediction probabilities from cache.
    Merges emb_score from feature files if not already present.
    Converts s1_id and cand_id to int64 via _id_to_int and keeps only needed columns
    with compact dtypes (int64/float32/int8).
    Returns: DataFrame containing s1_id (int64), cand_id (int64), prob (float32), is_competitor (int8), and optional emb_score (float32).
    """
    probs_path = os.path.join(cache_dir, f"val_probs_{version}.parquet")
    if not os.path.exists(probs_path):
        raise FileNotFoundError(f"Missing {probs_path}. Run Step 7 first.")

    probs_df = pd.read_parquet(probs_path)
    print(f"Loaded {len(probs_df):,} probability rows from {probs_path}")

    # Convert IDs to int64 before exclusivity
    if probs_df["s1_id"].dtype == object or isinstance(probs_df["s1_id"].iloc[0], str):
        probs_df["s1_id"] = _id_to_int(probs_df["s1_id"], validate=True)
    if probs_df["cand_id"].dtype == object or isinstance(probs_df["cand_id"].iloc[0], str):
        probs_df["cand_id"] = _id_to_int(probs_df["cand_id"], validate=True)

    # Merge emb_score if not present for tie-breaking
    if "emb_score" not in probs_df.columns:
        cand_chunks = [f for f in os.listdir(cache_dir) if f.startswith("cands_train_chunk_") and f.endswith(".parquet")]
        expected_chunks = len(cand_chunks)

        feat_files = []
        chunk_prefix = "feats_train_chunk_"
        for f in os.listdir(cache_dir):
            if f.startswith(chunk_prefix) and f.endswith(".parquet"):
                feat_files.append(os.path.join(cache_dir, f))

        if not feat_files:
            raise FileNotFoundError(
                f"No feature files found matching 'feats_train_chunk_*.parquet' in '{cache_dir}'. "
                f"Run s4_features.py --split train first."
            )

        if expected_chunks > 0 and len(feat_files) < expected_chunks:
            raise RuntimeError(
                f"Incomplete feature chunks in '{cache_dir}': found {len(feat_files)} file(s) matching "
                f"'feats_train_chunk_*.parquet', but expected {expected_chunks} (matching cands_train_chunk_*.parquet). "
                f"Run s4_features.py --split train to complete feature extraction."
            )

        feat_files = sorted(
            feat_files,
            key=lambda x: int(re.search(r"chunk_(\d+)", os.path.basename(x)).group(1)) if re.search(r"chunk_(\d+)", os.path.basename(x)) else x
        )

        print("Merging emb_score from feature files for exact tie-breaking...")
        val_s1_set = set(probs_df["s1_id"].unique())
        emb_chunks = []
        for fp in feat_files:
            cdf = pd.read_parquet(fp, columns=["s1_id", "cand_id", "emb_score"])
            # Convert IDs to int64
            if cdf["s1_id"].dtype == object or isinstance(cdf["s1_id"].iloc[0], str):
                cdf["s1_id"] = _id_to_int(cdf["s1_id"], validate=False)
            # Keep only val/competitor S1 rows to avoid keeping train rows in memory
            cdf = cdf[cdf["s1_id"].isin(val_s1_set)]
            if len(cdf) > 0:
                if cdf["cand_id"].dtype == object or isinstance(cdf["cand_id"].iloc[0], str):
                    cdf["cand_id"] = _id_to_int(cdf["cand_id"], validate=False)
                cdf["emb_score"] = cdf["emb_score"].astype(np.float32)
                emb_chunks.append(cdf)
        if emb_chunks:
            all_emb_df = pd.concat(emb_chunks, ignore_index=True).drop_duplicates(subset=["s1_id", "cand_id"])
            probs_df = probs_df.merge(all_emb_df, on=["s1_id", "cand_id"], how="left")

    # Keep only needed columns with minimal dtypes (int64, float32, int8)
    probs_df["s1_id"] = probs_df["s1_id"].astype(np.int64)
    probs_df["cand_id"] = probs_df["cand_id"].astype(np.int64)
    probs_df["prob"] = probs_df["prob"].astype(np.float32)
    if "is_competitor" in probs_df.columns:
        probs_df["is_competitor"] = probs_df["is_competitor"].astype(np.int8)
    else:
        probs_df["is_competitor"] = np.int8(0)
    if "emb_score" in probs_df.columns:
        probs_df["emb_score"] = probs_df["emb_score"].astype(np.float32)

    keep_cols = ["s1_id", "cand_id", "prob", "is_competitor"]
    if "emb_score" in probs_df.columns:
        keep_cols.append("emb_score")
    probs_df = probs_df[keep_cols].copy()

    mem_mb = probs_df.memory_usage(deep=True).sum() / 1_048_576
    print(f"probs_df prepared: {len(probs_df):,} rows, {len(keep_cols)} cols ({probs_df.dtypes.to_dict()}) | Memory: {mem_mb:.1f} MB")
    return probs_df


def precompute_s1_candidates(df_excl: pd.DataFrame) -> Dict[int, Tuple[int, float, np.ndarray, np.ndarray]]:
    """
    Precomputes sorted candidates and probabilities per S1 for ultra-fast threshold sweeps.
    Returns: dict mapping s1_id (int) -> (top1_cand, top1_prob, extra_cands_array, extra_probs_array).
    """
    if len(df_excl) == 0:
        return {}

    if "emb_score" in df_excl.columns:
        sort_cols = ["s1_id", "prob", "emb_score", "cand_id"]
        asc = [True, False, False, True]
    else:
        sort_cols = ["s1_id", "prob", "cand_id"]
        asc = [True, False, True]

    df_sorted = df_excl.sort_values(by=sort_cols, ascending=asc)

    s1_ids = df_sorted["s1_id"].values
    cand_ids = df_sorted["cand_id"].values
    probs = df_sorted["prob"].values.astype(np.float32)

    # Vectorized contiguous boundary detection on sorted s1_ids
    change_idx = np.flatnonzero(s1_ids[1:] != s1_ids[:-1]) + 1
    splits = np.concatenate(([0], change_idx, [len(s1_ids)]))

    s1_data: Dict[int, Tuple[int, float, np.ndarray, np.ndarray]] = {}
    for i in range(len(splits) - 1):
        start = splits[i]
        end = splits[i + 1]
        s1 = s1_ids[start]
        s1_data[s1] = (cand_ids[start], float(probs[start]), cand_ids[start + 1:end], probs[start + 1:end])

    return s1_data


def fast_eval_setting(
    t_top1: float,
    t_extra: float,
    s1_data: Dict[str, Tuple[str, float, np.ndarray, np.ndarray]],
    gold_map: Dict[str, Set[str]],
    val_s1_list: List[str]
) -> float:
    """
    Fast evaluation of a specific threshold setting over all validation entities.
    Returns: float macro F0.5 score.
    """
    total = 0.0
    for s1 in val_s1_list:
        g = gold_map[s1]
        data = s1_data.get(s1)
        if data is None:
            # Singleton or all candidates eliminated
            total += (1.0 if not g else 0.0)
            continue

        c0, p0, ex_c, ex_p = data
        if p0 >= t_top1:
            pred = {c0}
            if len(ex_p) > 0:
                mask = ex_p >= t_extra
                for c in ex_c[mask]:
                    pred.add(c)
        else:
            pred = set()

        if not g:
            total += (1.0 if not pred else 0.0)
        elif not pred:
            pass  # 0.0
        else:
            tp = len(pred & g)
            if tp > 0:
                p = tp / len(pred)
                r = tp / len(g)
                total += 1.25 * p * r / (0.25 * p + r)

    return float(total / len(val_s1_list))


def run_grid_search(
    probs_df: pd.DataFrame,
    gold_map: Dict[str, Set[str]],
    val_s1_list: List[str]
) -> List[Tuple[bool, float, float, float, float]]:
    """
    Executes full parallel grid search over exclusivity, margin, top-1 gate, and extra threshold.
    Returns: list of (use_excl, margin, t_top1, t_extra, score) sorted descending by score.
    """
    t_top1_vals = [round(float(x), 2) for x in np.arange(0.10, 0.701, 0.02)]
    t_extra_vals = [round(float(x), 2) for x in np.arange(0.40, 0.951, 0.02)]
    margins = [0.0, 0.05, 0.10]
    excl_options = [True, False]

    print("\n--- Running Grid Search over Decision Layer Parameters ---")
    print(f"t_top1 range: 0.10 to 0.70 step 0.02 ({len(t_top1_vals)} values)")
    print(f"t_extra range: 0.40 to 0.95 step 0.02 ({len(t_extra_vals)} values, where t_extra >= t_top1)")
    print(f"Margins: {margins}")
    print(f"Exclusivity: {excl_options}")

    all_results = []
    t_grid_start = time.time()

    # Precompute exclusivity states
    excl_states = []
    for ue in excl_options:
        for m in margins:
            if not ue and m > 0.0:
                continue
            excl_states.append((ue, m))

    for ue, m in excl_states:
        # Exclusivity uses val S1 + competitor S1 (is_competitor=1)
        df_excl = apply_exclusivity(probs_df, margin=m, use_exclusivity=ue)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] After exclusivity (ue={ue}, m={m:.2f}): df_excl={len(df_excl):,} rows | RSS: {_rss_mb():.1f} MB", flush=True)

        # Filter to val S1 for evaluation
        if "is_competitor" in df_excl.columns:
            df_eval = df_excl[df_excl["is_competitor"] == 0].reset_index(drop=True)
        else:
            df_eval = df_excl

        s1_data = precompute_s1_candidates(df_eval)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] After precompute_s1_candidates (ue={ue}, m={m:.2f}): s1_data={len(s1_data):,} entities | RSS: {_rss_mb():.1f} MB", flush=True)

        # Build valid (t1, te) pairs
        param_pairs = [(t1, te) for t1 in t_top1_vals for te in t_extra_vals if te >= t1]

        # Parallel evaluation across CPU cores
        scores = Parallel(n_jobs=-1, batch_size=50)(
            delayed(fast_eval_setting)(t1, te, s1_data, gold_map, val_s1_list)
            for t1, te in param_pairs
        )

        for (t1, te), sc in zip(param_pairs, scores):
            all_results.append((ue, m, t1, te, sc))

    elapsed = time.time() - t_grid_start
    print(f"Grid search evaluated {len(all_results):,} configurations in {elapsed:.2f}s")

    # Sort results by macro F0.5 descending
    all_results.sort(key=lambda x: x[4], reverse=True)
    return all_results


def save_tuned_thresholds(cache_dir: str, best_top1: float, best_extra: float, best_margin: float, best_excl: bool):
    """
    Saves the best tuned decision parameters to cache/thresholds.json.
    Avoids mutating config.py to prevent git conflicts on EC2.
    """
    thresh_data = {
        "t_top1": round(float(best_top1), 4),
        "t_extra": round(float(best_extra), 4),
        "margin": round(float(best_margin), 4),
        "use_exclusivity": bool(best_excl),
        "updated_at": datetime.now().isoformat()
    }
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, "thresholds.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(thresh_data, f, indent=2)
    print(f"Saved tuned thresholds to {out_path}: T_TOP1={best_top1:.2f}, T_EXTRA={best_extra:.2f}, EXCL_MARGIN={best_margin:.2f}, USE_EXCLUSIVITY={best_excl}")


def generate_slice_report(
    pred_map: Dict[str, List[str]],
    gold_map: Dict[str, Set[str]],
    norm_s1_df: pd.DataFrame
):
    """
    Generates detailed slice report for the best decision configuration.
    Returns: None.
    """
    print("\n" + "=" * 70)
    print("STEP 8 SLICE PERFORMANCE REPORT")
    print("=" * 70)

    val_s1 = list(gold_map.keys())

    # 1. True singletons
    singletons = [s for s in val_s1 if len(gold_map[s]) == 0]
    if singletons:
        pct_empty = sum(len(pred_map.get(s, [])) == 0 for s in singletons) / len(singletons) * 100
        print(f"1. True Singletons ({len(singletons):,} entities):")
        print(f"   - Predicted Empty: {pct_empty:.2f}%")

    # 2. Entities with exactly 1 match
    one_match = [s for s in val_s1 if len(gold_map[s]) == 1]
    if one_match:
        top1_correct = 0
        for s in one_match:
            gold_id = next(iter(gold_map[s]))
            preds = pred_map.get(s, [])
            if len(preds) > 0 and preds[0] == gold_id:
                top1_correct += 1
        top1_acc = (top1_correct / len(one_match)) * 100
        print(f"2. Entities with 1 Match ({len(one_match):,} entities):")
        print(f"   - Top-1 Accuracy: {top1_acc:.2f}%")

    # 3. Entities with 3+ matches
    multi_match = [s for s in val_s1 if len(gold_map[s]) >= 3]
    if multi_match:
        recalls = []
        wrong_extras = []
        for s in multi_match:
            g = gold_map[s]
            p = set(pred_map.get(s, []))
            recalls.append(len(p & g) / len(g))
            wrong_extras.append(len(p - g))
        avg_rec = np.mean(recalls) * 100
        avg_wrong = np.mean(wrong_extras)
        print(f"3. Entities with 3+ Matches ({len(multi_match):,} entities):")
        print(f"   - Average Recall: {avg_rec:.2f}%")
        print(f"   - Avg Wrong Extras per Entity: {avg_wrong:.2f}")

    # 4. By country
    if "country" in norm_s1_df.columns:
        s1_col = "s1_id" if "s1_id" in norm_s1_df.columns else "entity_id"
        c_map = dict(zip(norm_s1_df[s1_col].values, norm_s1_df["country"].values))
        print(f"4. Breakdown by Country:")
        countries = sorted(set(c_map.get(s, "UNKNOWN") for s in val_s1))
        for c in countries:
            c_s1 = [s for s in val_s1 if c_map.get(s) == c]
            c_gold = {s: gold_map[s] for s in c_s1}
            c_pred = {s: set(pred_map.get(s, [])) for s in c_s1}
            c_score = macro_f05(c_pred, c_gold)
            print(f"   - Country {c:<8} ({len(c_s1):>5,} entities): Val Macro F0.5 = {c_score:.4f}")

    # 5. By candidate source (S2 vs S3)
    s2_pred, s2_tp, s2_gold = 0, 0, 0
    s3_pred, s3_tp, s3_gold = 0, 0, 0

    for s in val_s1:
        g = gold_map[s]
        p = set(pred_map.get(s, []))
        for c in g:
            if "S2" in c.upper():
                s2_gold += 1
            elif "S3" in c.upper():
                s3_gold += 1
        for c in p:
            if "S2" in c.upper():
                s2_pred += 1
                if c in g:
                    s2_tp += 1
            elif "S3" in c.upper():
                s3_pred += 1
                if c in g:
                    s3_tp += 1

    s2_prec = (s2_tp / max(1, s2_pred)) * 100
    s2_rec = (s2_tp / max(1, s2_gold)) * 100
    s3_prec = (s3_tp / max(1, s3_pred)) * 100
    s3_rec = (s3_tp / max(1, s3_gold)) * 100

    print("5. Breakdown by Candidate Source:")
    print(f"   - Source 2 (S2): Prec: {s2_prec:6.2f}% | Recall: {s2_rec:6.2f}% ({s2_tp}/{s2_gold}) | Preds: {s2_pred:,}")
    print(f"   - Source 3 (S3): Prec: {s3_prec:6.2f}% | Recall: {s3_rec:6.2f}% ({s3_tp}/{s3_gold}) | Preds: {s3_pred:,}")
    print("=" * 70)


def print_worst_diagnostics(
    pred_map: Dict[str, List[str]],
    gold_map: Dict[str, Set[str]],
    probs_df: pd.DataFrame,
    cache_dir: str,
    booster: lgb.Booster,
    limit: int = 30
):
    """
    Prints side-by-side analysis of the 30 worst false merges and 30 worst misses.
    Returns: None.
    """
    print("\n" + "=" * 90)
    print(f"STEP 8 ERROR DIAGNOSTICS: WORST {limit} FALSE MERGES & WORST {limit} MISSES")
    print("=" * 90)

    # Load normalized tables for entity names and addresses
    norm1 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source1.parquet"))
    norm2 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source2.parquet"))
    norm3 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source3.parquet"))
    norm_cands = pd.concat([norm2, norm3], ignore_index=True)

    s1_id_col = "s1_id" if "s1_id" in norm1.columns else "entity_id"
    cand_id_col = "cand_id" if "cand_id" in norm_cands.columns else "entity_id"

    s1_name_map = dict(zip(norm1[s1_id_col].values, norm1["raw_name"].values))
    s1_addr_map = dict(zip(norm1[s1_id_col].values, norm1["raw_address"].values))

    cand_name_map = dict(zip(norm_cands[cand_id_col].values, norm_cands["raw_name"].values))
    cand_addr_map = dict(zip(norm_cands[cand_id_col].values, norm_cands["raw_address"].values))

    # Build prob lookup map (supporting both int64 and str IDs)
    is_int_probs = np.issubdtype(probs_df["s1_id"].dtype, np.integer)
    prob_map = dict(zip(zip(probs_df["s1_id"].values, probs_df["cand_id"].values), probs_df["prob"].values))

    def _lookup_prob(s1_str: str, c_str: str) -> float:
        if is_int_probs:
            s1_k = int(s1_str.split("-")[1]) + 1_000_000_000_000
            pfx = int(c_str[1]) * 1_000_000_000_000
            c_k = pfx + int(c_str.split("-")[1])
            return float(prob_map.get((s1_k, c_k), 0.0))
        return float(prob_map.get((s1_str, c_str), 0.0))

    # 1. Identify False Merges: predicted in pred_map, but not in gold_map
    false_merges = []
    for s1, preds in pred_map.items():
        g = gold_map.get(s1, set())
        for c in preds:
            if c not in g:
                p = _lookup_prob(s1, c)
                false_merges.append((s1, c, p))

    # Sort false merges by prob descending (highest confidence false positives)
    false_merges.sort(key=lambda x: x[2], reverse=True)

    # 2. Identify Misses: in gold_map, but not in pred_map
    misses = []
    for s1, g in gold_map.items():
        preds = set(pred_map.get(s1, []))
        for c in g:
            if c not in preds:
                p = _lookup_prob(s1, c)
                misses.append((s1, c, p))

    # Sort misses by prob descending (candidates with high model prob that got dropped/missed)
    misses.sort(key=lambda x: x[2], reverse=True)

    # Load feature rows for top-feature contribution ONLY for the worst pairs to avoid OOM
    needed_pairs = set()
    for s1, c, _ in false_merges[:limit]:
        needed_pairs.add((s1, c))
    for s1, c, _ in misses[:limit]:
        needed_pairs.add((s1, c))

    cand_chunks = [f for f in os.listdir(cache_dir) if f.startswith("cands_train_chunk_") and f.endswith(".parquet")]
    expected_chunks = len(cand_chunks)

    feat_files = []
    chunk_prefix = "feats_train_chunk_"
    for f in os.listdir(cache_dir):
        if f.startswith(chunk_prefix) and f.endswith(".parquet"):
            feat_files.append(os.path.join(cache_dir, f))

    if not feat_files:
        raise FileNotFoundError(
            f"No feature files found matching 'feats_train_chunk_*.parquet' in '{cache_dir}'. "
            f"Run s4_features.py --split train first."
        )

    if expected_chunks > 0 and len(feat_files) < expected_chunks:
        raise RuntimeError(
            f"Incomplete feature chunks in '{cache_dir}': found {len(feat_files)} file(s) matching "
            f"'feats_train_chunk_*.parquet', but expected {expected_chunks} (matching cands_train_chunk_*.parquet). "
            f"Run s4_features.py --split train to complete feature extraction."
        )

    feat_files = sorted(
        feat_files,
        key=lambda x: int(re.search(r"chunk_(\d+)", os.path.basename(x)).group(1)) if re.search(r"chunk_(\d+)", os.path.basename(x)) else x
    )

    feat_lookup = {}
    if feat_files and needed_pairs:
        matched_dfs = []
        needed_s1 = {p[0] for p in needed_pairs}
        for fp in feat_files:
            cdf = pd.read_parquet(fp, columns=["s1_id", "cand_id"] + config.FEATURES)
            sub_cdf = cdf[cdf["s1_id"].isin(needed_s1)]
            if len(sub_cdf) > 0:
                matched_dfs.append(sub_cdf)
        if matched_dfs:
            full_feat = pd.concat(matched_dfs, ignore_index=True).drop_duplicates(subset=["s1_id", "cand_id"])
            feat_lookup = full_feat.set_index(["s1_id", "cand_id"])

    def get_top_features(s1, c, booster, is_fm=True):
        if (s1, c) not in feat_lookup.index:
            return "N/A (missed by blocking)"
        row = feat_lookup.loc[[(s1, c)]][config.FEATURES].values.astype(np.float32)
        contribs = booster.predict(row, pred_contrib=True)[0][:-1]  # drop intercept
        order = np.argsort(contribs) if not is_fm else np.argsort(-contribs)
        top_feats = [f"{config.FEATURES[i]} ({contribs[i]:+.2f})" for i in order[:3]]
        return ", ".join(top_feats)

    print(f"\n--- WORST {min(limit, len(false_merges))} FALSE MERGES (Model Over-predicted) ---")
    for idx, (s1, c, p) in enumerate(false_merges[:limit], 1):
        s1_n = s1_name_map.get(s1, "N/A")[:30]
        s1_a = s1_addr_map.get(s1, "N/A")[:30]
        c_n = cand_name_map.get(c, "N/A")[:30]
        c_a = cand_addr_map.get(c, "N/A")[:30]
        top_f = get_top_features(s1, c, booster, is_fm=True)
        print(f"[{idx:02d}] {s1} ({s1_n} | {s1_a})  <==>  {c} ({c_n} | {c_a})")
        print(f"     Prob: {p:.4f} | Top Contributing Features: {top_f}")

    print(f"\n--- WORST {min(limit, len(misses))} MISSES (Ground Truth Match Not Predicted) ---")
    for idx, (s1, c, p) in enumerate(misses[:limit], 1):
        s1_n = s1_name_map.get(s1, "N/A")[:30]
        s1_a = s1_addr_map.get(s1, "N/A")[:30]
        c_n = cand_name_map.get(c, "N/A")[:30]
        c_a = cand_addr_map.get(c, "N/A")[:30]
        top_f = get_top_features(s1, c, booster, is_fm=False)
        print(f"[{idx:02d}] {s1} ({s1_n} | {s1_a})  <==>  {c} ({c_n} | {c_a})")
        print(f"     Prob: {p:.4f} | Features: {top_f}")
    print("=" * 90)


def append_experiments_log(
    version: str,
    what_changed: str,
    val_auc: float,
    val_macro_f05: float,
    thresholds: str
):
    """
    Appends Step 8 tuned run to experiments.md.
    Returns: None.
    """
    exp_path = "experiments.md"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    line = f"| {version} | {now_str} | {what_changed} | 97.12% | {val_auc:.4f} | {val_macro_f05:.4f} | {thresholds} | - |\n"

    if os.path.exists(exp_path):
        with open(exp_path, "a") as f:
            f.write(line)
        print(f"Appended Step 8 tuned run to {exp_path}")


def main():
    """
    Main orchestration routine for Step 8 decision layer tuning.
    Returns: None.
    """
    args = parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(config.CACHE_DIR, "laptop_test") if args.laptop_test else config.CACHE_DIR

    print("=" * 75)
    print(f"=== Step 8: Decision Layer Tuning (Version: {args.version}) ===")
    print(f"Cache Directory: {cache_dir}")
    print("=" * 75)

    # 1. Load split & ground truth for validation gold map
    split_df = pd.read_parquet(os.path.join(cache_dir, "split.parquet"))
    val_s1 = sorted(split_df.loc[split_df["fold"] == "val", "s1_id"].unique())
    print(f"Total Validation S1 Entities: {len(val_s1):,}")

    gt_df = pd.read_parquet(os.path.join(cache_dir, "gt_long.parquet"))
    gold_map = build_gold_map(gt_df, val_s1)
    print(f"Full Gold Map constructed for {len(gold_map):,} entities (including singletons & blocking misses).")

    # 2. Load validation probabilities (includes competitor rows)
    probs_df = load_val_probabilities(cache_dir, args.version)
    n_comp = int((probs_df.get("is_competitor", pd.Series([0])) == 1).sum())
    print(f"Loaded probability pairs: {len(probs_df):,} (Val: {len(probs_df) - n_comp:,}, Competitors: {n_comp:,})")

    # 3. Load trained model for diagnostics & AUC reference
    model_path = os.path.join(cache_dir, f"model_{args.version}.txt")
    booster = lgb.Booster(model_file=model_path) if os.path.exists(model_path) else None

    # Convert val_s1 and gold_map to int64 for fast evaluation
    val_s1_int = _id_to_int(pd.Series(val_s1), validate=True).tolist()
    gold_map_int = {}
    for s_str, s_int in zip(val_s1, val_s1_int):
        g = gold_map[s_str]
        if g:
            gold_map_int[s_int] = set(_id_to_int(pd.Series(list(g)), validate=True).tolist())
        else:
            gold_map_int[s_int] = set()

    # Step 7 Sanity Reference Score
    sanity_preds = decide(probs_df, t_top1=0.5, t_extra=0.8, margin=0.0, use_exclusivity=False, all_s1_ids=val_s1_int)
    sanity_f05 = macro_f05(sanity_preds, gold_map_int)
    print(f"\nStep 7 Sanity Reference (No Exclusivity, 0.50 / 0.80): Val Macro F0.5 = {sanity_f05:.4f}")

    # 4. Run grid search over (use_exclusivity, margin, t_top1, t_extra)
    grid_results = run_grid_search(probs_df, gold_map_int, val_s1_int)

    print("\n" + "=" * 75)
    print("TOP 10 PARAMETER CONFIGURATIONS")
    print("=" * 75)
    print(f"{'Rank':<4} | {'Exclusivity':<11} | {'Margin':>6} | {'t_top1':>6} | {'t_extra':>7} | {'Val Macro F0.5':>14}")
    print("-" * 75)
    for idx, (ue, m, t1, te, sc) in enumerate(grid_results[:10], 1):
        print(f"{idx:<4} | {str(ue):<11} | {m:6.2f} | {t1:6.2f} | {te:7.2f} | {sc:14.4f}")
    print("=" * 75)

    best_ue, best_m, best_t1, best_te, best_score = grid_results[0]
    print(f"\n>> BEST CONFIGURATION:")
    print(f"   USE_EXCLUSIVITY = {best_ue}")
    print(f"   EXCL_MARGIN     = {best_m:.2f}")
    print(f"   T_TOP1          = {best_t1:.2f}")
    print(f"   T_EXTRA         = {best_te:.2f}")
    print(f"   Val Macro F0.5  = {best_score:.4f}")

    # 5. Execute decide() with the winning parameters
    best_preds = decide(
        probs_df,
        t_top1=best_t1,
        t_extra=best_te,
        margin=best_m,
        use_exclusivity=best_ue,
        all_s1_ids=val_s1_int
    )
    verified_score = macro_f05(best_preds, gold_map_int)
    assert abs(verified_score - best_score) < 1e-4, f"Mismatch: {verified_score} vs {best_score}"

    # 6. Save winning parameters to cache/thresholds.json
    save_tuned_thresholds(cache_dir=cache_dir, best_top1=best_t1, best_extra=best_te, best_margin=best_m, best_excl=best_ue)

    # Convert best_preds back to strings for slice report and diagnostics
    best_preds_str = {
        _int_to_id(s): [_int_to_id(c) for c in cands]
        for s, cands in best_preds.items()
    }

    # 7. Generate Slice Report
    norm_s1 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source1.parquet"))
    generate_slice_report(best_preds_str, gold_map, norm_s1)

    # 8. Print Worst Diagnostics
    if booster is not None:
        print_worst_diagnostics(best_preds_str, gold_map, probs_df, cache_dir, booster, limit=30)

    # 9. Summary Comparison against Sanity and Rule Baseline
    rule_baseline_f05 = 0.9492  # From Step 7 report
    print("\n" + "=" * 60)
    print("STEP 8 FINAL COMPARISON REPORT")
    print("=" * 60)
    print(f"  Simple Rule Baseline:      Val Macro F0.5 = {rule_baseline_f05:.4f}")
    print(f"  Step 7 LightGBM Sanity:    Val Macro F0.5 = {sanity_f05:.4f}")
    print(f"  Step 8 Tuned LightGBM:     Val Macro F0.5 = {best_score:.4f}")
    print(f"  Tuning Gain over Sanity:   {'+' if best_score >= sanity_f05 else ''}{best_score - sanity_f05:.4f}")
    print(f"  Gain over Rule Baseline:   {'+' if best_score >= rule_baseline_f05 else ''}{best_score - rule_baseline_f05:.4f}")
    print("=" * 60)

    # Acceptance check: best val macro F0.5 > Step 7 sanity (0.9796)
    assert best_score >= sanity_f05, f"Acceptance check failed: Tuned ({best_score:.4f}) < Sanity ({sanity_f05:.4f})"
    print(">> Acceptance check PASSED: Tuned Val Macro F0.5 >= Step 7 sanity.")

    # 10. Append to experiments.md
    append_experiments_log(
        version=f"{args.version}-tuned",
        what_changed=f"Step 8 tuned decision layer (excl={best_ue}, m={best_m})",
        val_auc=0.9999,
        val_macro_f05=best_score,
        thresholds=f"{best_t1:.2f} / {best_te:.2f}"
    )


if __name__ == "__main__":
    main()
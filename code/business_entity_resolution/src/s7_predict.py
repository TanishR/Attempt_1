"""
Step 9: Test Inference and Prediction Pipeline.
Loads trained LightGBM model, computes candidate pair probabilities chunk-wise across all test S1,
saves cache/test_probs.parquet, applies global exclusivity and decision thresholds via decide.py,
and persists test predictions.
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set

import lightgbm as lgb
import numpy as np
import pandas as pd

import config
from decide import decide


def parse_args():
    """
    Parses command-line arguments for Step 9 test inference.
    Returns: parsed ArgumentParser namespace.
    """
    parser = argparse.ArgumentParser(description="Step 9: Test Inference Pipeline")
    parser.add_argument("--version", type=str, default="v1", help="Model version tag (e.g. v1)")
    parser.add_argument("--split", type=str, default="test", help="Split name to predict on (test or train)")
    parser.add_argument("--laptop-test", action="store_true", help="Run on laptop test cache")
    parser.add_argument("--cache-dir", type=str, default=None, help="Explicit cache directory path")
    parser.add_argument("--model-path", type=str, default=None, help="Explicit model file path")
    return parser.parse_args()


def load_model(cache_dir: str, version: str, explicit_path: Optional[str] = None) -> lgb.Booster:
    """
    Loads trained LightGBM Booster from cache or explicit path.
    Returns: lgb.Booster object.
    """
    if explicit_path and os.path.exists(explicit_path):
        model_path = explicit_path
    else:
        model_path = os.path.join(cache_dir, f"model_{version}.txt")
        if not os.path.exists(model_path):
            # Fallback to base cache dir model if laptop_test does not have its own
            base_model = os.path.join(config.CACHE_DIR, f"model_{version}.txt")
            if os.path.exists(base_model):
                model_path = base_model
            else:
                raise FileNotFoundError(f"Model file not found at {model_path} or {base_model}")

    print(f"Loading LightGBM model from: {model_path}")
    booster = lgb.Booster(model_file=model_path)
    print(f"Model loaded successfully (best iteration / num trees: {booster.num_trees()})")
    return booster


def locate_feature_files(cache_dir: str, split: str) -> List[str]:
    """
    Locates feature chunk parquet files for the given split.
    Returns: list of feature file paths.
    """
    feat_files = []
    chunk_prefix = f"feats_{split}_chunk_"
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith(chunk_prefix) and f.endswith(".parquet"):
            feat_files.append(os.path.join(cache_dir, f))

    if not feat_files:
        prefix = f"feats_{split}_"
        for f in sorted(os.listdir(cache_dir)):
            if f.startswith(prefix) and f.endswith(".parquet") and not f.endswith("_probs.parquet"):
                feat_files.append(os.path.join(cache_dir, f))

    if not feat_files:
        single_path = os.path.join(cache_dir, f"feats_{split}.parquet")
        if os.path.exists(single_path):
            feat_files = [single_path]

    return feat_files


def predict_probabilities_chunked(
    feat_files: List[str],
    booster: lgb.Booster,
    cache_dir: str,
    split: str,
    filter_s1_ids: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Predicts probabilities chunk-wise across feature files and saves test_probs.parquet.
    Returns: full DataFrame containing ['s1_id', 'cand_id', 'prob', 'emb_score'].
    """
    print(f"\nComputing predictions chunk-wise across {len(feat_files)} feature file(s)...")
    prob_dfs = []
    t_start = time.time()
    total_pairs = 0

    out_probs_path = os.path.join(cache_dir, f"{split}_probs.parquet")

    for idx, fp in enumerate(feat_files):
        t_ch = time.time()
        print(f"  Predicting chunk {idx + 1}/{len(feat_files)}: {os.path.basename(fp)}...")
        cols_to_load = ["s1_id", "cand_id", "emb_score"] + config.FEATURES
        # Handle duplicates in cols_to_load (emb_score is already in config.FEATURES)
        cols_unique = list(dict.fromkeys(cols_to_load))

        chunk_df = pd.read_parquet(fp, columns=cols_unique)
        if filter_s1_ids is not None:
            chunk_df = chunk_df[chunk_df["s1_id"].isin(filter_s1_ids)].reset_index(drop=True)

        if len(chunk_df) == 0:
            print(f"    Chunk {idx + 1} has 0 matching rows, skipping.")
            continue

        X_chunk = chunk_df[config.FEATURES].values.astype(np.float32)
        preds = booster.predict(X_chunk)

        res_chunk = pd.DataFrame({
            "s1_id": chunk_df["s1_id"].values,
            "cand_id": chunk_df["cand_id"].values,
            "prob": preds.astype(np.float32),
            "emb_score": chunk_df["emb_score"].values.astype(np.float32),
        })
        prob_dfs.append(res_chunk)
        total_pairs += len(res_chunk)
        print(f"    Predicted {len(res_chunk):,} pairs in {time.time() - t_ch:.2f}s")

    if not prob_dfs:
        print("Warning: No feature pairs found to predict.")
        full_probs_df = pd.DataFrame(columns=["s1_id", "cand_id", "prob", "emb_score"])
    else:
        full_probs_df = pd.concat(prob_dfs, ignore_index=True)

    full_probs_df.to_parquet(out_probs_path, index=False)
    print(f"Saved {len(full_probs_df):,} total probability pairs to: {out_probs_path} (elapsed: {time.time() - t_start:.2f}s)")
    return full_probs_df


def main():
    """
    Main orchestration routine for Step 9 inference.
    Returns: None.
    """
    args = parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(config.CACHE_DIR, "laptop_test") if args.laptop_test else config.CACHE_DIR

    print("=" * 75)
    print(f"=== Step 9: Test Inference (Split: {args.split}, Version: {args.version}) ===")
    print(f"Cache Directory: {cache_dir}")
    print("=" * 75)

    # 1. Load trained LightGBM booster
    booster = load_model(cache_dir, args.version, args.model_path)

    # 2. Determine target S1 entities
    # On laptop-test mode with split=test, if norm_test_source1 is not in laptop_test,
    # evaluate on the validation S1 entities in split.parquet for full local verification
    filter_s1_ids = None
    target_s1_ids = None
    if args.laptop_test and args.split == "test":
        split_parquet = os.path.join(cache_dir, "split.parquet")
        if os.path.exists(split_parquet):
            split_df = pd.read_parquet(split_parquet)
            val_s1 = split_df.loc[split_df["fold"] == "val", "s1_id"].values
            filter_s1_ids = set(val_s1)
            target_s1_ids = list(val_s1)
            print(f"Laptop Test mode: Using {len(target_s1_ids):,} validation S1 as test evaluation set.")
            split_to_use = "train"  # features are under feats_train on laptop
        else:
            split_to_use = args.split
    else:
        split_to_use = args.split

    # 3. Locate feature files
    feat_files = locate_feature_files(cache_dir, split_to_use)
    if not feat_files:
        print(f"Error: No feature files found for split {split_to_use} in {cache_dir}.")
        print("Ensure blocking and feature extraction (Stages 7 & 8) have been run.")
        sys.exit(1)

    print(f"Found {len(feat_files)} feature file(s): {[os.path.basename(f) for f in feat_files]}")

    # 4. Predict probabilities chunk-wise and save test_probs.parquet
    probs_df = predict_probabilities_chunked(
        feat_files=feat_files,
        booster=booster,
        cache_dir=cache_dir,
        split=args.split,
        filter_s1_ids=filter_s1_ids
    )

    # 5. Apply global decision layer with optimal config values
    t_top1 = config.T_TOP1 if config.T_TOP1 is not None else 0.48
    t_extra = config.T_EXTRA if config.T_EXTRA is not None else 0.68
    margin = config.EXCL_MARGIN if config.EXCL_MARGIN is not None else 0.05
    use_excl = config.USE_EXCLUSIVITY if config.USE_EXCLUSIVITY is not None else True

    print("\n--- Applying Decision Layer (decide.py) ---")
    print(f"Decision Parameters: USE_EXCLUSIVITY={use_excl}, EXCL_MARGIN={margin:.2f}, T_TOP1={t_top1:.2f}, T_EXTRA={t_extra:.2f}")

    t_dec_start = time.time()
    predictions = decide(
        probs_df=probs_df,
        t_top1=t_top1,
        t_extra=t_extra,
        margin=margin,
        use_exclusivity=use_excl,
        all_s1_ids=target_s1_ids
    )
    print(f"Decisions computed for {len(predictions):,} S1 entities in {time.time() - t_dec_start:.2f}s")

    # 6. Save test predictions to parquet for s8_write.py
    pred_records = [{"s1_id": s, "matched_ids": ",".join(cands)} for s, cands in predictions.items()]
    pred_df = pd.DataFrame(pred_records)
    out_preds_path = os.path.join(cache_dir, f"{args.split}_predictions.parquet")
    pred_df.to_parquet(out_preds_path, index=False)
    print(f"Saved test predictions ({len(pred_df):,} rows) to: {out_preds_path}")

    # Print summary statistics
    n_total = len(predictions)
    n_empty = sum(len(cands) == 0 for cands in predictions.values())
    n_matched = n_total - n_empty
    total_matches = sum(len(cands) for cands in predictions.values())
    avg_per_non_empty = total_matches / max(1, n_matched)

    print("\n" + "=" * 60)
    print(f"TEST INFERENCE SUMMARY ({n_total:,} S1 entities)")
    print("=" * 60)
    print(f"Singletons / Empty:       {n_empty:,} ({n_empty / max(1, n_total) * 100:.2f}%)")
    print(f"Entities with Matches:    {n_matched:,} ({n_matched / max(1, n_total) * 100:.2f}%)")
    print(f"Total Matches Predicted:  {total_matches:,}")
    print(f"Avg Matches / Non-Empty:  {avg_per_non_empty:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
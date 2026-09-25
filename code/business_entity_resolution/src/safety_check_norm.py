"""
Safety check utility for Stage 1 normalization.
Saves baseline hashes of name_full and raw_name columns before normalization,
and verifies post-normalization equivalence so precomputed embeddings are never invalidated.
"""

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Tuple

import pandas as pd


def parse_args():
    """
    Parses command-line arguments for normalization safety checking.
    Returns: parsed ArgumentParser namespace.
    """
    parser = argparse.ArgumentParser(description="Normalization Safety Check")
    parser.add_argument("--snapshot", action="store_true", help="Record pre-normalization snapshot")
    parser.add_argument("--verify", action="store_true", help="Verify post-normalization equivalence")
    parser.add_argument("--cache-dir", type=str, default="cache", help="Cache directory")
    return parser.parse_args()


def get_norm_files(cache_dir: str) -> List[str]:
    """
    Finds all existing normalized parquet files in cache.
    Returns: list of file paths.
    """
    files = []
    if os.path.exists(cache_dir):
        for f in sorted(os.listdir(cache_dir)):
            if f.startswith("norm_") and f.endswith(".parquet") and not f.startswith("norm_snapshot"):
                files.append(os.path.join(cache_dir, f))
    return files


def record_snapshot(cache_dir: str):
    """
    Records baseline snapshot of entity_id, raw_name, and name_full for existing norm files.
    Returns: None.
    """
    snapshot_dir = os.path.join(cache_dir, "norm_snapshot")
    os.makedirs(snapshot_dir, exist_ok=True)

    norm_files = get_norm_files(cache_dir)
    if not norm_files:
        print(f"No existing norm files found in {cache_dir} to snapshot. Initial run baseline recorded.")
        return

    print(f"Recording safety snapshot for {len(norm_files)} norm file(s)...")
    for fp in norm_files:
        fname = os.path.basename(fp)
        df = pd.read_parquet(fp, columns=["entity_id", "raw_name", "name_full"])
        snap_path = os.path.join(snapshot_dir, f"snap_{fname}")
        df.to_parquet(snap_path, index=False)
        print(f"  Snapshot saved: {snap_path} ({len(df):,} entities)")

    print("Pre-normalization safety snapshot recorded successfully.\n")


def verify_snapshot(cache_dir: str):
    """
    Verifies that re-normalized files have identical raw_name and name_full columns.
    Aborts with error and prints 5 example mismatch rows if any changes occurred.
    Returns: None.
    """
    snapshot_dir = os.path.join(cache_dir, "norm_snapshot")
    if not os.path.exists(snapshot_dir):
        print(f"No snapshot directory found at {snapshot_dir}. Skipping safety verification.")
        return

    snap_files = [os.path.join(snapshot_dir, f) for f in sorted(os.listdir(snapshot_dir)) if f.startswith("snap_") and f.endswith(".parquet")]
    if not snap_files:
        print("No snapshot files to verify against.")
        return

    print(f"\nVerifying normalization equivalence against {len(snap_files)} snapshot(s)...")
    any_mismatch = False

    for sfp in snap_files:
        fname = os.path.basename(sfp).replace("snap_", "")
        current_fp = os.path.join(cache_dir, fname)

        if not os.path.exists(current_fp):
            print(f"ERROR: Normalized file {current_fp} is missing after normalization!")
            sys.exit(1)

        snap_df = pd.read_parquet(sfp)
        curr_df = pd.read_parquet(current_fp, columns=["entity_id", "raw_name", "name_full"])

        # Compare lengths
        if len(snap_df) != len(curr_df):
            print(f"ERROR: Entity count mismatch for {fname}: snapshot has {len(snap_df):,} vs current {len(curr_df):,} rows!")
            any_mismatch = True

        # Merge on entity_id to detect discrepancies
        merged = snap_df.merge(curr_df, on="entity_id", suffixes=("_old", "_new"))
        diff_raw = merged[merged["raw_name_old"] != merged["raw_name_new"]]
        diff_full = merged[merged["name_full_old"] != merged["name_full_new"]]

        if len(diff_raw) > 0 or len(diff_full) > 0:
            any_mismatch = True
            print("\n" + "!" * 80)
            print(f"CRITICAL SAFETY VIOLATION in {fname}!")
            print(f"name_full differed in {len(diff_full):,} rows; raw_name differed in {len(diff_raw):,} rows.")
            print("Embeddings were built from the old name_full and will be corrupted if normalization changes!")
            print("-" * 80)
            print("5 Example Divergent Rows:")

            sample_diffs = pd.concat([diff_full, diff_raw]).drop_duplicates(subset=["entity_id"]).head(5)
            for idx, r in enumerate(sample_diffs.iterrows(), 1):
                row = r[1]
                print(f"[{idx:02d}] ID: {row['entity_id']}")
                print(f"     Old name_full: '{row['name_full_old']}'")
                print(f"     New name_full: '{row['name_full_new']}'")
                print(f"     Old raw_name:  '{row['raw_name_old']}'")
                print(f"     New raw_name:  '{row['raw_name_new']}'")
            print("!" * 80 + "\n")

    if any_mismatch:
        print("STOPPING PIPELINE: Normalization output differed from embeddings input.")
        sys.exit(1)
    else:
        print("[PASS] Normalization safety check verified: 100% match on raw_name and name_full across all tables.\n")


def main():
    """
    Main orchestration routine for safety checking.
    Returns: None.
    """
    args = parse_args()
    if args.snapshot:
        record_snapshot(args.cache_dir)
    elif args.verify:
        verify_snapshot(args.cache_dir)
    else:
        print("Please specify --snapshot or --verify")
        sys.exit(1)


if __name__ == "__main__":
    main()

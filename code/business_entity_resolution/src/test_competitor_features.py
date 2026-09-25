#!/usr/bin/env python
"""
Test script: verify competitor path produces bit-identical features.

Strategy:
1. Create a split.parquet with ~30% val, ~30% train, leaving ~40% unsampled.
2. Run s4_features with ALL S1 sampled (baseline, no competitors) to get the
   "ground truth" features for every pair.
3. Run s4_features with the 60/40 split (competitor filtering active).
4. For the competitor rows in run 3, compare against the same (s1_id, cand_id)
   rows from run 1. They must be bit-identical on ALL 36 columns.
"""
import os, sys, shutil, time
import numpy as np
import pandas as pd

# Setup paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
SRC_DIR = os.path.join(REPO_ROOT, 'code', 'business_entity_resolution', 'src')
LAPTOP_CACHE = os.path.join(REPO_ROOT, 'cache', 'laptop_test')
COMP_CACHE = os.path.join(REPO_ROOT, 'cache', 'competitor_test')

sys.path.insert(0, SRC_DIR)
import config

def create_competitor_test_cache():
    """Create a test cache with a split that produces competitors."""
    print("=" * 80)
    print("STEP 1: Creating competitor test cache")
    print("=" * 80)

    os.makedirs(COMP_CACHE, exist_ok=True)

    # Copy necessary files from laptop test
    for f in os.listdir(LAPTOP_CACHE):
        src = os.path.join(LAPTOP_CACHE, f)
        dst = os.path.join(COMP_CACHE, f)
        if f.startswith('feats_') or f == 'split.parquet' or f == 'addr_df_train.parquet' or f == 'baseline_feats.parquet':
            continue  # Skip these, we'll create our own split
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    b_path = os.path.join(COMP_CACHE, 'baseline_feats.parquet')
    if os.path.exists(b_path):
        os.remove(b_path)

    # Load the candidate chunk to understand the S1 IDs
    cands = pd.read_parquet(os.path.join(LAPTOP_CACHE, 'cands_train_chunk_0.parquet'))
    all_s1 = sorted(cands['s1_id'].unique())
    n_s1 = len(all_s1)
    print(f"Total S1 entities: {n_s1}")

    # Create a split with ~30% val, ~30% train, ~40% unsampled
    np.random.seed(42)
    perm = np.random.permutation(n_s1)
    n_val = int(n_s1 * 0.30)
    n_train = int(n_s1 * 0.30)

    val_ids = [all_s1[i] for i in perm[:n_val]]
    train_ids = [all_s1[i] for i in perm[n_val:n_val + n_train]]
    unsampled_ids = [all_s1[i] for i in perm[n_val + n_train:]]

    sampled_ids = val_ids + train_ids
    folds = ['val'] * len(val_ids) + ['train'] * len(train_ids)

    split_df = pd.DataFrame({'s1_id': sampled_ids, 'fold': folds})
    split_path = os.path.join(COMP_CACHE, 'split.parquet')
    split_df.to_parquet(split_path, index=False)

    print(f"Created split.parquet: {len(val_ids)} val, {len(train_ids)} train, {len(unsampled_ids)} unsampled")
    print(f"Sampled: {len(sampled_ids)}, Unsampled: {len(unsampled_ids)}")

    # Predict how many competitors there will be
    val_cands = set(cands.loc[cands['s1_id'].isin(val_ids), 'cand_id'].values)
    unsampled_mask = ~cands['s1_id'].isin(sampled_ids)
    comp_pairs = cands[unsampled_mask & cands['cand_id'].isin(val_cands)]
    comp_s1 = comp_pairs['s1_id'].nunique()
    print(f"Expected competitors: {comp_s1} S1 entities, {len(comp_pairs)} pairs")
    print(f"Expected sampled feature rows: {len(cands[cands['s1_id'].isin(sampled_ids)])}")
    print()

    return val_ids, train_ids, unsampled_ids


def run_baseline_all_sampled():
    """Run s4_features with ALL S1 sampled (no competitors) to get ground truth."""
    print("=" * 80)
    print("STEP 2: Running baseline with ALL S1 sampled (ground truth)")
    print("=" * 80)

    baseline_feats_path = os.path.join(COMP_CACHE, 'baseline_feats.parquet')
    if os.path.exists(baseline_feats_path):
        print(f"Baseline already exists at {baseline_feats_path}, loading...")
        return pd.read_parquet(baseline_feats_path)

    # Create a split where ALL S1 are sampled (some val, rest train)
    cands = pd.read_parquet(os.path.join(LAPTOP_CACHE, 'cands_train_chunk_0.parquet'))
    all_s1 = sorted(cands['s1_id'].unique())

    # Read the competitor test split to know which are val
    comp_split = pd.read_parquet(os.path.join(COMP_CACHE, 'split.parquet'))
    val_ids = set(comp_split.loc[comp_split['fold'] == 'val', 's1_id'].values)

    # Create "all sampled" split: same val IDs, everyone else is train
    all_sampled_split = pd.DataFrame({
        's1_id': all_s1,
        'fold': ['val' if s in val_ids else 'train' for s in all_s1]
    })
    all_split_path = os.path.join(COMP_CACHE, 'split_all_sampled.parquet')
    all_sampled_split.to_parquet(all_split_path, index=False)

    # Temporarily swap split.parquet
    orig_split = os.path.join(COMP_CACHE, 'split.parquet')
    backup_split = os.path.join(COMP_CACHE, 'split_comp.parquet')
    os.rename(orig_split, backup_split)
    shutil.copy2(all_split_path, orig_split)

    # Run s4_features
    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, 's4_features.py'),
         '--split', 'train', '--cache-dir', COMP_CACHE, '--force'],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError("Baseline run failed!")

    # Restore original split
    os.rename(backup_split, orig_split)

    # Read and save baseline features
    feats_path = os.path.join(COMP_CACHE, 'feats_train_chunk_0.parquet')
    baseline_df = pd.read_parquet(feats_path)
    baseline_df.to_parquet(baseline_feats_path, index=False)
    print(f"\nBaseline features: {len(baseline_df)} rows saved to {baseline_feats_path}")
    return baseline_df


def run_competitor_filtered():
    """Run s4_features with the 60/40 split (competitor filtering active)."""
    print("\n" + "=" * 80)
    print("STEP 3: Running with competitor filtering (60% sampled, 40% unsampled)")
    print("=" * 80)

    # Clean old feature files
    for f in os.listdir(COMP_CACHE):
        if f.startswith('feats_train_'):
            os.remove(os.path.join(COMP_CACHE, f))

    import subprocess
    result = subprocess.run(
        [sys.executable, os.path.join(SRC_DIR, 's4_features.py'),
         '--split', 'train', '--cache-dir', COMP_CACHE, '--force'],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-1000:])
        raise RuntimeError("Competitor run failed!")

    feats_path = os.path.join(COMP_CACHE, 'feats_train_chunk_0.parquet')
    comp_df = pd.read_parquet(feats_path)
    print(f"\nCompetitor-filtered features: {len(comp_df)} rows")
    n_comp = int((comp_df['is_competitor'] == 1).sum())
    n_samp = int((comp_df['is_competitor'] == 0).sum())
    print(f"  Sampled: {n_samp}, Competitor: {n_comp}")
    return comp_df


def compare_competitor_features(baseline_df, comp_df):
    """Compare competitor rows from filtered run against baseline (all-sampled) run."""
    print("\n" + "=" * 80)
    print("STEP 4: Comparing competitor features")
    print("=" * 80)

    # Get competitor rows from filtered run
    comp_rows = comp_df[comp_df['is_competitor'] == 1].copy()
    n_comp = len(comp_rows)
    if n_comp == 0:
        print("ERROR: No competitor rows found! Test is invalid.")
        sys.exit(1)

    print(f"Competitor rows in filtered run: {n_comp}")
    print(f"Competitor S1 entities: {comp_rows['s1_id'].nunique()}")

    # Find matching rows in baseline by (s1_id, cand_id)
    comp_keys = set(zip(comp_rows['s1_id'], comp_rows['cand_id']))
    baseline_keys = set(zip(baseline_df['s1_id'], baseline_df['cand_id']))
    missing = comp_keys - baseline_keys
    if missing:
        print(f"ERROR: {len(missing)} competitor pairs not found in baseline!")
        sys.exit(1)

    # Join on (s1_id, cand_id)
    comp_rows = comp_rows.set_index(['s1_id', 'cand_id']).sort_index()
    baseline_match = baseline_df.set_index(['s1_id', 'cand_id']).sort_index()
    baseline_match = baseline_match.loc[comp_rows.index]

    assert len(comp_rows) == len(baseline_match), "Row count mismatch after join!"

    # Compare every column
    mismatches = 0
    cols_to_check = [c for c in comp_rows.columns if c not in ('is_competitor',)]

    for col in cols_to_check:
        s_comp = comp_rows[col]
        s_base = baseline_match[col]

        if pd.api.types.is_float_dtype(s_comp):
            v_comp = s_comp.to_numpy()
            v_base = s_base.to_numpy()
            nan_c = np.isnan(v_comp)
            nan_b = np.isnan(v_base)
            if not np.array_equal(nan_c, nan_b):
                print(f"  FAIL {col}: NaN mask mismatch")
                mismatches += 1
                continue
            valid = ~nan_c
            if valid.any():
                diff = np.abs(v_comp[valid] - v_base[valid])
                max_diff = np.max(diff)
                if max_diff > 0:
                    print(f"  FAIL {col}: max_diff = {max_diff}")
                    # Show some details for context features
                    if col in ('gap_to_best', 'n_cands', 'support'):
                        bad_idx = np.where(diff > 0)[0][:3]
                        for bi in bad_idx:
                            print(f"       Row {bi}: comp={v_comp[valid][bi]:.6f} base={v_base[valid][bi]:.6f}")
                    mismatches += 1
                else:
                    print(f"  OK   {col}: bit-exact float match")
            else:
                print(f"  OK   {col}: all NaN (trivially identical)")
        else:
            if not (s_comp.values == s_base.values).all():
                print(f"  FAIL {col}: value mismatch")
                mismatches += 1
            else:
                print(f"  OK   {col}: exact match")

    print()
    if mismatches == 0:
        print(f"✓ ALL {len(cols_to_check)} columns are 100% BIT-IDENTICAL on {n_comp} competitor rows!")
    else:
        print(f"✗ {mismatches} column(s) have mismatches on competitor rows!")
        sys.exit(1)


if __name__ == '__main__':
    t0 = time.time()

    val_ids, train_ids, unsampled_ids = create_competitor_test_cache()
    baseline_df = run_baseline_all_sampled()
    comp_df = run_competitor_filtered()
    compare_competitor_features(baseline_df, comp_df)

    # Cleanup
    print(f"\nTotal test time: {time.time() - t0:.1f}s")
    print("COMPETITOR TEST PASSED!")

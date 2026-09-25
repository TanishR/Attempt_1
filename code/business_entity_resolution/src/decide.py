"""
Decision layer logic for Business Entity Resolution.
Implements GUIDEBOOK.md Step 8.1 and 8.2:
Exclusivity resolution and per-S1 thresholding.
"""

from typing import Dict, Iterable, List, Optional, Set
import numpy as np
import pandas as pd


def apply_exclusivity(
    probs_df: pd.DataFrame,
    margin: float = 0.0,
    use_exclusivity: bool = True
) -> pd.DataFrame:
    """
    Applies exclusivity rule across all candidate pairs:
    For each cand_id appearing in multiple S1 lists, keeps it only for the S1
    with the highest probability.
    Tie-breaks: higher emb_score (if available), then smaller s1_id.
    If margin > 0 and top-2 gap < margin, drops cand_id from all S1s.
    Returns: filtered DataFrame of candidate pairs.
    """
    if not use_exclusivity or len(probs_df) == 0:
        return probs_df.copy()

    df = probs_df.copy()

    # Determine sorting columns for tie-breaking
    # Primary: cand_id, Secondary: prob DESC, Tertiary: emb_score DESC (if present), Quaternary: s1_id ASC
    if "emb_score" in df.columns:
        sort_cols = ["cand_id", "prob", "emb_score", "s1_id"]
        ascending = [True, False, False, True]
    else:
        sort_cols = ["cand_id", "prob", "s1_id"]
        ascending = [True, False, True]

    df_sorted = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)

    # Rank within cand_id (0-indexed: 0 is top-1, 1 is top-2)
    cand_rank = df_sorted.groupby("cand_id").cumcount().values

    if margin > 0.0:
        # Vectorized margin check: find cand_ids where (prob_top1 - prob_top2) < margin
        rank0_mask = (cand_rank == 0)
        rank1_mask = (cand_rank == 1)

        rank0_df = df_sorted.loc[rank0_mask, ["cand_id", "prob"]].set_index("cand_id")
        rank1_df = df_sorted.loc[rank1_mask, ["cand_id", "prob"]].set_index("cand_id")

        # Inner join to only compare cand_ids with at least 2 appearances
        top2_df = rank0_df.join(rank1_df, lsuffix="_top1", rsuffix="_top2", how="inner")
        gap = top2_df["prob_top1"] - top2_df["prob_top2"]
        ambiguous_cands = set(top2_df.index[gap < margin])

        if ambiguous_cands:
            keep_mask = (cand_rank == 0) & (~df_sorted["cand_id"].isin(ambiguous_cands))
            return df_sorted[keep_mask].reset_index(drop=True)

    # Keep only rank 0 (top candidate per cand_id)
    return df_sorted[cand_rank == 0].reset_index(drop=True)


def apply_thresholds(
    df_excl: pd.DataFrame,
    t_top1: float,
    t_extra: float,
    all_s1_ids: Optional[Iterable[str]] = None
) -> Dict[str, List[str]]:
    """
    Applies per-S1 thresholding:
    Sorts each S1's candidates by probability descending.
    If top-1 prob < t_top1 -> predicts empty list.
    Else keeps top-1 candidate plus all other candidates with prob >= t_extra.
    Returns: dict mapping s1_id -> list of predicted cand_ids.
    """
    # Initialize dictionary for all expected S1 entities
    if all_s1_ids is not None:
        pred_dict: Dict[str, List[str]] = {s: [] for s in all_s1_ids}
    elif len(df_excl) > 0:
        pred_dict = {s: [] for s in df_excl["s1_id"].unique()}
    else:
        return {}

    if len(df_excl) == 0:
        return pred_dict

    # Sort within each s1_id by prob descending (tie-break by emb_score then cand_id)
    if "emb_score" in df_excl.columns:
        s1_sort_cols = ["s1_id", "prob", "emb_score", "cand_id"]
        s1_asc = [True, False, False, True]
    else:
        s1_sort_cols = ["s1_id", "prob", "cand_id"]
        s1_asc = [True, False, True]

    df_sorted = df_excl.sort_values(by=s1_sort_cols, ascending=s1_asc).reset_index(drop=True)

    # Fast vectorized ranking per s1_id
    s1_rank = df_sorted.groupby("s1_id").cumcount().values
    top1_probs = df_sorted.groupby("s1_id")["prob"].transform("first").values

    # Keep condition:
    # 1. S1 must pass top-1 gate: top1_probs >= t_top1
    # 2. For rank 0: kept automatically once gate passes
    # 3. For rank > 0: kept only if candidate prob >= t_extra
    cand_probs = df_sorted["prob"].values
    keep_mask = (top1_probs >= t_top1) & ((s1_rank == 0) | (cand_probs >= t_extra))

    df_kept = df_sorted[keep_mask]
    if len(df_kept) > 0:
        # Group kept candidates into list
        grouped = df_kept.groupby("s1_id")["cand_id"].agg(list).to_dict()
        pred_dict.update(grouped)

    return pred_dict


import json
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

import config


def load_decision_thresholds(cache_dir: Optional[str] = None) -> Tuple[float, float, float, bool, str]:
    """
    Loads decision thresholds:
    1. Checks <cache_dir>/thresholds.json or cache/thresholds.json.
    2. Falls back to config.py if thresholds.json is not found.
    Returns: (t_top1, t_extra, margin, use_exclusivity, source_description)
    """
    candidate_paths = []
    if cache_dir:
        candidate_paths.append(os.path.join(cache_dir, "thresholds.json"))
    if hasattr(config, "CACHE_DIR"):
        candidate_paths.append(os.path.join(config.CACHE_DIR, "thresholds.json"))
    candidate_paths.append("cache/thresholds.json")

    for path in candidate_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                t_top1 = float(data["t_top1"])
                t_extra = float(data["t_extra"])
                margin = float(data.get("margin", 0.05))
                use_excl = bool(data.get("use_exclusivity", True))
                return t_top1, t_extra, margin, use_excl, path
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

    # Fallback to config.py
    t_top1 = getattr(config, "T_TOP1", 0.48)
    t_top1 = 0.48 if t_top1 is None else float(t_top1)

    t_extra = getattr(config, "T_EXTRA", 0.68)
    t_extra = 0.68 if t_extra is None else float(t_extra)

    margin = getattr(config, "EXCL_MARGIN", 0.05)
    margin = 0.05 if margin is None else float(margin)

    use_excl = getattr(config, "USE_EXCLUSIVITY", True)
    use_excl = True if use_excl is None else bool(use_excl)

    return t_top1, t_extra, margin, use_excl, "config.py"


def decide(
    probs_df: pd.DataFrame,
    t_top1: Optional[float] = None,
    t_extra: Optional[float] = None,
    margin: Optional[float] = None,
    use_exclusivity: Optional[bool] = None,
    all_s1_ids: Optional[Iterable[Any]] = None,
    cache_dir: Optional[str] = None
) -> Dict[Any, List[Any]]:
    """
    Main decision function executing full Step 8 decision layer in order:
    1. Loads thresholds from cache/thresholds.json (falling back to config.py) if not passed.
    2. Exclusivity resolution across cand_id (with tie-breaks and margin check).
    3. Per-S1 thresholding (top-1 gate and extra threshold).
    Returns: dict mapping s1_id -> list of predicted cand_ids.
    """
    if t_top1 is None or t_extra is None or margin is None or use_exclusivity is None:
        def_t1, def_te, def_m, def_ue, _ = load_decision_thresholds(cache_dir)
        if t_top1 is None:
            t_top1 = def_t1
        if t_extra is None:
            t_extra = def_te
        if margin is None:
            margin = def_m
        if use_exclusivity is None:
            use_exclusivity = def_ue

    # Determine complete set of expected S1 IDs from input before exclusivity
    if all_s1_ids is None and len(probs_df) > 0:
        if "is_competitor" in probs_df.columns:
            all_s1_ids = probs_df.loc[probs_df["is_competitor"] == 0, "s1_id"].unique()
        else:
            all_s1_ids = probs_df["s1_id"].unique()

    # 1. Apply exclusivity
    df_excl = apply_exclusivity(probs_df, margin=margin, use_exclusivity=use_exclusivity)

    # 2. Filter to eval S1 entities if is_competitor column is present
    if "is_competitor" in df_excl.columns:
        df_excl = df_excl[df_excl["is_competitor"] == 0]

    # 3. Apply thresholds
    return apply_thresholds(df_excl, t_top1=t_top1, t_extra=t_extra, all_s1_ids=all_s1_ids)


def run_unit_tests():
    """
    Runs unit tests for exclusivity, margin, tie-breaking, and threshold rules.
    Returns: None. Raises AssertionError on failure.
    """
    print("Running decide.py unit tests...")

    # Test 1: Exclusivity without margin
    df1 = pd.DataFrame({
        "s1_id": ["S1_A", "S1_B"],
        "cand_id": ["C1", "C1"],
        "prob": [0.90, 0.80],
        "emb_score": [0.85, 0.85],
    })
    preds1 = decide(df1, t_top1=0.5, t_extra=0.8, margin=0.0, use_exclusivity=True)
    assert preds1["S1_A"] == ["C1"], f"Test 1 failed S1_A: {preds1}"
    assert preds1["S1_B"] == [], f"Test 1 failed S1_B: {preds1}"
    print("  [PASS] Test 1: Exclusivity without margin")

    # Test 2: Exclusivity tie-break on emb_score
    df2 = pd.DataFrame({
        "s1_id": ["S1_A", "S1_B"],
        "cand_id": ["C1", "C1"],
        "prob": [0.85, 0.85],
        "emb_score": [0.70, 0.90],
    })
    preds2 = decide(df2, t_top1=0.5, t_extra=0.8, margin=0.0, use_exclusivity=True)
    assert preds2["S1_B"] == ["C1"], f"Test 2 failed S1_B: {preds2}"
    assert preds2["S1_A"] == [], f"Test 2 failed S1_A: {preds2}"
    print("  [PASS] Test 2: Exclusivity tie-break on emb_score")

    # Test 3: Exclusivity margin drop
    df3 = pd.DataFrame({
        "s1_id": ["S1_A", "S1_B"],
        "cand_id": ["C1", "C1"],
        "prob": [0.82, 0.80],  # gap = 0.02 < margin 0.05
        "emb_score": [0.80, 0.80],
    })
    preds3 = decide(df3, t_top1=0.5, t_extra=0.8, margin=0.05, use_exclusivity=True)
    assert preds3["S1_A"] == [], f"Test 3 failed S1_A: {preds3}"
    assert preds3["S1_B"] == [], f"Test 3 failed S1_B: {preds3}"
    print("  [PASS] Test 3: Exclusivity margin ambiguous drop")

    # Test 4: Per-S1 thresholding (top-1 gate and extras)
    df4 = pd.DataFrame({
        "s1_id": ["S1_A", "S1_A", "S1_A", "S1_B", "S1_B"],
        "cand_id": ["C1", "C2", "C3", "C4", "C5"],
        "prob": [0.90, 0.82, 0.70, 0.45, 0.40],
        "emb_score": [0.9, 0.8, 0.7, 0.6, 0.5],
    })
    preds4 = decide(df4, t_top1=0.50, t_extra=0.80, margin=0.0, use_exclusivity=False)
    # S1_A: top-1 is C1 (0.90 >= 0.5), extra C2 (0.82 >= 0.80), C3 (0.70 < 0.80 dropped)
    assert set(preds4["S1_A"]) == {"C1", "C2"}, f"Test 4 failed S1_A: {preds4}"
    # S1_B: top-1 is C4 (0.45 < 0.50) -> empty list
    assert preds4["S1_B"] == [], f"Test 4 failed S1_B: {preds4}"
    print("  [PASS] Test 4: Top-1 gate and extra thresholding")

    # Test 5: Competitor filtering on exclusivity
    df5 = pd.DataFrame({
        "s1_id": ["S1_val", "S1_comp"],
        "cand_id": ["C1", "C1"],
        "prob": [0.70, 0.95],
        "is_competitor": [0, 1],
    })
    # S1_comp wins C1 because prob 0.95 > 0.70.
    # When scored on val entities, S1_val loses C1 and has []
    preds5 = decide(df5, t_top1=0.5, t_extra=0.8, margin=0.0, use_exclusivity=True, all_s1_ids=["S1_val"])
    assert preds5["S1_val"] == [], f"Test 5 failed: {preds5}"
    print("  [PASS] Test 5: Competitor exclusivity resolution")

    print("All decide.py unit tests passed successfully!\n")


if __name__ == "__main__":
    run_unit_tests()

"""
Entity-level F0.5 evaluation metric functions and unit tests.
Implements GUIDEBOOK.md Step 7.1 entity-level precision-weighted F0.5.
"""

from typing import Dict, Set, Iterable
import pandas as pd


def f05_entity(pred: Set[str], gold: Set[str]) -> float:
    """
    Computes entity-level F0.5 score for a single entity.
    Returns: float F0.5 score in [0.0, 1.0].
    """
    if not gold:                       # true singleton
        return 1.0 if not pred else 0.0
    if not pred:                       # had matches, predicted nothing
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map: Dict[str, Set[str]], gold_map: Dict[str, Set[str]]) -> float:
    """
    Computes macro-averaged F0.5 over all entities in gold_map.
    gold_map must contain every eval S1, including empty sets (singletons).
    Returns: float macro F0.5 score.
    """
    if not gold_map:
        return 0.0
    total = sum(f05_entity(pred_map.get(s, set()), g) for s, g in gold_map.items())
    return float(total / len(gold_map))


def build_gold_map(gt_df: pd.DataFrame, eval_s1_ids: Iterable[str]) -> Dict[str, Set[str]]:
    """
    Builds the full gold map for every evaluation S1 entity, including true singletons
    and ground-truth matches that were missed by blocking.
    Returns: dict mapping s1_id -> set of true matching cand_ids.
    """
    eval_set = set(eval_s1_ids)
    gold_map: Dict[str, Set[str]] = {s: set() for s in eval_set}
    
    if gt_df is not None and len(gt_df) > 0:
        # Determine column names (source1_id / source2_id or s1_id / cand_id)
        s1_col = 'source1_id' if 'source1_id' in gt_df.columns else ('s1_id' if 's1_id' in gt_df.columns else gt_df.columns[0])
        cand_col = 'cand_id' if 'cand_id' in gt_df.columns else (gt_df.columns[1])
        
        # Filter to pairs where s1 is in eval_set
        mask = gt_df[s1_col].isin(eval_set)
        filtered_gt = gt_df[mask]
        
        for s1, cand in zip(filtered_gt[s1_col].values, filtered_gt[cand_col].values):
            gold_map[s1].add(str(cand))
            
    return gold_map


def run_unit_tests():
    """
    Executes unit tests for f05_entity and macro_f05.
    Returns: None. Raises AssertionError on failure.
    """
    print("Running metrics.py unit tests...")

    # Test 1: Official example: pred 3, gold 2, both gold in pred -> 0.714 (3 decimals)
    pred_1 = {"c1", "c2", "c3"}
    gold_1 = {"c1", "c2"}
    score_1 = f05_entity(pred_1, gold_1)
    assert round(score_1, 3) == 0.714, f"Test 1 failed: expected 0.714, got {score_1:.6f}"
    print(f"  [PASS] Test 1 (official example: pred 3, gold 2, tp 2): {score_1:.4f} -> {round(score_1, 3)}")

    # Test 2: Singleton + empty pred -> 1.0; singleton + any pred -> 0.0
    score_2a = f05_entity(set(), set())
    assert score_2a == 1.0, f"Test 2a failed: expected 1.0, got {score_2a}"
    score_2b = f05_entity({"c1"}, set())
    assert score_2b == 0.0, f"Test 2b failed: expected 0.0, got {score_2b}"
    print(f"  [PASS] Test 2 (singleton empty: {score_2a}, singleton non-empty: {score_2b})")

    # Test 3: Non-empty gold + empty pred -> 0.0
    score_3 = f05_entity(set(), {"c1"})
    assert score_3 == 0.0, f"Test 3 failed: expected 0.0, got {score_3}"
    print(f"  [PASS] Test 3 (gold non-empty + pred empty): {score_3}")

    # Test 4: Gold 2, pred = 1 correct -> 0.833
    pred_4 = {"c1"}
    gold_4 = {"c1", "c2"}
    score_4 = f05_entity(pred_4, gold_4)
    assert round(score_4, 3) == 0.833, f"Test 4 failed: expected 0.833, got {score_4:.6f}"
    print(f"  [PASS] Test 4 (gold 2, pred 1 correct): {score_4:.4f} -> {round(score_4, 3)}")

    # Test 5: macro_f05 across multiple entities including singletons and missed blocking
    gold_map = {
        "s1": {"c1", "c2"},   # pred 3 -> 0.7142857
        "s2": set(),          # pred empty -> 1.0
        "s3": {"c10"},        # pred empty (missed) -> 0.0
        "s4": {"c1", "c2"},   # pred 1 -> 0.8333333
    }
    pred_map = {
        "s1": {"c1", "c2", "c3"},
        "s2": set(),
        "s3": set(),
        "s4": {"c1"},
    }
    macro_score = macro_f05(pred_map, gold_map)
    expected_macro = (5/7 + 1.0 + 0.0 + 5/6) / 4.0
    assert abs(macro_score - expected_macro) < 1e-6, f"Test 5 macro failed: expected {expected_macro}, got {macro_score}"
    print(f"  [PASS] Test 5 (macro_f05 multi-entity): {macro_score:.4f} == {expected_macro:.4f}")

    print("All metrics.py unit tests passed successfully!\n")


if __name__ == "__main__":
    run_unit_tests()

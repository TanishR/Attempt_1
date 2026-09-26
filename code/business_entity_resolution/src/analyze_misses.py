#!/usr/bin/env python3
"""
Standalone diagnostic script to analyze missed Ground Truth (GT) pairs from blocking.
Evaluates why candidate generation missed them, computes pair features, tests 5 candidate
blocking keys (K1-K5) individually and in union, measures block-size distributions on full
S2+S3, and outputs 50 random missed examples to logs/misses_sample.tsv.

Peak RSS is strictly controlled to stay under 6 GB by:
1. Streaming candidate chunk files to gather val candidates and 20th-best embedding scores.
2. Restricting normalized tables to only the IDs involved in missed pairs.
3. Memory-mapping embeddings (np.load with mmap_mode='r') for dot-product similarity.
4. Streaming the full S2+S3 pool in batches when computing block-size distributions.

Usage:
  python code/business_entity_resolution/src/analyze_misses.py [--laptop-test] [--limit 200]
  python code/business_entity_resolution/src/analyze_misses.py --cache-dir cache
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rapidfuzz import fuzz


def _rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


# ==============================================================================
# CANDIDATE KEYS K1 - K5 DEFINITIONS
# ==============================================================================

def extract_k1(row: dict[str, Any]) -> set[str]:
    """
    K1: (country, two consecutive numeric tokens in the address, each of length >= 2)
    Uses normalized num_tokens, which preserves the sequence of numeric tokens in the address.
    """
    country = str(row.get("country", "")).strip()
    if not country:
        return set()
    num_str = str(row.get("num_tokens", "")).strip()
    if not num_str or num_str == "nan":
        return set()
    nums = [t for t in num_str.split() if len(t) >= 2]
    keys = set()
    for i in range(len(nums) - 1):
        keys.add(f"K1_{country}_{nums[i]}_{nums[i+1]}")
    return keys


def extract_k2(row: dict[str, Any]) -> set[str]:
    """
    K2: (country, zip_pin, first alphabetic address token of length >= 4)
    """
    country = str(row.get("country", "")).strip()
    zip_pin = str(row.get("zip_pin", "")).strip()
    if not country or not zip_pin or zip_pin == "nan":
        return set()
    addr_toks = str(row.get("addr_tokens", "") or row.get("addr_norm", "")).split()
    first_alpha = None
    for t in addr_toks:
        if t.isalpha() and len(t) >= 4:
            first_alpha = t
            break
    if not first_alpha:
        return set()
    return {f"K2_{country}_{zip_pin}_{first_alpha}"}


def extract_k3(row: dict[str, Any]) -> set[str]:
    """
    K3: (country, sorted set of all numeric address tokens)
    """
    country = str(row.get("country", "")).strip()
    if not country:
        return set()
    num_str = str(row.get("num_tokens", "")).strip()
    if not num_str or num_str == "nan":
        return set()
    nums = sorted(set(num_str.split()))
    if not nums:
        return set()
    return {f"K3_{country}_" + "_".join(nums)}


def extract_k4(row: dict[str, Any]) -> set[str]:
    """
    K4: (country, first 2 sorted name_skel tokens)
    """
    country = str(row.get("country", "")).strip()
    if not country:
        return set()
    skel = str(row.get("name_skel", "")).strip()
    if not skel or skel == "nan":
        return set()
    toks = sorted(skel.split())
    if len(toks) < 2:
        return set()
    return {f"K4_{country}_{toks[0]}_{toks[1]}"}


def extract_k5(row: dict[str, Any], doc_freqs: dict[tuple[str, str], int]) -> set[str]:
    """
    K5: (country, any numeric token of length >= 3, rarest alphabetic address token of length >= 5)
    """
    country = str(row.get("country", "")).strip()
    if not country:
        return set()
    num_str = str(row.get("num_tokens", "")).strip()
    if not num_str or num_str == "nan":
        return set()
    nums_ge3 = [t for t in set(num_str.split()) if len(t) >= 3]
    if not nums_ge3:
        return set()
    addr_toks = str(row.get("addr_tokens", "") or row.get("addr_norm", "")).split()
    alpha_toks = [t for t in set(addr_toks) if t.isalpha() and len(t) >= 5]
    if not alpha_toks:
        return set()
    rarest = min(alpha_toks, key=lambda t: (doc_freqs.get((country, t), 0), t))
    keys = set()
    for n in nums_ge3:
        keys.add(f"K5_{country}_{n}_{rarest}")
    return keys


# ==============================================================================
# GROUND TRUTH & CANDIDATE STREAMING
# ==============================================================================

def load_val_s1_ids(cache_dir: str, limit: Optional[int] = None) -> list[str]:
    """Loads val-fold S1 IDs from split.parquet."""
    split_path = os.path.join(cache_dir, "split.parquet")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing {split_path}")
    split_df = pd.read_parquet(split_path)
    if "fold" in split_df.columns:
        val_df = split_df[split_df["fold"] == "val"]
    else:
        val_df = split_df
    s1_ids = val_df["s1_id"].tolist()
    if limit is not None and limit > 0:
        s1_ids = s1_ids[:limit]
        print(f"Limiting val S1 to first {limit:,} entities.")
    return s1_ids


def load_ground_truth(cache_dir: str, root_dir: str, val_s1_set: set[str]) -> pd.DataFrame:
    """Loads Ground Truth table restricted to val_s1_set."""
    gt_path = os.path.join(cache_dir, "gt_long.parquet")
    if os.path.exists(gt_path):
        print(f"Loading ground truth from {gt_path}...")
        gt_df = pd.read_parquet(gt_path)
    else:
        raw_gt = os.path.join(root_dir, "dataset", "train", "train_ground_truth.tsv")
        if not os.path.exists(raw_gt):
            raise FileNotFoundError(f"Neither gt_long.parquet nor {raw_gt} found.")
        print(f"Loading raw ground truth from {raw_gt}...")
        raw_df = pd.read_csv(raw_gt, sep="\t")
        records = []
        for _, row in raw_df.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            matches = str(row["matched_entity_ids"]).split(",")
            for m in matches:
                m = m.strip()
                if m:
                    src = "S2" if m.startswith("S2") else ("S3" if m.startswith("S3") else "other")
                    records.append({"s1_id": s1_id, "match_id": m, "match_source": src})
        gt_df = pd.DataFrame(records)
    
    gt_val = gt_df[gt_df["s1_id"].isin(val_s1_set)].copy()
    if "cand_id" in gt_val.columns and "match_id" not in gt_val.columns:
        gt_val = gt_val.rename(columns={"cand_id": "match_id"})
    print(f"Loaded GT val pairs: {len(gt_val):,} pairs across {gt_val['s1_id'].nunique():,} unique S1 (RSS {_rss_mb():.1f} MB)")
    return gt_val


def stream_val_candidates_and_find_misses(
    cache_dir: str,
    val_s1_set: set[str],
    gt_val: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Streams cands_train_chunk_*.parquet reading only val S1 rows.
    Identifies which GT pairs are missed, and records each S1's 20th-best emb_score.
    """
    chunk_files = sorted(glob.glob(os.path.join(cache_dir, "cands_train_chunk_*.parquet")))
    if not chunk_files:
        raise FileNotFoundError(f"No cands_train_chunk_*.parquet files found in {cache_dir}")
    print(f"\nStreaming {len(chunk_files)} candidate chunk(s) (val rows only)...")
    
    val_cands_map: dict[str, set[str]] = defaultdict(set)
    val_emb_scores: dict[str, list[float]] = defaultdict(list)
    
    total_val_cands_rows = 0
    t0 = time.time()
    for i, c_path in enumerate(chunk_files):
        # Read minimal columns
        cols = ["s1_id", "cand_id", "emb_score"]
        df_chunk = pd.read_parquet(c_path, columns=cols)
        df_val_sub = df_chunk[df_chunk["s1_id"].isin(val_s1_set)]
        total_val_cands_rows += len(df_val_sub)
        
        for s1, cid, score in zip(df_val_sub["s1_id"], df_val_sub["cand_id"], df_val_sub["emb_score"]):
            val_cands_map[s1].add(cid)
            if pd.notna(score):
                val_emb_scores[s1].append(float(score))
                
        del df_chunk, df_val_sub
        if (i + 1) % 5 == 0 or (i + 1) == len(chunk_files):
            print(f"  Processed {i+1}/{len(chunk_files)} chunks | Val cand rows: {total_val_cands_rows:,} | RSS: {_rss_mb():.1f} MB", flush=True)
            gc.collect()

    t_cands = time.time() - t0
    print(f"Streamed candidate chunks in {t_cands:.1f}s. Val candidates gathered for {len(val_cands_map):,} S1s.")

    # Compute 20th-best emb_score per S1
    s1_20th_best: dict[str, float] = {}
    for s1, scores in val_emb_scores.items():
        if not scores:
            s1_20th_best[s1] = float("nan")
        else:
            scores.sort(reverse=True)
            if len(scores) >= 20:
                s1_20th_best[s1] = scores[19]
            else:
                s1_20th_best[s1] = scores[-1]

    # Find missed GT pairs
    missed_records: list[dict[str, Any]] = []
    total_gt = len(gt_val)
    found_gt = 0
    for _, row in gt_val.iterrows():
        s1 = str(row["s1_id"])
        match_id = str(row["match_id"])
        src = str(row.get("match_source", "S2" if match_id.startswith("S2") else "S3"))
        
        cands_for_s1 = val_cands_map.get(s1, set())
        if match_id in cands_for_s1:
            found_gt += 1
        else:
            missed_records.append({
                "s1_id": s1,
                "cand_id": match_id,
                "cand_source": src,
                "s1_20th_emb_score": s1_20th_best.get(s1, float("nan")),
            })

    recall = (found_gt / total_gt * 100.0) if total_gt > 0 else 0.0
    print(f"\nGround Truth Pairs: {total_gt:,} total, {found_gt:,} caught ({recall:.2f}%), {len(missed_records):,} missed ({100.0 - recall:.2f}%)")
    print(f"RSS after finding misses: {_rss_mb():.1f} MB")
    
    return missed_records, s1_20th_best


# ==============================================================================
# RESTRICTED NORMALIZED TABLES LOADING
# ==============================================================================

def load_restricted_norm_tables(
    cache_dir: str,
    needed_s1: set[str],
    needed_cand: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Loads normalized tables restricted strictly to the entity IDs needed.
    Keeps memory tiny (< 50 MB) by using pyarrow.dataset pushdown filters.
    """
    print(f"\nLoading normalized tables restricted to {len(needed_s1):,} S1 IDs and {len(needed_cand):,} Cand IDs...")
    cols = [
        "entity_id", "country", "name_full", "core_name", "name_skel",
        "addr_norm", "addr_tokens", "house_no", "num_tokens", "zip_pin", "state_code",
    ]
    
    s1_dict: dict[str, dict[str, Any]] = {}
    cand_dict: dict[str, dict[str, Any]] = {}
    
    # Load S1
    p1 = os.path.join(cache_dir, "norm_train_source1.parquet")
    if os.path.exists(p1):
        ds1 = ds.dataset(p1, format="parquet")
        avail_cols = [c for c in cols if c in ds1.schema.names]
        tbl1 = ds1.to_table(
            columns=avail_cols,
            filter=pc.is_in(pc.field("entity_id"), pa.array(list(needed_s1))),
        )
        df1 = tbl1.to_pandas()
        for r in df1.to_dict("records"):
            s1_dict[r["entity_id"]] = r
        del ds1, tbl1, df1
        print(f"  Loaded {len(s1_dict):,} restricted S1 entities (RSS {_rss_mb():.1f} MB)")
        
    # Load S2 and S3
    for s_name in ["norm_train_source2.parquet", "norm_train_source3.parquet"]:
        p_cand = os.path.join(cache_dir, s_name)
        if os.path.exists(p_cand):
            dsc = ds.dataset(p_cand, format="parquet")
            avail_cols = [c for c in cols if c in dsc.schema.names]
            tbl_c = dsc.to_table(
                columns=avail_cols,
                filter=pc.is_in(pc.field("entity_id"), pa.array(list(needed_cand))),
            )
            df_c = tbl_c.to_pandas()
            for r in df_c.to_dict("records"):
                cand_dict[r["entity_id"]] = r
            del dsc, tbl_c, df_c
            print(f"  Loaded restricted entities from {s_name} (Total cands now {len(cand_dict):,}, RSS {_rss_mb():.1f} MB)")

    gc.collect()
    return s1_dict, cand_dict


# ==============================================================================
# EMBEDDINGS LOOKUP FOR MISSED PAIRS
# ==============================================================================

def compute_missed_pair_embeddings(
    missed_pairs: list[dict[str, Any]],
    cache_dir: str,
) -> list[float]:
    """
    Computes embedding cosine similarity for missed pairs using memory-mapped .npy arrays.
    Virtual memory only — RSS remains < 100 MB.
    """
    print(f"\nComputing embedding similarities for {len(missed_pairs):,} missed pairs (mmap mode)...")
    
    # Helper to load mmap array and ID map
    def _load_emb_and_map(src: str) -> tuple[Any, dict[str, int], Any, dict[str, int]]:
        m_p = os.path.join(cache_dir, f"emb_train_{src}.npy")
        m_id_p = os.path.join(cache_dir, f"ids_train_{src}.npy")
        a_p = os.path.join(cache_dir, f"emb_train_{src}_alt.npy")
        a_id_p = os.path.join(cache_dir, f"ids_train_{src}_alt.npy")
        
        m_emb = np.load(m_p, mmap_mode="r") if os.path.exists(m_p) else None
        m_ids = np.load(m_id_p, allow_pickle=True) if os.path.exists(m_id_p) else []
        m_map = {eid: idx for idx, eid in enumerate(m_ids)}
        
        a_emb = np.load(a_p, mmap_mode="r") if os.path.exists(a_p) else None
        a_ids = np.load(a_id_p, allow_pickle=True) if os.path.exists(a_id_p) else []
        a_map = {eid: idx for idx, eid in enumerate(a_ids)}
        
        return m_emb, m_map, a_emb, a_map

    s1_m, s1_mm, s1_a, s1_am = _load_emb_and_map("source1")
    s2_m, s2_mm, s2_a, s2_am = _load_emb_and_map("source2")
    s3_m, s3_mm, s3_a, s3_am = _load_emb_and_map("source3")

    pair_scores = []
    for r in missed_pairs:
        s1 = r["s1_id"]
        cid = r["cand_id"]
        
        # Select cand source arrays
        if cid.startswith("S2"):
            cm_emb, cm_map, ca_emb, ca_map = s2_m, s2_mm, s2_a, s2_am
        else:
            cm_emb, cm_map, ca_emb, ca_map = s3_m, s3_mm, s3_a, s3_am
            
        if s1_m is None or cm_emb is None or s1 not in s1_mm or cid not in cm_map:
            pair_scores.append(float("nan"))
            continue
            
        v1_m = s1_m[s1_mm[s1]].astype(np.float32)
        v2_m = cm_emb[cm_map[cid]].astype(np.float32)
        best_s = float(np.dot(v1_m, v2_m))
        
        has_s1_a = s1_a is not None and s1 in s1_am
        has_c_a = ca_emb is not None and cid in ca_map
        
        if has_c_a:
            v2_a = ca_emb[ca_map[cid]].astype(np.float32)
            s_ma = float(np.dot(v1_m, v2_a))
            if s_ma > best_s:
                best_s = s_ma
                
        if has_s1_a:
            v1_a = s1_a[s1_am[s1]].astype(np.float32)
            s_am = float(np.dot(v1_a, v2_m))
            if s_am > best_s:
                best_s = s_am
            if has_c_a:
                s_aa = float(np.dot(v1_a, v2_a))
                if s_aa > best_s:
                    best_s = s_aa
                    
        pair_scores.append(best_s)

    del s1_m, s1_mm, s1_a, s1_am, s2_m, s2_mm, s2_a, s2_am, s3_m, s3_mm, s3_a, s3_am
    gc.collect()
    print(f"Embedding scores computed (RSS: {_rss_mb():.1f} MB)")
    return pair_scores


# ==============================================================================
# DOCUMENT FREQUENCY LOADING
# ==============================================================================

def load_or_compute_doc_freqs(cache_dir: str) -> dict[tuple[str, str], int]:
    """Loads addr_df_train.parquet or returns empty dict if missing."""
    p = os.path.join(cache_dir, "addr_df_train.parquet")
    doc_freqs: dict[tuple[str, str], int] = {}
    if os.path.exists(p):
        print(f"Loading address token document frequencies from {p}...")
        df_tf = pd.read_parquet(p)
        for ctry, tok, df_val in zip(df_tf["country"], df_tf["token"], df_tf["doc_freq"]):
            doc_freqs[(str(ctry), str(tok))] = int(df_val)
        print(f"  Loaded {len(doc_freqs):,} token frequencies (RSS: {_rss_mb():.1f} MB)")
    else:
        print("Notice: addr_df_train.parquet not found; rarest token frequency will fallback to 0.")
    return doc_freqs


# ==============================================================================
# FEATURE COMPUTATION FOR MISSED PAIRS
# ==============================================================================

def compute_missed_pair_features(
    missed_records: list[dict[str, Any]],
    s1_dict: dict[str, dict[str, Any]],
    cand_dict: dict[str, dict[str, Any]],
    pair_emb_scores: list[float],
    doc_freqs: dict[tuple[str, str], int],
) -> pd.DataFrame:
    """Computes all required features and candidate keys for every missed GT pair."""
    print(f"\nComputing similarity features and candidate keys for {len(missed_records):,} missed pairs...")
    
    rows = []
    for i, r in enumerate(missed_records):
        s1_id = r["s1_id"]
        cand_id = r["cand_id"]
        cand_src = r["cand_source"]
        s1_20th = r["s1_20th_emb_score"]
        emb_score = pair_emb_scores[i]
        
        s1 = s1_dict.get(s1_id, {})
        cand = cand_dict.get(cand_id, {})
        
        country = str(s1.get("country") or cand.get("country") or "Unknown")
        s1_name = str(s1.get("name_full", "")).strip()
        cand_name = str(cand.get("name_full", "")).strip()
        s1_core = str(s1.get("core_name", "")).strip()
        cand_core = str(cand.get("core_name", "")).strip()
        s1_addr = str(s1.get("addr_norm", "")).strip()
        cand_addr = str(cand.get("addr_norm", "")).strip()
        
        # Name and Address fuzzy metrics
        name_token_sort = float(fuzz.token_sort_ratio(s1_name, cand_name)) if (s1_name and cand_name) else 0.0
        core_ratio = float(fuzz.ratio(s1_core, cand_core)) if (s1_core and cand_core) else 0.0
        addr_token_set = float(fuzz.token_set_ratio(s1_addr, cand_addr)) if (s1_addr and cand_addr) else 0.0
        
        # House number match (1/0/-1)
        s1_h = str(s1.get("house_no", "")).strip()
        c_h = str(cand.get("house_no", "")).strip()
        if not s1_h or not c_h:
            house_no_match = -1
        elif s1_h == c_h:
            house_no_match = 1
        else:
            house_no_match = 0
            
        # Count of shared numeric tokens
        s1_nums = set(str(s1.get("num_tokens", "")).split())
        c_nums = set(str(cand.get("num_tokens", "")).split())
        s1_nums.discard("")
        s1_nums.discard("nan")
        c_nums.discard("")
        c_nums.discard("nan")
        shared_nums = s1_nums & c_nums
        shared_num_tokens_count = len(shared_nums)
        
        # Zip match (1/0/-1)
        s1_z = str(s1.get("zip_pin", "")).strip()
        c_z = str(cand.get("zip_pin", "")).strip()
        if not s1_z or not c_z or s1_z == "nan" or c_z == "nan":
            zip_match = -1
        elif s1_z == c_z:
            zip_match = 1
        else:
            zip_match = 0
            
        # Candidate keys K1 - K5
        k1_s1, k1_cand = extract_k1(s1), extract_k1(cand)
        k2_s1, k2_cand = extract_k2(s1), extract_k2(cand)
        k3_s1, k3_cand = extract_k3(s1), extract_k3(cand)
        k4_s1, k4_cand = extract_k4(s1), extract_k4(cand)
        k5_s1, k5_cand = extract_k5(s1, doc_freqs), extract_k5(cand, doc_freqs)
        
        k1_shared = k1_s1 & k1_cand
        k2_shared = k2_s1 & k2_cand
        k3_shared = k3_s1 & k3_cand
        k4_shared = k4_s1 & k4_cand
        k5_shared = k5_s1 & k5_cand
        
        k1_caught = 1 if bool(k1_shared) else 0
        k2_caught = 1 if bool(k2_shared) else 0
        k3_caught = 1 if bool(k3_shared) else 0
        k4_caught = 1 if bool(k4_shared) else 0
        k5_caught = 1 if bool(k5_shared) else 0
        union_caught = 1 if (k1_caught or k2_caught or k3_caught or k4_caught or k5_caught) else 0
        
        emb_diff = (emb_score - s1_20th) if (pd.notna(emb_score) and pd.notna(s1_20th)) else float("nan")

        rows.append({
            "s1_id": s1_id,
            "cand_id": cand_id,
            "country": country,
            "cand_source": cand_src,
            "s1_name": s1_name,
            "cand_name": cand_name,
            "s1_addr": s1_addr,
            "cand_addr": cand_addr,
            "name_token_sort": name_token_sort,
            "core_ratio": core_ratio,
            "addr_token_set": addr_token_set,
            "house_no_match": house_no_match,
            "shared_num_tokens_count": shared_num_tokens_count,
            "zip_match": zip_match,
            "k1_caught": k1_caught,
            "k2_caught": k2_caught,
            "k3_caught": k3_caught,
            "k4_caught": k4_caught,
            "k5_caught": k5_caught,
            "union_caught": union_caught,
            "k1_shared_keys": list(k1_shared),
            "k2_shared_keys": list(k2_shared),
            "k3_shared_keys": list(k3_shared),
            "k4_shared_keys": list(k4_shared),
            "k5_shared_keys": list(k5_shared),
            "pair_emb_score": emb_score,
            "s1_20th_emb_score": s1_20th,
            "emb_score_diff": emb_diff,
        })

    df = pd.DataFrame(rows)
    print(f"Computed features for {len(df):,} pairs (RSS: {_rss_mb():.1f} MB)")
    return df


# ==============================================================================
# STREAM S2+S3 TO COUNT BLOCK SIZES
# ==============================================================================

def stream_s23_block_sizes(
    cache_dir: str,
    doc_freqs: dict[tuple[str, str], int],
) -> tuple[Counter, Counter, Counter, Counter, Counter]:
    """
    Streams full train S2+S3 pool chunk by chunk to count block sizes for K1-K5.
    Memory footprint remains minimal (< 300 MB) because it never loads all tables.
    """
    print("\n--- Streaming full train S2+S3 pool to count block sizes for K1-K5 ---")
    t0 = time.time()
    
    k1_counts: Counter[str] = Counter()
    k2_counts: Counter[str] = Counter()
    k3_counts: Counter[str] = Counter()
    k4_counts: Counter[str] = Counter()
    k5_counts: Counter[str] = Counter()
    
    total_records = 0
    for src_name in ["norm_train_source2.parquet", "norm_train_source3.parquet"]:
        p = os.path.join(cache_dir, src_name)
        if not os.path.exists(p):
            print(f"Warning: {p} not found, skipping.", flush=True)
            continue
            
        print(f"Streaming {src_name}...", flush=True)
        pf = pq.ParquetFile(p)
        cols = ["country", "name_skel", "addr_tokens", "num_tokens", "zip_pin"]
        avail_cols = [c for c in cols if c in pf.schema.names]
        
        for batch in pf.iter_batches(batch_size=100_000, columns=avail_cols):
            b_df = batch.to_pandas()
            total_records += len(b_df)
            
            c_arr = b_df["country"].to_numpy()
            sk_arr = b_df["name_skel"].to_numpy() if "name_skel" in b_df else np.array([""] * len(b_df))
            ad_arr = b_df["addr_tokens"].to_numpy() if "addr_tokens" in b_df else np.array([""] * len(b_df))
            num_arr = b_df["num_tokens"].to_numpy() if "num_tokens" in b_df else np.array([""] * len(b_df))
            zp_arr = b_df["zip_pin"].to_numpy() if "zip_pin" in b_df else np.array([""] * len(b_df))
            
            for ctry, sk, ad, num, zp in zip(c_arr, sk_arr, ad_arr, num_arr, zp_arr):
                c = str(ctry).strip()
                if not c or c == "nan":
                    continue
                    
                # K1: (country, two consecutive numeric tokens in address, each len >= 2)
                num_str = str(num).strip()
                if num_str and num_str != "nan":
                    nums = [t for t in num_str.split() if len(t) >= 2]
                    for idx in range(len(nums) - 1):
                        k1_counts[f"K1_{c}_{nums[idx]}_{nums[idx+1]}"] += 1
                        
                # K2: (country, zip_pin, first alphabetic address token of length >= 4)
                z = str(zp).strip()
                if z and z != "nan":
                    for t in str(ad).split():
                        if t.isalpha() and len(t) >= 4:
                            k2_counts[f"K2_{c}_{z}_{t}"] += 1
                            break
                            
                # K3: (country, sorted set of all numeric address tokens)
                if num_str and num_str != "nan":
                    nums_all = sorted(set(num_str.split()))
                    if nums_all:
                        k3_counts[f"K3_{c}_" + "_".join(nums_all)] += 1
                        
                # K4: (country, first 2 sorted name_skel tokens)
                sk_str = str(sk).strip()
                if sk_str and sk_str != "nan":
                    sk_toks = sorted(sk_str.split())
                    if len(sk_toks) >= 2:
                        k4_counts[f"K4_{c}_{sk_toks[0]}_{sk_toks[1]}"] += 1
                        
                # K5: (country, any numeric token len >= 3, rarest alphabetic address token len >= 5)
                if num_str and num_str != "nan":
                    nums_ge3 = [t for t in set(num_str.split()) if len(t) >= 3]
                    if nums_ge3:
                        alpha_toks = [t for t in set(str(ad).split()) if t.isalpha() and len(t) >= 5]
                        if alpha_toks:
                            rarest = min(alpha_toks, key=lambda t: (doc_freqs.get((c, t), 0), t))
                            for n in nums_ge3:
                                k5_counts[f"K5_{c}_{n}_{rarest}"] += 1
                                
            del b_df, batch
        gc.collect()
        print(f"  Finished {src_name} | Total pool records: {total_records:,} | RSS: {_rss_mb():.1f} MB", flush=True)

    t_s23 = time.time() - t0
    print(f"Streamed full pool ({total_records:,} records) in {t_s23:.1f}s.")
    print(f"Unique blocks: K1={len(k1_counts):,}, K2={len(k2_counts):,}, K3={len(k3_counts):,}, K4={len(k4_counts):,}, K5={len(k5_counts):,}")
    return k1_counts, k2_counts, k3_counts, k4_counts, k5_counts


# ==============================================================================
# REPORT GENERATION (A, B, C)
# ==============================================================================

def generate_report_a(missed_df: pd.DataFrame) -> str:
    """Generates Report (a): % of missed pairs caught by each key alone and in union, by country."""
    lines = [
        "",
        "=" * 90,
        "(a) PERCENTAGE OF MISSED PAIRS CAUGHT BY CANDIDATE KEYS (SPLIT BY COUNTRY)",
        "=" * 90,
    ]
    
    countries = ["India", "US"]
    total_all = len(missed_df)
    n_in = len(missed_df[missed_df["country"] == "India"])
    n_us = len(missed_df[missed_df["country"] == "US"])
    
    header = f"{'Candidate Key':<46} | {'India (N=' + str(n_in) + ')':<18} | {'US (N=' + str(n_us) + ')':<18} | {'Overall (N=' + str(total_all) + ')':<18}"
    lines.append(header)
    lines.append("-" * len(header))
    
    key_configs = [
        ("K1 (2 consec num in addr, len>=2)", "k1_caught"),
        ("K2 (zip_pin + 1st alpha addr, len>=4)", "k2_caught"),
        ("K3 (sorted set of all num addr tokens)", "k3_caught"),
        ("K4 (first 2 sorted name_skel tokens)", "k4_caught"),
        ("K5 (num len>=3 + rarest alpha len>=5)", "k5_caught"),
        ("Union (K1 - K5)", "union_caught"),
    ]
    
    for label, col in key_configs:
        if label.startswith("Union"):
            lines.append("-" * len(header))
            
        c_in = missed_df[missed_df["country"] == "India"][col].sum() if n_in > 0 else 0
        pct_in = (c_in / n_in * 100.0) if n_in > 0 else 0.0
        
        c_us = missed_df[missed_df["country"] == "US"][col].sum() if n_us > 0 else 0
        pct_us = (c_us / n_us * 100.0) if n_us > 0 else 0.0
        
        c_all = missed_df[col].sum() if total_all > 0 else 0
        pct_all = (c_all / total_all * 100.0) if total_all > 0 else 0.0
        
        s_in = f"{pct_in:6.2f}% ({c_in:,})"
        s_us = f"{pct_us:6.2f}% ({c_us:,})"
        s_all = f"{pct_all:6.2f}% ({c_all:,})"
        
        lines.append(f"{label:<46} | {s_in:<18} | {s_us:<18} | {s_all:<18}")

    lines.append("=" * 90)
    return "\n".join(lines)


def generate_report_b(
    missed_df: pd.DataFrame,
    counters: dict[str, Counter[str]],
) -> str:
    """
    Generates Report (b): Block-size distribution on full S2+S3 pool (p50, p95, p99, max)
    and % of missed pairs still caught if blocks larger than 50 or 100 are dropped.
    """
    lines = [
        "",
        "=" * 116,
        "(b) BLOCK-SIZE DISTRIBUTION (FULL TRAIN S2+S3 POOL) & MISSED PAIRS RECOVERED UNDER BLOCK CAPS",
        "=" * 116,
    ]
    
    header = (
        f"{'Key':<8} | {'Unique Blocks':<14} | {'p50':<6} | {'p95':<6} | {'p99':<6} | {'Max':<6} | "
        f"{'Caught (No Cap)':<17} | {'Caught (Cap<=100)':<17} | {'Caught (Cap<=50)':<17}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    
    total_missed = len(missed_df)
    
    keys_order = ["K1", "K2", "K3", "K4", "K5"]
    col_map = {
        "K1": ("k1_caught", "k1_shared_keys"),
        "K2": ("k2_caught", "k2_shared_keys"),
        "K3": ("k3_caught", "k3_shared_keys"),
        "K4": ("k4_caught", "k4_shared_keys"),
        "K5": ("k5_caught", "k5_shared_keys"),
    }
    
    # Track union under caps
    union_caught_no_cap = missed_df["union_caught"].sum()
    union_caught_100 = 0
    union_caught_50 = 0
    
    all_block_sizes: list[int] = []
    
    for k_name in keys_order:
        c = counters[k_name]
        sizes = list(c.values())
        all_block_sizes.extend(sizes)
        
        n_blocks = len(sizes)
        if n_blocks > 0:
            p50 = float(np.percentile(sizes, 50))
            p95 = float(np.percentile(sizes, 95))
            p99 = float(np.percentile(sizes, 99))
            m_val = int(np.max(sizes))
        else:
            p50, p95, p99, m_val = 0.0, 0.0, 0.0, 0
            
        caught_col, shared_col = col_map[k_name]
        c_no_cap = missed_df[caught_col].sum()
        pct_no_cap = (c_no_cap / total_missed * 100.0) if total_missed > 0 else 0.0
        
        # Check cap 100 and cap 50
        c_100 = 0
        c_50 = 0
        for keys_list in missed_df[shared_col]:
            if any(c[k] <= 100 for k in keys_list):
                c_100 += 1
            if any(c[k] <= 50 for k in keys_list):
                c_50 += 1
                
        pct_100 = (c_100 / total_missed * 100.0) if total_missed > 0 else 0.0
        pct_50 = (c_50 / total_missed * 100.0) if total_missed > 0 else 0.0
        
        s_no_cap = f"{pct_no_cap:6.2f}% ({c_no_cap:,})"
        s_100 = f"{pct_100:6.2f}% ({c_100:,})"
        s_50 = f"{pct_50:6.2f}% ({c_50:,})"
        
        lines.append(
            f"{k_name:<8} | {n_blocks:<14,} | {p50:<6.1f} | {p95:<6.1f} | {p99:<6.1f} | {m_val:<6,} | "
            f"{s_no_cap:<17} | {s_100:<17} | {s_50:<17}"
        )

    # Union row
    lines.append("-" * len(header))
    for _, row in missed_df.iterrows():
        # Caught under cap 100 by ANY key
        caught_u100 = any(
            any(counters[k_name][k] <= 100 for k in row[col_map[k_name][1]])
            for k_name in keys_order
        )
        if caught_u100:
            union_caught_100 += 1
            
        # Caught under cap 50 by ANY key
        caught_u50 = any(
            any(counters[k_name][k] <= 50 for k in row[col_map[k_name][1]])
            for k_name in keys_order
        )
        if caught_u50:
            union_caught_50 += 1

    total_union_blocks = len(all_block_sizes)
    if total_union_blocks > 0:
        u_p50 = float(np.percentile(all_block_sizes, 50))
        u_p95 = float(np.percentile(all_block_sizes, 95))
        u_p99 = float(np.percentile(all_block_sizes, 99))
        u_max = int(np.max(all_block_sizes))
    else:
        u_p50, u_p95, u_p99, u_max = 0.0, 0.0, 0.0, 0

    pct_u_no_cap = (union_caught_no_cap / total_missed * 100.0) if total_missed > 0 else 0.0
    pct_u_100 = (union_caught_100 / total_missed * 100.0) if total_missed > 0 else 0.0
    pct_u_50 = (union_caught_50 / total_missed * 100.0) if total_missed > 0 else 0.0
    
    s_u_no_cap = f"{pct_u_no_cap:6.2f}% ({union_caught_no_cap:,})"
    s_u_100 = f"{pct_u_100:6.2f}% ({union_caught_100:,})"
    s_u_50 = f"{pct_u_50:6.2f}% ({union_caught_50:,})"
    
    lines.append(
        f"{'Union':<8} | {total_union_blocks:<14,} | {u_p50:<6.1f} | {u_p95:<6.1f} | {u_p99:<6.1f} | {u_max:<6,} | "
        f"{s_u_no_cap:<17} | {s_u_100:<17} | {s_u_50:<17}"
    )
    lines.append("=" * 116)
    return "\n".join(lines)


def generate_summary_stats(missed_df: pd.DataFrame) -> str:
    """Generates overall summary of missed pairs, feature distributions, and embedding gaps."""
    lines = [
        "",
        "=" * 90,
        "SUMMARY OF MISSED PAIR SIMILARITIES & EMBEDDING COMPARISONS",
        "=" * 90,
    ]
    total = len(missed_df)
    n_in = len(missed_df[missed_df["country"] == "India"])
    n_us = len(missed_df[missed_df["country"] == "US"])
    lines.append(f"Total Missed Pairs: {total:,} (India: {n_in:,}, US: {n_us:,})")
    lines.append("")
    
    # Feature distributions
    def _dist_str(vals: pd.Series) -> str:
        v = vals.dropna().to_numpy()
        if len(v) == 0:
            return "N/A"
        return f"mean={np.mean(v):.2f}, p25={np.percentile(v, 25):.2f}, median={np.percentile(v, 50):.2f}, p75={np.percentile(v, 75):.2f}"

    lines.append(f"Name token_sort ratio:       {_dist_str(missed_df['name_token_sort'])}")
    lines.append(f"Core name ratio:             {_dist_str(missed_df['core_ratio'])}")
    lines.append(f"Address token_set ratio:     {_dist_str(missed_df['addr_token_set'])}")
    lines.append(f"Shared numeric tokens count: {_dist_str(missed_df['shared_num_tokens_count'])}")
    
    # House match breakdown
    h_m1 = (missed_df["house_no_match"] == 1).mean() * 100.0
    h_0 = (missed_df["house_no_match"] == 0).mean() * 100.0
    h_missing = (missed_df["house_no_match"] == -1).mean() * 100.0
    lines.append(f"House number match:          +1 (exact match): {h_m1:.1f}%,  0 (mismatch): {h_0:.1f}%,  -1 (missing): {h_missing:.1f}%")

    # Zip match breakdown
    z_m1 = (missed_df["zip_match"] == 1).mean() * 100.0
    z_0 = (missed_df["zip_match"] == 0).mean() * 100.0
    z_missing = (missed_df["zip_match"] == -1).mean() * 100.0
    lines.append(f"Zip match:                   +1 (exact match): {z_m1:.1f}%,  0 (mismatch): {z_0:.1f}%,  -1 (missing): {z_missing:.1f}%")
    lines.append("")

    # Embedding score comparison vs S1's 20th-best
    valid_emb = missed_df[missed_df["pair_emb_score"].notna()]
    lines.append(f"Missed Pair emb_score:       {_dist_str(missed_df['pair_emb_score'])}")
    lines.append(f"S1 20th-best emb_score:      {_dist_str(missed_df['s1_20th_emb_score'])}")
    lines.append(f"emb_score difference (pair - 20th): {_dist_str(missed_df['emb_score_diff'])}")
    
    higher_count = (missed_df["emb_score_diff"] >= 0).sum()
    higher_pct = (higher_count / total * 100.0) if total > 0 else 0.0
    lines.append(f"Pairs with emb_score >= S1's 20th-best: {higher_count:,} / {total:,} ({higher_pct:.2f}%)")
    lines.append("=" * 90)
    return "\n".join(lines)


def save_misses_sample_tsv(
    missed_df: pd.DataFrame,
    output_path: str,
    n_samples: int = 50,
    seed: int = 42,
) -> None:
    """Saves 50 random missed examples to TSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if len(missed_df) <= n_samples:
        sample_df = missed_df.copy()
    else:
        sample_df = missed_df.sample(n=n_samples, random_state=seed).copy()
        
    out_cols = [
        "s1_id", "cand_id", "country", "cand_source",
        "s1_name", "cand_name", "s1_addr", "cand_addr",
        "name_token_sort", "core_ratio", "addr_token_set",
        "house_no_match", "shared_num_tokens_count", "zip_match",
        "k1_caught", "k2_caught", "k3_caught", "k4_caught", "k5_caught", "union_caught",
        "pair_emb_score", "s1_20th_emb_score", "emb_score_diff",
    ]
    sample_df[out_cols].to_csv(output_path, sep="\t", index=False)
    print(f"\nSaved {len(sample_df)} random missed examples to {output_path}")


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze missed blocking ground truth pairs.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--laptop-test", action="store_true", help="Use cache/laptop_test")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of val S1 queries for fast local testing")
    parser.add_argument("--output-tsv", type=str, default=None, help="Output path for misses sample TSV")
    args = parser.parse_args()

    start_time = time.time()
    initial_rss = _rss_mb()
    print(f"=== Starting analyze_misses.py (Initial RSS: {initial_rss:.1f} MB) ===")

    # Determine directories
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.laptop_test:
        cache_dir = os.path.join(root_dir, "cache", "laptop_test")
    else:
        cache_dir = os.path.join(root_dir, "cache")
        
    logs_dir = os.path.join(root_dir, "logs")
    output_tsv = args.output_tsv or os.path.join(logs_dir, "misses_sample.tsv")
    
    print(f"Cache Directory: {cache_dir}")
    print(f"Output Sample TSV: {output_tsv}")
    if args.limit:
        print(f"Limit: {args.limit} val S1 queries")

    # 1. Load val S1 IDs
    val_s1_list = load_val_s1_ids(cache_dir, limit=args.limit)
    val_s1_set = set(val_s1_list)
    print(f"Val S1 set size: {len(val_s1_set):,} entities (RSS: {_rss_mb():.1f} MB)")

    # 2. Load Ground Truth restricted to val S1
    gt_val = load_ground_truth(cache_dir, root_dir, val_s1_set)

    # 3. Stream cands_train_chunk_*.parquet to gather candidates and S1 20th-best emb_scores
    missed_records, s1_20th_best = stream_val_candidates_and_find_misses(cache_dir, val_s1_set, gt_val)
    if not missed_records:
        print("All GT pairs were caught! No missed pairs to analyze.")
        return

    # 4. Restrict normalized tables strictly to needed IDs
    needed_s1 = {r["s1_id"] for r in missed_records}
    needed_cand = {r["cand_id"] for r in missed_records}
    s1_dict, cand_dict = load_restricted_norm_tables(cache_dir, needed_s1, needed_cand)

    # 5. Compute embedding similarities for missed pairs (mmap mode)
    pair_emb_scores = compute_missed_pair_embeddings(missed_records, cache_dir)

    # 6. Load address token document frequencies (for K5)
    doc_freqs = load_or_compute_doc_freqs(cache_dir)

    # 7. Compute pair features and candidate keys
    missed_df = compute_missed_pair_features(
        missed_records, s1_dict, cand_dict, pair_emb_scores, doc_freqs
    )
    del s1_dict, cand_dict, pair_emb_scores, missed_records
    gc.collect()

    # 8. Stream S2+S3 normalized tables to count block sizes for K1-K5
    k1_c, k2_c, k3_c, k4_c, k5_c = stream_s23_block_sizes(cache_dir, doc_freqs)
    counters = {
        "K1": k1_c,
        "K2": k2_c,
        "K3": k3_c,
        "K4": k4_c,
        "K5": k5_c,
    }

    # 9. Output Report (a): % caught alone and in union by country
    report_a = generate_report_a(missed_df)
    print(report_a)

    # 10. Output Report (b): block-size distribution and % caught under cap 50 and 100
    report_b = generate_report_b(missed_df, counters)
    print(report_b)

    # 11. Output Feature and Embedding Summary
    summary_stats = generate_summary_stats(missed_df)
    print(summary_stats)

    # 12. Save 50 random missed examples to logs/misses_sample.tsv
    save_misses_sample_tsv(missed_df, output_tsv, n_samples=50, seed=42)

    total_time = time.time() - start_time
    peak_rss = _rss_mb()
    print(f"\nCompleted analyze_misses.py in {total_time:.1f}s. Final Peak RSS: {peak_rss:.1f} MB (strictly < 6,000 MB)")


if __name__ == "__main__":
    main()

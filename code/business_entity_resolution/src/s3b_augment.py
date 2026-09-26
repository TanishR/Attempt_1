#!/usr/bin/env python3
"""
Candidate Augmentation (Stage 3b): Adds high-precision candidate pairs on top of the
existing candidate pairs in cands_{split}_chunk_*.parquet to resolve recall bottlenecks.

Channels:
  Channel F (Wide Name Retrieval + Address Re-ranking, GPU):
    - Top 200 nearest neighbors by name embedding similarity (main + alt max, like Channel A).
    - Re-ranked on GPU using int32 address token ID arrays (numeric tokens + alpha tokens len >= 4).
    - Scored by sum of IDF of shared tokens; tie-break by emb_score.
    - Top 10 per source (not already in existing candidates) kept.
  Channel G (Address Keys):
    - Keys K5, K3, K1 within country and per source.
    - Block size cap <= 50 (larger blocks dropped).
    - At most 10 per S1 per key, prioritized by emb_score.

Merge Rules:
  - Add on top of existing candidate rows, max 20 extra per S1.
  - Priority: number of new channels that found the pair, then F score, then emb_score.
  - Never remove or alter existing rows.
  - New columns: ch_rerank, ch_k1, ch_k3, ch_k5 (int8; 0 for existing rows).
  - New pairs get emb_score as stored embedding dot product, and emb_rank = 999.0.

Safety:
  - Requires cache/cands_backup_{split}/ containing all original chunk files.
  - Idempotent: skips chunks that already have 'ch_rerank'.
  - One chunk at a time, resuming gracefully.
  - Peak RSS < 20 GB, GPU memory < 20 GB.

Usage:
  python code/business_entity_resolution/src/s3b_augment.py --split train [--laptop-test] [--max-chunks 1]
  python code/business_entity_resolution/src/s3b_augment.py --split test
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
import torch

from config import CAND_CAP, EMB_DIM, K_PER_SOURCE


def _rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


# ==============================================================================
# CANDIDATE KEYS K1, K3, K5
# ==============================================================================

def extract_k1(country: str, num_tokens_str: str) -> set[str]:
    """K1: (country, two consecutive numeric tokens in the address, each of length >= 2)."""
    if not country or not num_tokens_str or num_tokens_str == "nan":
        return set()
    nums = [t for t in num_tokens_str.split() if len(t) >= 2]
    keys = set()
    for i in range(len(nums) - 1):
        keys.add(f"K1_{country}_{nums[i]}_{nums[i+1]}")
    return keys


def extract_k3(country: str, num_tokens_str: str) -> set[str]:
    """K3: (country, sorted set of all numeric address tokens)."""
    if not country or not num_tokens_str or num_tokens_str == "nan":
        return set()
    nums = sorted(set(num_tokens_str.split()))
    if not nums:
        return set()
    return {f"K3_{country}_" + "_".join(nums)}


def extract_k5(country: str, num_tokens_str: str, addr_norm_str: str, doc_freqs: dict[tuple[str, str], int]) -> set[str]:
    """K5: (country, any numeric token of length >= 3, rarest alphabetic address token of length >= 5)."""
    if not country or not num_tokens_str or num_tokens_str == "nan":
        return set()
    nums_ge3 = [t for t in set(num_tokens_str.split()) if len(t) >= 3]
    if not nums_ge3:
        return set()
    alpha_toks = [t for t in set(str(addr_norm_str).split()) if t.isalpha() and len(t) >= 5]
    if not alpha_toks:
        return set()
    rarest = min(alpha_toks, key=lambda t: (doc_freqs.get((country, t), 0), t))
    keys = set()
    for n in nums_ge3:
        keys.add(f"K5_{country}_{n}_{rarest}")
    return keys


# ==============================================================================
# TOKENIZATION & IDF COMPUTATION FOR CHANNEL F
# ==============================================================================

def tokenize_address_f(addr_str: str) -> list[str]:
    """Numeric tokens + alpha tokens of length >= 4, deduplicated in order of appearance."""
    if not addr_str or not isinstance(addr_str, str) or addr_str == "nan":
        return []
    toks = []
    seen = set()
    for t in addr_str.split():
        if t.isdigit() or (t.isalpha() and len(t) >= 4):
            if t not in seen:
                seen.add(t)
                toks.append(t)
    return toks


def load_doc_freqs(cache_dir: str, split: str) -> dict[tuple[str, str], int]:
    """Loads cache/addr_df_{split}.parquet."""
    path = os.path.join(cache_dir, f"addr_df_{split}.parquet")
    if not os.path.exists(path) and split == "test":
        train_path = os.path.join(cache_dir, "addr_df_train.parquet")
        if os.path.exists(train_path):
            path = train_path
    doc_freqs: dict[tuple[str, str], int] = {}
    if os.path.exists(path):
        print(f"Loading document frequencies from {path}...")
        df_p = pd.read_parquet(path)
        for ctry, tok, df_val in zip(df_p["country"], df_p["token"], df_p["doc_freq"]):
            doc_freqs[(str(ctry), str(tok))] = int(df_val)
        print(f"  Loaded {len(doc_freqs):,} document frequencies (RSS: {_rss_mb():.1f} MB)")
    else:
        print(f"Warning: {path} not found. IDF scores will use default frequencies.")
    return doc_freqs


# ==============================================================================
# EMBEDDINGS LOADING HELPER
# ==============================================================================

def load_embeddings(cache_dir: str, split: str, source: str) -> tuple[np.ndarray, dict[str, int], np.ndarray, np.ndarray, dict[str, int], np.ndarray]:
    """Loads main and alt embeddings for source."""
    m_p = os.path.join(cache_dir, f"emb_{split}_{source}.npy")
    m_id_p = os.path.join(cache_dir, f"ids_{split}_{source}.npy")
    
    if not os.path.exists(m_p) or not os.path.exists(m_id_p):
        return (np.zeros((0, EMB_DIM), dtype=np.float16), {}, np.array([], dtype=object),
                np.zeros((0, EMB_DIM), dtype=np.float16), {}, np.array([], dtype=object))
        
    main_emb = np.load(m_p)
    main_ids = np.load(m_id_p, allow_pickle=True)
    main_map = {eid: idx for idx, eid in enumerate(main_ids)}
    
    a_p = os.path.join(cache_dir, f"emb_{split}_{source}_alt.npy")
    a_id_p = os.path.join(cache_dir, f"ids_{split}_{source}_alt.npy")
    
    if os.path.exists(a_p) and os.path.exists(a_id_p):
        alt_emb = np.load(a_p)
        alt_ids = np.load(a_id_p, allow_pickle=True)
        alt_map = {eid: idx for idx, eid in enumerate(alt_ids)}
    else:
        alt_emb = np.zeros((0, EMB_DIM), dtype=np.float16)
        alt_ids = np.array([], dtype=object)
        alt_map = {}
        
    return main_emb, main_map, main_ids, alt_emb, alt_map, alt_ids


# ==============================================================================
# SAFETY CHECK: BACKUP VERIFICATION
# ==============================================================================

def verify_backup_safety(cache_dir: str, split: str) -> list[str]:
    """Verifies that cache/cands_backup_{split}/ exists and contains all original chunk files."""
    backup_dir = os.path.join(cache_dir, f"cands_backup_{split}")
    if not os.path.isdir(backup_dir):
        raise RuntimeError(
            f"SAFETY CHECK FAILED: Required backup directory '{backup_dir}' does not exist! "
            f"You must create this backup containing all original chunk files before running augmentation."
        )

    chunk_pattern = os.path.join(cache_dir, f"cands_{split}_chunk_*.parquet")
    chunk_files = sorted(glob.glob(chunk_pattern))
    if not chunk_files:
        raise FileNotFoundError(f"No candidate chunk files found matching '{chunk_pattern}'.")

    backup_chunk_files = sorted(glob.glob(os.path.join(backup_dir, f"cands_{split}_chunk_*.parquet")))
    if len(backup_chunk_files) != len(chunk_files):
        raise RuntimeError(
            f"SAFETY CHECK FAILED: Backup directory '{backup_dir}' contains {len(backup_chunk_files)} chunk files, "
            f"but {len(chunk_files)} chunk files exist in '{cache_dir}'. All original chunk files must be backed up!"
        )

    for cf in chunk_files:
        bf = os.path.join(backup_dir, os.path.basename(cf))
        if not os.path.exists(bf) or os.path.getsize(bf) == 0:
            raise RuntimeError(f"SAFETY CHECK FAILED: Backup file '{bf}' is missing or empty!")

    print(f"Safety Check PASSED: Backup directory '{backup_dir}' verified ({len(backup_chunk_files)} chunk files).")
    return chunk_files


# ==============================================================================
# S2 & S3 TARGET INDEX PREPARATION
# ==============================================================================

class TargetSourceIndex:
    """Precomputed target index for a source (source2 or source3)."""

    def __init__(
        self,
        source_name: str,
        df: pd.DataFrame,
        main_emb: np.ndarray,
        main_map: dict[str, int],
        main_ids: np.ndarray,
        alt_emb: np.ndarray,
        alt_map: dict[str, int],
        alt_ids: np.ndarray,
        doc_freqs: dict[tuple[str, str], int],
        max_pad_len: int = 24,
    ) -> None:
        self.source_name = source_name
        self.max_pad_len = max_pad_len
        self.main_emb = main_emb
        self.main_map = main_map
        self.main_ids = main_ids
        self.alt_emb = alt_emb
        self.alt_map = alt_map
        self.alt_ids = alt_ids
        
        # Candidate table data
        self.df = df
        self.entity_ids = df["entity_id"].values
        self.countries = df["country"].values
        self.id_to_row_idx = {eid: idx for idx, eid in enumerate(self.entity_ids)}

        # Country partitions
        self.cand_indices_by_country: dict[str, list[int]] = defaultdict(list)
        for idx, c in enumerate(self.countries):
            c_str = str(c).strip()
            if c_str:
                self.cand_indices_by_country[c_str].append(idx)

        # Inverted index for Channel G (K5, K3, K1)
        print(f"Building Channel G inverted indices for {source_name}...", flush=True)
        t0 = time.time()
        raw_k1: dict[str, list[str]] = defaultdict(list)
        raw_k3: dict[str, list[str]] = defaultdict(list)
        raw_k5: dict[str, list[str]] = defaultdict(list)
        
        num_arr = df["num_tokens"].to_numpy() if "num_tokens" in df else np.array([""] * len(df))
        ad_arr = df["addr_norm"].to_numpy() if "addr_norm" in df else np.array([""] * len(df))

        for eid, ctry, num, ad in zip(self.entity_ids, self.countries, num_arr, ad_arr):
            c = str(ctry).strip()
            if not c:
                continue
            n_str = str(num).strip()
            a_str = str(ad).strip()
            
            # K1
            for k in extract_k1(c, n_str):
                raw_k1[k].append(eid)
            # K3
            for k in extract_k3(c, n_str):
                raw_k3[k].append(eid)
            # K5
            for k in extract_k5(c, n_str, a_str, doc_freqs):
                raw_k5[k].append(eid)

        # Prune blocks larger than 50
        self.k1_index = {k: v for k, v in raw_k1.items() if len(v) <= 50}
        self.k3_index = {k: v for k, v in raw_k3.items() if len(v) <= 50}
        self.k5_index = {k: v for k, v in raw_k5.items() if len(v) <= 50}
        print(f"  {source_name} Channel G indices built in {time.time() - t0:.1f}s | "
              f"K1={len(self.k1_index):,} (dropped {len(raw_k1) - len(self.k1_index):,}), "
              f"K3={len(self.k3_index):,} (dropped {len(raw_k3) - len(self.k3_index):,}), "
              f"K5={len(self.k5_index):,} (dropped {len(raw_k5) - len(self.k5_index):,}) | "
              f"RSS: {_rss_mb():.1f} MB", flush=True)
        del raw_k1, raw_k3, raw_k5
        gc.collect()

        # Token arrays for Channel F address re-ranking
        print(f"Tokenizing addresses for {source_name} Channel F...", flush=True)
        self.toks_list = [tokenize_address_f(a) for a in ad_arr]
        print(f"  Tokenized {len(self.toks_list):,} addresses (RSS: {_rss_mb():.1f} MB)", flush=True)


# ==============================================================================
# MAIN AUGMENTATION PIPELINE
# ==============================================================================

def run_augmentation(
    cache_dir: str,
    split: str,
    chunk_files: list[str],
    max_chunks: Optional[int] = None,
    batch_size: int = 5000,
) -> None:
    """Executes Stage 3b candidate augmentation chunk by chunk."""
    total_start = time.time()
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== Running Stage 3b Candidate Augmentation ({split.upper()}) ===")
    print(f"Compute Device: {device} | Initial RSS: {_rss_mb():.1f} MB")

    if max_chunks is not None and max_chunks > 0:
        chunk_files = chunk_files[:max_chunks]
        print(f"Limiting execution to first {len(chunk_files)} chunk(s).")

    # 1. Load document frequencies
    doc_freqs = load_doc_freqs(cache_dir, split)

    # 2. Load Source 2 & Source 3 normalized tables
    print("\nLoading normalized tables for Source 2 and Source 3...")
    cols = ["entity_id", "country", "num_tokens", "addr_norm"]
    p_s2 = os.path.join(cache_dir, f"norm_{split}_source2.parquet")
    p_s3 = os.path.join(cache_dir, f"norm_{split}_source3.parquet")
    
    def _read_norm_cols(p: str) -> pd.DataFrame:
        if not os.path.exists(p):
            return pd.DataFrame(columns=cols)
        schema_names = pq.ParquetFile(p).schema.names
        load_cols = [c for c in cols if c in schema_names]
        return pd.read_parquet(p, columns=load_cols)

    df_s2 = _read_norm_cols(p_s2)
    df_s3 = _read_norm_cols(p_s3)
    print(f"Loaded normalized records: S2={len(df_s2):,}, S3={len(df_s3):,} (RSS: {_rss_mb():.1f} MB)")

    # 3. Load embeddings for Source 2 & Source 3
    print("Loading embeddings for Source 2 and Source 3...")
    s2_m_emb, s2_m_map, s2_m_ids, s2_a_emb, s2_a_map, s2_a_ids = load_embeddings(cache_dir, split, "source2")
    s3_m_emb, s3_m_map, s3_m_ids, s3_a_emb, s3_a_map, s3_a_ids = load_embeddings(cache_dir, split, "source3")
    print(f"Loaded embedding vectors (RSS: {_rss_mb():.1f} MB)")

    # 4. Build Target Indices for S2 and S3
    idx_s2 = TargetSourceIndex("source2", df_s2, s2_m_emb, s2_m_map, s2_m_ids, s2_a_emb, s2_a_map, s2_a_ids, doc_freqs)
    idx_s3 = TargetSourceIndex("source3", df_s3, s3_m_emb, s3_m_map, s3_m_ids, s3_a_emb, s3_a_map, s3_a_ids, doc_freqs)
    del df_s2, df_s3
    gc.collect()

    # 5. Load S1 normalized table and embeddings
    print("\nLoading Source 1 normalized table and embeddings...")
    p_s1 = os.path.join(cache_dir, f"norm_{split}_source1.parquet")
    s1_norm = _read_norm_cols(p_s1)
    s1_norm_map = {r["entity_id"]: r for r in s1_norm.to_dict("records")}
    s1_m_emb, s1_m_map, s1_m_ids, s1_a_emb, s1_a_map, s1_a_ids = load_embeddings(cache_dir, split, "source1")
    print(f"Loaded Source 1: {len(s1_norm):,} records (RSS: {_rss_mb():.1f} MB)")

    # 6. Load Ground Truth for label assignment if train
    gt_pairs: set[tuple[str, str]] = set()
    if split == "train":
        gt_path = os.path.join(cache_dir, "gt_long.parquet")
        if os.path.exists(gt_path):
            gt_df = pd.read_parquet(gt_path, columns=["s1_id", "match_id"])
            gt_pairs = set(zip(gt_df["s1_id"], gt_df["match_id"]))
            print(f"Loaded ground truth: {len(gt_pairs):,} pairs")

    # 7. Helper: compute pairwise dot product embedding score
    def _calc_emb_score(s1_id: str, cand_id: str, is_s2: bool) -> float:
        cm_emb, cm_map, ca_emb, ca_map = (s2_m_emb, s2_m_map, s2_a_emb, s2_a_map) if is_s2 else (s3_m_emb, s3_m_map, s3_a_emb, s3_a_map)
        if s1_id not in s1_m_map or cand_id not in cm_map:
            return 0.0
        v1_m = s1_m_emb[s1_m_map[s1_id]].astype(np.float32)
        v2_m = cm_emb[cm_map[cand_id]].astype(np.float32)
        best_s = float(np.dot(v1_m, v2_m))
        
        has_s1_a = s1_id in s1_a_map
        has_c_a = cand_id in ca_map
        if has_c_a:
            v2_a = ca_emb[ca_map[cand_id]].astype(np.float32)
            s_ma = float(np.dot(v1_m, v2_a))
            if s_ma > best_s: best_s = s_ma
        if has_s1_a:
            v1_a = s1_a_emb[s1_a_map[s1_id]].astype(np.float32)
            s_am = float(np.dot(v1_a, v2_m))
            if s_am > best_s: best_s = s_am
            if has_c_a:
                s_aa = float(np.dot(v1_a, v2_a))
                if s_aa > best_s: best_s = s_aa
        return best_s

    # 8. Process each chunk
    print(f"\nProcessing {len(chunk_files)} chunk(s)...")
    for ch_idx, chunk_path in enumerate(chunk_files):
        t_ch_start = time.time()
        print(f"\n--- Chunk {ch_idx + 1}/{len(chunk_files)}: {chunk_path} ---", flush=True)

        # Check idempotence: if already has ch_rerank, skip
        pf = pq.ParquetFile(chunk_path)
        if "ch_rerank" in pf.schema.names:
            print(f"Chunk already contains 'ch_rerank', skipping (idempotent).")
            continue

        chunk_df = pd.read_parquet(chunk_path)
        existing_rows_count = len(chunk_df)
        chunk_s1_ids = chunk_df["s1_id"].unique()
        print(f"  Existing rows: {existing_rows_count:,} across {len(chunk_s1_ids):,} unique S1s (RSS: {_rss_mb():.1f} MB)")

        # Map existing candidates per S1
        existing_cands_map: dict[str, set[str]] = defaultdict(set)
        for s1, cid in zip(chunk_df["s1_id"], chunk_df["cand_id"]):
            existing_cands_map[s1].add(cid)

        # Group this chunk's S1 IDs by country
        s1_by_country: dict[str, list[str]] = defaultdict(list)
        for sid in chunk_s1_ids:
            r = s1_norm_map.get(sid)
            c = str(r.get("country", "")).strip() if r else ""
            if c:
                s1_by_country[c].append(sid)

        # Candidate collectors for this chunk: s1_id -> cand_id -> candidate dict
        # candidate dict: {'ch_rerank': 0, 'ch_k1': 0, 'ch_k3': 0, 'ch_k5': 0, 'f_score': 0.0, 'emb_score': float, 'is_s2': bool}
        augmented_candidates: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

        # ----------------------------------------------------------------------
        # CHANNEL F: Wide Retrieval (top 200) + Address Re-ranking (GPU)
        # ----------------------------------------------------------------------
        print(f"  Running Channel F (Wide 200 + GPU Re-ranking)...", flush=True)
        t_f0 = time.time()
        
        for ctry, s1_ids_c in s1_by_country.items():
            for target_idx, is_s2 in [(idx_s2, True), (idx_s3, False)]:
                cand_pool_indices = target_idx.cand_indices_by_country.get(ctry, [])
                if not cand_pool_indices or not s1_ids_c:
                    continue

                cand_ids_pool = target_idx.entity_ids[cand_pool_indices]
                
                # Stack target embeddings (main + alt)
                c_m_indices = [target_idx.main_map[cid] for cid in cand_ids_pool if cid in target_idx.main_map]
                X_main = target_idx.main_emb[c_m_indices]
                X_main_cids = cand_ids_pool[[cid in target_idx.main_map for cid in cand_ids_pool]]

                c_a_cids = [cid for cid in cand_ids_pool if cid in target_idx.alt_map]
                if c_a_cids:
                    c_a_indices = [target_idx.alt_map[cid] for cid in c_a_cids]
                    X_alt = target_idx.alt_emb[c_a_indices]
                    X_alt_cids = np.array(c_a_cids, dtype=object)
                    X_stacked = np.vstack([X_main, X_alt])
                    row_to_cid = np.concatenate([X_main_cids, X_alt_cids])
                else:
                    X_stacked = X_main
                    row_to_cid = X_main_cids

                N_index = len(X_stacked)
                if N_index == 0:
                    continue

                X_gpu = torch.from_numpy(X_stacked).to(device)
                k_search = min(200, N_index)

                # Token vocabulary & IDF for this country
                max_docs_ctry = max(1, len(cand_pool_indices))
                tok_to_id: dict[str, int] = {}
                idf_weights_list: list[float] = [0.0]  # index 0 = padding
                
                # Build token arrays for candidates in pool
                cand_token_matrix = np.zeros((len(cand_pool_indices), 24), dtype=np.int32)
                for local_i, global_i in enumerate(cand_pool_indices):
                    toks = target_idx.toks_list[global_i][:24]
                    for t_pos, t_str in enumerate(toks):
                        if t_str not in tok_to_id:
                            tid = len(tok_to_id) + 1
                            tok_to_id[t_str] = tid
                            df_val = doc_freqs.get((ctry, t_str), 0)
                            idf_val = float(np.log(1.0 + (max_docs_ctry + 1.0) / (df_val + 1.0)))
                            idf_weights_list.append(idf_val)
                        cand_token_matrix[local_i, t_pos] = tok_to_id[t_str]

                cand_id_to_pool_idx = {cid: idx for idx, cid in enumerate(cand_ids_pool)}
                idf_weights_tensor = torch.tensor(idf_weights_list, dtype=torch.float32, device=device)

                # Build token array & weights for S1 queries in this country
                s1_token_matrix = np.zeros((len(s1_ids_c), 24), dtype=np.int32)
                s1_weights_matrix = np.zeros((len(s1_ids_c), 24), dtype=np.float32)
                for q_i, sid in enumerate(s1_ids_c):
                    r_s1 = s1_norm_map.get(sid, {})
                    ad_s1 = r_s1.get("addr_norm", "")
                    toks_s1 = tokenize_address_f(ad_s1)[:24]
                    for t_pos, t_str in enumerate(toks_s1):
                        if t_str not in tok_to_id:
                            tid = len(tok_to_id) + 1
                            tok_to_id[t_str] = tid
                            df_val = doc_freqs.get((ctry, t_str), 0)
                            idf_val = float(np.log(1.0 + (max_docs_ctry + 1.0) / (df_val + 1.0)))
                            idf_weights_list.append(idf_val)
                        t_id = tok_to_id[t_str]
                        s1_token_matrix[q_i, t_pos] = t_id
                        s1_weights_matrix[q_i, t_pos] = idf_weights_list[t_id]

                # Update idf tensor if new S1 tokens added
                if len(idf_weights_list) > len(idf_weights_tensor):
                    idf_weights_tensor = torch.tensor(idf_weights_list, dtype=torch.float32, device=device)

                # Calculate query chunk size so chunk_size * N_index * 2 bytes <= 3 GB (exactly like Channel A)
                max_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
                bytes_per_query = N_index * 2
                max_q_chunk = max(1, max_bytes // max(1, bytes_per_query))
                q_chunk_size = min(max_q_chunk, 5000)
                if batch_size:
                    q_chunk_size = min(q_chunk_size, batch_size)
                score_mat_gb = (q_chunk_size * bytes_per_query) / (1024.0 ** 3)
                print(f"    [{ctry} - {target_idx.source_name}] Index rows: {N_index:,} | "
                      f"Query chunk size: {q_chunk_size} (score matrix {score_mat_gb:.2f} GB <= 3.00 GB)", flush=True)

                # Process S1 queries in batches
                for b_start in range(0, len(s1_ids_c), q_chunk_size):
                    b_end = min(b_start + q_chunk_size, len(s1_ids_c))
                    sub_sids = s1_ids_c[b_start:b_end]

                    # Query embeddings
                    sub_m_idx = [s1_m_map[sid] for sid in sub_sids if sid in s1_m_map]
                    if len(sub_m_idx) != len(sub_sids):
                        continue
                    Q_m = torch.from_numpy(s1_m_emb[sub_m_idx]).to(device)
                    S = Q_m @ X_gpu.T

                    has_alt_mask = np.array([sid in s1_a_map for sid in sub_sids])
                    if has_alt_mask.any():
                        sub_alt_sids = [sid for sid in sub_sids if sid in s1_a_map]
                        sub_alt_idx = [s1_a_map[sid] for sid in sub_alt_sids]
                        Q_a = torch.from_numpy(s1_a_emb[sub_alt_idx]).to(device)
                        S_a = Q_a @ X_gpu.T
                        S[has_alt_mask] = torch.maximum(S[has_alt_mask], S_a)

                    # Top 400 embedding scores
                    top_scores, top_indices = torch.topk(S, k=k_search, dim=1)
                    top_scores = top_scores.float().cpu().numpy()
                    top_indices = top_indices.cpu().numpy()
                    del S

                    # Deduplicate candidates to 200 unique
                    B_curr = len(sub_sids)
                    top200_cand_ids: list[list[str]] = []
                    top200_cand_scores: list[list[float]] = []
                    top200_pool_indices: list[list[int]] = []

                    for b in range(B_curr):
                        b_cids = row_to_cid[top_indices[b]]
                        b_scs = top_scores[b]
                        seen_c: dict[str, float] = {}
                        for cid, sc in zip(b_cids, b_scs):
                            if cid not in seen_c or sc > seen_c[cid]:
                                seen_c[cid] = float(sc)
                                if len(seen_c) >= 200:
                                    break
                        c_list = list(seen_c.keys())
                        top200_cand_ids.append(c_list)
                        top200_cand_scores.append(list(seen_c.values()))
                        top200_pool_indices.append([cand_id_to_pool_idx[c] for c in c_list])

                    # Prepare batch tensor for GPU address re-rank
                    max_c_len = max(len(cl) for cl in top200_cand_ids) if top200_cand_ids else 0
                    if max_c_len == 0:
                        continue

                    T_cand_sub = np.zeros((B_curr, max_c_len, 24), dtype=np.int32)
                    for b in range(B_curr):
                        for j, p_idx in enumerate(top200_pool_indices[b]):
                            T_cand_sub[b, j] = cand_token_matrix[p_idx]

                    # GPU Re-rank
                    T_cand_gpu = torch.from_numpy(T_cand_sub).to(device)
                    T_s1_gpu = torch.from_numpy(s1_token_matrix[b_start:b_end]).to(device)
                    W_s1_gpu = torch.from_numpy(s1_weights_matrix[b_start:b_end]).to(device)

                    # (B, 1, L_s1, 1) == (B, 200, 1, L_cand) -> any over dim=-1
                    has_tok = (T_s1_gpu.unsqueeze(1).unsqueeze(-1) == T_cand_gpu.unsqueeze(2)).any(dim=-1)
                    has_tok = has_tok & (T_s1_gpu.unsqueeze(1) != 0)
                    rerank_scores = (has_tok.float() * W_s1_gpu.unsqueeze(1)).sum(dim=-1).cpu().numpy()

                    del T_cand_gpu, T_s1_gpu, W_s1_gpu, has_tok

                    # Select top 10 per S1 not already in candidates
                    for b, sid in enumerate(sub_sids):
                        cids_b = top200_cand_ids[b]
                        emb_scs_b = top200_cand_scores[b]
                        rr_scs_b = rerank_scores[b]
                        
                        existing_set = existing_cands_map[sid]

                        # Sort by (rerank_score, emb_score) descending
                        ranked_order = sorted(
                            range(len(cids_b)),
                            key=lambda j: (rr_scs_b[j], emb_scs_b[j]),
                            reverse=True,
                        )

                        kept_count = 0
                        for j in ranked_order:
                            cid = cids_b[j]
                            if cid in existing_set:
                                continue
                            
                            c_dict = augmented_candidates[sid].setdefault(cid, {
                                "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0,
                                "f_score": 0.0, "emb_score": emb_scs_b[j], "is_s2": is_s2,
                            })
                            c_dict["ch_rerank"] = 1
                            c_dict["f_score"] = max(c_dict["f_score"], float(rr_scs_b[j]))
                            c_dict["emb_score"] = max(c_dict["emb_score"], float(emb_scs_b[j]))

                            kept_count += 1
                            if kept_count >= 10:
                                break

                del X_gpu, cand_token_matrix, idf_weights_tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        print(f"  Channel F completed in {time.time() - t_f0:.1f}s | RSS: {_rss_mb():.1f} MB", flush=True)

        # ----------------------------------------------------------------------
        # CHANNEL G: Address Keys K5, K3, K1
        # ----------------------------------------------------------------------
        print(f"  Running Channel G (Address Keys K5, K3, K1)...", flush=True)
        t_g0 = time.time()
        for sid in chunk_s1_ids:
            r_s1 = s1_norm_map.get(sid)
            if not r_s1:
                continue
            c = str(r_s1.get("country", "")).strip()
            num_str = str(r_s1.get("num_tokens", "")).strip()
            addr_str = str(r_s1.get("addr_norm", "")).strip()
            existing_set = existing_cands_map[sid]

            for target_idx, is_s2 in [(idx_s2, True), (idx_s3, False)]:
                # Key K5
                k5_keys = extract_k5(c, num_str, addr_str, doc_freqs)
                cands_k5: set[str] = set()
                for k in k5_keys:
                    cands_k5.update(target_idx.k5_index.get(k, []))
                cands_k5.difference_update(existing_set)
                if cands_k5:
                    scored_k5 = [(cid, _calc_emb_score(sid, cid, is_s2)) for cid in cands_k5]
                    scored_k5.sort(key=lambda x: x[1], reverse=True)
                    for cid, sc in scored_k5[:10]:
                        c_dict = augmented_candidates[sid].setdefault(cid, {
                            "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0,
                            "f_score": 0.0, "emb_score": sc, "is_s2": is_s2,
                        })
                        c_dict["ch_k5"] = 1
                        c_dict["emb_score"] = max(c_dict["emb_score"], sc)

                # Key K3
                k3_keys = extract_k3(c, num_str)
                cands_k3: set[str] = set()
                for k in k3_keys:
                    cands_k3.update(target_idx.k3_index.get(k, []))
                cands_k3.difference_update(existing_set)
                if cands_k3:
                    scored_k3 = [(cid, _calc_emb_score(sid, cid, is_s2)) for cid in cands_k3]
                    scored_k3.sort(key=lambda x: x[1], reverse=True)
                    for cid, sc in scored_k3[:10]:
                        c_dict = augmented_candidates[sid].setdefault(cid, {
                            "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0,
                            "f_score": 0.0, "emb_score": sc, "is_s2": is_s2,
                        })
                        c_dict["ch_k3"] = 1
                        c_dict["emb_score"] = max(c_dict["emb_score"], sc)

                # Key K1
                k1_keys = extract_k1(c, num_str)
                cands_k1: set[str] = set()
                for k in k1_keys:
                    cands_k1.update(target_idx.k1_index.get(k, []))
                cands_k1.difference_update(existing_set)
                if cands_k1:
                    scored_k1 = [(cid, _calc_emb_score(sid, cid, is_s2)) for cid in cands_k1]
                    scored_k1.sort(key=lambda x: x[1], reverse=True)
                    for cid, sc in scored_k1[:10]:
                        c_dict = augmented_candidates[sid].setdefault(cid, {
                            "ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0,
                            "f_score": 0.0, "emb_score": sc, "is_s2": is_s2,
                        })
                        c_dict["ch_k1"] = 1
                        c_dict["emb_score"] = max(c_dict["emb_score"], sc)

        print(f"  Channel G completed in {time.time() - t_g0:.1f}s | RSS: {_rss_mb():.1f} MB", flush=True)

        # ----------------------------------------------------------------------
        # MERGE & ATOM-WRITE CHUNK
        # ----------------------------------------------------------------------
        print(f"  Merging new candidates (max 20 per S1)...", flush=True)
        new_rows: list[dict[str, Any]] = []
        for sid in chunk_s1_ids:
            cands_dict = augmented_candidates.get(sid, {})
            if not cands_dict:
                continue

            # Sort new candidates: (n_new_channels, f_score, emb_score) descending
            sorted_new_cands = sorted(
                cands_dict.items(),
                key=lambda item: (
                    item[1]["ch_rerank"] + item[1]["ch_k1"] + item[1]["ch_k3"] + item[1]["ch_k5"],
                    item[1]["f_score"],
                    item[1]["emb_score"],
                ),
                reverse=True,
            )

            # Cap to at most 20 extra candidates per S1
            for cid, info in sorted_new_cands[:20]:
                is_s2 = info["is_s2"]
                row_dict: dict[str, Any] = {
                    "s1_id": sid,
                    "cand_id": cid,
                    "cand_source": 0 if is_s2 else 1,
                    "emb_score": np.float32(info["emb_score"]),
                    "emb_rank": np.float32(999.0),
                    "ch_emb": np.int8(0),
                    "ch_addr": np.int8(0),
                    "ch_skel": np.int8(0),
                    "ch_rare": np.int8(0),
                    "ch_rev": np.int8(0),
                    "ch_rerank": np.int8(info["ch_rerank"]),
                    "ch_k1": np.int8(info["ch_k1"]),
                    "ch_k3": np.int8(info["ch_k3"]),
                    "ch_k5": np.int8(info["ch_k5"]),
                }
                if split == "train":
                    row_dict["label"] = np.int32(1 if (sid, cid) in gt_pairs else 0)
                new_rows.append(row_dict)

        # Add 0 flags to existing rows
        chunk_df["ch_rerank"] = np.int8(0)
        chunk_df["ch_k1"] = np.int8(0)
        chunk_df["ch_k3"] = np.int8(0)
        chunk_df["ch_k5"] = np.int8(0)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            # Align column types with existing chunk_df
            for col in chunk_df.columns:
                if col in new_df.columns:
                    new_df[col] = new_df[col].astype(chunk_df[col].dtype)
            augmented_df = pd.concat([chunk_df, new_df], ignore_index=True)
        else:
            augmented_df = chunk_df

        # Atomic write back to chunk_path
        tmp_out = chunk_path + ".tmp"
        augmented_df.to_parquet(tmp_out, index=False)
        os.replace(tmp_out, chunk_path)

        added_cnt = len(new_rows)
        avg_extra = added_cnt / max(1, len(chunk_s1_ids))
        t_ch_elapsed = time.time() - t_ch_start
        print(f"  Chunk {ch_idx + 1} Saved: {len(augmented_df):,} total rows "
              f"(+{added_cnt:,} new rows, avg +{avg_extra:.2f}/S1) in {t_ch_elapsed:.1f}s | "
              f"RSS: {_rss_mb():.1f} MB", flush=True)

        del chunk_df, augmented_df, augmented_candidates, new_rows
        gc.collect()

    print(f"\nAll chunk processing completed in {time.time() - total_start:.1f}s.")

    # 9. Validation Recall Report (train only)
    if split == "train" and len(gt_pairs) > 0:
        run_validation_report(cache_dir, chunk_files, s1_norm_map, gt_pairs)


# ==============================================================================
# VALIDATION RECALL REPORT (BEFORE vs AFTER)
# ==============================================================================

def run_validation_report(
    cache_dir: str,
    processed_chunk_files: list[str],
    s1_norm_map: dict[str, dict[str, Any]],
    gt_pairs: set[tuple[str, str]],
) -> None:
    """Computes and displays pair recall, entity recall, and unique channel contributions on val S1."""
    print("\n" + "=" * 90)
    print("                 VAL BLOCKING RECALL REPORT (BEFORE vs AFTER)")
    print("=" * 90)

    # Load val S1 set
    split_path = os.path.join(cache_dir, "split.parquet")
    if not os.path.exists(split_path):
        print("split.parquet not found, skipping validation report.")
        return
        
    split_df = pd.read_parquet(split_path)
    val_s1_set = set(split_df[split_df["fold"] == "val"]["s1_id"]) if "fold" in split_df.columns else set(split_df["s1_id"])
    print(f"Total Val S1 entities: {len(val_s1_set):,}")

    # Load GT table for val
    gt_path = os.path.join(cache_dir, "gt_long.parquet")
    gt_df = pd.read_parquet(gt_path)
    if "match_id" not in gt_df.columns and "cand_id" in gt_df.columns:
        gt_df = gt_df.rename(columns={"cand_id": "match_id"})
    gt_val = gt_df[gt_df["s1_id"].isin(val_s1_set)].copy()
    if "match_source" not in gt_val.columns:
        gt_val["match_source"] = ["S2" if str(m).startswith("S2") else "S3" for m in gt_val["match_id"]]
        
    gt_val["country"] = [str(s1_norm_map.get(sid, {}).get("country", "")).strip() for sid in gt_val["s1_id"]]

    total_gt = len(gt_val)
    print(f"Total Val GT pairs: {total_gt:,} across {gt_val['s1_id'].nunique():,} unique S1")

    # Read processed chunks, filter to val S1 rows
    before_cands: set[tuple[str, str]] = set()
    after_cands: set[tuple[str, str]] = set()
    new_cands_flags: dict[tuple[str, str], dict[str, int]] = {}

    for cf in processed_chunk_files:
        df_c = pd.read_parquet(cf, columns=["s1_id", "cand_id", "ch_rerank", "ch_k1", "ch_k3", "ch_k5"])
        df_val = df_c[df_c["s1_id"].isin(val_s1_set)]
        for s1, cid, rrk, k1, k3, k5 in zip(df_val["s1_id"], df_val["cand_id"], df_val["ch_rerank"], df_val["ch_k1"], df_val["ch_k3"], df_val["ch_k5"]):
            pair = (s1, cid)
            after_cands.add(pair)
            if rrk == 0 and k1 == 0 and k3 == 0 and k5 == 0:
                before_cands.add(pair)
            else:
                new_cands_flags[pair] = {"ch_rerank": rrk, "ch_k1": k1, "ch_k3": k3, "ch_k5": k5}

    # Helper for recall statistics
    def _calc_stats(cands_set: set[tuple[str, str]]) -> dict[str, Any]:
        found_mask = [pair in cands_set for pair in zip(gt_val["s1_id"], gt_val["match_id"])]
        gt_found = gt_val[found_mask]
        
        # Overall
        total_p = total_gt
        found_p = len(gt_found)
        pct_p = (found_p / total_p * 100.0) if total_p > 0 else 0.0

        # By Country
        in_tot = len(gt_val[gt_val["country"] == "India"])
        in_fnd = len(gt_found[gt_found["country"] == "India"])
        in_pct = (in_fnd / in_tot * 100.0) if in_tot > 0 else 0.0

        us_tot = len(gt_val[gt_val["country"] == "US"])
        us_fnd = len(gt_found[gt_found["country"] == "US"])
        us_pct = (us_fnd / us_tot * 100.0) if us_tot > 0 else 0.0

        # By Source
        s2_tot = len(gt_val[gt_val["match_source"] == "S2"])
        s2_fnd = len(gt_found[gt_found["match_source"] == "S2"])
        s2_pct = (s2_fnd / s2_tot * 100.0) if s2_tot > 0 else 0.0

        s3_tot = len(gt_val[gt_val["match_source"] == "S3"])
        s3_fnd = len(gt_found[gt_found["match_source"] == "S3"])
        s3_pct = (s3_fnd / s3_tot * 100.0) if s3_tot > 0 else 0.0

        # Entity-level (100% matches found)
        gt_grouped = gt_val.groupby("s1_id")["match_id"].apply(set)
        cands_by_s1: dict[str, set[str]] = defaultdict(set)
        for s1, cid in cands_set:
            cands_by_s1[s1].add(cid)
            
        full_entities = sum(gold_set.issubset(cands_by_s1[sid]) for sid, gold_set in gt_grouped.items())
        tot_entities = len(gt_grouped)
        ent_pct = (full_entities / tot_entities * 100.0) if tot_entities > 0 else 0.0

        avg_cands_s1 = len(cands_set) / max(1, len(val_s1_set))

        return {
            "found_p": found_p, "pct_p": pct_p,
            "in_fnd": in_fnd, "in_tot": in_tot, "in_pct": in_pct,
            "us_fnd": us_fnd, "us_tot": us_tot, "us_pct": us_pct,
            "s2_fnd": s2_fnd, "s2_tot": s2_tot, "s2_pct": s2_pct,
            "s3_fnd": s3_fnd, "s3_tot": s3_tot, "s3_pct": s3_pct,
            "full_ent": full_entities, "tot_ent": tot_entities, "ent_pct": ent_pct,
            "avg_cands": avg_cands_s1,
        }

    st_before = _calc_stats(before_cands)
    st_after = _calc_stats(after_cands)

    header = f"{'Metric':<36} | {'Before Augment':<20} | {'After Augment':<20} | {'Delta':<10}"
    print(header)
    print("-" * len(header))

    def _fmt_row(name: str, key_p: str, key_cnt: str, key_tot: Optional[str] = None) -> str:
        pct_b, pct_a = st_before[key_p], st_after[key_p]
        delta = pct_a - pct_b
        if key_tot:
            s_b = f"{pct_b:6.2f}% ({st_before[key_cnt]:,}/{st_before[key_tot]:,})"
            s_a = f"{pct_a:6.2f}% ({st_after[key_cnt]:,}/{st_after[key_tot]:,})"
        else:
            s_b = f"{pct_b:6.2f}% ({st_before[key_cnt]:,})"
            s_a = f"{pct_a:6.2f}% ({st_after[key_cnt]:,})"
        s_d = f"+{delta:.2f}%" if delta >= 0 else f"{delta:.2f}%"
        return f"{name:<36} | {s_b:<20} | {s_a:<20} | {s_d:<10}"

    print(_fmt_row("Overall Pair Recall", "pct_p", "found_p", None))
    print(_fmt_row("  - India Pair Recall", "in_pct", "in_fnd", "in_tot"))
    print(_fmt_row("  - US Pair Recall", "us_pct", "us_fnd", "us_tot"))
    print(_fmt_row("  - S2 Pair Recall", "s2_pct", "s2_fnd", "s2_tot"))
    print(_fmt_row("  - S3 Pair Recall", "s3_pct", "s3_fnd", "s3_tot"))
    print(_fmt_row("Entity-Level Recall (100% matches)", "ent_pct", "full_ent", "tot_ent"))
    
    avg_b = st_before["avg_cands"]
    avg_a = st_after["avg_cands"]
    d_avg = avg_a - avg_b
    print(f"{'Average Candidates per S1':<36} | {avg_b:<20.2f} | {avg_a:<20.2f} | +{d_avg:.2f}")

    # Unique contributions among newly recovered GT pairs
    newly_recovered = [pair for pair in zip(gt_val["s1_id"], gt_val["match_id"]) if (pair in after_cands and pair not in before_cands)]
    print("\n" + "-" * len(header))
    print(f"Unique Channel Contributions on {len(newly_recovered):,} Newly Recovered GT Pairs:")

    only_rr = 0
    only_k1 = 0
    only_k3 = 0
    only_k5 = 0
    shared_new = 0

    for pair in newly_recovered:
        flags = new_cands_flags.get(pair, {"ch_rerank": 0, "ch_k1": 0, "ch_k3": 0, "ch_k5": 0})
        rr = flags["ch_rerank"]
        k1 = flags["ch_k1"]
        k3 = flags["ch_k3"]
        k5 = flags["ch_k5"]
        n_ch = rr + k1 + k3 + k5
        if n_ch > 1:
            shared_new += 1
        elif rr == 1:
            only_rr += 1
        elif k1 == 1:
            only_k1 += 1
        elif k3 == 1:
            only_k3 += 1
        elif k5 == 1:
            only_k5 += 1

    tot_rec = max(1, len(newly_recovered))
    print(f"  - Channel F (Address Re-rank):       {only_rr:5,} ({only_rr/tot_rec*100:6.2f}%)")
    print(f"  - Channel G (Key K1):                {only_k1:5,} ({only_k1/tot_rec*100:6.2f}%)")
    print(f"  - Channel G (Key K3):                {only_k3:5,} ({only_k3/tot_rec*100:6.2f}%)")
    print(f"  - Channel G (Key K5):                {only_k5:5,} ({only_k5/tot_rec*100:6.2f}%)")
    print(f"  - Shared across new channels:        {shared_new:5,} ({shared_new/tot_rec*100:6.2f}%)")
    print("=" * 90)


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3b Candidate Augmentation")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="train", help="Dataset split")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--laptop-test", action="store_true", help="Use cache/laptop_test")
    parser.add_argument("--max-chunks", type=int, default=None, help="Limit to N chunks (for timing test)")
    parser.add_argument("--batch-size", type=int, default=5000, help="S1 query batch size for GPU re-rank")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.laptop_test:
        cache_dir = os.path.join(root_dir, "cache", "laptop_test")
    else:
        cache_dir = os.path.join(root_dir, "cache")

    print(f"=== Starting Stage 3b Augmentation ({args.split}) ===")
    print(f"Cache Directory: {cache_dir}")

    # Safety: backup verification
    chunk_files = verify_backup_safety(cache_dir, args.split)

    # Run augmentation
    run_augmentation(
        cache_dir=cache_dir,
        split=args.split,
        chunk_files=chunk_files,
        max_chunks=args.max_chunks,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

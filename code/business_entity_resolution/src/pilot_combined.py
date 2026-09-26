#!/usr/bin/env python3
"""
Pilot: Combined Name + Address Embeddings for India S3.
Evaluates 256-dim combined embeddings (name_full + " | " + addr_norm) vs existing name-only embeddings.

Tasks:
1. Embed the India S3 pool and the India val S1 (from cache/split.parquet).
2. Run GPU top-20 search (chunked, score matrix strictly <= 3 GB).
3. Report:
   - recall@5 / 10 / 20 of India S3 GT pairs: Combined vs Name-only.
   - Percentage of currently MISSED India S3 GT pairs (not in cands_backup_train chunks)
     caught by combined top-20.
4. Keep Peak RSS under 8 GB and GPU memory under 10 GB.

Usage:
  # On laptop test:
  python code/business_entity_resolution/src/pilot_combined.py --laptop-test --limit 200

  # On EC2 full run:
  python code/business_entity_resolution/src/pilot_combined.py
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
from sentence_transformers import SentenceTransformer


def _rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pilot for combined name + address embeddings on India S3.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train).")
    parser.add_argument("--country", type=str, default="India", help="Country filter (default: India).")
    parser.add_argument("--source", type=str, default="source3", help="Candidate source (default: source3).")
    parser.add_argument("--limit", type=int, default=None, help="Row limit for quick testing.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory.")
    parser.add_argument("--laptop-test", action="store_true", help="Use cache/laptop_test.")
    parser.add_argument("--batch-size", type=int, default=512, help="Embedding batch size.")
    parser.add_argument("--force-embed", action="store_true", help="Re-embed even if cached.")
    return parser.parse_args()


def load_model(device: str) -> SentenceTransformer:
    """Loads Qwen3-Embedding-0.6B with 256 truncation and max_seq_length 64."""
    if device == "cuda":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    print(f"Loading Qwen/Qwen3-Embedding-0.6B on {device} ({torch_dtype})...", flush=True)
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device=device,
        model_kwargs={"torch_dtype": torch_dtype},
        truncate_dim=256,
    )
    model.max_seq_length = 64
    print("Model initialized: max_seq_length = 64, truncate_dim = 256", flush=True)
    return model


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 512) -> np.ndarray:
    """Encodes texts into L2-normalized float16 embeddings with OOM fallback."""
    if len(texts) == 0:
        return np.zeros((0, 256), dtype=np.float16)

    current_bs = batch_size
    while current_bs >= 64:
        try:
            sub_slice = 25000
            all_v = []
            for s_idx in range(0, len(texts), sub_slice):
                slice_texts = texts[s_idx : s_idx + sub_slice]
                v = model.encode(
                    slice_texts,
                    batch_size=current_bs,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=len(slice_texts) > 500,
                )
                norms = np.linalg.norm(v, axis=1, keepdims=True).clip(1e-6)
                v = v / norms
                all_v.append(v.astype(np.float16))

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            return np.concatenate(all_v, axis=0)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and current_bs > 64:
                current_bs = current_bs // 2
                print(f"GPU OOM! Halving batch size to {current_bs} and retrying...", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            else:
                raise e

    return np.zeros((0, 256), dtype=np.float16)


def get_combined_embeddings(
    texts: np.ndarray,
    ids: np.ndarray,
    cache_file_prefix: str,
    device: str,
    batch_size: int = 512,
    force_embed: bool = False,
    model: Optional[SentenceTransformer] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieves or generates L2-normalized 256-dim combined embeddings."""
    emb_path = f"{cache_file_prefix}_emb.npy"
    ids_path = f"{cache_file_prefix}_ids.npy"

    if not force_embed and os.path.exists(emb_path) and os.path.exists(ids_path):
        saved_ids = np.load(ids_path, allow_pickle=True)
        if len(saved_ids) == len(ids) and np.array_equal(saved_ids, ids):
            print(f"Loading cached embeddings from {emb_path} ({len(saved_ids):,} items)...", flush=True)
            return np.load(emb_path), saved_ids

    # Deduplicate before encoding
    t0 = time.time()
    uniques, inverse = np.unique(texts, return_inverse=True)
    uniques_list = uniques.tolist()
    dedup_time = time.time() - t0
    print(f"  Deduplication: {len(texts):,} records -> {len(uniques_list):,} unique strings ({len(uniques_list)/max(1, len(texts))*100:.1f}%) in {dedup_time:.2f}s", flush=True)

    if model is None:
        model = load_model(device)

    t_enc_start = time.time()
    unique_embs = embed_texts(model, uniques_list, batch_size=batch_size)
    t_enc = time.time() - t_enc_start
    throughput = len(uniques_list) / max(1e-4, t_enc)
    print(f"  Encoded {len(uniques_list):,} unique texts in {t_enc:.1f}s ({throughput:.1f} texts/sec) | RSS: {_rss_mb():.1f} MB", flush=True)

    final_emb = unique_embs[inverse]
    final_ids = np.array(ids, dtype=object)

    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
    np.save(emb_path, final_emb)
    np.save(ids_path, final_ids)
    print(f"  Saved embeddings to {emb_path} ({final_emb.shape})", flush=True)
    return final_emb, final_ids


def load_existing_candidates(cache_dir: str, val_s1_set: set[str]) -> dict[str, set[str]]:
    """Loads existing candidate pairs for val S1 from cands_backup_train or cands_train_chunk_*.parquet."""
    backup_dir = os.path.join(cache_dir, "cands_backup_train")
    if os.path.exists(backup_dir):
        chunk_files = sorted(glob.glob(os.path.join(backup_dir, "cands_train_chunk_*.parquet")))
        source_dir = "cands_backup_train"
    else:
        chunk_files = sorted(glob.glob(os.path.join(cache_dir, "cands_train_chunk_*.parquet")))
        source_dir = "cache"

    if not chunk_files:
        print(f"Warning: No candidate chunk files found in {source_dir}. Assuming 0 existing candidates.", flush=True)
        return {}

    print(f"\nLoading existing candidate sets from {len(chunk_files)} chunk(s) in {source_dir}...", flush=True)
    existing_cands: dict[str, set[str]] = {}
    total_cand_rows = 0
    t0 = time.time()

    for ch_path in chunk_files:
        df_ch = pd.read_parquet(ch_path, columns=["s1_id", "cand_id"])
        df_sub = df_ch[df_ch["s1_id"].isin(val_s1_set)]
        for s1, cid in zip(df_sub["s1_id"], df_sub["cand_id"]):
            s1_str = str(s1)
            cid_str = str(cid)
            if s1_str not in existing_cands:
                existing_cands[s1_str] = set()
            existing_cands[s1_str].add(cid_str)
            total_cand_rows += 1
        del df_ch, df_sub

    print(f"Loaded existing candidates: {total_cand_rows:,} candidate pairs across {len(existing_cands):,} S1 entities in {time.time() - t0:.1f}s | RSS: {_rss_mb():.1f} MB", flush=True)
    return existing_cands


def run_gpu_topk_search(
    q_emb: np.ndarray,
    q_ids: np.ndarray,
    pool_emb: np.ndarray,
    pool_ids: np.ndarray,
    device: str,
    k: int = 20,
    batch_size: int = 5000,
    q_alt_emb: Optional[np.ndarray] = None,
    q_alt_map: Optional[dict[str, int]] = None,
    desc: str = "Search",
) -> dict[str, list[str]]:
    """
    Performs chunked top-k cosine similarity search on GPU/device.
    Guarantees score matrix (chunk x pool_rows x 2 bytes) <= 3 GB.
    """
    N_pool = len(pool_emb)
    N_q = len(q_ids)
    if N_pool == 0 or N_q == 0:
        return {}

    print(f"\n--- Running GPU top-{k} search: {desc} ---", flush=True)
    print(f"  Queries: {N_q:,} | Pool rows: {N_pool:,}", flush=True)

    # 3 GB memory cap for score matrix
    max_bytes = 3 * 1024 * 1024 * 1024  # 3 GB
    bytes_per_query = N_pool * 2  # float16
    max_q_chunk = max(1, max_bytes // max(1, bytes_per_query))
    q_chunk_size = min(max_q_chunk, 5000, batch_size)
    score_gb = (q_chunk_size * bytes_per_query) / (1024.0 ** 3)
    print(f"  Chosen query chunk size: {q_chunk_size:,} (score matrix {score_gb:.2f} GB <= 3.00 GB)", flush=True)

    X_gpu = torch.from_numpy(pool_emb).to(device)
    k_actual = min(k, N_pool)
    topk_results: dict[str, list[str]] = {}

    t_start = time.time()
    for b_start in range(0, N_q, q_chunk_size):
        b_end = min(b_start + q_chunk_size, N_q)
        sub_sids = q_ids[b_start:b_end]
        Q_sub = torch.from_numpy(q_emb[b_start:b_end]).to(device)

        S = Q_sub @ X_gpu.T

        # If alt embeddings provided, take element-wise max
        if q_alt_emb is not None and q_alt_map is not None:
            has_alt = [sid in q_alt_map for sid in sub_sids]
            if any(has_alt):
                alt_indices = [q_alt_map[sid] for sid, h in zip(sub_sids, has_alt) if h]
                Q_alt_sub = torch.from_numpy(q_alt_emb[alt_indices]).to(device)
                S_alt = Q_alt_sub @ X_gpu.T
                has_alt_tensor = torch.tensor(has_alt, dtype=torch.bool, device=device)
                S[has_alt_tensor] = torch.maximum(S[has_alt_tensor], S_alt)
                del Q_alt_sub, S_alt

        # Top-k
        top_scores, top_indices = torch.topk(S, k=k_actual, dim=1)
        top_indices = top_indices.cpu().numpy()
        del S, Q_sub, top_scores

        for i, sid in enumerate(sub_sids):
            c_indices = top_indices[i]
            # Deduplicate preserving order
            seen: set[str] = set()
            ranked_cids: list[str] = []
            for c_idx in c_indices:
                cid = str(pool_ids[c_idx])
                if cid not in seen:
                    seen.add(cid)
                    ranked_cids.append(cid)
                    if len(ranked_cids) >= k:
                        break
            topk_results[str(sid)] = ranked_cids

        if (b_end // q_chunk_size) % 5 == 0 or b_end == N_q:
            print(f"  Processed {b_end:,}/{N_q:,} queries ({b_end/max(1e-4, time.time() - t_start):.1f} q/s) | RSS: {_rss_mb():.1f} MB", flush=True)

    del X_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return topk_results


def evaluate_recall(
    gt_pairs: list[tuple[str, str]],
    topk_map: dict[str, list[str]],
    missed_gt_set: set[tuple[str, str]],
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, Any]:
    """Computes recall@k on all GT pairs and on currently missed GT pairs."""
    hits = {k: 0 for k in ks}
    missed_hits = {k: 0 for k in ks}

    total_gt = len(gt_pairs)
    total_missed = len(missed_gt_set)

    for s1, cid in gt_pairs:
        retrieved = topk_map.get(s1, [])
        is_missed = (s1, cid) in missed_gt_set

        for k in ks:
            if cid in retrieved[:k]:
                hits[k] += 1
                if is_missed:
                    missed_hits[k] += 1

    recalls = {f"recall@{k}": (hits[k] / total_gt * 100.0) if total_gt > 0 else 0.0 for k in ks}
    missed_caught = {
        f"missed_caught@{k}": (missed_hits[k] / total_missed * 100.0) if total_missed > 0 else 0.0
        for k in ks
    }
    missed_counts = {f"missed_hits@{k}": missed_hits[k] for k in ks}

    return {
        "total_gt": total_gt,
        "hits": hits,
        "recalls": recalls,
        "total_missed": total_missed,
        "missed_counts": missed_counts,
        "missed_caught_pct": missed_caught,
    }


def main() -> None:
    args = parse_args()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.laptop_test:
        cache_dir = os.path.join(root_dir, "cache", "laptop_test")
    else:
        cache_dir = os.path.join(root_dir, "cache")

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("PILOT: COMBINED NAME + ADDRESS EMBEDDINGS (India S3)")
    print(f"Device: {device} | Cache: {cache_dir} | Split: {args.split} | Country: {args.country}")
    print("=" * 70, flush=True)

    # 1. Load Split and Filter India Val S1
    split_path = os.path.join(cache_dir, "split.parquet")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    split_df = pd.read_parquet(split_path)
    s1_col = "s1_id" if "s1_id" in split_df.columns else "entity_id"
    val_s1_set = set(split_df[split_df["fold"] == "val"][s1_col].astype(str))
    print(f"Loaded val fold S1 entities: {len(val_s1_set):,}")

    # Load S1 table
    p_norm_s1 = os.path.join(cache_dir, f"norm_{args.split}_source1.parquet")
    if not os.path.exists(p_norm_s1):
        raise FileNotFoundError(f"Normalized S1 table not found: {p_norm_s1}")
    df_s1 = pd.read_parquet(p_norm_s1, columns=["entity_id", "country", "name_full", "addr_norm"])
    df_s1["entity_id"] = df_s1["entity_id"].astype(str)
    val_s1_india = df_s1[(df_s1["country"] == args.country) & (df_s1["entity_id"].isin(val_s1_set))].copy()

    # Load S3 table (India pool)
    source_name = "source3"
    p_norm_s3 = os.path.join(cache_dir, f"norm_{args.split}_{source_name}.parquet")
    if not os.path.exists(p_norm_s3):
        raise FileNotFoundError(f"Normalized S3 table not found: {p_norm_s3}")
    df_s3 = pd.read_parquet(p_norm_s3, columns=["entity_id", "country", "name_full", "addr_norm"])
    df_s3["entity_id"] = df_s3["entity_id"].astype(str)
    s3_india = df_s3[df_s3["country"] == args.country].copy()

    # 2. Load Ground Truth
    gt_path = os.path.join(cache_dir, "gt_long.parquet")
    if os.path.exists(gt_path):
        gt_df = pd.read_parquet(gt_path)
    else:
        raw_gt = os.path.join(root_dir, "dataset", "train", "train_ground_truth.tsv")
        if not os.path.exists(raw_gt):
            raise FileNotFoundError(f"Ground truth not found at {gt_path} or {raw_gt}")
        raw_df = pd.read_csv(raw_gt, sep="\t")
        records = []
        for _, row in raw_df.iterrows():
            sid = str(row["source1_entity_id"]).strip()
            for m in str(row["matched_entity_ids"]).split(","):
                m = m.strip()
                if m.startswith("S3") or m.startswith("source3"):
                    records.append({"s1_id": sid, "match_id": m})
        gt_df = pd.DataFrame(records)

    cand_col = "match_id" if "match_id" in gt_df.columns else "cand_id"
    gt_df["s1_id"] = gt_df["s1_id"].astype(str)
    gt_df[cand_col] = gt_df[cand_col].astype(str)

    val_s1_india_ids = set(val_s1_india["entity_id"])
    s3_india_ids = set(s3_india["entity_id"])
    gt_india_s3 = gt_df[gt_df["s1_id"].isin(val_s1_india_ids) & gt_df[cand_col].isin(s3_india_ids)].copy()

    # Apply limit if requested
    if args.limit is not None and args.limit > 0:
        print(f"Applying limit = {args.limit} to val S1 entities...")
        # Prioritize S1 entities that have GT pairs for meaningful evaluation
        gt_s1_sub = set(gt_india_s3["s1_id"].iloc[: args.limit // 2])
        other_s1 = [sid for sid in val_s1_india["entity_id"] if sid not in gt_s1_sub]
        chosen_s1 = list(gt_s1_sub) + other_s1[: max(0, args.limit - len(gt_s1_sub))]
        val_s1_india = val_s1_india[val_s1_india["entity_id"].isin(set(chosen_s1))].copy()

        # Keep GT matches in pool plus random up to limit * 10
        val_gt_matches = set(gt_india_s3[gt_india_s3["s1_id"].isin(set(chosen_s1))][cand_col])
        other_s3 = [cid for cid in s3_india["entity_id"] if cid not in val_gt_matches]
        keep_s3 = list(val_gt_matches) + other_s3[: max(0, args.limit * 10 - len(val_gt_matches))]
        s3_india = s3_india[s3_india["entity_id"].isin(set(keep_s3))].copy()

        # Re-filter GT
        val_s1_india_ids = set(val_s1_india["entity_id"])
        s3_india_ids = set(s3_india["entity_id"])
        gt_india_s3 = gt_df[gt_df["s1_id"].isin(val_s1_india_ids) & gt_df[cand_col].isin(s3_india_ids)].copy()

    print(f"India Val S1 entities: {len(val_s1_india):,}")
    print(f"India S3 pool candidates: {len(s3_india):,}")
    gt_pairs_list = [(str(r["s1_id"]), str(r[cand_col])) for _, r in gt_india_s3.iterrows()]
    print(f"Total India S3 GT pairs in scope: {len(gt_pairs_list):,}")

    # 3. Load Existing Candidates to determine currently missed GT pairs
    existing_cands = load_existing_candidates(cache_dir, val_s1_india_ids)
    missed_gt_set = set()
    for s1, cid in gt_pairs_list:
        if cid not in existing_cands.get(s1, set()):
            missed_gt_set.add((s1, cid))

    pct_missed = (len(missed_gt_set) / max(1, len(gt_pairs_list))) * 100.0
    print(f"Currently MISSED India S3 GT pairs: {len(missed_gt_set):,} / {len(gt_pairs_list):,} ({pct_missed:.1f}% missed)")

    # 4. Generate / Load Combined Embeddings (name_full + " | " + addr_norm)
    print("\n--- Preparing Combined Name + Address Embeddings ---", flush=True)
    shared_model = load_model(device)

    # India S3 pool combined texts
    s3_name = s3_india["name_full"].fillna("").astype(str).str.strip()
    s3_addr = s3_india["addr_norm"].fillna("").astype(str).str.strip()
    s3_combined_texts = (s3_name + " | " + s3_addr).values
    s3_ids = s3_india["entity_id"].values

    s3_cache_prefix = os.path.join(cache_dir, f"embc_{args.split}_source3_india_{len(s3_ids)}")
    s3_embc, s3_idsc = get_combined_embeddings(
        texts=s3_combined_texts,
        ids=s3_ids,
        cache_file_prefix=s3_cache_prefix,
        device=device,
        batch_size=args.batch_size,
        force_embed=args.force_embed,
        model=shared_model,
    )

    # India val S1 combined texts
    s1_name = val_s1_india["name_full"].fillna("").astype(str).str.strip()
    s1_addr = val_s1_india["addr_norm"].fillna("").astype(str).str.strip()
    s1_combined_texts = (s1_name + " | " + s1_addr).values
    s1_ids = val_s1_india["entity_id"].values

    s1_cache_prefix = os.path.join(cache_dir, f"embc_{args.split}_vals1_india_{len(s1_ids)}")
    s1_embc, s1_idsc = get_combined_embeddings(
        texts=s1_combined_texts,
        ids=s1_ids,
        cache_file_prefix=s1_cache_prefix,
        device=device,
        batch_size=args.batch_size,
        force_embed=args.force_embed,
        model=shared_model,
    )

    # Free text arrays
    del s3_name, s3_addr, s3_combined_texts, s1_name, s1_addr, s1_combined_texts
    gc.collect()

    # 5. Load Existing Name-Only Embeddings
    print("\n--- Preparing Name-Only Embeddings (Baseline) ---", flush=True)
    emb_s1_base_path = os.path.join(cache_dir, f"emb_{args.split}_source1.npy")
    ids_s1_base_path = os.path.join(cache_dir, f"ids_{args.split}_source1.npy")
    emb_s3_base_path = os.path.join(cache_dir, f"emb_{args.split}_source3.npy")
    ids_s3_base_path = os.path.join(cache_dir, f"ids_{args.split}_source3.npy")

    has_baseline_files = (
        os.path.exists(emb_s1_base_path)
        and os.path.exists(ids_s1_base_path)
        and os.path.exists(emb_s3_base_path)
        and os.path.exists(ids_s3_base_path)
    )

    if has_baseline_files and not args.force_embed:
        print(f"Loading existing name embeddings from {cache_dir}...", flush=True)
        all_s1_emb = np.load(emb_s1_base_path)
        all_s1_ids = np.load(ids_s1_base_path, allow_pickle=True)
        s1_map = {str(sid): i for i, sid in enumerate(all_s1_ids)}

        all_s3_emb = np.load(emb_s3_base_path)
        all_s3_ids = np.load(ids_s3_base_path, allow_pickle=True)
        s3_map = {str(cid): i for i, cid in enumerate(all_s3_ids)}

        # Filter down to current S1 and S3
        s1_name_idx = [s1_map[str(sid)] for sid in s1_ids if str(sid) in s1_map]
        s1_emb_name = all_s1_emb[s1_name_idx]
        s1_ids_name = np.array([sid for sid in s1_ids if str(sid) in s1_map], dtype=object)

        s3_name_idx = [s3_map[str(cid)] for cid in s3_ids if str(cid) in s3_map]
        s3_emb_name = all_s3_emb[s3_name_idx]
        s3_ids_name = np.array([cid for cid in s3_ids if str(cid) in s3_map], dtype=object)

        del all_s1_emb, all_s1_ids, all_s3_emb, all_s3_ids
        gc.collect()
    else:
        print("Name-only embedding files not found or re-embed requested; encoding name_full...", flush=True)
        s3_name_texts = s3_india["name_full"].fillna("").astype(str).tolist()
        s3_emb_name = embed_texts(shared_model, s3_name_texts, batch_size=args.batch_size)
        s3_ids_name = np.array(s3_ids, dtype=object)

        s1_name_texts = val_s1_india["name_full"].fillna("").astype(str).tolist()
        s1_emb_name = embed_texts(shared_model, s1_name_texts, batch_size=args.batch_size)
        s1_ids_name = np.array(s1_ids, dtype=object)

    # Alt embeddings for S1 name-only if present
    alt_s1_path = os.path.join(cache_dir, f"emb_{args.split}_source1_alt.npy")
    alt_ids_path = os.path.join(cache_dir, f"ids_{args.split}_source1_alt.npy")
    q_alt_emb = None
    q_alt_map = None
    if os.path.exists(alt_s1_path) and os.path.exists(alt_ids_path):
        alt_ids = np.load(alt_ids_path, allow_pickle=True)
        if len(alt_ids) > 0:
            q_alt_emb = np.load(alt_s1_path)
            q_alt_map = {str(sid): i for i, sid in enumerate(alt_ids)}

    # Free model if no longer needed
    del shared_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    # 6. Run Top-20 Searches
    # Search A: Name-Only Top-20
    name_top20 = run_gpu_topk_search(
        q_emb=s1_emb_name,
        q_ids=s1_ids_name,
        pool_emb=s3_emb_name,
        pool_ids=s3_ids_name,
        device=device,
        k=20,
        batch_size=5000,
        q_alt_emb=q_alt_emb,
        q_alt_map=q_alt_map,
        desc="Name-Only Embeddings",
    )

    del s1_emb_name, s3_emb_name, q_alt_emb, q_alt_map
    gc.collect()

    # Search B: Combined Name + Address Top-20
    combined_top20 = run_gpu_topk_search(
        q_emb=s1_embc,
        q_ids=s1_idsc,
        pool_emb=s3_embc,
        pool_ids=s3_idsc,
        device=device,
        k=20,
        batch_size=5000,
        desc="Combined Name + Address Embeddings",
    )

    del s1_embc, s3_embc
    gc.collect()

    # 7. Evaluate and Compare
    eval_name = evaluate_recall(gt_pairs_list, name_top20, missed_gt_set, ks=(5, 10, 20))
    eval_comb = evaluate_recall(gt_pairs_list, combined_top20, missed_gt_set, ks=(5, 10, 20))

    # 8. Print Formatted Report
    print("\n" + "=" * 70)
    print("PILOT EVALUATION REPORT: INDIA S3 (Combined vs Name-Only)")
    print("=" * 70)
    print(f"Total Val S1 Queries Evaluated:    {len(val_s1_india):,}")
    print(f"Total S3 Candidate Pool Evaluated: {len(s3_india):,}")
    print(f"Total Evaluated GT Pairs:          {len(gt_pairs_list):,}")
    print(f"Currently MISSED GT Pairs:         {len(missed_gt_set):,} ({pct_missed:.1f}%)")
    print("-" * 70)
    print(f"{'Metric':<25} | {'Name-Only':<15} | {'Combined (Name+Addr)':<20} | {'Delta':<10}")
    print("-" * 70)

    for k in [5, 10, 20]:
        m = f"recall@{k}"
        r_name = eval_name["recalls"][m]
        r_comb = eval_comb["recalls"][m]
        delta = r_comb - r_name
        sign = "+" if delta >= 0 else ""
        print(f"{m:<25} | {r_name:>13.2f}% | {r_comb:>18.2f}% | {sign}{delta:>8.2f}%")

    print("-" * 70)
    print("RECOVERY OF CURRENTLY MISSED GT PAIRS (Combined Top-K):")
    for k in [5, 10, 20]:
        caught_cnt = eval_comb["missed_counts"][f"missed_hits@{k}"]
        caught_pct = eval_comb["missed_caught_pct"][f"missed_caught@{k}"]
        print(f"  Combined Top-{k:<2} caught {caught_cnt:,} / {len(missed_gt_set):,} missed GT pairs ({caught_pct:.2f}%)")

    print("=" * 70)
    print(f"Peak RSS: {_rss_mb():.1f} MB (Budget: 8,192 MB)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

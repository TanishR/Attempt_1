#!/usr/bin/env python3
"""
Combined Name + Address Embedding Generator (Stage 2b).
Encodes combined text: name_full + " | " + addr_norm into 256-dim L2-normalized embeddings.

Model: Qwen/Qwen3-Embedding-0.6B
Configuration:
  - fp16 (on CUDA) / float32 (on CPU/MPS)
  - truncate_dim = 256
  - max_seq_length = 64
  - L2-normalized after truncation
  - Whole-file deduplication with parts saved for resume in cache/parts/
  - Output: cache/embc_{split}_{source}.npy and cache/idsc_{split}_{source}.npy

Usage:
  python code/business_entity_resolution/src/s2b_embed_combined.py --split train --source s3 --country India [--limit 1000]
  python code/business_entity_resolution/src/s2b_embed_combined.py --split train --source s1 --laptop-test
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
from sentence_transformers import SentenceTransformer


def _rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in megabytes."""
    return psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 256-dim combined name+address embeddings.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--source", type=str, required=True, choices=["source1", "source2", "source3", "s1", "s2", "s3"], help="Data source.")
    parser.add_argument("--country", type=str, default=None, help="Optional country filter (e.g. India, US).")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for testing.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory.")
    parser.add_argument("--laptop-test", action="store_true", help="Use cache/laptop_test.")
    parser.add_argument("--batch-size", type=int, default=512, help="Encoding batch size (default 512 for GPU < 10 GB).")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Unique text part size for resume.")
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
    print("Set model.max_seq_length = 64 and truncate_dim = 256")
    return model


def embed_batch_texts(model: SentenceTransformer, texts: list[str], batch_size: int = 512) -> np.ndarray:
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
                    show_progress_bar=True,
                )
                # L2 normalize after truncation
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


def encode_uniques_with_resume(
    model: SentenceTransformer,
    texts: np.ndarray,
    ids: np.ndarray,
    tag: str,
    cache_dir: str,
    emb_out_path: str,
    ids_out_path: str,
    chunk_size: int = 200000,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicates texts across whole file, encodes in parts with resume, and maps back."""
    if len(texts) == 0:
        empty_emb = np.zeros((0, 256), dtype=np.float16)
        empty_ids = np.array([], dtype=object)
        np.save(emb_out_path, empty_emb)
        np.save(ids_out_path, empty_ids)
        return empty_emb, empty_ids

    # 1. Whole-file deduplication
    print(f"[{tag}] Running whole-file deduplication on {len(texts):,} records...", flush=True)
    t_dedup = time.time()
    uniques, inverse = np.unique(texts, return_inverse=True)
    uniques_list = uniques.tolist()
    dedup_pct = (len(uniques_list) / len(texts)) * 100.0
    print(f"[{tag}] Deduplication done in {time.time() - t_dedup:.1f}s: {len(uniques_list):,} unique strings ({dedup_pct:.1f}% of {len(texts):,})", flush=True)

    parts_dir = os.path.join(cache_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    num_parts = (len(uniques_list) + chunk_size - 1) // chunk_size
    all_part_embs: list[np.ndarray] = []

    total_encoded_texts = 0
    encode_time_total = 0.0

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(uniques_list))
        part_path = os.path.join(parts_dir, f"{tag}_unique_part{i}.npy")

        if os.path.exists(part_path):
            print(f"[{tag}] Part {i+1}/{num_parts} ({start_idx:,} to {end_idx:,}) exists on disk, loading...", flush=True)
            part_emb = np.load(part_path)
        else:
            print(f"[{tag}] Encoding part {i+1}/{num_parts} ({start_idx:,} to {end_idx:,})...", flush=True)
            sub_uniques = uniques_list[start_idx:end_idx]
            t0 = time.time()
            part_emb = embed_batch_texts(model, sub_uniques, batch_size=batch_size)
            t_part = time.time() - t0
            encode_time_total += t_part
            total_encoded_texts += len(sub_uniques)
            throughput = len(sub_uniques) / max(1e-4, t_part)
            print(f"[{tag}] Part {i+1} encoded in {t_part:.1f}s ({throughput:.1f} texts/sec). Saving to {part_path}...", flush=True)
            np.save(part_path, part_emb)

        all_part_embs.append(part_emb)

    if total_encoded_texts > 0 and encode_time_total > 0:
        overall_throughput = total_encoded_texts / encode_time_total
        print(f"\n=======================================================")
        print(f"[{tag}] Total unique texts newly encoded: {total_encoded_texts:,}")
        print(f"[{tag}] Total encode time: {encode_time_total:.1f}s")
        print(f"[{tag}] Overall Throughput: {overall_throughput:.1f} texts/sec")
        print(f"=======================================================\n")

    uniques_emb = np.concatenate(all_part_embs, axis=0) if num_parts > 0 else np.zeros((0, 256), dtype=np.float16)
    assert uniques_emb.shape[0] == len(uniques_list), f"Part size mismatch: {uniques_emb.shape[0]} vs {len(uniques_list)}"

    print(f"[{tag}] Mapping {len(uniques_emb):,} unique embeddings back to {len(texts):,} entity rows...", flush=True)
    final_emb = uniques_emb[inverse]
    final_ids = np.array(ids, dtype=object)

    assert final_emb.shape[0] == final_ids.shape[0], f"Row count mismatch: {final_emb.shape[0]} vs {final_ids.shape[0]}"
    
    # Save final arrays
    os.makedirs(os.path.dirname(emb_out_path), exist_ok=True)
    np.save(emb_out_path, final_emb)
    np.save(ids_out_path, final_ids)
    print(f"[{tag}] Saved final combined embeddings: {emb_out_path} ({final_emb.shape}) and {ids_out_path}")
    return final_emb, final_ids


def embed_and_save_combined(
    split: str,
    source: str,
    cache_dir: str,
    country: Optional[str] = None,
    limit: Optional[int] = None,
    batch_size: int = 512,
    chunk_size: int = 200000,
    model: Optional[SentenceTransformer] = None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Public helper to embed combined texts and save arrays."""
    source_map = {"s1": "source1", "s2": "source2", "s3": "source3"}
    source_norm = source_map.get(source, source)

    p_norm = os.path.join(cache_dir, f"norm_{split}_{source_norm}.parquet")
    if not os.path.exists(p_norm):
        raise FileNotFoundError(f"Normalized table not found: {p_norm}")

    print(f"\nLoading normalized table {p_norm}...", flush=True)
    cols = ["entity_id", "country", "name_full", "addr_norm"]
    df = pd.read_parquet(p_norm, columns=cols)
    print(f"  Loaded {len(df):,} records (RSS: {_rss_mb():.1f} MB)")

    if country:
        print(f"  Filtering to country = '{country}'...")
        df = df[df["country"] == country].copy()
        print(f"  Records after country filter: {len(df):,}")

    if limit is not None and limit > 0:
        print(f"  Applying limit: {limit:,} records...")
        df = df.iloc[:limit].copy()

    # Text = name_full + " | " + addr_norm (name_full stays frozen)
    name_full = df["name_full"].fillna("").astype(str).str.strip()
    addr_norm = df["addr_norm"].fillna("").astype(str).str.strip()
    texts = (name_full + " | " + addr_norm).values
    entity_ids = df["entity_id"].values

    tag_parts = [split, source_norm]
    if country:
        tag_parts.append(country.lower())
    tag = "_".join(tag_parts)

    emb_out_path = os.path.join(cache_dir, f"embc_{split}_{source_norm}.npy")
    ids_out_path = os.path.join(cache_dir, f"idsc_{split}_{source_norm}.npy")

    if country:
        # Also create country-specific name for distinct preservation
        emb_ctry_path = os.path.join(cache_dir, f"embc_{split}_{source_norm}_{country.lower()}.npy")
        ids_ctry_path = os.path.join(cache_dir, f"idsc_{split}_{source_norm}_{country.lower()}.npy")
    else:
        emb_ctry_path = emb_out_path
        ids_ctry_path = ids_out_path

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if model is None:
        model = load_model(device)

    final_emb, final_ids = encode_uniques_with_resume(
        model=model,
        texts=texts,
        ids=entity_ids,
        tag=tag,
        cache_dir=cache_dir,
        emb_out_path=emb_ctry_path,
        ids_out_path=ids_ctry_path,
        chunk_size=chunk_size,
        batch_size=batch_size,
    )

    # Ensure both source name variations (e.g. s3 and source3) exist
    for s_name in {source, source_norm}:
        p_emb = os.path.join(cache_dir, f"embc_{split}_{s_name}.npy")
        p_ids = os.path.join(cache_dir, f"idsc_{split}_{s_name}.npy")
        if p_emb != emb_ctry_path and not os.path.exists(p_emb):
            np.save(p_emb, final_emb)
        if p_ids != ids_ctry_path and not os.path.exists(p_ids):
            np.save(p_ids, final_ids)

    return final_emb, final_ids, emb_out_path, ids_out_path


def main() -> None:
    args = parse_args()
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if args.cache_dir:
        cache_dir = args.cache_dir
    elif args.laptop_test:
        cache_dir = os.path.join(root_dir, "cache", "laptop_test")
    else:
        cache_dir = os.path.join(root_dir, "cache")

    embed_and_save_combined(
        split=args.split,
        source=args.source,
        cache_dir=cache_dir,
        country=args.country,
        limit=args.limit,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

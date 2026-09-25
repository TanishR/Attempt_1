import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
import re
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import CACHE_DIR

def get_args():
    """
    Parses command-line arguments for s2_embed.py.
    Returns: argparse.Namespace with split, source, ids_file, and benchmark options.
    """
    parser = argparse.ArgumentParser(description="Encode names to 256-dim embeddings using Qwen3-Embedding-0.6B.")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--source", type=str, required=True, choices=["source1", "source2", "source3", "s1", "s2", "s3"], help="Data source.")
    parser.add_argument("--ids-file", type=str, default=None, help="Optional parquet file with entity_ids to filter.")
    parser.add_argument("--all-s1", action="store_true", help="Embed ALL train S1 (ignore --ids-file filter). Fixes train/test distribution shift.")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on random 50k unique names and exit.")
    return parser.parse_args()

def has_non_ascii(text):
    """
    Checks whether a given text contains any non-ASCII character.
    Returns: bool (True if non-ASCII character present, else False).
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r'[^\x00-\x7F]', text))

def embed_texts(model, texts, batch_size=1024):
    """
    Encodes a list of string texts into 256-dim L2-normalized float16 embeddings with OOM fallback.
    Returns: numpy.ndarray of shape (len(texts), 256) in float16.
    """
    if len(texts) == 0:
        return np.zeros((0, 256), dtype=np.float16)
        
    current_bs = batch_size
    while current_bs >= 128:
        try:
            slice_size = 50000
            all_v = []
            for s_idx in range(0, len(texts), slice_size):
                sub_texts = texts[s_idx : s_idx + slice_size]
                v_sub = model.encode(
                    sub_texts,
                    batch_size=current_bs,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=True
                )
                v_sub = v_sub / np.linalg.norm(v_sub, axis=1, keepdims=True).clip(1e-6)
                all_v.append(v_sub.astype(np.float16))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            return np.concatenate(all_v, axis=0)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower() and current_bs > 128:
                current_bs = current_bs // 2
                print(f"OOM encountered! Clearing cache and falling back to batch_size={current_bs}...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            else:
                raise e

def encode_uniques_and_save(model, texts, ids, emb_prefix, tag, chunk_size=500000, batch_size=1024):
    """
    Deduplicates texts across the whole file, encodes unique strings in parts with resume, and maps back.
    Returns: None (persists final .npy arrays and part files in cache/parts/).
    """
    final_emb_path = f"{emb_prefix}.npy"
    final_ids_path = f"{emb_prefix.replace('emb_', 'ids_')}.npy"

    if len(texts) == 0:
        np.save(final_emb_path, np.zeros((0, 256), dtype=np.float16))
        np.save(final_ids_path, np.array([], dtype=object))
        print(f"Saved empty arrays for {tag}: {final_emb_path} and {final_ids_path}")
        return

    # Deduplicate across the whole file
    uniques, inverse = np.unique(texts, return_inverse=True)
    uniques_list = uniques.tolist()
    print(f"[{tag}] Whole-file dedup: {len(uniques_list)} uniques / {len(texts)} rows ({len(uniques_list)/len(texts)*100:.1f}%)")

    parts_dir = os.path.join(os.path.dirname(emb_prefix), "parts")
    os.makedirs(parts_dir, exist_ok=True)

    num_parts = (len(uniques_list) + chunk_size - 1) // chunk_size
    all_part_embs = []

    for i in range(num_parts):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(uniques_list))
        part_path = os.path.join(parts_dir, f"{tag}_unique_part{i}.npy")

        if os.path.exists(part_path):
            print(f"[{tag}] Unique part {i+1}/{num_parts} already exists, loading from disk...")
            part_emb = np.load(part_path)
        else:
            print(f"[{tag}] Encoding unique part {i+1}/{num_parts} ({start_idx} to {end_idx})...")
            chunk_uniques = uniques_list[start_idx:end_idx]
            part_emb = embed_texts(model, chunk_uniques, batch_size=batch_size)
            np.save(part_path, part_emb)
            print(f"[{tag}] Saved unique part {i+1} to {part_path}")

        all_part_embs.append(part_emb)

    uniques_emb = np.concatenate(all_part_embs, axis=0) if num_parts > 0 else np.zeros((0, 256), dtype=np.float16)
    assert uniques_emb.shape[0] == len(uniques_list), f"Unique embedding mismatch: {uniques_emb.shape[0]} vs {len(uniques_list)}"

    print(f"[{tag}] Mapping {len(uniques_emb)} unique embeddings back to {len(texts)} entity rows...")
    final_emb = uniques_emb[inverse]
    final_ids = np.array(ids, dtype=object)

    assert final_emb.shape[0] == final_ids.shape[0], f"Row count mismatch: {final_emb.shape[0]} vs {final_ids.shape[0]}"
    norms = np.linalg.norm(final_emb.astype(np.float32), axis=1)
    assert np.all(norms > 0.99) and np.all(norms < 1.01), "Norms are not unit normalized (0.99-1.01)"

    np.save(final_emb_path, final_emb)
    np.save(final_ids_path, final_ids)
    print(f"[{tag}] Successfully saved: {final_emb_path} ({final_emb.shape}) and {final_ids_path}")

def main():
    """
    Main execution pipeline: loads data, sets up model with sequence length cap, runs benchmark or full encoding.
    Returns: None.
    """
    args = get_args()

    source_map = {"s1": "source1", "s2": "source2", "s3": "source3"}
    source = source_map.get(args.source, args.source)

    file_path = os.path.join(CACHE_DIR, f"norm_{args.split}_{source}.parquet")
    print(f"Loading {file_path}...")
    df = pd.read_parquet(file_path)

    if args.all_s1:
        print("--all-s1 set: embedding ALL S1 records (ignoring --ids-file filter).")
    elif args.ids_file:
        print(f"Filtering using IDs from {args.ids_file}...")
        ids_df = pd.read_parquet(args.ids_file)
        if 'source1_entity_id' in ids_df.columns:
            valid_ids = set(ids_df['source1_entity_id'])
        elif 'entity_id' in ids_df.columns:
            valid_ids = set(ids_df['entity_id'])
        elif 's1_id' in ids_df.columns:
            valid_ids = set(ids_df['s1_id'])
        else:
            raise ValueError(f"Could not find ID column in {args.ids_file}")
        df = df[df['entity_id'].isin(valid_ids)].copy()
        print(f"Filtered down to {len(df)} rows.")

    raw_names = df['raw_name'].fillna("").astype(str).values
    name_fulls = df['name_full'].fillna("").astype(str).values
    entity_ids = df['entity_id'].values

    # Determine non-ASCII rows
    is_non_ascii_mask = np.array([has_non_ascii(t) for t in raw_names])

    # Main texts: raw_name if non-ASCII, else name_full
    main_texts = np.where(is_non_ascii_mask, raw_names, name_fulls)

    # Alt texts: name_full for non-ASCII records only
    alt_texts = name_fulls[is_non_ascii_mask]
    alt_ids = entity_ids[is_non_ascii_mask]

    # Select compute device
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print(f"Loading Qwen3-Embedding-0.6B on {device} ({torch_dtype})...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device=device,
        model_kwargs={"torch_dtype": torch_dtype},
        truncate_dim=256,
    )
    # Cap sequence length to 48 for business names
    model.max_seq_length = 48
    print("Set model.max_seq_length = 48")

    # Benchmark mode: test on random 50k unique names and exit
    if args.benchmark:
        all_unique_main = np.unique(main_texts)
        all_unique_alt = np.unique(alt_texts) if len(alt_texts) > 0 else np.array([], dtype=str)
        total_unique = len(all_unique_main) + len(all_unique_alt)
        print(f"\n--- BENCHMARK MODE ---")
        print(f"Total unique names in file: {total_unique} (Main: {len(all_unique_main)}, Alt: {len(all_unique_alt)})")

        sample_size = min(50000, len(all_unique_main))
        print(f"Sampling {sample_size} RANDOM unique names for benchmark (seed=42)...")
        np.random.seed(42)
        sample_texts = np.random.choice(all_unique_main, size=sample_size, replace=False).tolist()

        start_t = time.time()
        _ = embed_texts(model, sample_texts, batch_size=1024)
        elapsed = time.time() - start_t

        rate = sample_size / elapsed if elapsed > 0 else 0
        est_seconds = total_unique / rate if rate > 0 else 0
        print(f"\n================ BENCHMARK RESULTS ================")
        print(f"Throughput: {rate:.2f} texts/sec")
        print(f"Time for sample ({sample_size} names): {elapsed:.2f}s")
        print(f"Estimated full runtime for {total_unique} unique names: {est_seconds / 60:.2f} minutes ({est_seconds / 3600:.2f} hours)")
        print(f"===================================================")
        print("Benchmark complete. Exiting without full run as requested.")
        sys.exit(0)

    # Full encoding mode
    tag_prefix = f"{args.split}_{source}"
    emb_prefix_main = os.path.join(CACHE_DIR, f"emb_{tag_prefix}")
    print(f"\nProcessing MAIN embeddings ({len(main_texts)} rows)...")
    encode_uniques_and_save(model, main_texts, entity_ids, emb_prefix_main, f"{tag_prefix}_main", batch_size=1024)

    emb_prefix_alt = os.path.join(CACHE_DIR, f"emb_{tag_prefix}_alt")
    print(f"\nProcessing ALT embeddings ({len(alt_texts)} rows)...")
    encode_uniques_and_save(model, alt_texts, alt_ids, emb_prefix_alt, f"{tag_prefix}_alt", batch_size=1024)

if __name__ == "__main__":
    main()

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--source", type=str, required=True, choices=["source1", "source2", "source3", "s1", "s2", "s3"])
    parser.add_argument("--ids-file", type=str, default=None)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()

def embed_texts(model, texts, batch_size=512):
    if len(texts) == 0:
        return np.zeros((0, 256), dtype=np.float16)
    v = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-6)
    return v.astype(np.float16)

def has_non_ascii(text):
    if not isinstance(text, str): return False
    return bool(re.search(r'[^\x00-\x7F]', text))

def process_and_save(model, texts, ids, emb_prefix, chunk_size=500000):
    if len(texts) == 0:
        np.save(f"{emb_prefix}.npy", np.zeros((0, 256), dtype=np.float16))
        np.save(f"{emb_prefix.replace('emb_', 'ids_')}.npy", np.array([], dtype=object))
        return
        
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size
    all_embs = []
    all_ids = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(texts))
        
        emb_path = f"{emb_prefix}_part{i}.npy"
        ids_prefix = emb_prefix.replace("emb_", "ids_")
        ids_path = f"{ids_prefix}_part{i}.npy"
        
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            print(f"Part {i} already exists, skipping...")
            all_embs.append(np.load(emb_path))
            all_ids.append(np.load(ids_path, allow_pickle=True))
            continue
            
        print(f"Processing part {i+1}/{num_chunks} ({start_idx} to {end_idx})...")
        
        chunk_texts = texts[start_idx:end_idx]
        chunk_ids = ids[start_idx:end_idx]
        
        c_unique, c_inv = np.unique(chunk_texts, return_inverse=True)
        c_unique = c_unique.tolist()
        
        print(f"Chunk unique ratio: {len(c_unique)} / {len(chunk_texts)}")
        
        v = embed_texts(model, c_unique, batch_size=512)
        chunk_emb = v[c_inv]
        
        np.save(emb_path, chunk_emb)
        np.save(ids_path, chunk_ids)
        print(f"Saved part {i}")
        
        all_embs.append(chunk_emb)
        all_ids.append(chunk_ids)
        
    if num_chunks > 0:
        final_emb = np.concatenate(all_embs, axis=0)
        final_ids = np.concatenate(all_ids, axis=0)
        
        final_emb_path = f"{emb_prefix}.npy"
        final_ids_path = f"{emb_prefix.replace('emb_', 'ids_')}.npy"
        
        np.save(final_emb_path, final_emb)
        np.save(final_ids_path, final_ids)
        print(f"Saved final arrays to {final_emb_path} and {final_ids_path}")
        
        assert final_emb.shape[0] == final_ids.shape[0], "Row count mismatch!"
        norms = np.linalg.norm(final_emb.astype(np.float32), axis=1)
        assert np.all(norms > 0.99) and np.all(norms < 1.01), "Norms not 1.0"

def main():
    args = get_args()
    
    source_map = {"s1": "source1", "s2": "source2", "s3": "source3"}
    source = source_map.get(args.source, args.source)
    
    file_path = os.path.join(CACHE_DIR, f"norm_{args.split}_{source}.parquet")
    print(f"Loading {file_path}...")
    df = pd.read_parquet(file_path)
    
    if args.ids_file:
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
    
    # Determine which are non-ascii
    is_non_ascii_mask = np.array([has_non_ascii(t) for t in raw_names])
    
    # Main texts: raw_name if non-ascii, else name_full
    main_texts = np.where(is_non_ascii_mask, raw_names, name_fulls)
    
    # Alt texts: name_full for non-ascii only
    alt_texts = name_fulls[is_non_ascii_mask]
    alt_ids = entity_ids[is_non_ascii_mask]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Loading Qwen3-Embedding-0.6B on {device} ({torch_dtype})...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device=device,
        model_kwargs={"torch_dtype": torch_dtype},
        truncate_dim=256,
    )
    
    if args.benchmark:
        unique_names = np.unique(main_texts).tolist()
        print(f"Total unique main names in file: {len(unique_names)}")
        print("Running benchmark on first 50k names...")
        sample_size = min(50000, len(unique_names))
        sample_texts = unique_names[:sample_size]
        
        start_t = time.time()
        _ = embed_texts(model, sample_texts, batch_size=512)
        elapsed = time.time() - start_t
        
        rate = sample_size / elapsed if elapsed > 0 else 0
        print(f"Benchmark: {rate:.2f} texts/sec")
        
        total_unique = len(unique_names) + len(np.unique(alt_texts))
        est_seconds = total_unique / rate if rate > 0 else 0
        print(f"Estimated full runtime for {total_unique} unique names: {est_seconds / 60:.2f} minutes")

    print(f"\nProcessing MAIN embeddings ({len(main_texts)} rows)...")
    emb_prefix_main = os.path.join(CACHE_DIR, f"emb_{args.split}_{source}")
    process_and_save(model, main_texts, entity_ids, emb_prefix_main)
    
    print(f"\nProcessing ALT embeddings ({len(alt_texts)} rows)...")
    emb_prefix_alt = os.path.join(CACHE_DIR, f"emb_{args.split}_{source}_alt")
    process_and_save(model, alt_texts, alt_ids, emb_prefix_alt)

if __name__ == "__main__":
    main()

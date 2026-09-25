import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import torch
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
    v = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-6)
    return v.astype(np.float16)

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

    names = df['name_full'].fillna("").astype(str).values
    entity_ids = df['entity_id'].values
    
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
        unique_names = np.unique(names).tolist()
        print(f"Total unique names in file: {len(unique_names)}")
        print("Running benchmark on first 50k names...")
        sample_size = min(50000, len(unique_names))
        sample_texts = unique_names[:sample_size]
        
        start_t = time.time()
        _ = embed_texts(model, sample_texts, batch_size=512)
        elapsed = time.time() - start_t
        
        rate = sample_size / elapsed if elapsed > 0 else 0
        print(f"Benchmark: {rate:.2f} texts/sec")
        
        total_unique = len(unique_names)
        est_seconds = total_unique / rate if rate > 0 else 0
        print(f"Estimated full runtime for {total_unique} unique names: {est_seconds / 60:.2f} minutes")

    chunk_size = 500000
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        
        emb_path = os.path.join(CACHE_DIR, f"emb_{args.split}_{source}_part{i}.npy")
        ids_path = os.path.join(CACHE_DIR, f"ids_{args.split}_{source}_part{i}.npy")
        
        if os.path.exists(emb_path) and os.path.exists(ids_path):
            print(f"Part {i} already exists, skipping...")
            continue
            
        print(f"Processing part {i+1}/{num_chunks} ({start_idx} to {end_idx})...")
        
        chunk_names = names[start_idx:end_idx]
        chunk_ids = entity_ids[start_idx:end_idx]
        
        c_unique, c_inv = np.unique(chunk_names, return_inverse=True)
        c_unique = c_unique.tolist()
        
        print(f"Chunk unique ratio: {len(c_unique)} / {len(chunk_names)}")
        
        v = embed_texts(model, c_unique, batch_size=512)
        chunk_emb = v[c_inv]
        
        np.save(emb_path, chunk_emb)
        np.save(ids_path, chunk_ids)
        print(f"Saved part {i}")
        
    print("Combining parts...")
    all_embs = []
    all_ids = []
    
    for i in range(num_chunks):
        emb_path = os.path.join(CACHE_DIR, f"emb_{args.split}_{source}_part{i}.npy")
        ids_path = os.path.join(CACHE_DIR, f"ids_{args.split}_{source}_part{i}.npy")
        
        if not os.path.exists(emb_path) or not os.path.exists(ids_path):
            print(f"Warning: Missing part {i}!")
            continue
            
        all_embs.append(np.load(emb_path))
        all_ids.append(np.load(ids_path, allow_pickle=True))
        
    if num_chunks > 0 and len(all_embs) == num_chunks:
        final_emb = np.concatenate(all_embs, axis=0)
        final_ids = np.concatenate(all_ids, axis=0)
        
        final_emb_path = os.path.join(CACHE_DIR, f"emb_{args.split}_{source}.npy")
        final_ids_path = os.path.join(CACHE_DIR, f"ids_{args.split}_{source}.npy")
        
        np.save(final_emb_path, final_emb)
        np.save(final_ids_path, final_ids)
        print(f"Saved final arrays to {final_emb_path} and {final_ids_path}")
        
        # Sanity Checks
        assert final_emb.shape[0] == final_ids.shape[0], f"Row count mismatch! {final_emb.shape[0]} != {final_ids.shape[0]}"
        print(f"Sanity Check: Embedding row count ({final_emb.shape[0]}) == ids row count ({final_ids.shape[0]})")
        
        norms = np.linalg.norm(final_emb.astype(np.float32), axis=1)
        assert np.all(norms > 0.99) and np.all(norms < 1.01), "Norms are not around 1.0"
        print("Sanity Check: Norms of saved vectors are within 0.99-1.01")
        
    if args.benchmark:
        print("\nSanity Cosine Test:")
        test_texts = [
            "ram marketing private limited",
            "raam maarketing praaivet limited",
            "summit holdings inc"
        ]
        v_test = embed_texts(model, test_texts, batch_size=32).astype(np.float32)
        sim_12 = np.dot(v_test[0], v_test[1])
        sim_13 = np.dot(v_test[0], v_test[2])
        print(f'cosine("ram marketing private limited", "raam maarketing praaivet limited") = {sim_12:.4f}')
        print(f'cosine("ram marketing private limited", "summit holdings inc") = {sim_13:.4f}')
        if sim_12 > sim_13:
            print("Sanity Check PASS: Match is closer than non-match")
        else:
            print("Sanity Check FAIL: Match is farther than non-match")

if __name__ == "__main__":
    main()

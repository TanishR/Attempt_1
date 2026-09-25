import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import (
    CACHE_DIR, CAND_CAP, K_PER_SOURCE, MAX_ADDR_CANDS,
    MAX_SKEL_CANDS, MAX_BLOCK_SIZE, BLOCK_BY_COUNTRY
)

def get_args():
    """
    Parses command-line arguments for candidate generation.
    Returns: argparse.Namespace with split, cache_dir, laptop_test, and chunk_size.
    """
    parser = argparse.ArgumentParser(description="Multi-channel candidate generation (blocking).")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"], help="Dataset split.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory holding cached parquet and npy files.")
    parser.add_argument("--laptop-test", action="store_true", help="Run on cache/laptop_test isolated small dataset.")
    parser.add_argument("--chunk-size", type=int, default=100000, help="S1 query chunk size for memory-safe processing.")
    return parser.parse_args()

def load_embedding_arrays(cache_dir, split, source):
    """
    Loads main and alt embeddings as pure 2D float16 numpy arrays with ID-to-index maps.
    Returns: tuple of (main_emb, main_id_map, main_ids, alt_emb, alt_id_map, alt_ids).
    """
    m_p = os.path.join(cache_dir, f"emb_{split}_{source}.npy")
    m_id_p = os.path.join(cache_dir, f"ids_{split}_{source}.npy")
    
    if not os.path.exists(m_p) or not os.path.exists(m_id_p):
        return (np.zeros((0, 256), dtype=np.float16), {}, np.array([], dtype=object),
                np.zeros((0, 256), dtype=np.float16), {}, np.array([], dtype=object))
        
    main_emb = np.load(m_p)
    main_ids = np.load(m_id_p, allow_pickle=True)
    main_id_map = {eid: idx for idx, eid in enumerate(main_ids)}
    
    a_p = os.path.join(cache_dir, f"emb_{split}_{source}_alt.npy")
    a_id_p = os.path.join(cache_dir, f"ids_{split}_{source}_alt.npy")
    
    if os.path.exists(a_p) and os.path.exists(a_id_p):
        alt_emb = np.load(a_p)
        alt_ids = np.load(a_id_p, allow_pickle=True)
        alt_id_map = {eid: idx for idx, eid in enumerate(alt_ids)}
    else:
        alt_emb = np.zeros((0, 256), dtype=np.float16)
        alt_ids = np.array([], dtype=object)
        alt_id_map = {}
        
    return main_emb, main_id_map, main_ids, alt_emb, alt_id_map, alt_ids

def load_blocking_keys(cache_dir, split, source):
    """
    Loads normalized table and constructs composite blocking keys for Channel B and Channel C.
    Returns: pandas.DataFrame with entity_id, country, key_addr1, key_addr2, and key_skel.
    """
    path = os.path.join(cache_dir, f"norm_{split}_{source}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = ["entity_id", "country", "house_no", "street_token", "zip_pin", "name_skel"]
    df = pd.read_parquet(path, columns=cols)
    
    df["skel_sorted"] = df["name_skel"].fillna("").astype(str).apply(lambda x: " ".join(sorted(x.split())))
    
    # Channel B keys
    has_h = df["house_no"].astype(bool)
    has_st = df["street_token"].astype(bool)
    has_z = df["zip_pin"].astype(bool)
    
    df["key_addr1"] = np.where(has_h & has_st, df["country"] + "_" + df["house_no"] + "_" + df["street_token"], None)
    df["key_addr2"] = np.where(has_h & has_z, df["country"] + "_" + df["zip_pin"] + "_" + df["house_no"], None)
    
    # Channel C key
    has_sk = df["skel_sorted"].astype(bool)
    df["key_skel"] = np.where(has_sk, df["country"] + "_" + df["skel_sorted"], None)
    
    return df[["entity_id", "country", "key_addr1", "key_addr2", "key_skel"]]

def get_valid_keys(df, col, max_block_size):
    """
    Filters out keys that appear in more than max_block_size records to avoid combinatorial explosion.
    Returns: pandas.Index containing valid keys.
    """
    counts = df[col].dropna().value_counts()
    return counts[counts <= max_block_size].index

def search_channel_a_gpu(
    s1_ids, s1_main_emb, s1_main_map, s1_alt_emb, s1_alt_map,
    cand_ids, c_main_emb, c_main_map, c_alt_emb, c_alt_map,
    k_per_source, device
):
    """
    Performs GPU fp16 matrix-multiplication top-k search with memory-safe query chunking and deduplication.
    Returns: pandas.DataFrame with columns [s1_id, cand_id, emb_score, emb_rank, ch_emb].
    """
    if len(s1_ids) == 0 or len(cand_ids) == 0:
        return pd.DataFrame(columns=['s1_id', 'cand_id', 'emb_score', 'emb_rank', 'ch_emb'])

    # Build stacked target index (main + alt vectors)
    c_m_indices = [c_main_map[cid] for cid in cand_ids if cid in c_main_map]
    X_main = c_main_emb[c_m_indices]
    X_main_ids = cand_ids[[cid in c_main_map for cid in cand_ids]]
    
    c_a_cids = [cid for cid in cand_ids if cid in c_alt_map]
    if c_a_cids:
        c_a_indices = [c_alt_map[cid] for cid in c_a_cids]
        X_alt = c_alt_emb[c_a_indices]
        X_alt_ids = np.array(c_a_cids, dtype=object)
        X_stacked = np.vstack([X_main, X_alt])
        row_to_cid = np.concatenate([X_main_ids, X_alt_ids])
    else:
        X_stacked = X_main
        row_to_cid = X_main_ids
        
    N_index = len(X_stacked)
    if N_index == 0:
        return pd.DataFrame(columns=['s1_id', 'cand_id', 'emb_score', 'emb_rank', 'ch_emb'])
        
    X_gpu = torch.from_numpy(X_stacked).to(device)
    
    # Calculate query chunk size so chunk_size * N_index * 2 bytes < 4 GB
    max_bytes = 4 * 1024 * 1024 * 1024  # 4 GB
    bytes_per_query = N_index * 2
    max_q_chunk = max(1, max_bytes // max(1, bytes_per_query))
    q_chunk_size = min(20000, max_q_chunk)
    
    results = []
    k_search = min(2 * k_per_source, N_index)
    
    for q_start in range(0, len(s1_ids), q_chunk_size):
        q_end = min(q_start + q_chunk_size, len(s1_ids))
        sub_s1_ids = s1_ids[q_start:q_end]
        
        # Main query matrix
        sub_m_idx = [s1_main_map[sid] for sid in sub_s1_ids]
        Q_m = torch.from_numpy(s1_main_emb[sub_m_idx]).to(device)
        S = Q_m @ X_gpu.T
        
        # Alt query: only compute for S1 rows that have an alt vector
        has_alt_mask = np.array([sid in s1_alt_map for sid in sub_s1_ids])
        if has_alt_mask.any():
            sub_alt_sids = [sid for sid in sub_s1_ids if sid in s1_alt_map]
            sub_alt_idx = [s1_alt_map[sid] for sid in sub_alt_sids]
            Q_a = torch.from_numpy(s1_alt_emb[sub_alt_idx]).to(device)
            S_a = Q_a @ X_gpu.T
            S[has_alt_mask] = torch.maximum(S[has_alt_mask], S_a)
            
        top_scores, top_indices = torch.topk(S, k=k_search, dim=1)
        top_scores = top_scores.float().cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        for b, sid in enumerate(sub_s1_ids):
            b_scores = top_scores[b]
            b_cids = row_to_cid[top_indices[b]]
            
            # Deduplicate by entity id after top-k
            seen = {}
            for cid, score in zip(b_cids, b_scores):
                if cid not in seen or score > seen[cid]:
                    seen[cid] = score
                    if len(seen) >= k_per_source:
                        break
                        
            for rank, (cid, score) in enumerate(seen.items(), start=1):
                results.append((sid, cid, float(score), rank, 1))
                
    del X_gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    df_res = pd.DataFrame(results, columns=['s1_id', 'cand_id', 'emb_score', 'emb_rank', 'ch_emb'])
    return df_res

def compute_missing_scores_vectorized(
    df_missing, s1_main_emb, s1_main_map, s1_alt_emb, s1_alt_map,
    c_main_emb, c_main_map, c_alt_emb, c_alt_map
):
    """
    Computes dot-product cosine similarity for candidate pairs missing Channel A scores.
    Returns: numpy.ndarray of scores for each row in df_missing.
    """
    if df_missing.empty:
        return np.array([], dtype=np.float32)

    s1_ids = df_missing['s1_id'].values
    c_ids = df_missing['cand_id'].values
    
    s1_m_idx = np.array([s1_main_map.get(sid, 0) for sid in s1_ids])
    c_m_idx = np.array([c_main_map.get(cid, 0) for cid in c_ids])
    
    v1_m = s1_main_emb[s1_m_idx]
    v2_m = c_main_emb[c_m_idx]
    s_mm = (v1_m.astype(np.float32) * v2_m.astype(np.float32)).sum(axis=1)
    
    scores = s_mm.copy()
    
    # Check pairs where S1 or Cand has alt
    for i, (sid, cid) in enumerate(zip(s1_ids, c_ids)):
        best_s = s_mm[i]
        has_s1_a = sid in s1_alt_map
        has_c_a = cid in c_alt_map
        
        if has_c_a:
            v2_a = c_alt_emb[c_alt_map[cid]].astype(np.float32)
            s_ma = np.dot(v1_m[i].astype(np.float32), v2_a)
            if s_ma > best_s: best_s = s_ma
            
        if has_s1_a:
            v1_a = s1_alt_emb[s1_alt_map[sid]].astype(np.float32)
            s_am = np.dot(v1_a, v2_m[i].astype(np.float32))
            if s_am > best_s: best_s = s_am
            if has_c_a:
                s_aa = np.dot(v1_a, v2_a)
                if s_aa > best_s: best_s = s_aa
                
        scores[i] = best_s
        
    return scores

def run_recall_report(df_cands, gt, norm_s1, norm_s2, norm_s3):
    """
    Computes and prints comprehensive recall analytics, channel contributions, and side-by-side missed pairs.
    """
    print("\n" + "="*70)
    print("               STEP 5 CANDIDATE GENERATION RECALL REPORT")
    print("="*70)
    
    s1_in_cands = set(df_cands['s1_id'].unique())
    gt_eval = gt[gt['s1_id'].isin(s1_in_cands)].copy()
    total_gt_pairs = len(gt_eval)
    
    # 1. Pair recall overall
    cands_set = set(zip(df_cands['s1_id'], df_cands['cand_id']))
    gt_eval['found'] = [pair in cands_set for pair in zip(gt_eval['s1_id'], gt_eval['match_id'])]
    found_gt_pairs = gt_eval['found'].sum()
    overall_recall = (found_gt_pairs / max(1, total_gt_pairs)) * 100
    print(f"\n1. Overall Pair Recall: {found_gt_pairs} / {total_gt_pairs} ({overall_recall:.2f}%)")
    
    # Pair recall by source
    for src in ['S2', 'S3']:
        gt_src = gt_eval[gt_eval['match_source'].str.upper().str.contains(src)]
        f_src = gt_src['found'].sum()
        tot_src = len(gt_src)
        pct_src = (f_src / max(1, tot_src)) * 100 if tot_src > 0 else 0
        print(f"   - {src} Pair Recall: {f_src} / {tot_src} ({pct_src:.2f}%)")
        
    # Pair recall by country
    country_map = norm_s1.set_index('entity_id')['country'].to_dict()
    gt_eval['country'] = gt_eval['s1_id'].map(country_map)
    for c in sorted(gt_eval['country'].dropna().unique()):
        gt_c = gt_eval[gt_eval['country'] == c]
        f_c = gt_c['found'].sum()
        tot_c = len(gt_c)
        pct_c = (f_c / max(1, tot_c)) * 100 if tot_c > 0 else 0
        print(f"   - [{c}] Pair Recall: {f_c} / {tot_c} ({pct_c:.2f}%)")
        
    # 2. Entity-level recall
    gt_grouped = gt_eval.groupby('s1_id')['match_id'].apply(set)
    cands_grouped = df_cands.groupby('s1_id')['cand_id'].apply(set).to_dict()
    
    fully_found_entities = 0
    for sid, gold_set in gt_grouped.items():
        cand_set = cands_grouped.get(sid, set())
        if gold_set.issubset(cand_set):
            fully_found_entities += 1
            
    total_entities = len(gt_grouped)
    entity_recall = (fully_found_entities / max(1, total_entities)) * 100
    print(f"\n2. Entity-Level Recall (100% matches in candidates): {fully_found_entities} / {total_entities} ({entity_recall:.2f}%)")
    
    # 3. Candidates per S1 statistics
    cands_per_s1 = df_cands.groupby('s1_id').size()
    avg_cands = cands_per_s1.mean()
    p95_cands = cands_per_s1.quantile(0.95)
    max_cands = cands_per_s1.max()
    print(f"\n3. Candidate Count Statistics:")
    print(f"   - Mean candidates per S1: {avg_cands:.2f}")
    print(f"   - 95th percentile candidates per S1: {p95_cands:.1f}")
    print(f"   - Max candidates per S1: {max_cands}")
    
    # 4. Unique contribution per channel
    # Merge candidates with GT to inspect channel tags of found GT pairs
    gt_pairs_df = gt_eval[gt_eval['found']][['s1_id', 'match_id']].rename(columns={'match_id': 'cand_id'})
    found_cands = df_cands.merge(gt_pairs_df, on=['s1_id', 'cand_id'], how='inner')
    
    only_emb = len(found_cands[(found_cands['ch_emb'] == 1) & (found_cands['ch_addr'] == 0) & (found_cands['ch_skel'] == 0)])
    only_addr = len(found_cands[(found_cands['ch_emb'] == 0) & (found_cands['ch_addr'] == 1) & (found_cands['ch_skel'] == 0)])
    only_skel = len(found_cands[(found_cands['ch_emb'] == 0) & (found_cands['ch_addr'] == 0) & (found_cands['ch_skel'] == 1)])
    
    print(f"\n4. Unique Channel Contributions on Recovered GT Pairs:")
    print(f"   - Unique to Channel A (Embedding): {only_emb} ({only_emb/max(1, found_gt_pairs)*100:.2f}%)")
    print(f"   - Unique to Channel B (Address):   {only_addr} ({only_addr/max(1, found_gt_pairs)*100:.2f}%)")
    print(f"   - Unique to Channel C (Skeleton):  {only_skel} ({only_skel/max(1, found_gt_pairs)*100:.2f}%)")
    
    # 5. Channel A Recall@k curve
    print(f"\n5. Channel A (Embedding) Recall@k Curve:")
    ch_a_cands = df_cands[df_cands['ch_emb'] == 1]
    for k in [5, 10, 15, 20]:
        ch_a_k = ch_a_cands[ch_a_cands['emb_rank'] <= k]
        pairs_k = set(zip(ch_a_k['s1_id'], ch_a_k['cand_id']))
        f_k = sum(pair in pairs_k for pair in zip(gt_eval['s1_id'], gt_eval['match_id']))
        print(f"   - Channel A Recall@{k}: {f_k} / {total_gt_pairs} ({(f_k / max(1, total_gt_pairs))*100:.2f}%)")
        
    # 6. Sample of 20 missed pairs side by side
    missed_gt = gt_eval[~gt_eval['found']]
    print(f"\n6. Side-by-Side Sample of Missed Pairs (Total Missed: {len(missed_gt)}):")
    print("-" * 100)
    
    # Prepare text lookups
    s1_name = norm_s1.set_index('entity_id')['name_full'].to_dict()
    s1_addr = norm_s1.set_index('entity_id')['addr_norm'].to_dict()
    c_name = {**norm_s2.set_index('entity_id')['name_full'].to_dict(), **norm_s3.set_index('entity_id')['name_full'].to_dict()}
    c_addr = {**norm_s2.set_index('entity_id')['addr_norm'].to_dict(), **norm_s3.set_index('entity_id')['addr_norm'].to_dict()}
    
    sample_missed = missed_gt.head(20)
    for idx, (_, row) in enumerate(sample_missed.iterrows(), start=1):
        sid = row['s1_id']
        mid = row['match_id']
        c = row['country']
        src = row['match_source']
        print(f"[{idx:02d}] ({c} | {src})")
        print(f"  S1 [{sid}]: {s1_name.get(sid, '')} || {s1_addr.get(sid, '')}")
        print(f"  GT [{mid}]: {c_name.get(mid, '')} || {c_addr.get(mid, '')}")
        print("-" * 100)

def main():
    """
    Main blocking entry point: loads keys and embeddings, generates candidates across 3 channels,
    performs capping and score assignment, attaches labels, and persists chunked candidate outputs.
    """
    args = get_args()
    
    if args.laptop_test:
        cache_dir = os.path.join(CACHE_DIR, "laptop_test")
    elif args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = CACHE_DIR
        
    print(f"--- Running Step 5 Blocking ---")
    print(f"Cache Directory: {cache_dir}")
    print(f"Split: {args.split}")
    print(f"K_PER_SOURCE={K_PER_SOURCE}, CAND_CAP={CAND_CAP}, MAX_BLOCK_SIZE={MAX_BLOCK_SIZE}")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Compute Device: {device}")

    # Load S1 keys and embeddings
    print("\nLoading Source 1 keys and embeddings...")
    s1_keys_df = load_blocking_keys(cache_dir, args.split, "source1")
    s1_m_emb, s1_m_map, s1_m_ids, s1_a_emb, s1_a_map, s1_a_ids = load_embedding_arrays(cache_dir, args.split, "source1")
    
    if len(s1_m_ids) == 0:
        print("Error: Source 1 embeddings not found. Please run Step 4 first.")
        return

    # Load S2 and S3 keys and embeddings
    print("Loading Source 2 & 3 keys and embeddings...")
    c2_keys_df = load_blocking_keys(cache_dir, args.split, "source2")
    c2_m_emb, c2_m_map, c2_m_ids, c2_a_emb, c2_a_map, c2_a_ids = load_embedding_arrays(cache_dir, args.split, "source2")
    
    c3_keys_df = load_blocking_keys(cache_dir, args.split, "source3")
    c3_m_emb, c3_m_map, c3_m_ids, c3_a_emb, c3_a_map, c3_a_ids = load_embedding_arrays(cache_dir, args.split, "source3")

    # Load GT for label assignment if train/val split
    gt_df = None
    if args.split == "train":
        gt_path = os.path.join(cache_dir, "gt_long.parquet")
        if os.path.exists(gt_path):
            gt_df = pd.read_parquet(gt_path)
            print(f"Loaded ground truth: {len(gt_df)} rows")

    # Partition S1 into chunks of args.chunk_size (100k queries) with resume
    all_s1_ids = s1_keys_df['entity_id'].values
    chunk_size = args.chunk_size
    num_chunks = (len(all_s1_ids) + chunk_size - 1) // chunk_size
    print(f"\nProcessing {len(all_s1_ids)} S1 queries in {num_chunks} chunk(s) of {chunk_size}...")

    chunk_files = []
    start_total_time = time.time()

    for ch_idx in range(num_chunks):
        chunk_out_path = os.path.join(cache_dir, f"cands_{args.split}_chunk_{ch_idx}.parquet")
        chunk_files.append(chunk_out_path)
        
        if os.path.exists(chunk_out_path):
            print(f"Chunk {ch_idx + 1}/{num_chunks} already exists ({chunk_out_path}), skipping...")
            continue
            
        print(f"\n>>> Processing Chunk {ch_idx + 1}/{num_chunks}...")
        c_s1_ids = all_s1_ids[ch_idx * chunk_size : (ch_idx + 1) * chunk_size]
        sub_s1_keys = s1_keys_df[s1_keys_df['entity_id'].isin(c_s1_ids)].copy()
        
        chunk_cands_list = []
        
        # Sources to block against
        cand_sources = [
            ("source2", 0, c2_keys_df, c2_m_emb, c2_m_map, c2_m_ids, c2_a_emb, c2_a_map, c2_a_ids),
            ("source3", 1, c3_keys_df, c3_m_emb, c3_m_map, c3_m_ids, c3_a_emb, c3_a_map, c3_a_ids)
        ]
        
        countries = sub_s1_keys['country'].dropna().unique() if BLOCK_BY_COUNTRY else ["ALL"]
        
        for src_name, src_code, c_keys_df, c_m_emb, c_m_map, c_m_ids, c_a_emb, c_a_map, c_a_ids in cand_sources:
            if len(c_m_ids) == 0: continue
            
            for country in countries:
                if BLOCK_BY_COUNTRY:
                    s1_c_df = sub_s1_keys[sub_s1_keys['country'] == country]
                    c_c_df = c_keys_df[c_keys_df['country'] == country]
                else:
                    s1_c_df = sub_s1_keys
                    c_c_df = c_keys_df
                    
                s1_c_ids = s1_c_df['entity_id'].values
                cand_c_ids = c_c_df['entity_id'].values
                
                if len(s1_c_ids) == 0 or len(cand_c_ids) == 0:
                    continue
                    
                # --- Channel A: GPU Embedding Search ---
                df_a = search_channel_a_gpu(
                    s1_c_ids, s1_m_emb, s1_m_map, s1_a_emb, s1_a_map,
                    cand_c_ids, c_m_emb, c_m_map, c_a_emb, c_a_map,
                    K_PER_SOURCE, device
                )
                
                # --- Channel B: Address Hash Join ---
                valid_a1 = get_valid_keys(c_c_df, 'key_addr1', MAX_BLOCK_SIZE)
                df_b1 = s1_c_df[['entity_id', 'key_addr1']].dropna().merge(
                    c_c_df[c_c_df['key_addr1'].isin(valid_a1)][['entity_id', 'key_addr1']],
                    on='key_addr1'
                ).rename(columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'})[['s1_id', 'cand_id']]
                
                valid_a2 = get_valid_keys(c_c_df, 'key_addr2', MAX_BLOCK_SIZE)
                df_b2 = s1_c_df[['entity_id', 'key_addr2']].dropna().merge(
                    c_c_df[c_c_df['key_addr2'].isin(valid_a2)][['entity_id', 'key_addr2']],
                    on='key_addr2'
                ).rename(columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'})[['s1_id', 'cand_id']]
                
                df_b = pd.concat([df_b1, df_b2], ignore_index=True).drop_duplicates()
                df_b = df_b.groupby('s1_id').head(MAX_ADDR_CANDS).reset_index(drop=True)
                df_b['ch_addr'] = 1
                
                # --- Channel C: Name Skeleton Hash Join ---
                valid_sk = get_valid_keys(c_c_df, 'key_skel', MAX_BLOCK_SIZE)
                df_c = s1_c_df[['entity_id', 'key_skel']].dropna().merge(
                    c_c_df[c_c_df['key_skel'].isin(valid_sk)][['entity_id', 'key_skel']],
                    on='key_skel'
                ).rename(columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'})[['s1_id', 'cand_id']]
                
                df_c = df_c.drop_duplicates()
                df_c = df_c.groupby('s1_id').head(MAX_SKEL_CANDS).reset_index(drop=True)
                df_c['ch_skel'] = 1
                
                # --- Union Channels ---
                merged_cands = pd.concat([df_a, df_b, df_c], ignore_index=True)
                if merged_cands.empty:
                    continue
                    
                agg_dict = {
                    'ch_emb': 'max', 'ch_addr': 'max', 'ch_skel': 'max',
                    'emb_score': 'max', 'emb_rank': 'min'
                }
                merged_cands = merged_cands.groupby(['s1_id', 'cand_id']).agg(agg_dict).reset_index()
                merged_cands[['ch_emb', 'ch_addr', 'ch_skel']] = merged_cands[['ch_emb', 'ch_addr', 'ch_skel']].fillna(0).astype(int)
                
                # Compute missing emb_score for Channel B/C candidates
                missing_mask = merged_cands['emb_score'].isna()
                if missing_mask.any():
                    df_miss = merged_cands[missing_mask]
                    missing_scores = compute_missing_scores_vectorized(
                        df_miss, s1_m_emb, s1_m_map, s1_a_emb, s1_a_map,
                        c_m_emb, c_m_map, c_a_emb, c_a_map
                    )
                    merged_cands.loc[missing_mask, 'emb_score'] = missing_scores
                    merged_cands.loc[missing_mask, 'emb_rank'] = 999
                    
                merged_cands['cand_source'] = src_code
                chunk_cands_list.append(merged_cands)
                
        if not chunk_cands_list:
            chunk_df = pd.DataFrame(columns=['s1_id', 'cand_id', 'cand_source', 'emb_score', 'emb_rank', 'ch_emb', 'ch_addr', 'ch_skel'])
        else:
            chunk_df = pd.concat(chunk_cands_list, ignore_index=True)
            # Cap candidates per S1 to CAND_CAP (top by emb_score)
            chunk_df = chunk_df.sort_values(['s1_id', 'emb_score'], ascending=[True, False])
            chunk_df = chunk_df.groupby('s1_id').head(CAND_CAP).reset_index(drop=True)
            
        # Add label column if ground truth exists
        if gt_df is not None:
            gt_pairs = gt_df[['s1_id', 'match_id']].rename(columns={'match_id': 'cand_id'})
            gt_pairs['label'] = 1
            chunk_df = chunk_df.merge(gt_pairs, on=['s1_id', 'cand_id'], how='left')
            chunk_df['label'] = chunk_df['label'].fillna(0).astype(int)
            
        chunk_df.to_parquet(chunk_out_path, index=False)
        print(f"Saved chunk {ch_idx + 1} ({len(chunk_df)} candidate pairs) to {chunk_out_path}")

    # Merge all chunk parquets into final cands_{split}.parquet
    final_cands_path = os.path.join(cache_dir, f"cands_{args.split}.parquet")
    all_chunks_df = [pd.read_parquet(cp) for cp in chunk_files if os.path.exists(cp)]
    
    if all_chunks_df:
        final_df = pd.concat(all_chunks_df, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=['s1_id', 'cand_id', 'cand_source', 'emb_score', 'emb_rank', 'ch_emb', 'ch_addr', 'ch_skel'])
        
    final_df.to_parquet(final_cands_path, index=False)
    elapsed_total = time.time() - start_total_time
    print(f"\nFinal candidates saved to: {final_cands_path}")
    print(f"Total candidate pairs: {len(final_df)} in {elapsed_total:.2f}s")

    # If training split, run full recall report
    if args.split == "train" and gt_df is not None:
        norm_s1 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source1.parquet"))
        norm_s2 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source2.parquet"))
        norm_s3 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source3.parquet"))
        run_recall_report(final_df, gt_df, norm_s1, norm_s2, norm_s3)

if __name__ == "__main__":
    main()

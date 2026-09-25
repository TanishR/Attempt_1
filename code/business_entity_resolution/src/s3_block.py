import os
import sys
import gc
import time
import argparse
from collections import Counter
import numpy as np
import pandas as pd
import torch
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _rss_mb() -> float:
    """Returns current process RSS in MB, or -1 if psutil unavailable."""
    if _HAS_PSUTIL:
        return psutil.Process(os.getpid()).memory_info().rss / 1_048_576
    return -1.0

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import (
    CACHE_DIR, CAND_CAP, K_PER_SOURCE, MAX_ADDR_CANDS,
    MAX_SKEL_CANDS, MAX_RARE_CANDS, MAX_BLOCK_SIZE, BLOCK_BY_COUNTRY
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
    parser.add_argument("--k-channel-a", type=int, default=K_PER_SOURCE, help="Number of nearest neighbors to retrieve per source in Channel A.")
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
    Loads normalized table columns needed for symbolic blocking (Channels B, C, and E).
    Returns: pandas.DataFrame with entity_id, country, house_cands, num_tokens, zip_pin, name_skel, addr_norm.
    """
    path = os.path.join(cache_dir, f"norm_{split}_{source}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = ["entity_id", "country", "house_cands", "num_tokens", "zip_pin", "name_skel", "addr_norm"]
    # Fallback if some columns missing
    available_cols = pd.read_parquet(path).columns
    load_cols = [c for c in cols if c in available_cols]
    df = pd.read_parquet(path, columns=load_cols)
    return df

def build_channel_b_keys(df):
    """
    Channel B v2: builds keys (country, number, next_alpha) for up to 3 house candidates
    plus (country, zip_pin, number) for each number token.
    Returns: pandas.DataFrame with columns ['entity_id', 'b_key'].
    """
    keys_list = []
    
    # 1. House candidates: (country, number, next_alpha)
    if 'house_cands' in df.columns:
        df_h = df[['entity_id', 'country', 'house_cands']].dropna()
        df_h = df_h[df_h['house_cands'] != '']
        if not df_h.empty:
            df_h = df_h.assign(b_key=df_h['house_cands'].str.split(';')).explode('b_key')
            df_h = df_h[df_h['b_key'].str.len() > 0]
            df_h['b_key'] = "H_" + df_h['country'] + "_" + df_h['b_key']
            keys_list.append(df_h[['entity_id', 'b_key']])
            
    # 2. Zip + number: (country, zip_pin, number)
    if 'zip_pin' in df.columns and 'num_tokens' in df.columns:
        df_z = df[['entity_id', 'country', 'zip_pin', 'num_tokens']].dropna()
        df_z = df_z[(df_z['zip_pin'] != '') & (df_z['num_tokens'] != '')]
        if not df_z.empty:
            df_z = df_z.assign(b_key=df_z['num_tokens'].str.split()).explode('b_key')
            df_z = df_z[df_z['b_key'].str.len() > 0]
            df_z['b_key'] = "Z_" + df_z['country'] + "_" + df_z['zip_pin'] + "_" + df_z['b_key']
            keys_list.append(df_z[['entity_id', 'b_key']])
            
    if keys_list:
        return pd.concat(keys_list, ignore_index=True).drop_duplicates()
    return pd.DataFrame(columns=['entity_id', 'b_key'])

def build_channel_c_keys(df):
    """
    Channel C: builds keys (country, sorted name_skel) for typo/cross-script near-exact matches.
    Returns: pandas.DataFrame with columns ['entity_id', 'c_key'].
    """
    if 'name_skel' not in df.columns:
        return pd.DataFrame(columns=['entity_id', 'c_key'])
    df_sk = df[['entity_id', 'country', 'name_skel']].dropna()
    df_sk = df_sk[df_sk['name_skel'] != '']
    if df_sk.empty:
        return pd.DataFrame(columns=['entity_id', 'c_key'])
    sk_sorted = df_sk['name_skel'].astype(str).apply(lambda x: " ".join(sorted(x.split())))
    df_sk = df_sk.assign(c_key="SK_" + df_sk['country'] + "_" + sk_sorted)
    return df_sk[['entity_id', 'c_key']].drop_duplicates()

def compute_address_token_df(s1_df, s2_df, s3_df):
    """
    Computes per-country document frequencies of address tokens (len >= 5) across S1+S2+S3.
    Returns: dict mapping country string to Counter of {token: doc_count}.
    """
    df_tokens = {}
    all_countries = set(s1_df['country']).union(set(s2_df['country'])).union(set(s3_df['country']))
    for c in all_countries:
        c_counter = Counter()
        for df_src in [s1_df, s2_df, s3_df]:
            if df_src.empty or 'country' not in df_src.columns or 'addr_norm' not in df_src.columns:
                continue
            sub = df_src[df_src['country'] == c]
            for addr in sub['addr_norm'].dropna():
                if not isinstance(addr, str) or not addr.strip():
                    continue
                toks = set(t for t in addr.split() if len(t) >= 5)
                for t in toks:
                    c_counter[t] += 1
        df_tokens[c] = c_counter
    return df_tokens

def build_channel_e_keys(df, df_tokens):
    """
    Channel E: builds keys (country, tok1) and (country, sorted(tok1, tok2)) for 2 rarest address tokens (len >= 5).
    Returns: pandas.DataFrame with columns ['entity_id', 'e_key'].
    """
    records = []
    if 'addr_norm' not in df.columns or 'country' not in df.columns:
        return pd.DataFrame(columns=['entity_id', 'e_key'])
        
    for eid, c, addr in zip(df['entity_id'], df['country'], df['addr_norm']):
        if not isinstance(addr, str) or not addr.strip():
            continue
        toks = list(set(t for t in addr.split() if len(t) >= 5))
        if not toks:
            continue
        c_counter = df_tokens.get(c, {})
        toks.sort(key=lambda t: (c_counter.get(t, 0), t))
        tok1 = toks[0]
        records.append((eid, f"E1_{c}_{tok1}"))
        if len(toks) >= 2:
            tok2 = toks[1]
            pair_key = "_".join(sorted([tok1, tok2]))
            records.append((eid, f"E2_{c}_{pair_key}"))
            
    if records:
        return pd.DataFrame(records, columns=['entity_id', 'e_key']).drop_duplicates()
    return pd.DataFrame(columns=['entity_id', 'e_key'])

def get_valid_keys(df, col, max_block_size):
    """
    Filters out keys that appear in more than max_block_size records to avoid combinatorial explosion.
    Returns: pandas.Index containing valid keys.
    """
    counts = df[col].dropna().value_counts()
    return set(counts[counts <= max_block_size].index)


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

def search_channel_d_reverse_gpu(
    s1_keys_df, c_keys_df, s1_main_emb, s1_main_map, s1_main_ids, s1_alt_emb, s1_alt_map, s1_alt_ids,
    c_main_emb, c_main_map, c_alt_emb, c_alt_map, topk, device, src_code
):
    """
    Channel D (reverse search): for each S2/S3 record, finds top-k S1 entities within country using main+alt embeddings.
    Returns: pandas.DataFrame with columns ['s1_id', 'cand_id', 'cand_source', 'emb_score', 'ch_rev'].
    """
    rev_results = []
    countries = s1_keys_df['country'].dropna().unique() if BLOCK_BY_COUNTRY else ["ALL"]
    
    for country in countries:
        if BLOCK_BY_COUNTRY:
            s1_sub = s1_keys_df[s1_keys_df['country'] == country]
            c_sub = c_keys_df[c_keys_df['country'] == country]
        else:
            s1_sub = s1_keys_df
            c_sub = c_keys_df
            
        s1_c_ids = s1_sub['entity_id'].values
        cand_c_ids = c_sub['entity_id'].values
        
        if len(s1_c_ids) == 0 or len(cand_c_ids) == 0:
            continue
            
        # Build S1 index: main + alt stacked
        s1_m_idx = [s1_main_map[sid] for sid in s1_c_ids if sid in s1_main_map]
        X_main = s1_main_emb[s1_m_idx]
        X_main_ids = s1_c_ids[[sid in s1_main_map for sid in s1_c_ids]]
        
        s1_a_sids = [sid for sid in s1_c_ids if sid in s1_alt_map]
        if s1_a_sids:
            s1_a_idx = [s1_alt_map[sid] for sid in s1_a_sids]
            X_alt = s1_alt_emb[s1_a_idx]
            X_alt_ids = np.array(s1_a_sids, dtype=object)
            X_stacked = np.vstack([X_main, X_alt])
            row_to_sid = np.concatenate([X_main_ids, X_alt_ids])
        else:
            X_stacked = X_main
            row_to_sid = X_main_ids
            
        N_index = len(X_stacked)
        if N_index == 0:
            continue
            
        X_gpu = torch.from_numpy(X_stacked).to(device)
        
        # Calculate query chunk size
        max_bytes = 4 * 1024 * 1024 * 1024
        bytes_per_query = N_index * 2
        max_q_chunk = max(1, max_bytes // max(1, bytes_per_query))
        q_chunk_size = min(20000, max_q_chunk)
        k_search = min(2 * topk, N_index)
        
        for q_start in range(0, len(cand_c_ids), q_chunk_size):
            q_end = min(q_start + q_chunk_size, len(cand_c_ids))
            sub_q_cids = cand_c_ids[q_start:q_end]
            
            sub_m_idx = [c_main_map[cid] for cid in sub_q_cids]
            Q_m = torch.from_numpy(c_main_emb[sub_m_idx]).to(device)
            S = Q_m @ X_gpu.T
            
            has_alt = np.array([cid in c_alt_map for cid in sub_q_cids])
            if has_alt.any():
                sub_a_cids = [cid for cid in sub_q_cids if cid in c_alt_map]
                sub_a_idx = [c_alt_map[cid] for cid in sub_a_cids]
                Q_a = torch.from_numpy(c_alt_emb[sub_a_idx]).to(device)
                S_a = Q_a @ X_gpu.T
                S[has_alt] = torch.maximum(S[has_alt], S_a)
                
            top_scores, top_indices = torch.topk(S, k=k_search, dim=1)
            top_scores = top_scores.float().cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            for b, cid in enumerate(sub_q_cids):
                b_scores = top_scores[b]
                b_sids = row_to_sid[top_indices[b]]
                seen = {}
                for sid, score in zip(b_sids, b_scores):
                    if sid not in seen or score > seen[sid]:
                        seen[sid] = score
                        if len(seen) >= topk:
                            break
                for sid, score in seen.items():
                    rev_results.append((sid, cid, float(score), src_code, 1))
                    
        del X_gpu
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    df_rev = pd.DataFrame(rev_results, columns=['s1_id', 'cand_id', 'emb_score', 'cand_source', 'ch_rev'])
    return df_rev

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

def run_recall_report(df_cands, gt, norm_s1, norm_s2, norm_s3, cand_cap=CAND_CAP):
    """
    Computes and prints comprehensive recall analytics, channel contributions, and side-by-side missed pairs.
    """
    print("\n" + "="*70)
    print("               STEP 5 CANDIDATE GENERATION RECALL REPORT")
    print("="*70)
    
    # Evaluate primary metrics at configured CAND_CAP
    is_sym = (df_cands['ch_addr'] == 1) | (df_cands['ch_skel'] == 1) | (df_cands.get('ch_rare', 0) == 1)
    df_sorted = df_cands.assign(is_sym=is_sym).sort_values(['s1_id', 'is_sym', 'emb_score'], ascending=[True, False, False])
    df_eval_cands = df_sorted.groupby('s1_id').head(cand_cap).reset_index(drop=True)
    
    s1_in_cands = set(df_eval_cands['s1_id'].unique())
    gt_eval = gt[gt['s1_id'].isin(s1_in_cands)].copy()
    total_gt_pairs = len(gt_eval)
    
    # 1. Pair recall overall
    cands_set = set(zip(df_eval_cands['s1_id'], df_eval_cands['cand_id']))
    gt_eval['found'] = [pair in cands_set for pair in zip(gt_eval['s1_id'], gt_eval['match_id'])]
    found_gt_pairs = gt_eval['found'].sum()
    overall_recall = (found_gt_pairs / max(1, total_gt_pairs)) * 100
    print(f"\n1. Overall Pair Recall (CAND_CAP={cand_cap}): {found_gt_pairs} / {total_gt_pairs} ({overall_recall:.2f}%)")
    
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
    cands_grouped = df_eval_cands.groupby('s1_id')['cand_id'].apply(set).to_dict()
    
    fully_found_entities = 0
    for sid, gold_set in gt_grouped.items():
        cand_set = cands_grouped.get(sid, set())
        if gold_set.issubset(cand_set):
            fully_found_entities += 1
            
    total_entities = len(gt_grouped)
    entity_recall = (fully_found_entities / max(1, total_entities)) * 100
    print(f"\n2. Entity-Level Recall (100% matches in candidates at CAND_CAP={cand_cap}): {fully_found_entities} / {total_entities} ({entity_recall:.2f}%)")
    
    # 3. Candidates per S1 statistics
    cands_per_s1 = df_eval_cands.groupby('s1_id').size()
    avg_cands = cands_per_s1.mean()
    p95_cands = cands_per_s1.quantile(0.95)
    max_cands = cands_per_s1.max()
    print(f"\n3. Candidate Count Statistics (at CAND_CAP={cand_cap}):")
    print(f"   - Mean candidates per S1: {avg_cands:.2f}")
    print(f"   - 95th percentile candidates per S1: {p95_cands:.1f}")
    print(f"   - Max candidates per S1: {max_cands}")
    
    # 4. Unique contribution per channel
    # Merge candidates with GT to inspect channel tags of found GT pairs
    gt_pairs_df = gt_eval[gt_eval['found']][['s1_id', 'match_id']].rename(columns={'match_id': 'cand_id'})
    found_cands = df_eval_cands.merge(gt_pairs_df, on=['s1_id', 'cand_id'], how='inner')
    
    only_emb = len(found_cands[(found_cands['ch_emb'] == 1) & (found_cands['ch_addr'] == 0) & (found_cands['ch_skel'] == 0) & (found_cands.get('ch_rare', 0) == 0) & (found_cands.get('ch_rev', 0) == 0)])
    only_addr = len(found_cands[(found_cands['ch_emb'] == 0) & (found_cands['ch_addr'] == 1) & (found_cands['ch_skel'] == 0) & (found_cands.get('ch_rare', 0) == 0) & (found_cands.get('ch_rev', 0) == 0)])
    only_skel = len(found_cands[(found_cands['ch_emb'] == 0) & (found_cands['ch_addr'] == 0) & (found_cands['ch_skel'] == 1) & (found_cands.get('ch_rare', 0) == 0) & (found_cands.get('ch_rev', 0) == 0)])
    only_rare = len(found_cands[(found_cands['ch_emb'] == 0) & (found_cands['ch_addr'] == 0) & (found_cands['ch_skel'] == 0) & (found_cands.get('ch_rare', 0) == 1) & (found_cands.get('ch_rev', 0) == 0)])
    only_rev = len(found_cands[(found_cands['ch_emb'] == 0) & (found_cands['ch_addr'] == 0) & (found_cands['ch_skel'] == 0) & (found_cands.get('ch_rare', 0) == 0) & (found_cands.get('ch_rev', 0) == 1)])
    
    print(f"\n4. Unique Channel Contributions on Recovered GT Pairs:")
    print(f"   - Unique to Channel A (Embedding): {only_emb} ({only_emb/max(1, found_gt_pairs)*100:.2f}%)")
    print(f"   - Unique to Channel B (Address):   {only_addr} ({only_addr/max(1, found_gt_pairs)*100:.2f}%)")
    print(f"   - Unique to Channel C (Skeleton):  {only_skel} ({only_skel/max(1, found_gt_pairs)*100:.2f}%)")
    print(f"   - Unique to Channel E (Rare Addr): {only_rare} ({only_rare/max(1, found_gt_pairs)*100:.2f}%)")
    print(f"   - Unique to Channel D (Reverse):   {only_rev} ({only_rev/max(1, found_gt_pairs)*100:.2f}%)")
    
    # 5. Channel A Recall@k curve
    print(f"\n5. Channel A (Embedding) Recall@k Curve:")
    ch_a_cands = df_cands[df_cands['ch_emb'] == 1]
    for k in [5, 10, 15, 20, 30, 50]:
        ch_a_k = ch_a_cands[ch_a_cands['emb_rank'] <= k]
        pairs_k = set(zip(ch_a_k['s1_id'], ch_a_k['cand_id']))
        f_k = sum(pair in pairs_k for pair in zip(gt_eval['s1_id'], gt_eval['match_id']))
        print(f"   - Channel A Recall@{k}: {f_k} / {total_gt_pairs} ({(f_k / max(1, total_gt_pairs))*100:.2f}%)")
        
    # 6. Multi-CAND_CAP recall evaluation
    print(f"\n6. Recall Across Multiple Candidate Caps (CAND_CAP = 30, 40, 50):")
    is_sym = (df_cands['ch_addr'] == 1) | (df_cands['ch_skel'] == 1) | (df_cands.get('ch_rare', 0) == 1)
    df_sorted = df_cands.assign(is_sym=is_sym).sort_values(['s1_id', 'is_sym', 'emb_score'], ascending=[True, False, False])
    for cap in [30, 40, 50]:
        df_cap = df_sorted.groupby('s1_id').head(cap)
        pairs_cap = set(zip(df_cap['s1_id'], df_cap['cand_id']))
        f_cap = sum(pair in pairs_cap for pair in zip(gt_eval['s1_id'], gt_eval['match_id']))
        print(f"   - Recall @ CAND_CAP={cap}: {f_cap} / {total_gt_pairs} ({(f_cap / max(1, total_gt_pairs))*100:.2f}%)")
        
    # 7. Sample of 20 missed pairs side by side
    missed_gt = gt_eval[~gt_eval['found']]
    print(f"\n7. Side-by-Side Sample of Missed Pairs (Total Missed: {len(missed_gt)}):")
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

    # Document frequency of address tokens (len >= 5) across S1+S2+S3 for Channel E
    print("Computing address token document frequencies across S1+S2+S3...")
    df_tokens = compute_address_token_df(s1_keys_df, c2_keys_df, c3_keys_df)
    c2_e_keys = build_channel_e_keys(c2_keys_df, df_tokens)
    c3_e_keys = build_channel_e_keys(c3_keys_df, df_tokens)

    # Channel D (Reverse Search): compute top-3 S1 per S2 and S3 record within country
    print("Running Channel D (Reverse Embedding Search: top-3 S1 per S2/S3 record)...")
    rev_cands_s2 = search_channel_d_reverse_gpu(
        s1_keys_df, c2_keys_df, s1_m_emb, s1_m_map, s1_m_ids, s1_a_emb, s1_a_map, s1_a_ids,
        c2_m_emb, c2_m_map, c2_a_emb, c2_a_map, 3, device, 0
    )
    rev_cands_s3 = search_channel_d_reverse_gpu(
        s1_keys_df, c3_keys_df, s1_m_emb, s1_m_map, s1_m_ids, s1_a_emb, s1_a_map, s1_a_ids,
        c3_m_emb, c3_m_map, c3_a_emb, c3_a_map, 3, device, 1
    )
    all_rev_cands = pd.concat([rev_cands_s2, rev_cands_s3], ignore_index=True)
    all_rev_cands['emb_rank'] = 999
    all_rev_cands['ch_emb'] = 0
    all_rev_cands['ch_addr'] = 0
    all_rev_cands['ch_skel'] = 0
    all_rev_cands['ch_rare'] = 0

    # Load GT for label assignment if train/val split
    gt_df = None
    if args.split == "train":
        gt_path = os.path.join(cache_dir, "gt_long.parquet")
        if os.path.exists(gt_path):
            gt_df = pd.read_parquet(gt_path)
            print(f"Loaded ground truth: {len(gt_df)} rows")

    # Partition S1 into chunks of args.chunk_size (100k queries) with resume
    # NOTE: For split=train, this is ALL S1 from the norm file (not just the sampled
    # train/val split). This ensures Channel D, reverse_rank and exclusivity see
    # realistic competition matching the test-time distribution.
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
            print(f"Chunk {ch_idx + 1}/{num_chunks} already exists ({chunk_out_path}), skipping...", flush=True)
            continue

        elapsed_so_far = time.time() - start_total_time
        rss = _rss_mb()
        print(f"\n>>> Processing Chunk {ch_idx + 1}/{num_chunks}  (elapsed {elapsed_so_far:.0f}s, RSS {rss:.0f} MB)", flush=True)
        c_s1_ids = all_s1_ids[ch_idx * chunk_size : (ch_idx + 1) * chunk_size]
        sub_s1_keys = s1_keys_df[s1_keys_df['entity_id'].isin(c_s1_ids)].copy()
        
        chunk_cands_list = []
        
        # Sources to block against
        cand_sources = [
            ("source2", 0, c2_keys_df, c2_e_keys, c2_m_emb, c2_m_map, c2_m_ids, c2_a_emb, c2_a_map, c2_a_ids),
            ("source3", 1, c3_keys_df, c3_e_keys, c3_m_emb, c3_m_map, c3_m_ids, c3_a_emb, c3_a_map, c3_a_ids)
        ]
        
        countries = sub_s1_keys['country'].dropna().unique() if BLOCK_BY_COUNTRY else ["ALL"]
        
        for src_name, src_code, c_keys_df, c_e_keys, c_m_emb, c_m_map, c_m_ids, c_a_emb, c_a_map, c_a_ids in cand_sources:
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
                    args.k_channel_a, device
                )
                
                # --- Channel B: Address Hash Join (v2) ---
                s1_b_keys = build_channel_b_keys(s1_c_df)
                c_b_keys = build_channel_b_keys(c_c_df)
                if not s1_b_keys.empty and not c_b_keys.empty:
                    valid_b = get_valid_keys(c_b_keys, 'b_key', MAX_BLOCK_SIZE)
                    s1_b_valid = s1_b_keys[s1_b_keys['b_key'].isin(valid_b)]
                    c_b_valid = c_b_keys[c_b_keys['b_key'].isin(valid_b)]
                    df_b = s1_b_valid.merge(c_b_valid, on='b_key').rename(
                        columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'}
                    )[['s1_id', 'cand_id']].drop_duplicates()
                    df_b = df_b.groupby('s1_id').head(MAX_ADDR_CANDS).reset_index(drop=True)
                    df_b['ch_addr'] = 1
                else:
                    df_b = pd.DataFrame(columns=['s1_id', 'cand_id', 'ch_addr'])
                
                # --- Channel C: Name Skeleton Hash Join ---
                s1_c_keys = build_channel_c_keys(s1_c_df)
                c_c_keys = build_channel_c_keys(c_c_df)
                if not s1_c_keys.empty and not c_c_keys.empty:
                    valid_c = get_valid_keys(c_c_keys, 'c_key', MAX_BLOCK_SIZE)
                    s1_c_valid = s1_c_keys[s1_c_keys['c_key'].isin(valid_c)]
                    c_c_valid = c_c_keys[c_c_keys['c_key'].isin(valid_c)]
                    df_c = s1_c_valid.merge(c_c_valid, on='c_key').rename(
                        columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'}
                    )[['s1_id', 'cand_id']].drop_duplicates()
                    df_c = df_c.groupby('s1_id').head(MAX_SKEL_CANDS).reset_index(drop=True)
                    df_c['ch_skel'] = 1
                else:
                    df_c = pd.DataFrame(columns=['s1_id', 'cand_id', 'ch_skel'])

                # --- Channel E: Rare Address Tokens ---
                s1_e_keys = build_channel_e_keys(s1_c_df, df_tokens)
                c_e_keys_c = c_e_keys[c_e_keys['entity_id'].isin(cand_c_ids)]
                if not s1_e_keys.empty and not c_e_keys_c.empty:
                    valid_e = get_valid_keys(c_e_keys_c, 'e_key', MAX_BLOCK_SIZE)
                    s1_e_valid = s1_e_keys[s1_e_keys['e_key'].isin(valid_e)]
                    c_e_valid = c_e_keys_c[c_e_keys_c['e_key'].isin(valid_e)]
                    df_e = s1_e_valid.merge(c_e_valid, on='e_key').rename(
                        columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'}
                    )[['s1_id', 'cand_id']].drop_duplicates()
                    df_e = df_e.groupby('s1_id').head(MAX_RARE_CANDS).reset_index(drop=True)
                    df_e['ch_rare'] = 1
                else:
                    df_e = pd.DataFrame(columns=['s1_id', 'cand_id', 'ch_rare'])

                # --- Union Channels A, B, C, E ---
                merged_cands = pd.concat([df_a, df_b, df_c, df_e], ignore_index=True)
                if merged_cands.empty:
                    continue
                    
                agg_dict = {
                    'ch_emb': 'max', 'ch_addr': 'max', 'ch_skel': 'max', 'ch_rare': 'max',
                    'emb_score': 'max', 'emb_rank': 'min'
                }
                merged_cands = merged_cands.groupby(['s1_id', 'cand_id']).agg(agg_dict).reset_index()
                merged_cands[['ch_emb', 'ch_addr', 'ch_skel', 'ch_rare']] = merged_cands[['ch_emb', 'ch_addr', 'ch_skel', 'ch_rare']].fillna(0).astype(int)
                
                # Compute missing emb_score for non-Channel A candidates
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
                
        # Append Channel D reverse candidates for S1 in this chunk
        df_d_chunk = all_rev_cands[all_rev_cands['s1_id'].isin(c_s1_ids)].copy()
        if not df_d_chunk.empty:
            chunk_cands_list.append(df_d_chunk)
            
        if not chunk_cands_list:
            chunk_df = pd.DataFrame(columns=['s1_id', 'cand_id', 'cand_source', 'emb_score', 'emb_rank', 'ch_emb', 'ch_addr', 'ch_skel', 'ch_rare', 'ch_rev'])
        else:
            chunk_df = pd.concat(chunk_cands_list, ignore_index=True)
            agg_chunk = {
                'ch_emb': 'max', 'ch_addr': 'max', 'ch_skel': 'max', 'ch_rare': 'max', 'ch_rev': 'max',
                'emb_score': 'max', 'emb_rank': 'min', 'cand_source': 'first'
            }
            chunk_df = chunk_df.groupby(['s1_id', 'cand_id']).agg(agg_chunk).reset_index()
            chunk_df[['ch_emb', 'ch_addr', 'ch_skel', 'ch_rare', 'ch_rev']] = chunk_df[['ch_emb', 'ch_addr', 'ch_skel', 'ch_rare', 'ch_rev']].fillna(0).astype(int)
            
            # Cap candidates per S1 to CAND_CAP, prioritizing symbolic address blocks
            is_sym = (chunk_df['ch_addr'] == 1) | (chunk_df['ch_skel'] == 1) | (chunk_df['ch_rare'] == 1)
            chunk_df = chunk_df.assign(is_sym=is_sym).sort_values(['s1_id', 'is_sym', 'emb_score'], ascending=[True, False, False]).drop(columns=['is_sym'])
            chunk_df = chunk_df.groupby('s1_id').head(CAND_CAP).reset_index(drop=True)
            
        # Add label column if ground truth exists
        if gt_df is not None:
            gt_pairs = gt_df[['s1_id', 'match_id']].rename(columns={'match_id': 'cand_id'})
            gt_pairs['label'] = 1
            chunk_df = chunk_df.merge(gt_pairs, on=['s1_id', 'cand_id'], how='left')
            chunk_df['label'] = chunk_df['label'].fillna(0).astype(int)
            
        chunk_df.to_parquet(chunk_out_path, index=False)
        elapsed_ch = time.time() - start_total_time
        rss_ch = _rss_mb()
        print(f"Saved chunk {ch_idx + 1}/{num_chunks} ({len(chunk_df)} candidate pairs) to {chunk_out_path}  "
              f"(total elapsed {elapsed_ch:.0f}s, RSS {rss_ch:.0f} MB)", flush=True)
        del chunk_df, chunk_cands_list
        gc.collect()

    # --------------------------------------------------------------------------
    # Post-processing: NO combined 9-crore parquet.
    #
    # Problem with the previous approach: pd.concat on Categorical columns where
    # each chunk has different categories silently reverts to str/object dtype,
    # giving ZERO memory saving (confirmed: dtype='str' after concat in tests).
    #
    # Solution: never build the combined DataFrame at all.
    #   - Recall report reads only val-sample S1 rows, chunk by chunk.
    #   - s4_features.py already reads per-chunk files directly.
    #   - cands_{split}.parquet is NOT written (downstream must use chunk files).
    # --------------------------------------------------------------------------
    elapsed_total = time.time() - start_total_time
    rss_post = _rss_mb()
    print(f"\n--- Post-processing: {len(chunk_files)} chunk(s)  "
          f"(elapsed so far {elapsed_total:.0f}s, RSS {rss_post:.0f} MB) ---", flush=True)

    # Count total pairs from chunk metadata only (no full load)
    total_pairs_count = 0
    existing_chunks = [cp for cp in chunk_files if os.path.exists(cp)]
    for cp in existing_chunks:
        meta = pd.read_parquet(cp, columns=['s1_id']).shape[0]
        total_pairs_count += meta
    print(f"  Total candidate pairs across {len(existing_chunks)} chunks: {total_pairs_count:,}", flush=True)

    # Determine val-sample S1 ids for the recall report (train only)
    val_s1_set = set()
    if args.split == "train" and gt_df is not None:
        split_path = os.path.join(cache_dir, "split.parquet")
        if os.path.exists(split_path):
            split_df = pd.read_parquet(split_path, columns=['s1_id'])
            val_s1_set = set(split_df['s1_id'].values)
            print(f"  Val-sample S1 for recall report: {len(val_s1_set):,}", flush=True)

    # Recall report: filter to val-sample S1 per chunk, concat small result
    if args.split == "train" and gt_df is not None:
        print("\n--- Running recall report (val-sample rows only, no full concat) ---", flush=True)
        val_chunk_dfs = []
        for ci, cp in enumerate(existing_chunks):
            df_c = pd.read_parquet(cp)
            if val_s1_set:
                df_c = df_c[df_c['s1_id'].isin(val_s1_set)]
            val_chunk_dfs.append(df_c)
            rss_ci = _rss_mb()
            print(f"  Chunk {ci + 1}/{len(existing_chunks)}: kept {len(df_c):,} val rows  "
                  f"(RSS {rss_ci:.0f} MB)", flush=True)
            del df_c
            gc.collect()

        recall_df = pd.concat(val_chunk_dfs, ignore_index=True)
        rss_recall = _rss_mb()
        # After concat of uniform-schema small frames: show dtype so it is auditable
        print(f"  Recall df: {len(recall_df):,} rows, "
              f"s1_id dtype={recall_df['s1_id'].dtype}, "
              f"cand_id dtype={recall_df['cand_id'].dtype}  "
              f"(RSS {rss_recall:.0f} MB)", flush=True)
        del val_chunk_dfs
        gc.collect()

        norm_cols = ['entity_id', 'name_full', 'addr_norm', 'country']
        norm_s1 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source1.parquet"),
                                  columns=norm_cols)
        norm_s1 = norm_s1[norm_s1['entity_id'].isin(val_s1_set)] if val_s1_set else norm_s1
        norm_s2 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source2.parquet"),
                                  columns=norm_cols)
        norm_s3 = pd.read_parquet(os.path.join(cache_dir, "norm_train_source3.parquet"),
                                  columns=norm_cols)
        run_recall_report(recall_df, gt_df, norm_s1, norm_s2, norm_s3, cand_cap=CAND_CAP)
        del recall_df, norm_s1, norm_s2, norm_s3
        gc.collect()

    elapsed_total = time.time() - start_total_time
    rss_done = _rss_mb()
    print(f"\nBlocking complete: {total_pairs_count:,} candidate pairs in {len(existing_chunks)} chunks "
          f"({elapsed_total:.0f}s, RSS {rss_done:.0f} MB)", flush=True)

    # Write done-marker (must be last so resume only skips if truly complete)
    done_marker = os.path.join(cache_dir, f"cands_{args.split}.done")
    with open(done_marker, 'w') as _f:
        _f.write("done\n")
    print(f"Done-marker written: {done_marker}", flush=True)

if __name__ == "__main__":
    main()

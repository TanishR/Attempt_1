import os
import sys
import gc
import time
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import CACHE_DIR, CAND_CAP, K_PER_SOURCE, MAX_ADDR_CANDS, MAX_SKEL_CANDS, MAX_BLOCK_SIZE

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    return parser.parse_args()

def load_embeddings_df(split, source):
    """Loads main and alt embeddings, returns (df_main, df_alt) indexed by entity_id"""
    emb_path = os.path.join(CACHE_DIR, f"emb_{split}_{source}.npy")
    ids_path = os.path.join(CACHE_DIR, f"ids_{split}_{source}.npy")
    
    if not os.path.exists(emb_path):
        return pd.DataFrame(), pd.DataFrame()
        
    main_emb = np.load(emb_path)
    main_ids = np.load(ids_path, allow_pickle=True)
    df_main = pd.DataFrame({'entity_id': main_ids, 'emb': list(main_emb)}).set_index('entity_id')
    
    alt_emb_path = os.path.join(CACHE_DIR, f"emb_{split}_{source}_alt.npy")
    alt_ids_path = os.path.join(CACHE_DIR, f"ids_{split}_{source}_alt.npy")
    
    if os.path.exists(alt_emb_path):
        alt_emb = np.load(alt_emb_path)
        alt_ids = np.load(alt_ids_path, allow_pickle=True)
        df_alt = pd.DataFrame({'entity_id': alt_ids, 'emb': list(alt_emb)}).set_index('entity_id')
    else:
        df_alt = pd.DataFrame(columns=['emb'])
            
    return df_main, df_alt

def get_country_map(split, source):
    path = os.path.join(CACHE_DIR, f"norm_{split}_{source}.parquet")
    if not os.path.exists(path):
        return pd.Series(dtype=str)
    df = pd.read_parquet(path, columns=["entity_id", "country"])
    return df.set_index("entity_id")["country"]

def topk_search_vectorized(q_ids, q_main, q_alt, x_emb, x_ids, k, chunk=20000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(x_emb).to(device)
    
    dfs = []
    
    for i in range(0, len(q_main), chunk):
        Q_m = torch.from_numpy(q_main[i:i+chunk]).to(device)
        Q_a = torch.from_numpy(q_alt[i:i+chunk]).to(device)
        
        S_m = Q_m @ X.T
        S_a = Q_a @ X.T
        S = torch.maximum(S_m, S_a)
        
        # 2*k guarantees at least k unique candidate IDs
        s, idx = torch.topk(S, k=min(2 * k, X.shape[0]), dim=1)
        s = s.float().cpu().numpy()
        idx = idx.cpu().numpy()
        
        q_ids_chunk = q_ids[i:i+chunk]
        q_ids_expanded = np.repeat(q_ids_chunk, s.shape[1])
        cand_ids_expanded = x_ids[idx.flatten()]
        scores_expanded = s.flatten()
        
        df_chunk = pd.DataFrame({
            's1_id': q_ids_expanded,
            'cand_id': cand_ids_expanded,
            'emb_score': scores_expanded
        })
        dfs.append(df_chunk)
        
    if not dfs:
        return pd.DataFrame(columns=['s1_id', 'cand_id', 'emb_score'])
        
    df_emb = pd.concat(dfs, ignore_index=True)
    df_emb = df_emb.drop_duplicates(subset=['s1_id', 'cand_id'], keep='first')
    df_emb = df_emb.groupby('s1_id').head(k).reset_index(drop=True)
    return df_emb

def load_keys(split, source):
    path = os.path.join(CACHE_DIR, f"norm_{split}_{source}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = ["entity_id", "country", "house_no", "street_token", "zip_pin", "name_skel"]
    df = pd.read_parquet(path, columns=cols)
    
    df["skel_sorted"] = df["name_skel"].astype(str).apply(lambda x: " ".join(sorted(x.split())))
    
    df["key_addr1"] = df.apply(lambda x: f"{x['country']}_{x['house_no']}_{x['street_token']}" if bool(x.get('house_no')) and bool(x.get('street_token')) else None, axis=1)
    df["key_addr2"] = df.apply(lambda x: f"{x['country']}_{x['zip_pin']}_{x['house_no']}" if bool(x.get('zip_pin')) and bool(x.get('house_no')) else None, axis=1)
    df["key_skel"] = df.apply(lambda x: f"{x['country']}_{x['skel_sorted']}" if bool(x.get('skel_sorted')) else None, axis=1)
    
    return df

def get_valid_keys(df, col):
    counts = df[col].value_counts()
    return counts[counts <= MAX_BLOCK_SIZE].index

def get_scores_vectorized(s1_ids, cand_ids, s1_main_df, s1_alt_df, c2_main_df, c2_alt_df):
    if len(s1_ids) == 0:
        return np.array([])
        
    df = pd.DataFrame({'s1_id': s1_ids, 'cand_id': cand_ids})
    
    # Merge S1
    df = df.merge(s1_main_df, left_on='s1_id', right_index=True, how='left')
    df = df.merge(s1_alt_df, left_on='s1_id', right_index=True, how='left', suffixes=('_s1m', '_s1a'))
    df['emb_s1a'] = df['emb_s1a'].fillna(df['emb_s1m'])
    
    # Merge C2
    df = df.merge(c2_main_df, left_on='cand_id', right_index=True, how='left')
    df = df.merge(c2_alt_df, left_on='cand_id', right_index=True, how='left', suffixes=('_c2m', '_c2a'))
    df['emb_c2a'] = df['emb_c2a'].fillna(df['emb_c2m'])
    
    # Handle missing embeddings (should be extremely rare, but just in case)
    valid_mask = df['emb_s1m'].notnull() & df['emb_c2m'].notnull()
    
    scores = np.zeros(len(df))
    if valid_mask.any():
        v1_m = np.vstack(df.loc[valid_mask, 'emb_s1m'].values)
        v1_a = np.vstack(df.loc[valid_mask, 'emb_s1a'].values)
        v2_m = np.vstack(df.loc[valid_mask, 'emb_c2m'].values)
        v2_a = np.vstack(df.loc[valid_mask, 'emb_c2a'].values)
        
        s11 = (v1_m * v2_m).sum(axis=1)
        s12 = (v1_m * v2_a).sum(axis=1)
        s21 = (v1_a * v2_m).sum(axis=1)
        s22 = (v1_a * v2_a).sum(axis=1)
        
        scores[valid_mask] = np.maximum.reduce([s11, s12, s21, s22])
        
    return scores

def run_recall_report(df_cands, gt, cand_source, country_map=None):
    # df_cands already joined with GT to find labels
    
    print(f"\n--- Recall Report ({cand_source if cand_source else 'Overall'}) ---")
    gt_s = gt[gt['match_source'] == cand_source] if ('match_source' in gt.columns and cand_source is not None) else gt
    
    # Total GT pairs
    total_gt = len(gt_s[gt_s['s1_id'].isin(df_cands['s1_id'].unique())])
    
    # Found pairs
    df_cands_gt = df_cands.merge(gt_s[['s1_id', 'cand_id']], on=['s1_id', 'cand_id'], how='inner')
    found_gt = len(df_cands_gt)
    print(f"Pair Recall: {found_gt} / {total_gt} ({(found_gt/max(1, total_gt))*100:.2f}%)")
    
    # By country recall
    if country_map is not None:
        df_c_map = df_cands['s1_id'].map(country_map)
        for c in df_c_map.unique():
            if pd.isna(c): continue
            s1_in_c = df_cands[df_c_map == c]['s1_id'].unique()
            gt_c = gt_s[gt_s['s1_id'].isin(s1_in_c)]
            total_c = len(gt_c)
            found_c = len(df_cands_gt[df_cands_gt['s1_id'].isin(s1_in_c)])
            if total_c > 0:
                print(f"  {c} Recall: {found_c} / {total_c} ({(found_c/max(1, total_c))*100:.2f}%)")
    
    # Entity-level recall
    gt_s_grouped = gt_s.groupby('s1_id')['cand_id'].apply(set)
    cands_grouped = df_cands.groupby('s1_id')['cand_id'].apply(set)
    
    full_recall_count = 0
    for s1_id, gt_cands in gt_s_grouped.items():
        if s1_id in cands_grouped:
            if gt_cands.issubset(cands_grouped[s1_id]):
                full_recall_count += 1
    print(f"Entity-level Recall (100% matches found): {full_recall_count} / {len(gt_s_grouped)} ({(full_recall_count/max(1, len(gt_s_grouped)))*100:.2f}%)")
    
    # Avg / 95th candidates
    cands_per_s1 = df_cands.groupby('s1_id').size()
    print(f"Avg candidates per S1: {cands_per_s1.mean():.2f}")
    print(f"95th pct candidates per S1: {cands_per_s1.quantile(0.95):.0f}")
    
    # Unique contribution
    ch_emb_only = len(df_cands_gt[(df_cands_gt['ch_emb'] == 1) & (df_cands_gt['ch_addr'] == 0) & (df_cands_gt['ch_skel'] == 0)])
    ch_addr_only = len(df_cands_gt[(df_cands_gt['ch_emb'] == 0) & (df_cands_gt['ch_addr'] == 1) & (df_cands_gt['ch_skel'] == 0)])
    ch_skel_only = len(df_cands_gt[(df_cands_gt['ch_emb'] == 0) & (df_cands_gt['ch_addr'] == 0) & (df_cands_gt['ch_skel'] == 1)])
    
    print(f"Unique GT pairs found ONLY by Channel A (Emb): {ch_emb_only}")
    print(f"Unique GT pairs found ONLY by Channel B (Addr): {ch_addr_only}")
    print(f"Unique GT pairs found ONLY by Channel C (Skel): {ch_skel_only}")
    
    # Recall @ K for Channel A
    print("\nChannel A Recall@K on missed pairs:")
    df_ch_a = df_cands[df_cands['ch_emb'] == 1]
    
    # Missed pairs overall? Or we just print Recall@K for Channel A overall?
    # The prompt asks for: "recall@k curve for channel A at k=5,10,15,20, and 30 missed pairs"
    for k in [5, 10, 15, 20]:
        df_a_k = df_ch_a[df_ch_a['emb_rank'] <= k]
        found_k = len(df_a_k.merge(gt_s[['s1_id', 'cand_id']], on=['s1_id', 'cand_id'], how='inner'))
        print(f"  Recall@{k}: {found_k} / {total_gt} ({(found_k/max(1, total_gt))*100:.2f}%)")
        
    missed_pairs = gt_s.merge(df_cands_gt, on=['s1_id', 'cand_id'], how='left', indicator=True)
    missed_pairs = missed_pairs[missed_pairs['_merge'] == 'left_only']
    print(f"\nTotal missed pairs: {len(missed_pairs)}")
    if len(missed_pairs) > 0:
        print("Sample of 30 missed pairs:")
        for idx, row in missed_pairs.head(30).iterrows():
            print(f"  S1: {row['s1_id']} | Cand: {row['cand_id']}")

def main():
    args = get_args()
    
    print("Loading S1 embeddings...")
    s1_main_df, s1_alt_df = load_embeddings_df(args.split, "source1")
    if s1_main_df.empty:
        print("S1 embeddings not found.")
        return
        
    s1_country_map = get_country_map(args.split, "source1")
    s1_keys = load_keys(args.split, "source1")
    
    countries = list(set(s1_country_map.dropna().values))
    all_cands_dfs = []
    
    start_time = time.time()
    num_s1_processed = len(s1_main_df)
    
    for cand_source in ["source2", "source3"]:
        print(f"\n--- Processing {cand_source} ---")
        c2_main_df, c2_alt_df = load_embeddings_df(args.split, cand_source)
        if c2_main_df.empty: continue
        c2_country_map = get_country_map(args.split, cand_source)
        c2_keys = load_keys(args.split, cand_source)
        
        dfs_c_source = []
        
        for c in countries:
            s1_c_ids = s1_country_map[s1_country_map == c].index.values
            s1_c_ids = np.intersect1d(s1_c_ids, s1_main_df.index.values)
            if len(s1_c_ids) == 0: continue
            
            c2_c_ids = c2_country_map[c2_country_map == c].index.values
            c2_c_ids = np.intersect1d(c2_c_ids, c2_main_df.index.values)
            if len(c2_c_ids) == 0: continue
            
            # --- Channel A ---
            x_main_c = np.vstack(c2_main_df.loc[c2_c_ids, 'emb'].values)
            x_alt_mask = c2_alt_df.index.isin(c2_c_ids)
            if x_alt_mask.any():
                c2_alt_c_ids = c2_alt_df.index[x_alt_mask].values
                x_alt_c = np.vstack(c2_alt_df.loc[c2_alt_c_ids, 'emb'].values)
                X_full = np.vstack([x_main_c, x_alt_c])
                X_ids_full = np.concatenate([c2_c_ids, c2_alt_c_ids])
            else:
                X_full = x_main_c
                X_ids_full = c2_c_ids
                
            q_main = np.vstack(s1_main_df.loc[s1_c_ids, 'emb'].values)
            
            s1_alt_mask = s1_alt_df.index.isin(s1_c_ids)
            if s1_alt_mask.any():
                q_alt_df = pd.DataFrame(index=s1_c_ids)
                q_alt_df = q_alt_df.merge(s1_alt_df, left_index=True, right_index=True, how='left')
                q_alt_df = q_alt_df.merge(s1_main_df, left_index=True, right_index=True, how='left', suffixes=('_a', '_m'))
                q_alt_df['emb_a'] = q_alt_df['emb_a'].fillna(q_alt_df['emb_m'])
                q_alt = np.vstack(q_alt_df['emb_a'].values)
            else:
                q_alt = q_main
                
            print(f"[{c}] Channel A: {len(s1_c_ids)} S1 vs {len(X_ids_full)} C2")
            df_emb = topk_search_vectorized(s1_c_ids, q_main, q_alt, X_full, X_ids_full, k=K_PER_SOURCE)
            df_emb['ch_emb'] = 1
            df_emb['emb_rank'] = df_emb.groupby('s1_id').cumcount() + 1
            
            # --- Channel B & C ---
            s1_keys_c = s1_keys[s1_keys['entity_id'].isin(s1_c_ids)]
            c2_keys_c = c2_keys[c2_keys['entity_id'].isin(c2_c_ids)]
            
            # Addr1
            valid_a1 = get_valid_keys(c2_keys_c, "key_addr1")
            df_a1 = s1_keys_c[['entity_id', 'key_addr1']].merge(
                c2_keys_c[c2_keys_c['key_addr1'].isin(valid_a1)][['entity_id', 'key_addr1']],
                on='key_addr1'
            ).rename(columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'})
            
            # Addr2
            valid_a2 = get_valid_keys(c2_keys_c, "key_addr2")
            df_a2 = s1_keys_c[['entity_id', 'key_addr2']].merge(
                c2_keys_c[c2_keys_c['key_addr2'].isin(valid_a2)][['entity_id', 'key_addr2']],
                on='key_addr2'
            ).rename(columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'})
            
            df_addr = pd.concat([df_a1[['s1_id', 'cand_id']], df_a2[['s1_id', 'cand_id']]]).drop_duplicates()
            df_addr = df_addr.groupby('s1_id').head(MAX_ADDR_CANDS).reset_index(drop=True)
            df_addr['ch_addr'] = 1
            
            # Skel
            valid_skel = get_valid_keys(c2_keys_c, "key_skel")
            df_skel = s1_keys_c[['entity_id', 'key_skel']].merge(
                c2_keys_c[c2_keys_c['key_skel'].isin(valid_skel)][['entity_id', 'key_skel']],
                on='key_skel'
            ).rename(columns={'entity_id_x': 's1_id', 'entity_id_y': 'cand_id'})
            df_skel = df_skel[['s1_id', 'cand_id']].drop_duplicates()
            df_skel = df_skel.groupby('s1_id').head(MAX_SKEL_CANDS).reset_index(drop=True)
            df_skel['ch_skel'] = 1
            
            # --- Union ---
            df_all = pd.concat([df_emb, df_addr, df_skel], ignore_index=True)
            if df_all.empty:
                continue
                
            df_all = df_all.groupby(['s1_id', 'cand_id']).agg({
                'ch_emb': 'max', 'ch_addr': 'max', 'ch_skel': 'max', 
                'emb_score': 'max', 'emb_rank': 'min'
            }).reset_index()
            
            df_all[['ch_emb', 'ch_addr', 'ch_skel']] = df_all[['ch_emb', 'ch_addr', 'ch_skel']].fillna(0).astype(int)
            
            # Compute missing scores
            missing_mask = df_all['emb_score'].isna()
            if missing_mask.any():
                missing_s1 = df_all.loc[missing_mask, 's1_id'].values
                missing_c2 = df_all.loc[missing_mask, 'cand_id'].values
                scores = get_scores_vectorized(missing_s1, missing_c2, s1_main_df, s1_alt_df, c2_main_df, c2_alt_df)
                df_all.loc[missing_mask, 'emb_score'] = scores
                
            # Cap
            df_all = df_all.sort_values(['s1_id', 'emb_score'], ascending=[True, False])
            df_all = df_all.groupby('s1_id').head(CAND_CAP).reset_index(drop=True)
            
            df_all['cand_source'] = 0 if cand_source == "source2" else 1
            dfs_c_source.append(df_all)
            
        if dfs_c_source:
            df_source_full = pd.concat(dfs_c_source, ignore_index=True)
            all_cands_dfs.append(df_source_full)
            
            # Recall report for this source
            if args.split == "train":
                gt_path = os.path.join(CACHE_DIR, "gt_long.parquet")
                if os.path.exists(gt_path):
                    gt = pd.read_parquet(gt_path)
                    run_recall_report(df_source_full, gt, "source2" if cand_source == "source2" else "source3", s1_country_map)

    if not all_cands_dfs:
        df_cands = pd.DataFrame(columns=['s1_id', 'cand_id', 'cand_source', 'emb_score', 'emb_rank', 'ch_emb', 'ch_addr', 'ch_skel'])
    else:
        df_cands = pd.concat(all_cands_dfs, ignore_index=True)
        
    end_time = time.time()
    elapsed = end_time - start_time
    time_per_10k = (elapsed / num_s1_processed) * 10000 if num_s1_processed > 0 else 0
    time_17L = (elapsed / num_s1_processed) * 1700000 if num_s1_processed > 0 else 0
    
    print(f"\n--- Performance ---")
    print(f"Time per 10,000 S1: {time_per_10k:.2f} seconds")
    print(f"Estimate for 17 Lakh S1 (1.7M): {time_17L / 60:.2f} minutes")
    
    if args.split == "train":
        gt_path = os.path.join(CACHE_DIR, "gt_long.parquet")
        if os.path.exists(gt_path):
            gt = pd.read_parquet(gt_path)
            print("\n--- Overall Recall Report ---")
            run_recall_report(df_cands, gt, cand_source=None, country_map=s1_country_map)
    
    out_path = os.path.join(CACHE_DIR, f"cands_{args.split}.parquet")
    df_cands.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df_cands)} total candidates to {out_path}")

if __name__ == "__main__":
    main()

import os
import sys
import gc
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import CACHE_DIR

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    return parser.parse_args()

def load_embeddings(split, source):
    """Loads main and alt embeddings, returns (main_emb, main_ids, alt_dict)"""
    emb_path = os.path.join(CACHE_DIR, f"emb_{split}_{source}.npy")
    ids_path = os.path.join(CACHE_DIR, f"ids_{split}_{source}.npy")
    
    if not os.path.exists(emb_path):
        return None, None, {}
        
    main_emb = np.load(emb_path)
    main_ids = np.load(ids_path, allow_pickle=True)
    
    alt_emb_path = os.path.join(CACHE_DIR, f"emb_{split}_{source}_alt.npy")
    alt_ids_path = os.path.join(CACHE_DIR, f"ids_{split}_{source}_alt.npy")
    
    alt_dict = {}
    if os.path.exists(alt_emb_path):
        alt_emb = np.load(alt_emb_path)
        alt_ids = np.load(alt_ids_path, allow_pickle=True)
        for i, eid in enumerate(alt_ids):
            alt_dict[eid] = alt_emb[i]
            
    return main_emb, main_ids, alt_dict

def get_country_map(split, source):
    path = os.path.join(CACHE_DIR, f"norm_{split}_{source}.parquet")
    if not os.path.exists(path):
        return {}
    df = pd.read_parquet(path, columns=["entity_id", "country"])
    return dict(zip(df["entity_id"], df["country"]))

def topk_search(q_main, q_alt, x_emb, x_ids, k, chunk=20000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(x_emb).to(device)
    
    all_res = []
    
    for i in range(0, len(q_main), chunk):
        Q_m = torch.from_numpy(q_main[i:i+chunk]).to(device)
        Q_a = torch.from_numpy(q_alt[i:i+chunk]).to(device)
        
        S_m = Q_m @ X.T
        S_a = Q_a @ X.T
        S = torch.maximum(S_m, S_a)
        
        s, idx = torch.topk(S, k=min(2 * k, X.shape[0]), dim=1)
        s = s.float().cpu().numpy()
        idx = idx.cpu().numpy()
        
        for q_idx in range(len(s)):
            row_s = s[q_idx]
            row_i = idx[q_idx]
            
            seen = set()
            uniq_ids = []
            uniq_s = []
            
            for score, x_idx in zip(row_s, row_i):
                cid = x_ids[x_idx]
                if cid not in seen:
                    seen.add(cid)
                    uniq_ids.append(cid)
                    uniq_s.append(score)
                    if len(uniq_ids) == k:
                        break
            all_res.append((uniq_ids, uniq_s))
            
    return all_res

def load_keys(split, source):
    path = os.path.join(CACHE_DIR, f"norm_{split}_{source}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = ["entity_id", "country", "house_no", "street_token", "zip_pin", "name_skel"]
    df = pd.read_parquet(path, columns=cols)
    
    def sort_skel(x):
        if not isinstance(x, str): return ""
        return " ".join(sorted(x.split()))
        
    df["skel_sorted"] = df["name_skel"].apply(sort_skel)
    
    df["key_addr1"] = df.apply(lambda x: f"{x['country']}_{x['house_no']}_{x['street_token']}" if x['house_no'] and x['street_token'] else None, axis=1)
    df["key_addr2"] = df.apply(lambda x: f"{x['country']}_{x['zip_pin']}_{x['house_no']}" if x['zip_pin'] and x['house_no'] else None, axis=1)
    df["key_skel"] = df.apply(lambda x: f"{x['country']}_{x['skel_sorted']}" if x['skel_sorted'] else None, axis=1)
    
    return df

def build_hash_index(df, key_col):
    df_valid = df[df[key_col].notnull()]
    counts = df_valid[key_col].value_counts()
    valid_keys = counts[counts <= 50].index
    df_filtered = df_valid[df_valid[key_col].isin(valid_keys)]
    return df_filtered.groupby(key_col)["entity_id"].apply(list).to_dict()

def score_pairs(s1_ids, cand_ids, emb1_main, emb1_alt_dict, emb2_main_dict, emb2_alt_dict):
    scores = []
    for s1, c2 in zip(s1_ids, cand_ids):
        v1_m = emb1_main.get(s1)
        v1_a = emb1_alt_dict.get(s1, v1_m)
        
        v2_m = emb2_main_dict.get(c2)
        v2_a = emb2_alt_dict.get(c2, v2_m)
        
        if v1_m is None or v2_m is None:
            scores.append(0.0)
            continue
            
        s11 = np.dot(v1_m, v2_m)
        s12 = np.dot(v1_m, v2_a)
        s21 = np.dot(v1_a, v2_m)
        s22 = np.dot(v1_a, v2_a)
        scores.append(float(max(s11, s12, s21, s22)))
    return scores

def main():
    args = get_args()
    k_emb = 15
    max_cands = 40
    
    print("Loading S1 embeddings...")
    s1_main, s1_ids, s1_alt_dict = load_embeddings(args.split, "source1")
    if s1_main is None:
        print("S1 embeddings not found.")
        return
        
    s1_country_map = get_country_map(args.split, "source1")
    countries = list(set(s1_country_map.values()))
    
    s1_queries = {}
    for c in countries:
        mask = np.array([s1_country_map.get(eid) == c for eid in s1_ids])
        if not mask.any(): continue
        c_ids = s1_ids[mask]
        c_main = s1_main[mask]
        c_alt = np.array([s1_alt_dict.get(eid, c_main[i]) for i, eid in enumerate(c_ids)])
        s1_queries[c] = {"ids": c_ids, "main": c_main, "alt": c_alt}
        
    del s1_main
    gc.collect()
    
    s1_main_dict = {s1_queries[c]["ids"][i]: s1_queries[c]["main"][i] for c in s1_queries for i in range(len(s1_queries[c]["ids"]))}
    
    all_cands = []
    
    for cand_source in ["source2", "source3"]:
        print(f"\n--- Processing {cand_source} ---")
        c2_main, c2_ids, c2_alt_dict = load_embeddings(args.split, cand_source)
        if c2_main is None: continue
        c2_country_map = get_country_map(args.split, cand_source)
        
        c2_main_dict = {eid: c2_main[i] for i, eid in enumerate(c2_ids)}
        
        c2_keys = load_keys(args.split, cand_source)
        idx_addr1 = build_hash_index(c2_keys, "key_addr1")
        idx_addr2 = build_hash_index(c2_keys, "key_addr2")
        idx_skel = build_hash_index(c2_keys, "key_skel")
        del c2_keys
        gc.collect()
        
        s1_keys = load_keys(args.split, "source1")
        
        for c in countries:
            if c not in s1_queries: continue
            
            c_mask = np.array([c2_country_map.get(eid) == c for eid in c2_ids])
            if not c_mask.any(): continue
            
            x_main_c = c2_main[c_mask]
            x_ids_c = c2_ids[c_mask]
            
            x_alt_c = []
            x_alt_ids_c = []
            for eid in x_ids_c:
                if eid in c2_alt_dict:
                    x_alt_c.append(c2_alt_dict[eid])
                    x_alt_ids_c.append(eid)
            
            if x_alt_c:
                X_full = np.vstack([x_main_c, np.array(x_alt_c)])
                X_ids_full = np.concatenate([x_ids_c, x_alt_ids_c])
            else:
                X_full = x_main_c
                X_ids_full = x_ids_c
                
            q_ids = s1_queries[c]["ids"]
            q_main = s1_queries[c]["main"]
            q_alt = s1_queries[c]["alt"]
            
            print(f"[{c}] Channel A search: {len(q_ids)} S1 vs {len(X_ids_full)} C2")
            res = topk_search(q_main, q_alt, X_full, X_ids_full, k=k_emb)
            
            s1_keys_c = s1_keys[s1_keys["country"] == c]
            s1_keys_dict = s1_keys_c.set_index("entity_id").to_dict("index")
            
            for i, sid in enumerate(q_ids):
                c_dict = {}
                
                emb_cids, emb_scores = res[i]
                for r, (cid, sc) in enumerate(zip(emb_cids, emb_scores)):
                    c_dict[cid] = {"emb_score": float(sc), "emb_rank": r+1, "ch_emb": 1, "ch_addr": 0, "ch_skel": 0}
                    
                s_info = s1_keys_dict.get(sid, {})
                k_a1 = s_info.get("key_addr1")
                k_a2 = s_info.get("key_addr2")
                k_s = s_info.get("key_skel")
                
                addr_cands = []
                if k_a1 in idx_addr1: addr_cands.extend(idx_addr1[k_a1])
                if k_a2 in idx_addr2: addr_cands.extend(idx_addr2[k_a2])
                addr_cands = list(set(addr_cands))[:20]
                
                for cid in addr_cands:
                    if cid not in c_dict:
                        c_dict[cid] = {"emb_score": None, "emb_rank": 999, "ch_emb": 0, "ch_addr": 1, "ch_skel": 0}
                    else:
                        c_dict[cid]["ch_addr"] = 1
                        
                skel_cands = []
                if k_s in idx_skel: skel_cands.extend(idx_skel[k_s])
                skel_cands = list(set(skel_cands))[:20]
                
                for cid in skel_cands:
                    if cid not in c_dict:
                        c_dict[cid] = {"emb_score": None, "emb_rank": 999, "ch_emb": 0, "ch_addr": 0, "ch_skel": 1}
                    else:
                        c_dict[cid]["ch_skel"] = 1
                
                missing_score_cids = [cid for cid, v in c_dict.items() if v["emb_score"] is None]
                if missing_score_cids:
                    m_scores = score_pairs([sid]*len(missing_score_cids), missing_score_cids, s1_main_dict, s1_alt_dict, c2_main_dict, c2_alt_dict)
                    for cid, sc in zip(missing_score_cids, m_scores):
                        c_dict[cid]["emb_score"] = sc
                        
                sorted_cands = sorted(c_dict.items(), key=lambda x: x[1]["emb_score"], reverse=True)[:max_cands]
                
                for cid, info in sorted_cands:
                    all_cands.append({
                        "s1_id": sid,
                        "cand_id": cid,
                        "cand_source": 0 if cand_source == "source2" else 1,
                        "emb_score": info["emb_score"],
                        "emb_rank": info["emb_rank"],
                        "ch_emb": info["ch_emb"],
                        "ch_addr": info["ch_addr"],
                        "ch_skel": info["ch_skel"]
                    })
                    
        del c2_main, c2_main_dict, idx_addr1, idx_addr2, idx_skel
        gc.collect()

    print("\nSaving candidates...")
    df_cands = pd.DataFrame(all_cands)
    if df_cands.empty:
        df_cands = pd.DataFrame(columns=['s1_id', 'cand_id', 'cand_source', 'emb_score', 'emb_rank', 'ch_emb', 'ch_addr', 'ch_skel'])
    
    if args.split == "train":
        gt_path = os.path.join(CACHE_DIR, "gt_long.parquet")
        if os.path.exists(gt_path):
            gt = pd.read_parquet(gt_path)
            gt["label"] = 1
            gt = gt.rename(columns={"match_id": "cand_id"})
            df_cands = df_cands.merge(gt[["s1_id", "cand_id", "label"]], on=["s1_id", "cand_id"], how="left")
            df_cands["label"] = df_cands["label"].fillna(0).astype(int)
            
            total_gt = len(gt[gt["s1_id"].isin(df_cands["s1_id"].unique())])
            found_gt = df_cands["label"].sum()
            print(f"Recall: {found_gt} / {total_gt} ({found_gt/max(1, total_gt):.4f})")
            n_s1 = df_cands['s1_id'].nunique()
            print(f"Avg candidates per S1: {len(df_cands) / n_s1:.2f}" if n_s1 > 0 else "Avg candidates per S1: 0.00")
    
    out_path = os.path.join(CACHE_DIR, f"cands_{args.split}.parquet")
    df_cands.to_parquet(out_path, index=False)
    print(f"Saved {len(df_cands)} candidates to {out_path}")

if __name__ == "__main__":
    main()

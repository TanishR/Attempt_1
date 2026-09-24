import os
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import train_test_split
import sys
sys.path.append('.')
from config import CACHE_DIR, SEED, SAMPLE

def create_split():
    s1_path = os.path.join(CACHE_DIR, "raw_train_source1.parquet")
    if not os.path.exists(s1_path): return
    s1_df = pd.read_parquet(s1_path, columns=["entity_id"])
    
    train_ids, val_ids = train_test_split(s1_df["entity_id"], test_size=0.2, random_state=SEED)
    
    train_ids = train_ids.sample(n=min(300000, len(train_ids)), random_state=SEED, replace=False)
    val_ids = val_ids.sample(n=min(100000, len(val_ids)), random_state=SEED, replace=False)
    
    split_df = pd.concat([
        pd.DataFrame({"s1_id": train_ids.values, "fold": "train"}),
        pd.DataFrame({"s1_id": val_ids.values, "fold": "val"})
    ])
    
    split_df.to_parquet(os.path.join(CACHE_DIR, "split.parquet"), index=False)
    print(f"Created split.parquet: {len(train_ids)} train, {len(val_ids)} val")

def get_sample(split="train"):
    if SAMPLE is None or str(SAMPLE).lower() == "none":
        print("SAMPLE is None. Using full data.")
        return
        
    print(f"Sampling {SAMPLE} S1s from {split} fold...")
    split_df = pd.read_parquet(os.path.join(CACHE_DIR, "split.parquet"))
    gt_long = pd.read_parquet(os.path.join(CACHE_DIR, "gt_long.parquet"))
    
    sampled_s1 = split_df[split_df["fold"] == split].head(int(SAMPLE))["s1_id"].values
    
    matched_s23 = gt_long[gt_long["s1_id"].isin(sampled_s1)]["match_id"].unique()
    
    s1_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source1.parquet"), columns=["entity_id", "country"])
    s1_countries = set(s1_df[s1_df["entity_id"].isin(sampled_s1)]["country"].unique())
    del s1_df
    
    s2_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source2.parquet"), columns=["entity_id", "country"]) if os.path.exists(os.path.join(CACHE_DIR, "raw_train_source2.parquet")) else pd.DataFrame()
    s3_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source3.parquet"), columns=["entity_id", "country"]) if os.path.exists(os.path.join(CACHE_DIR, "raw_train_source3.parquet")) else pd.DataFrame()
    s23_all = pd.concat([s2_df, s3_df])
    del s2_df, s3_df
    
    distractors = s23_all[s23_all["country"].isin(s1_countries) & (~s23_all["entity_id"].isin(matched_s23))]
    n_distractors = min(len(distractors), int(SAMPLE) * 5)
    distractors_sampled = distractors.sample(n_distractors, random_state=SEED)["entity_id"].values
    
    final_s23 = np.concatenate([matched_s23, distractors_sampled])
    
    print(f"Sampled {len(sampled_s1)} S1, {len(matched_s23)} matches, {len(distractors_sampled)} distractors")
    
    for src in ["train_source1", "train_source2", "train_source3", "train_ground_truth"]:
        path = os.path.join(CACHE_DIR, f"raw_{src}.parquet")
        if not os.path.exists(path): continue
        df = pd.read_parquet(path)
        if "source1" in src:
            df = df[df["entity_id"].isin(sampled_s1)]
        elif "source2" in src or "source3" in src:
            df = df[df["entity_id"].isin(final_s23)]
        elif "ground_truth" in src:
            df = df[df["source1_entity_id"].isin(sampled_s1)]
            
        df.to_parquet(os.path.join(CACHE_DIR, f"sample_{src}.parquet"), index=False)
        del df
        gc.collect()

if __name__ == "__main__":
    create_split()
    get_sample("train")

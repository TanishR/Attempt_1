import os
import sys
import gc
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import CACHE_DIR, SEED, SAMPLE_BLOCK_TEST
from s1_normalize import process_chunk
from s2_embed import embed_texts, encode_uniques_and_save, has_non_ascii

def normalize_in_parallel(df, desc="data"):
    """
    Normalizes a DataFrame of entities using chunked multiprocessing.
    Returns: pandas.DataFrame with all normalized and extracted feature columns.
    """
    import multiprocessing as mp
    if "source1_entity_id" in df.columns:
        df["entity_id"] = df["source1_entity_id"]
    print(f"Normalizing {len(df)} rows for {desc}...")
    num_cores = max(1, mp.cpu_count() - 1)
    chunk_size = max(1, len(df) // (num_cores * 2) + 1)
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    with mp.Pool(num_cores) as pool:
        result_chunks = pool.map(process_chunk, chunks)
    res_df = pd.concat(result_chunks, ignore_index=True)
    return res_df

def main():
    """
    Builds the isolated laptop_test candidate dataset (5k train S1, 2k val S1, GT matches, 2x distractors)
    and computes embeddings for S1, S2, and S3 under cache/laptop_test/.
    """
    laptop_dir = os.path.join(CACHE_DIR, "laptop_test")
    os.makedirs(laptop_dir, exist_ok=True)
    print(f"Target directory: {laptop_dir}")

    # 1. Select 5000 train-fold + 2000 val-fold S1
    split_df = pd.read_parquet(os.path.join(CACHE_DIR, "split.parquet"))
    train_5k = split_df[split_df['fold'] == 'train'].head(SAMPLE_BLOCK_TEST)
    val_2k = split_df[split_df['fold'] == 'val'].head(2000)
    split_laptop = pd.concat([train_5k, val_2k], ignore_index=True)
    split_path = os.path.join(laptop_dir, "split.parquet")
    split_laptop.to_parquet(split_path, index=False)
    print(f"Saved {split_path}: {len(train_5k)} train + {len(val_2k)} val = {len(split_laptop)} total S1")

    s1_ids = set(split_laptop['s1_id'].values)

    # 2. Get GT matches
    gt_long = pd.read_parquet(os.path.join(CACHE_DIR, "gt_long.parquet"))
    gt_laptop = gt_long[gt_long['s1_id'].isin(s1_ids)].copy()
    gt_path = os.path.join(laptop_dir, "gt_long.parquet")
    gt_laptop.to_parquet(gt_path, index=False)
    print(f"Saved {gt_path}: {len(gt_laptop)} GT match rows")

    s2_matches = set(gt_laptop[gt_laptop['match_source'].str.upper().isin(['S2', 'SOURCE2'])]['match_id'].unique())
    s3_matches = set(gt_laptop[gt_laptop['match_source'].str.upper().isin(['S3', 'SOURCE3'])]['match_id'].unique())
    print(f"Unique GT matches: {len(s2_matches)} in S2, {len(s3_matches)} in S3")


    # 3. Load S1 raw records
    raw_s1 = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source1.parquet"))
    s1_sub = raw_s1[raw_s1['entity_id'].isin(s1_ids)].copy()
    del raw_s1
    gc.collect()

    country_counts = s1_sub['country'].value_counts()
    country_props = (country_counts / len(s1_sub)).to_dict()
    print(f"S1 Country distribution: {country_counts.to_dict()}")

    # 4. Sample S2 matches + 2x distractors
    raw_s2 = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source2.parquet"))
    s2_matched_df = raw_s2[raw_s2['entity_id'].isin(s2_matches)].copy()
    s2_distractor_pool = raw_s2[(~raw_s2['entity_id'].isin(s2_matches)) & (raw_s2['country'].isin(country_props.keys()))]
    del raw_s2
    gc.collect()

    n_s2_distractors = 2 * len(s2_matches)
    s2_dist_samples = []
    for c, prop in country_props.items():
        pool_c = s2_distractor_pool[s2_distractor_pool['country'] == c]
        n_c = min(len(pool_c), int(round(n_s2_distractors * prop)))
        if n_c > 0:
            s2_dist_samples.append(pool_c.sample(n=n_c, random_state=SEED))
    s2_dist_df = pd.concat(s2_dist_samples, ignore_index=True) if s2_dist_samples else pd.DataFrame()
    s2_sub = pd.concat([s2_matched_df, s2_dist_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"S2 selected: {len(s2_matched_df)} matches + {len(s2_dist_df)} distractors = {len(s2_sub)} total S2")
    del s2_matched_df, s2_dist_df, s2_distractor_pool
    gc.collect()

    # 5. Sample S3 matches + 2x distractors
    raw_s3 = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source3.parquet"))
    s3_matched_df = raw_s3[raw_s3['entity_id'].isin(s3_matches)].copy()
    s3_distractor_pool = raw_s3[(~raw_s3['entity_id'].isin(s3_matches)) & (raw_s3['country'].isin(country_props.keys()))]
    del raw_s3
    gc.collect()

    n_s3_distractors = 2 * len(s3_matches)
    s3_dist_samples = []
    for c, prop in country_props.items():
        pool_c = s3_distractor_pool[s3_distractor_pool['country'] == c]
        n_c = min(len(pool_c), int(round(n_s3_distractors * prop)))
        if n_c > 0:
            s3_dist_samples.append(pool_c.sample(n=n_c, random_state=SEED))
    s3_dist_df = pd.concat(s3_dist_samples, ignore_index=True) if s3_dist_samples else pd.DataFrame()
    s3_sub = pd.concat([s3_matched_df, s3_dist_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    print(f"S3 selected: {len(s3_matched_df)} matches + {len(s3_dist_df)} distractors = {len(s3_sub)} total S3")
    del s3_matched_df, s3_dist_df, s3_distractor_pool
    gc.collect()

    # 6. Normalize records and save
    norm_s1 = normalize_in_parallel(s1_sub, "Source1")
    norm_s1.to_parquet(os.path.join(laptop_dir, "norm_train_source1.parquet"), index=False)

    norm_s2 = normalize_in_parallel(s2_sub, "Source2")
    norm_s2.to_parquet(os.path.join(laptop_dir, "norm_train_source2.parquet"), index=False)

    norm_s3 = normalize_in_parallel(s3_sub, "Source3")
    norm_s3.to_parquet(os.path.join(laptop_dir, "norm_train_source3.parquet"), index=False)
    print("All normalized Parquet files written to cache/laptop_test/")

    # 7. Embed using Qwen3 model only if missing
    all_embs_exist = all(
        os.path.exists(os.path.join(laptop_dir, f"emb_train_{src}.npy")) and
        os.path.exists(os.path.join(laptop_dir, f"emb_train_{src}_alt.npy"))
        for src in ["source1", "source2", "source3"]
    )
    if all_embs_exist:
        print("\nAll embeddings in cache/laptop_test already exist, skipping re-embedding!")
    else:
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            torch_dtype = torch.float32
        else:
            device = "cpu"
            torch_dtype = torch.float32

        print(f"\nLoading Qwen3-Embedding-0.6B on {device} ({torch_dtype})...")
        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            device=device,
            model_kwargs={"torch_dtype": torch_dtype},
            truncate_dim=256,
        )
        model.max_seq_length = 48

        for src_name, norm_df in [("source1", norm_s1), ("source2", norm_s2), ("source3", norm_s3)]:
            print(f"\n=== Embedding {src_name} ({len(norm_df)} rows) ===")
            raw_names = norm_df['raw_name'].fillna("").astype(str).values
            name_fulls = norm_df['name_full'].fillna("").astype(str).values
            entity_ids = norm_df['entity_id'].values

            is_non_ascii = np.array([has_non_ascii(t) for t in raw_names])
            main_texts = np.where(is_non_ascii, raw_names, name_fulls)
            alt_texts = name_fulls[is_non_ascii]
            alt_ids = entity_ids[is_non_ascii]

            tag_prefix = f"train_{src_name}"
            emb_prefix_main = os.path.join(laptop_dir, f"emb_{tag_prefix}")
            encode_uniques_and_save(model, main_texts, entity_ids, emb_prefix_main, f"{tag_prefix}_main", batch_size=512)

            emb_prefix_alt = os.path.join(laptop_dir, f"emb_{tag_prefix}_alt")
            encode_uniques_and_save(model, alt_texts, alt_ids, emb_prefix_alt, f"{tag_prefix}_alt", batch_size=512)


    print("\n[laptop_test] All embeddings and data successfully created!")

if __name__ == "__main__":
    main()

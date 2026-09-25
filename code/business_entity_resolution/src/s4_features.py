import os
import sys
import time
import argparse
from collections import Counter
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

import config

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


def _id_to_int(series: pd.Series) -> pd.Series:
    """
    Converts 'S1-12345', 'S2-12345', 'S3-12345' style IDs to int64.
    Source prefix is encoded as the top 2 decimal digits: S1->10^12, S2->2*10^12, S3->3*10^12.
    Falls back to pandas Categorical codes if the format is unexpected.
    """
    s = series.astype(str)
    # Determine prefix multiplier from first char after 'S'
    prefix = s.str[1].map({'1': 1_000_000_000_000, '2': 2_000_000_000_000, '3': 3_000_000_000_000})
    numeric = s.str.split('-', n=1).str[1].astype(np.int64)
    return (prefix.fillna(0).astype(np.int64) + numeric)


def parse_args():
    """
    Parses command line arguments for feature engineering.
    Returns: argparse.Namespace object.
    """
    parser = argparse.ArgumentParser(description="Step 6: Pairwise Feature Engineering")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--laptop-test", action="store_true", help="Use small laptop test pool in cache/laptop_test/")
    parser.add_argument("--cache-dir", type=str, default=None, help="Custom cache directory path")
    parser.add_argument("--force", action="store_true", help="Overwrite existing feature parquet files")
    return parser.parse_args()

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

def extract_rare3_set(addr, country, df_tokens):
    """
    Extracts up to 3 rarest address tokens (len >= 5) based on country document frequency.
    Returns: frozenset of rarest token strings.
    """
    if not isinstance(addr, str) or not addr.strip():
        return frozenset()
    toks = list(set(t for t in addr.split() if len(t) >= 5))
    if not toks:
        return frozenset()
    ctr = df_tokens.get(country, {})
    toks.sort(key=lambda t: (ctr.get(t, 0), t))
    return frozenset(toks[:3])

def load_norm_table(cache_dir, split, source):
    """
    Loads normalized table for a source with fallback paths across train and test splits.
    Returns: pandas.DataFrame indexed by entity_id.
    """
    paths_to_try = [
        os.path.join(cache_dir, f"norm_{split}_{source}.parquet"),
        os.path.join(cache_dir, f"norm_train_{source}.parquet"),
        os.path.join(cache_dir, f"norm_{source}.parquet"),
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            df = pd.read_parquet(p)
            return df.set_index("entity_id")
    raise FileNotFoundError(f"Could not find normalized table for {source} in {cache_dir}")

def load_candidate_embeddings(cache_dir, split):
    """
    Loads candidate main embeddings for Source 2 and Source 3 into a stacked float32 matrix.
    Returns: tuple of (all_cand_emb, id_to_row_map).
    """
    s2_m_p = os.path.join(cache_dir, f"emb_{split}_source2.npy")
    if not os.path.exists(s2_m_p):
        s2_m_p = os.path.join(cache_dir, "emb_train_source2.npy")
    s2_id_p = os.path.join(cache_dir, f"ids_{split}_source2.npy")
    if not os.path.exists(s2_id_p):
        s2_id_p = os.path.join(cache_dir, "ids_train_source2.npy")

    s3_m_p = os.path.join(cache_dir, f"emb_{split}_source3.npy")
    if not os.path.exists(s3_m_p):
        s3_m_p = os.path.join(cache_dir, "emb_train_source3.npy")
    s3_id_p = os.path.join(cache_dir, f"ids_{split}_source3.npy")
    if not os.path.exists(s3_id_p):
        s3_id_p = os.path.join(cache_dir, "ids_train_source3.npy")

    s2_emb = np.load(s2_m_p)
    s2_ids = np.load(s2_id_p, allow_pickle=True)
    s3_emb = np.load(s3_m_p)
    s3_ids = np.load(s3_id_p, allow_pickle=True)

    all_cand_emb = np.vstack([s2_emb, s3_emb]).astype(np.float32)
    id_to_row = {cid: idx for idx, cid in enumerate(s2_ids)}
    n_s2 = len(s2_ids)
    for idx, cid in enumerate(s3_ids):
        id_to_row[cid] = n_s2 + idx

    return all_cand_emb, id_to_row

def compute_support_feature(chunk_df, all_cand_emb, id_to_row):
    """
    Computes max cosine similarity to other top-5 candidates (by emb_score) for each S1.
    Returns: numpy.ndarray of float32 support scores aligned with chunk_df rows.
    """
    n_rows = len(chunk_df)
    if n_rows == 0:
        return np.array([], dtype=np.float32)

    # Sort candidates per S1 by emb_score descending to guarantee top-5 order
    orig_indices = np.arange(n_rows)
    df_sorted = chunk_df[['s1_id', 'cand_id', 'emb_score']].assign(orig_idx=orig_indices).sort_values(
        ['s1_id', 'emb_score'], ascending=[True, False]
    )

    s1_sorted = df_sorted['s1_id'].values
    c_sorted = df_sorted['cand_id'].values
    sorted_orig_idx = df_sorted['orig_idx'].values

    c_rows = np.array([id_to_row.get(cid, 0) for cid in c_sorted], dtype=np.int32)
    cand_vecs = all_cand_emb[c_rows]

    unique_s1, start_indices, counts = np.unique(s1_sorted, return_index=True, return_counts=True)
    sorted_support = np.zeros(n_rows, dtype=np.float32)

    for start, count in zip(start_indices, counts):
        if count <= 1:
            continue
        k = min(count, 5)
        E = cand_vecs[start : start + count]
        T = E[:k]  # Exactly the top-5 candidates by emb_score
        M = np.dot(E, T.T)
        for i in range(k):
            M[i, i] = -999.0  # Exclude self-similarity
        sorted_support[start : start + count] = np.max(M, axis=1)

    support = np.zeros(n_rows, dtype=np.float32)
    support[sorted_orig_idx] = sorted_support
    return support

def compute_chunk_features(chunk_df, s1_df, cands_df, all_cand_emb, cand_id_map, global_reverse_rank):
    """
    Extracts all 31 features and rule_score for a candidate chunk without row-wise loops.
    Returns: pandas.DataFrame containing entity IDs, label, rule_score, and ordered config.FEATURES.
    """
    n_pairs = len(chunk_df)
    s1_ids = chunk_df['s1_id'].values
    cand_ids = chunk_df['cand_id'].values

    s1_sub = s1_df.loc[s1_ids]
    cand_sub = cands_df.loc[cand_ids]

    # 1. Embedding & Rank features
    emb_score = chunk_df['emb_score'].values.astype(np.float32)
    emb_rank = chunk_df['emb_rank'].values.astype(np.float32)

    # 2. Vectorized Name Similarities via RapidFuzz cpdist
    s1_nf = s1_sub['name_full'].tolist()
    c_nf = cand_sub['name_full'].tolist()
    s1_cn = s1_sub['core_name'].tolist()
    c_cn = cand_sub['core_name'].tolist()
    s1_sk = s1_sub['name_skel'].tolist()
    c_sk = cand_sub['name_skel'].tolist()
    s1_ad = s1_sub['addr_norm'].tolist()
    c_ad = cand_sub['addr_norm'].tolist()

    name_token_sort = process.cpdist(s1_nf, c_nf, scorer=fuzz.token_sort_ratio, workers=-1).astype(np.float32)
    name_token_set = process.cpdist(s1_nf, c_nf, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    core_ratio = process.cpdist(s1_cn, c_cn, scorer=fuzz.ratio, workers=-1).astype(np.float32)
    core_partial = process.cpdist(s1_cn, c_cn, scorer=fuzz.partial_ratio, workers=-1).astype(np.float32)

    skel_ratio = process.cpdist(s1_sk, c_sk, scorer=fuzz.ratio, workers=-1).astype(np.float32)
    empty_sk = (np.array(s1_sk) == '') | (np.array(c_sk) == '')
    skel_ratio[empty_sk] = np.nan

    # 3. name_jaccard over precomputed token frozensets
    s1_tsets = s1_sub['tok_set'].values
    c_tsets = cand_sub['tok_set'].values
    name_jaccard = np.array([
        len(a & b) / len(a | b) if (a or b) else 0.0
        for a, b in zip(s1_tsets, c_tsets)
    ], dtype=np.float32)

    # 4. legal_match: 1 if present and equal, 0 if present and different, -1 if missing
    s1_leg = s1_sub['legal'].values
    c_leg = cand_sub['legal'].values
    leg_miss = (s1_leg == '') | (c_leg == '')
    legal_match = np.where(leg_miss, -1, np.where(s1_leg == c_leg, 1, 0)).astype(np.float32)

    # 5. dba_max & aka_max
    dba_mask = (cand_sub['name_a'].values != '') | (s1_sub['name_a'].values != '')
    dba_max = np.full(n_pairs, np.nan, dtype=np.float32)
    if dba_mask.any():
        s_a = process.cpdist(s1_sub.loc[dba_mask, 'name_full'].tolist(), cand_sub.loc[dba_mask, 'name_a'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        s_b = process.cpdist(s1_sub.loc[dba_mask, 'name_full'].tolist(), cand_sub.loc[dba_mask, 'name_b'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        dba_max[dba_mask] = np.maximum(s_a, s_b)

    aka_mask = (cand_sub['name_aka_a'].values != '') | (s1_sub['name_aka_a'].values != '')
    aka_max = np.full(n_pairs, np.nan, dtype=np.float32)
    if aka_mask.any():
        s_aka_a = process.cpdist(s1_sub.loc[aka_mask, 'name_full'].tolist(), cand_sub.loc[aka_mask, 'name_aka_a'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        s_aka_b = process.cpdist(s1_sub.loc[aka_mask, 'name_full'].tolist(), cand_sub.loc[aka_mask, 'name_aka_b'].tolist(), scorer=fuzz.token_sort_ratio, workers=-1)
        aka_max[aka_mask] = np.maximum(s_aka_a, s_aka_b)

    # 6. len_diff on core_name
    len_diff = np.abs(s1_sub['core_len'].values - cand_sub['core_len'].values).astype(np.float32)

    # 7. addr_token_set
    addr_token_set = process.cpdist(s1_ad, c_ad, scorer=fuzz.token_set_ratio, workers=-1).astype(np.float32)
    empty_ad = (np.array(s1_ad) == '') | (np.array(c_ad) == '')
    addr_token_set[empty_ad] = np.nan

    # 8. house_match, zip_match, state_match: 1 same, 0 different, -1 missing
    s1_h = s1_sub['house_no'].values
    c_h = cand_sub['house_no'].values
    h_miss = (s1_h == '') | (c_h == '')
    house_match = np.where(h_miss, -1, np.where(s1_h == c_h, 1, 0)).astype(np.float32)

    s1_z = s1_sub['zip_pin'].values
    c_z = cand_sub['zip_pin'].values
    z_miss = (s1_z == '') | (c_z == '')
    zip_match = np.where(z_miss, -1, np.where(s1_z == c_z, 1, 0)).astype(np.float32)

    s1_st = s1_sub['state_code'].values
    c_st = cand_sub['state_code'].values
    st_miss = (s1_st == '') | (c_st == '')
    state_match = np.where(st_miss, -1, np.where(s1_st == c_st, 1, 0)).astype(np.float32)

    # 9. house_cand_match: 1 if any candidate matches, 0 if both exist but differ, -1 if missing
    s1_hc = s1_sub['hc_set'].values
    c_hc = cand_sub['hc_set'].values
    house_cand_match = np.array([
        -1 if (not a or not b) else (1 if bool(a & b) else 0)
        for a, b in zip(s1_hc, c_hc)
    ], dtype=np.float32)

    # 10. num_jaccard: Jaccard on numeric tokens (NaN if either side has no numbers)
    s1_num = s1_sub['num_set'].values
    c_num = cand_sub['num_set'].values
    num_jaccard = np.array([
        np.nan if (not a or not b) else (len(a & b) / len(a | b))
        for a, b in zip(s1_num, c_num)
    ], dtype=np.float32)

    # 11. rare_tok_overlap: count of shared tokens among each side's 3 rarest address tokens
    s1_r3 = s1_sub['rare3_set'].values
    c_r3 = cand_sub['rare3_set'].values
    rare_tok_overlap = np.array([
        len(a & b) for a, b in zip(s1_r3, c_r3)
    ], dtype=np.float32)

    # 12. addr_missing_any: 1 if address empty on either side, else 0
    addr_missing_any = ((np.array(s1_ad) == '') | (np.array(c_ad) == '')).astype(np.float32)

    # 13. Channel flags and cand_source
    cand_source = chunk_df['cand_source'].values.astype(np.float32)
    ch_emb = chunk_df['ch_emb'].values.astype(np.float32)
    ch_addr = chunk_df['ch_addr'].values.astype(np.float32)
    ch_skel = chunk_df['ch_skel'].values.astype(np.float32)
    ch_rare = chunk_df.get('ch_rare', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    ch_rev = chunk_df.get('ch_rev', pd.Series(0, index=chunk_df.index)).values.astype(np.float32)
    n_channels = (ch_emb + ch_addr + ch_skel + ch_rare + ch_rev).astype(np.float32)

    # 14. Context features: gap_to_best and n_cands per S1
    s1_best_score = chunk_df.groupby('s1_id')['emb_score'].transform('max').values.astype(np.float32)
    gap_to_best = s1_best_score - emb_score
    n_cands = chunk_df.groupby('s1_id')['cand_id'].transform('count').values.astype(np.float32)

    # 15. reverse_rank (passed from global split calculation)
    reverse_rank = global_reverse_rank.astype(np.float32)

    # 16. support
    support = compute_support_feature(chunk_df, all_cand_emb, cand_id_map)

    # 17. rule_score: 0.5*max(name_token_sort, core_ratio) + 0.3*addr_token_set + 20*(house_match==1)
    rule_score = (
        0.5 * np.maximum(name_token_sort, core_ratio)
        + 0.3 * np.nan_to_num(addr_token_set, nan=0.0)
        + 20.0 * (house_match == 1.0)
    ).astype(np.float32)

    feats_dict = {
        's1_id': s1_ids,
        'cand_id': cand_ids,
    }
    if 'label' in chunk_df.columns:
        feats_dict['label'] = chunk_df['label'].values.astype(np.int32)
    if 'is_competitor' in chunk_df.columns:
        feats_dict['is_competitor'] = chunk_df['is_competitor'].values.astype(np.int8)
    else:
        feats_dict['is_competitor'] = np.zeros(len(s1_ids), dtype=np.int8)

    feats_dict['rule_score'] = rule_score

    # Add features strictly in order of config.FEATURES
    computed_map = {
        'emb_score': emb_score, 'emb_rank': emb_rank, 'name_token_sort': name_token_sort,
        'name_token_set': name_token_set, 'core_ratio': core_ratio, 'core_partial': core_partial,
        'skel_ratio': skel_ratio, 'name_jaccard': name_jaccard, 'legal_match': legal_match,
        'dba_max': dba_max, 'aka_max': aka_max, 'len_diff': len_diff,
        'addr_token_set': addr_token_set, 'house_match': house_match, 'house_cand_match': house_cand_match,
        'num_jaccard': num_jaccard, 'rare_tok_overlap': rare_tok_overlap, 'zip_match': zip_match,
        'state_match': state_match, 'addr_missing_any': addr_missing_any, 'cand_source': cand_source,
        'ch_emb': ch_emb, 'ch_addr': ch_addr, 'ch_skel': ch_skel, 'ch_rare': ch_rare, 'ch_rev': ch_rev,
        'n_channels': n_channels, 'gap_to_best': gap_to_best, 'n_cands': n_cands,
        'reverse_rank': reverse_rank, 'support': support
    }

    for col in config.FEATURES:
        feats_dict[col] = computed_map[col]

    return pd.DataFrame(feats_dict)

def compute_global_reverse_ranks(cache_dir, split, chunk_files, sampled_s1_ids=None, val_s1_ids=None):
    """
    Computes global reverse rank per cand_id across all candidates of the split.
    Reads only (s1_id, cand_id, emb_score) per chunk with integer IDs to keep peak RAM
    well below 15 GB even for 9-crore-row train candidate tables.

    Returns: (list of numpy arrays containing reverse_rank for each chunk, set of competitor_s1_ids).
    """
    print("Computing global reverse_rank across all candidate chunks...", flush=True)
    t0 = time.time()

    # --- Pass 1: read 3 columns only, convert IDs to int64, track chunk lengths ---
    int_chunk_dfs = []
    chunk_lens = []
    for ci, cf in enumerate(chunk_files):
        df = pd.read_parquet(cf, columns=['s1_id', 'cand_id', 'emb_score'])
        chunk_lens.append(len(df))
        # Convert string IDs to int64 (strips 'S1-' etc.) — ~4× smaller than object dtype
        df['s1_int'] = _id_to_int(df['s1_id'])
        df['cid_int'] = _id_to_int(df['cand_id'])
        df = df.drop(columns=['s1_id', 'cand_id'])
        int_chunk_dfs.append(df)
        rss = _rss_mb()
        print(f"  [reverse_rank] Read chunk {ci + 1}/{len(chunk_files)}: {len(df):,} rows  "
              f"(RSS {rss:.0f} MB)", flush=True)

    # --- Concatenate int64 frames and rank globally ---
    print(f"  Concatenating {len(int_chunk_dfs)} int64 frames ({sum(chunk_lens):,} rows)...", flush=True)
    all_int = pd.concat(int_chunk_dfs, ignore_index=True)
    del int_chunk_dfs
    import gc; gc.collect()
    rss = _rss_mb()
    print(f"  All pairs loaded: {len(all_int):,} rows  (RSS {rss:.0f} MB)", flush=True)

    all_int['reverse_rank'] = (
        all_int.groupby('cid_int')['emb_score']
        .rank(ascending=False, method='min')
        .astype(np.float32)
    )
    print(f"  reverse_rank computed in {time.time() - t0:.2f}s  (RSS {_rss_mb():.0f} MB)", flush=True)

    # --- Slice back per chunk ---
    all_rr = all_int['reverse_rank'].values
    chunk_rr_list = []
    offset = 0
    for clen in chunk_lens:
        chunk_rr_list.append(all_rr[offset: offset + clen].copy())
        offset += clen

    # --- Identify competitor S1 ids (still using int64 -> original string mapping not needed) ---
    competitor_s1_ids = set()
    if val_s1_ids is not None and len(val_s1_ids) > 0:
        # Map val string IDs to int64
        val_int_ids = set(_id_to_int(pd.Series(list(val_s1_ids))).values)
        val_cid_mask = all_int['s1_int'].isin(val_int_ids)
        val_cid_ints = set(all_int.loc[val_cid_mask, 'cid_int'].values)
        if val_cid_ints:
            sharing_s1_ints = set(all_int.loc[all_int['cid_int'].isin(val_cid_ints), 's1_int'].values)
            # Convert sampled_s1_ids to int64 set for exclusion
            sampled_int = set(_id_to_int(pd.Series(list(sampled_s1_ids))).values) if sampled_s1_ids else set()
            comp_int = sharing_s1_ints - sampled_int
            # Map competitor int64 IDs back to original string IDs via the original chunk files
            # We need to find the string IDs that map to comp_int
            comp_s_prefix = {1_000_000_000_000: 'S1', 2_000_000_000_000: 'S2', 3_000_000_000_000: 'S3'}
            comp_strings = set()
            for cid_int in comp_int:
                src_mult = (cid_int // 1_000_000_000_000) * 1_000_000_000_000
                num_part = cid_int % 1_000_000_000_000
                pfx = comp_s_prefix.get(src_mult, 'S1')
                comp_strings.add(f"{pfx}-{num_part}")
            competitor_s1_ids = comp_strings

    del all_int
    import gc; gc.collect()
    print(f"Global reverse_rank done in {time.time() - t0:.2f}s. "
          f"Competitor S1 count: {len(competitor_s1_ids):,}  (RSS {_rss_mb():.0f} MB)", flush=True)
    return chunk_rr_list, competitor_s1_ids


def print_acceptance_report(feats_df, elapsed_time):
    """
    Prints the per-feature acceptance verification table and runtime statistics.
    Returns: None.
    """
    n_pairs = len(feats_df)
    labels = feats_df['label'].values if 'label' in feats_df.columns else None

    print("\n" + "=" * 90)
    print(f"STEP 6 FEATURE VERIFICATION REPORT ({n_pairs:,} candidate pairs)")
    print("=" * 90)
    if 'is_competitor' in feats_df.columns:
        n_comp_pairs = int((feats_df['is_competitor'] == 1).sum())
        n_comp_s1 = int(feats_df.loc[feats_df['is_competitor'] == 1, 's1_id'].nunique())
        print(f"Competitor S1: {n_comp_s1:,} entities ({n_comp_pairs:,} candidate pairs)")
        print("-" * 90)

    print(f"{'Feature':<20} | {'Min':>7} | {'Max':>7} | {'Mean':>8} | {'% NaN':>6} | {'Pos Mean':>9} | {'Neg Mean':>9}")
    print("-" * 90)

    for fname in config.FEATURES:
        vals = feats_df[fname].values.astype(np.float32)
        fmin = np.nanmin(vals)
        fmax = np.nanmax(vals)
        fmean = np.nanmean(vals)
        pct_nan = np.isnan(vals).mean() * 100
        if labels is not None:
            pos_mean = np.nanmean(vals[labels == 1])
            neg_mean = np.nanmean(vals[labels == 0])
            print(f"{fname:<20} | {fmin:7.2f} | {fmax:7.2f} | {fmean:8.2f} | {pct_nan:5.1f}% | {pos_mean:9.2f} | {neg_mean:9.2f}")
        else:
            print(f"{fname:<20} | {fmin:7.2f} | {fmax:7.2f} | {fmean:8.2f} | {pct_nan:5.1f}% | {'N/A':>9} | {'N/A':>9}")

    rs = feats_df['rule_score'].values
    print("-" * 90)
    if labels is not None:
        print(f"{'rule_score':<20} | {rs.min():7.2f} | {rs.max():7.2f} | {rs.mean():8.2f} | {'0.0%':>6} | {rs[labels == 1].mean():9.2f} | {rs[labels == 0].mean():9.2f}")
    else:
        print(f"{'rule_score':<20} | {rs.min():7.2f} | {rs.max():7.2f} | {rs.mean():8.2f} | {'0.0%':>6} | {'N/A':>9} | {'N/A':>9}")
    print("=" * 90)

    # Runtime and extrapolation
    time_per_1m = (elapsed_time / n_pairs) * 1_000_000 if n_pairs > 0 else 0
    print(f"\nTiming:")
    print(f"  Processed {n_pairs:,} candidate pairs in {elapsed_time:.2f}s")
    print(f"  Laptop CPU rate: {time_per_1m:.2f}s per 1M candidate pairs")
    
    # Train full: ~16M pairs, Test: ~8M to ~70M pairs
    est_train_sample = (16_000_000 / 1_000_000) * time_per_1m / 60.0
    est_test_8m = (8_000_000 / 1_000_000) * time_per_1m / 60.0
    est_test_70m = (70_000_000 / 1_000_000) * time_per_1m / 60.0
    print(f"  Estimated EC2 runtime (8 cores):")
    print(f"    - Full Train candidate pairs (~16M pairs): ~{est_train_sample:.1f} minutes")
    print(f"    - Test candidate pairs (capped ~8M pairs): ~{est_test_8m:.1f} minutes")
    print(f"    - Test candidate pairs (uncapped ~70M pairs): ~{est_test_70m:.1f} minutes")

def main():
    """
    Main orchestration function for Step 6 pairwise feature engineering.
    Returns: None.
    """
    args = parse_args()
    cache_dir = args.cache_dir
    if cache_dir is None:
        cache_dir = os.path.join(config.CACHE_DIR, "laptop_test") if args.laptop_test else config.CACHE_DIR

    print(f"=== Step 6: Pairwise Feature Engineering ===")
    print(f"Split: {args.split}")
    print(f"Cache Directory: {cache_dir}")
    print(f"Total features configured: {len(config.FEATURES)}")

    # 0. For train split, load split.parquet to determine which S1 get features.
    #    Blocking runs over ALL S1 (for realistic reverse_rank/Channel D),
    #    features are computed for sampled S1 + competitor S1.
    sampled_s1_ids = None  # None means "keep all" (test split)
    val_s1_ids = None
    if args.split == "train":
        split_path = os.path.join(cache_dir, "split.parquet")
        if os.path.exists(split_path):
            split_df = pd.read_parquet(split_path)
            sampled_s1_ids = set(split_df['s1_id'].values)
            val_s1_ids = set(split_df.loc[split_df['fold'] == 'val', 's1_id'].values)
            print(f"Loaded split.parquet: {len(sampled_s1_ids):,} sampled S1 ({len(val_s1_ids):,} val S1) for feature extraction.")
        else:
            print("WARNING: split.parquet not found; computing features for ALL S1.")

    # 1. Locate candidate files
    chunk_pattern_prefix = f"cands_{args.split}_chunk_"
    chunk_files = []
    for f in sorted(os.listdir(cache_dir)):
        if f.startswith(chunk_pattern_prefix) and f.endswith(".parquet"):
            chunk_files.append(os.path.join(cache_dir, f))

    if not chunk_files:
        # Fallback to single cands_{split}.parquet if chunks not found
        single_cands_path = os.path.join(cache_dir, f"cands_{args.split}.parquet")
        if os.path.exists(single_cands_path):
            chunk_files = [single_cands_path]
        else:
            print(f"Error: No candidate files found in {cache_dir} for split {args.split}.")
            sys.exit(1)

    print(f"Found {len(chunk_files)} candidate chunk file(s).")

    # 2. Check resume status
    pending_chunks = []
    for idx, cf in enumerate(chunk_files):
        # Naming: feats_<split>_<chunk>.parquet
        chunk_suffix = os.path.basename(cf).replace(f"cands_{args.split}_", "").replace(".parquet", "")
        # e.g. chunk_suffix is 'chunk_0' or '0'
        out_name1 = f"feats_{args.split}_{chunk_suffix}.parquet"
        out_name2 = f"feats_{args.split}_{idx}.parquet"
        out_p1 = os.path.join(cache_dir, out_name1)
        out_p2 = os.path.join(cache_dir, out_name2)
        
        if (os.path.exists(out_p1) or os.path.exists(out_p2)) and not args.force:
            print(f"Chunk {idx + 1}/{len(chunk_files)} already processed, skipping.")
        else:
            pending_chunks.append((idx, cf, out_p1, out_p2))

    if not pending_chunks:
        print("\nAll feature chunks already computed. Loading results for acceptance checks...")
        all_feats = []
        for idx, cf in enumerate(chunk_files):
            chunk_suffix = os.path.basename(cf).replace(f"cands_{args.split}_", "").replace(".parquet", "")
            out_p1 = os.path.join(cache_dir, f"feats_{args.split}_{chunk_suffix}.parquet")
            out_p2 = os.path.join(cache_dir, f"feats_{args.split}_{idx}.parquet")
            target_p = out_p1 if os.path.exists(out_p1) else out_p2
            all_feats.append(pd.read_parquet(target_p))
        df_full = pd.concat(all_feats, ignore_index=True)
        print_acceptance_report(df_full, elapsed_time=0.0)
        return

    # 3. Load normalized entity tables and compute token document frequencies
    t_start = time.time()
    print("\nLoading normalized tables...")
    s1_norm = load_norm_table(cache_dir, args.split, "source1")
    s2_norm = load_norm_table(cache_dir, args.split, "source2")
    s3_norm = load_norm_table(cache_dir, args.split, "source3")
    cands_norm = pd.concat([s2_norm, s3_norm])

    print("Computing address token document frequencies across S1+S2+S3...")
    df_tokens = compute_address_token_df(s1_norm, s2_norm, s3_norm)

    print("Precomputing token sets and lengths...")
    s1_norm['tok_set'] = [frozenset(str(x).split()) for x in s1_norm['name_full']]
    cands_norm['tok_set'] = [frozenset(str(x).split()) for x in cands_norm['name_full']]

    s1_norm['num_set'] = [frozenset(str(x).split()) if str(x).strip() else frozenset() for x in s1_norm['num_tokens']]
    cands_norm['num_set'] = [frozenset(str(x).split()) if str(x).strip() else frozenset() for x in cands_norm['num_tokens']]

    s1_norm['hc_set'] = [frozenset(str(x).split(';')) if str(x).strip() else frozenset() for x in s1_norm['house_cands']]
    cands_norm['hc_set'] = [frozenset(str(x).split(';')) if str(x).strip() else frozenset() for x in cands_norm['house_cands']]

    s1_norm['rare3_set'] = [extract_rare3_set(r['addr_norm'], r['country'], df_tokens) for _, r in s1_norm.iterrows()]
    cands_norm['rare3_set'] = [extract_rare3_set(r['addr_norm'], r['country'], df_tokens) for _, r in cands_norm.iterrows()]

    s1_norm['core_len'] = s1_norm['core_name'].str.len().astype(np.int32)
    cands_norm['core_len'] = cands_norm['core_name'].str.len().astype(np.int32)

    # 4. Load candidate embeddings
    print("Loading candidate embedding arrays...")
    all_cand_emb, cand_id_map = load_candidate_embeddings(cache_dir, args.split)

    # 5. Compute global reverse rank across ALL candidate pairs (full competition)
    #    and identify competitor S1s that share candidate(s) with val S1.
    chunk_rr_list, competitor_s1_ids = compute_global_reverse_ranks(
        cache_dir, args.split, chunk_files, sampled_s1_ids, val_s1_ids
    )
    print(f"Report: Competitor S1 count: {len(competitor_s1_ids):,}")

    target_s1_ids = None
    if sampled_s1_ids is not None:
        target_s1_ids = sampled_s1_ids | competitor_s1_ids

    # 6. Process each pending chunk — filter to sampled S1 + competitors (train) or all (test)
    processed_feats = []
    t_feat_start = time.time()
    for idx, cf, out_p1, out_p2 in pending_chunks:
        print(f"\nProcessing chunk {idx + 1}/{len(chunk_files)}: {cf}...")
        t_ch = time.time()
        chunk_cands = pd.read_parquet(cf)
        chunk_rr = chunk_rr_list[idx]

        # Filter to sampled S1 + competitors (for train split)
        if target_s1_ids is not None:
            keep_mask = chunk_cands['s1_id'].isin(target_s1_ids)
            n_before = len(chunk_cands)
            chunk_cands = chunk_cands[keep_mask].reset_index(drop=True)
            chunk_rr = chunk_rr[keep_mask.values]
            chunk_cands['is_competitor'] = chunk_cands['s1_id'].isin(competitor_s1_ids).astype(np.int8)
            print(f"  Filtered to sampled + competitor S1: {n_before} -> {len(chunk_cands)} candidate pairs")
        else:
            chunk_cands['is_competitor'] = np.zeros(len(chunk_cands), dtype=np.int8)

        if len(chunk_cands) == 0:
            print(f"  No sampled/competitor S1 in this chunk, skipping.")
            # Write empty file so resume sees it as done
            pd.DataFrame(columns=['s1_id', 'cand_id']).to_parquet(out_p1, index=False)
            continue

        feats_df = compute_chunk_features(
            chunk_cands, s1_norm, cands_norm, all_cand_emb, cand_id_map, chunk_rr
        )
        
        # Save to both target naming formats for 100% compatibility
        feats_df.to_parquet(out_p1, index=False)
        if out_p1 != out_p2:
            feats_df.to_parquet(out_p2, index=False)

        rss_aft = _rss_mb()
        print(f"Saved chunk features ({len(feats_df)} pairs) to {out_p1} "
              f"in {time.time() - t_ch:.2f}s  (RSS {rss_aft:.0f} MB)", flush=True)
        processed_feats.append(feats_df)

    t_feat_total = time.time() - t_feat_start

    # 7. Merge all chunks for summary report and validation
    all_feats = []
    for idx, cf in enumerate(chunk_files):
        chunk_suffix = os.path.basename(cf).replace(f"cands_{args.split}_", "").replace(".parquet", "")
        out_p1 = os.path.join(cache_dir, f"feats_{args.split}_{chunk_suffix}.parquet")
        out_p2 = os.path.join(cache_dir, f"feats_{args.split}_{idx}.parquet")
        target_p = out_p1 if os.path.exists(out_p1) else out_p2
        if os.path.exists(target_p):
            df_chunk = pd.read_parquet(target_p)
            if len(df_chunk) > 0:
                all_feats.append(df_chunk)
    if all_feats:
        df_full = pd.concat(all_feats, ignore_index=True)
    else:
        df_full = pd.DataFrame()

    # Acceptance report
    if len(df_full) > 0:
        print_acceptance_report(df_full, elapsed_time=t_feat_total)
    else:
        print("No feature rows produced.")

if __name__ == "__main__":
    main()
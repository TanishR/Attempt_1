import sys
import gc
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import DATA_DIR, CACHE_DIR, load_tsv, SEED
from utils_io import timer, save_parquet

def process_file(src, rel_path, is_gt=False):
    in_path = os.path.join(DATA_DIR, rel_path)
    out_name = f"raw_{src}.parquet"
    out_path = os.path.join(CACHE_DIR, out_name)
    if not os.path.exists(in_path):
        return None
    if not os.path.exists(out_path):
        with timer(f"Loading TSV to Parquet: {src}"):
            df = load_tsv(in_path)
            if is_gt:
                df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
            save_parquet(df, out_name)
            del df
            gc.collect()
    
    with timer(f"Reading {src} for EDA"):
        if is_gt:
            return pd.read_parquet(out_path)
        else:
            return pd.read_parquet(out_path, columns=["entity_id", "country", "business_name", "business_address"])

def run():
    files = {
        "train_source1": "train/train_source1.tsv",
        "train_source2": "train/train_source2.tsv",
        "train_source3": "train/train_source3.tsv",
        "train_ground_truth": "train/train_ground_truth.tsv",
        "test_source1": "test/test_source1.tsv",
        "test_source2": "test/test_source2.tsv",
        "test_source3": "test/test_source3.tsv",
    }

    report = []
    def log(msg):
        print(msg)
        report.append(msg)

    s1_countries = {}
    s23_countries = {}
    stats = {}

    for src, rel_path in files.items():
        if src == "train_ground_truth": continue
        df = process_file(src, rel_path)
        if df is None: continue
        
        n_rows = len(df)
        country_counts = df["country"].value_counts().to_dict()
        
        if "source1" in src:
            s1_countries.update(dict(zip(df["entity_id"], df["country"])))
        else:
            s23_countries.update(dict(zip(df["entity_id"], df["country"])))
            
        def has_nonascii(s):
            if pd.isna(s): return False
            return not str(s).isascii()
            
        nonascii_name = df["business_name"].apply(has_nonascii).mean() * 100
        nonascii_addr = df["business_address"].apply(has_nonascii).mean() * 100
        
        empty = df["business_address"].isna() | (df["business_address"].str.lower().str.strip() == "null") | (df["business_address"].str.strip() == "")
        null_addr_pct = empty.mean() * 100
        
        country_nonascii = {}
        for c in df["country"].unique():
            c_df = df[df["country"] == c]
            c_name = c_df["business_name"].apply(has_nonascii).mean() * 100
            c_addr = c_df["business_address"].apply(has_nonascii).mean() * 100
            country_nonascii[c] = {"name": c_name, "addr": c_addr}
            
        stats[src] = {
            "rows": n_rows,
            "country": country_counts,
            "nonascii_name": nonascii_name,
            "nonascii_addr": nonascii_addr,
            "null_addr_pct": null_addr_pct,
            "country_nonascii": country_nonascii
        }
        
        del df
        gc.collect()

    for src, s in stats.items():
        log(f"--- {src} ---")
        log(f"Rows: {s['rows']}")
        for c, count in s["country"].items():
            log(f"  Country {c}: {count} rows")
        log(f"Non-ASCII name: {s['nonascii_name']:.2f}%, Address: {s['nonascii_addr']:.2f}%")
        for c, v in s["country_nonascii"].items():
            log(f"  Country {c} Non-ASCII name: {v['name']:.2f}%, Address: {v['addr']:.2f}%")
        log(f"Empty/Null address: {s['null_addr_pct']:.2f}%")

    gt = process_file("train_ground_truth", files["train_ground_truth"], is_gt=True)
    if gt is not None:
        records = []
        sets_data = []
        for s1_id, match_str in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
            matches = [m.strip() for m in match_str.split(",")] if match_str else []
            sets_data.append({"s1_id": s1_id, "matches": matches})
            for m in matches:
                if m.startswith("S2"):
                    records.append({"s1_id": s1_id, "match_id": m, "match_source": "S2"})
                elif m.startswith("S3"):
                    records.append({"s1_id": s1_id, "match_id": m, "match_source": "S3"})
        gt_long = pd.DataFrame(records)
        gt_sets = pd.DataFrame(sets_data)
        
        save_parquet(gt_long, "gt_long.parquet")
        save_parquet(gt_sets, "gt_sets.parquet")
        
        del gt
        gc.collect()
        
        missing_s1 = len(set(gt_long["s1_id"]) - set(s1_countries.keys()))
        matched_s23 = set(gt_long["match_id"])
        s23_keys = set(s23_countries.keys())
        missing_s23 = len(matched_s23 - s23_keys)
        log("")
        log(f"Missing from files: GT S1s not in S1 file = {missing_s1}, GT matches not in S2/S3 files = {missing_s23}")

        gt_sets["is_singleton"] = gt_sets["matches"].apply(len) == 0
        gt_sets["country"] = gt_sets["s1_id"].map(s1_countries)
        
        singleton_overall = gt_sets["is_singleton"].mean() * 100
        log("")
        log(f"Singleton overall: {singleton_overall:.2f}%")
        for c in gt_sets["country"].dropna().unique():
            c_pct = gt_sets[gt_sets["country"] == c]["is_singleton"].mean() * 100
            log(f"Singleton {c}: {c_pct:.2f}%")
            
        match_counts = gt_long.groupby(["s1_id", "match_source"]).size().unstack(fill_value=0)
        all_s1 = pd.DataFrame(index=gt_sets["s1_id"])
        match_counts = match_counts.reindex(all_s1.index).fillna(0)
        if "S2" not in match_counts: match_counts["S2"] = 0
        if "S3" not in match_counts: match_counts["S3"] = 0
        match_counts["Total"] = match_counts["S2"] + match_counts["S3"]
        
        log("")
        log("Matches per S1:")
        for col in ["S2", "S3", "Total"]:
            log(f"  {col}: Mean={match_counts[col].mean():.2f}, Median={match_counts[col].median()}, 95th={np.percentile(match_counts[col], 95)}, Max={match_counts[col].max()}")
        
        log("Histogram of total matches:")
        hist = match_counts["Total"].value_counts().sort_index()
        for k, v in hist.items():
            if k <= 10:
                log(f"  {k} matches: {v}")
        log(f"  >10 matches: {hist[hist.index > 10].sum()}")

        match_id_counts = gt_long["match_id"].value_counts()
        exclusivity_violations = match_id_counts[match_id_counts > 1]
        log("")
        log(f"Exclusivity violations (S2/S3 matching >1 S1): {len(exclusivity_violations)}")
        if len(exclusivity_violations) > 0:
            log("Examples:")
            for k, v in exclusivity_violations.head(10).items():
                s1_list = gt_long[gt_long["match_id"] == k]["s1_id"].tolist()
                log(f"  {k} mapped to {s1_list}")

        gt_long["s1_country"] = gt_long["s1_id"].map(s1_countries)
        gt_long["match_country"] = gt_long["match_id"].map(s23_countries)
        cross = gt_long[gt_long["s1_country"] != gt_long["match_country"]]
        cross = cross.dropna(subset=["s1_country", "match_country"])
        log("")
        log(f"Cross-country matches: {len(cross)}")
        if len(cross) > 0:
            log("Examples:")
            for _, row in cross.head(10).iterrows():
                log(f"  {row['s1_id']} ({row['s1_country']}) -> {row['match_id']} ({row['match_country']})")

        total_s23_train = 0
        if "train_source2" in stats: total_s23_train += stats["train_source2"]["rows"]
        if "train_source3" in stats: total_s23_train += stats["train_source3"]["rows"]
        unmatched_s23 = total_s23_train - len(matched_s23)
        unmatched_pct = (unmatched_s23 / total_s23_train * 100) if total_s23_train > 0 else 0
        log("")
        log(f"Unmatched train S2/S3 (Distractors): {unmatched_s23} ({unmatched_pct:.2f}%)")

        log("")
        log("20 Random GT Pairs (10 US, 10 India):")
        s1_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source1.parquet"), columns=["entity_id", "business_name", "business_address"]).set_index("entity_id")
        try:
            s2_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source2.parquet"), columns=["entity_id", "business_name", "business_address"]).set_index("entity_id")
        except: s2_df = pd.DataFrame()
        try:
            s3_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source3.parquet"), columns=["entity_id", "business_name", "business_address"]).set_index("entity_id")
        except: s3_df = pd.DataFrame()
        s23_df = pd.concat([s2_df, s3_df])
        
        for c in ["US", "India"]:
            c_gt = gt_long[gt_long["s1_country"] == c]
            if not c_gt.empty:
                sample = c_gt.sample(n=min(10, len(c_gt)), random_state=SEED)
                for _, row in sample.iterrows():
                    try:
                        n1 = s1_df.loc[row["s1_id"], "business_name"]
                        a1 = s1_df.loc[row["s1_id"], "business_address"]
                        n2 = s23_df.loc[row["match_id"], "business_name"]
                        a2 = s23_df.loc[row["match_id"], "business_address"]
                        log(f"[{c}] {n1} | {a1} || {n2} | {a2}")
                    except KeyError:
                        pass
        del s1_df, s2_df, s3_df, s23_df
        gc.collect()

    with open(os.path.join(CACHE_DIR, "eda_report.txt"), "w") as f:
        f.write("\n".join(report))

if __name__ == "__main__":
    try:
        run()
    except MemoryError:
        print("EDA failed due to out of memory.")

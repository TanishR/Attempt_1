import os
import sys
import re
import gc
import pandas as pd
import anyascii
from rapidfuzz.fuzz import token_sort_ratio
import multiprocessing as mp
from time import time
import numpy as np

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import CACHE_DIR
from maps import (
    NAME_ABBREVIATIONS, LEGAL_SUFFIXES, WEAK_TOKENS, 
    ADDR_ABBREVIATIONS, UNIT_TOKENS, STATE_MAP
)

def transliterate_if_needed(text):
    if not isinstance(text, str): return ""
    if text.isascii(): return text
    return anyascii.anyascii(text)

def get_consonant_skeleton(text):
    if not isinstance(text, str) or not text: return ""
    vowels = set("aeiouy")
    words = text.split()
    skel_words = []
    for w in words:
        if not w: continue
        chars = []
        for i, c in enumerate(w):
            if c == '5': c = 's'
            elif c == '0': c = 'o'
            elif c == '1': c = 'l'
            elif c == '3': c = 'e'
            
            if i == 0 or c not in vowels:
                if not chars or chars[-1] != c:
                    chars.append(c)
        if w.isdigit():
            chars = []
            for i, c in enumerate(w):
                if not chars or chars[-1] != c: chars.append(c)
        skel_words.append("".join(chars))
    return " ".join(skel_words)

URL_CLEAN_RE = re.compile(r'\b(www\.)|(\.com|\.in|\.org|\.net|\.fr)\b')
TRAILING_ID_RE = re.compile(r'[-\s#]+[0-9]{4,}\s*$')
PUNCT_RE = re.compile(r'[^\w\s]')

def normalize_name(raw_name):
    if not isinstance(raw_name, str) or not raw_name.strip():
        return "", "", "", "", "", "", ""
        
    n = transliterate_if_needed(raw_name).lower()
    
    if '|' in n:
        parts = n.split('|')
        if 'www.' in parts[-1] or '.com' in parts[-1] or '.in' in parts[-1] or '.org' in parts[-1] or '.net' in parts[-1] or '.fr' in parts[-1]:
            n = parts[0]
        else:
            n = n.replace('|', ' ')
            
    n = URL_CLEAN_RE.sub('', n)
    
    name_a, name_b = "", ""
    if ' dba ' in n:
        parts = n.split(' dba ', 1)
        name_a = parts[0].strip()
        name_b = parts[1].strip()
    
    n = TRAILING_ID_RE.sub('', n)
    
    n = n.replace('&', ' and ')
    n = PUNCT_RE.sub(' ', n)
    
    tokens = n.split()
    norm_tokens = []
    for t in tokens:
        mapped = NAME_ABBREVIATIONS.get(t, t)
        if not norm_tokens or norm_tokens[-1] != mapped:
            norm_tokens.append(mapped)
            
    legal_tokens = []
    core_tokens_full = []
    for t in norm_tokens:
        if t in LEGAL_SUFFIXES:
            legal_tokens.append(t)
        else:
            core_tokens_full.append(t)
            
    name_full = " ".join(core_tokens_full)
    legal = " ".join(legal_tokens)
    
    core_tokens = [t for t in core_tokens_full if t not in WEAK_TOKENS]
    if not core_tokens and core_tokens_full: 
        core_tokens = core_tokens_full
        
    core_name = " ".join(core_tokens)
    core_sorted = " ".join(sorted(core_tokens))
    name_skel = get_consonant_skeleton(core_name)
    
    if name_a and name_b:
        name_a_norm = normalize_name(name_a)[3] 
        name_b_norm = normalize_name(name_b)[3] 
        name_a = name_a_norm
        name_b = name_b_norm
        
    return name_full, name_a, name_b, core_name, legal, core_sorted, name_skel

def normalize_address(raw_addr):
    if not isinstance(raw_addr, str) or not raw_addr.strip() or raw_addr.strip().lower() == "null":
        return "", "", "", 0, "", "", "", 1
        
    a = transliterate_if_needed(raw_addr).lower()
    
    a = a.replace('&', ' and ')
    a_punct_removed = PUNCT_RE.sub(' ', a)
    
    tokens = a_punct_removed.split()
    norm_tokens = []
    for t in tokens:
        mapped = ADDR_ABBREVIATIONS.get(t, t)
        norm_tokens.append(mapped)
        
    addr_tokens = []
    unit_tokens = []
    num_tokens = []
    
    for t in norm_tokens:
        if t in UNIT_TOKENS:
            unit_tokens.append(t)
        else:
            addr_tokens.append(t)
            
    for t in addr_tokens:
        if any(c.isdigit() for c in t):
            num_tokens.append(t.lstrip('0') or '0') 
            
    addr_norm = " ".join(addr_tokens)
    unit_toks_str = " ".join(unit_tokens)
    num_toks_str = " ".join(num_tokens)
    
    house_no = ""
    house_masked = 0
    
    raw_tokens = a.split()
    for rt in raw_tokens:
        if any(c.isdigit() for c in rt):
            if "#" in rt:
                digits_only = "".join(c for c in rt if c.isdigit())
                if digits_only:
                    house_no = digits_only.lstrip("0") or "0"
                    house_masked = 1
                    break
            else:
                digits_only = "".join(c for c in rt if c.isdigit())
                if digits_only:
                    house_no = digits_only.lstrip("0") or "0"
                    break

    zip_pin = ""
    for t in reversed(tokens):
        if len(t) in (5, 6) and t.isdigit():
            zip_pin = t
            break
            
    state_code = ""
    for i in range(len(addr_tokens)):
        # try 2-grams
        if i < len(addr_tokens) - 1:
            bigram = addr_tokens[i] + " " + addr_tokens[i+1]
            if bigram in STATE_MAP:
                state_code = STATE_MAP[bigram]
                break
        # try 1-grams
        if addr_tokens[i] in STATE_MAP:
            state_code = STATE_MAP[addr_tokens[i]]
            break
            
    addr_missing = 1 if not addr_norm else 0
    
    return addr_norm, unit_toks_str, house_no, house_masked, num_toks_str, zip_pin, state_code, addr_missing

def process_chunk(df_chunk):
    results = []
    for _, row in df_chunk.iterrows():
        eid = row.get("entity_id", row.get("source1_entity_id", ""))
        country = row.get("country", "")
        raw_name = row.get("business_name", "")
        raw_addr = row.get("business_address", "")
        
        n_full, n_a, n_b, core, legal, c_sorted, skel = normalize_name(raw_name)
        a_norm, a_unit, h_no, h_mask, num_tok, z_pin, st_code, a_miss = normalize_address(raw_addr)
        
        # If the name is completely empty after normalization but raw_name was not, fallback to transliterated raw name
        if not n_full and raw_name and str(raw_name).strip():
            fallback = transliterate_if_needed(raw_name).lower()
            fallback = PUNCT_RE.sub(' ', fallback).strip()
            n_full = fallback
            core = fallback
            c_sorted = " ".join(sorted(fallback.split()))
            skel = get_consonant_skeleton(fallback)
        
        results.append({
            "entity_id": eid,
            "country": country,
            "raw_name": raw_name,
            "raw_address": raw_addr,
            "name_full": n_full,
            "name_a": n_a,
            "name_b": n_b,
            "core_name": core,
            "legal": legal,
            "core_sorted": c_sorted,
            "name_skel": skel,
            "addr_norm": a_norm,
            "addr_tokens": a_norm, # same as addr_norm based on requirements
            "unit_tokens": a_unit,
            "house_no": h_no,
            "house_masked": h_mask,
            "num_tokens": num_tok,
            "zip_pin": z_pin,
            "state_code": st_code,
            "addr_missing": a_miss
        })
    return pd.DataFrame(results)

def normalize_file(file_path, out_path):
    print(f"Processing {file_path}...")
    df = pd.read_parquet(file_path)
    if "source1_entity_id" in df.columns:
        df["entity_id"] = df["source1_entity_id"]
        
    num_cores = mp.cpu_count()
    chunk_size = len(df) // (num_cores * 2) + 1
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    start_time = time()
    with mp.Pool(num_cores) as pool:
        result_chunks = pool.map(process_chunk, chunks)
        
    res_df = pd.concat(result_chunks, ignore_index=True)
    res_df.to_parquet(out_path, index=False)
    
    elapsed = time() - start_time
    rows = len(df)
    rate = rows / elapsed if elapsed > 0 else 0
    time_per_100k = 100000 / rate if rate > 0 else 0
    print(f"Processed {rows} rows in {elapsed:.2f}s (Time per 100k rows: {time_per_100k:.2f}s)")
    return rows, elapsed

if __name__ == "__main__":
    files = [
        "sample_train_source1.parquet",
        "sample_train_source2.parquet", 
        "sample_train_source3.parquet",
        "sample_test_source1.parquet",
        "sample_test_source2.parquet",
        "sample_test_source3.parquet"
    ]
    
    total_rows = 0
    total_time = 0
    
    for f in files:
        in_path = os.path.join(CACHE_DIR, f)
        if not os.path.exists(in_path):
            # Try raw if sample doesn't exist
            raw_f = f.replace("sample_", "raw_")
            in_path = os.path.join(CACHE_DIR, raw_f)
            if not os.path.exists(in_path):
                continue
                
        out_f = f.replace("sample_", "norm_").replace("raw_", "norm_")
        out_path = os.path.join(CACHE_DIR, out_f)
        
        r, t = normalize_file(in_path, out_path)
        total_rows += r
        total_time += t
        
    if total_rows > 0:
        overall_rate = total_rows / total_time
        full_data_rows = 24000000 # ~2.4 crore
        est_seconds = full_data_rows / overall_rate
        print(f"\nOverall rate: {overall_rate:.2f} rows/s")
        print(f"Estimated time for 2.4 crore rows: {est_seconds / 60:.2f} minutes")

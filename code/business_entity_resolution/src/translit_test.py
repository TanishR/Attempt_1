import os
import sys
import pandas as pd
from rapidfuzz.fuzz import token_sort_ratio
import anyascii
from indic_transliteration import sanscript

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from config import CACHE_DIR, SEED

def get_consonant_skeleton(text):
    if not isinstance(text, str): return ""
    vowels = set("aeiouy")
    words = text.split()
    skel_words = []
    for w in words:
        if not w: continue
        chars = []
        for i, c in enumerate(w):
            # map digits 5->s, 0->o, 1->l, 3->e inside a token (if token is not entirely numeric)
            if c == '5': c = 's'
            elif c == '0': c = 'o'
            elif c == '1': c = 'l'
            elif c == '3': c = 'e'
            if i == 0 or c not in vowels:
                if not chars or chars[-1] != c:
                    chars.append(c)
        # if token was entirely numeric and became string, we'll keep it. 
        # But wait, rule says "leave standalone numbers untouched".
        if w.isdigit():
            chars = []
            for i, c in enumerate(w):
                if not chars or chars[-1] != c: chars.append(c)
        skel_words.append("".join(chars))
    return " ".join(skel_words)


def run_test():
    gt_path = os.path.join(CACHE_DIR, "gt_long.parquet")
    if not os.path.exists(gt_path):
        print("gt_long not found!")
        return

    s1_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source1.parquet"), columns=["entity_id", "business_name"]).set_index("entity_id")
    s2_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source2.parquet"), columns=["entity_id", "business_name"]).set_index("entity_id")
    s3_df = pd.read_parquet(os.path.join(CACHE_DIR, "raw_train_source3.parquet"), columns=["entity_id", "business_name"]).set_index("entity_id")
    s23_df = pd.concat([s2_df, s3_df])
    
    gt = pd.read_parquet(gt_path)
    
    pairs = []
    for _, row in gt.iterrows():
        s1 = row["s1_id"]
        m = row["match_id"]
        try:
            n1 = str(s1_df.loc[s1, "business_name"])
            n2 = str(s23_df.loc[m, "business_name"])
            if n1.isascii() and not n2.isascii():
                pairs.append((n1, n2))
                if len(pairs) == 200:
                    break
        except KeyError:
            continue
            
    print(f"Found {len(pairs)} non-ASCII test pairs.")
    
    anyascii_scores = []
    indic_scores = []
    anyascii_skel = []
    indic_skel = []
    
    examples = []
    
    for n1, n2 in pairs:
        # anyascii
        n2_any = anyascii.anyascii(n2).lower()
        # indic
        try:
            n2_indic = sanscript.transliterate(n2, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
        except:
            n2_indic = n2_any
            
        n1_low = n1.lower()
        
        a_ts = token_sort_ratio(n1_low, n2_any)
        i_ts = token_sort_ratio(n1_low, n2_indic)
        
        a_sk = token_sort_ratio(get_consonant_skeleton(n1_low), get_consonant_skeleton(n2_any))
        i_sk = token_sort_ratio(get_consonant_skeleton(n1_low), get_consonant_skeleton(n2_indic))
        
        anyascii_scores.append(a_ts)
        indic_scores.append(i_ts)
        anyascii_skel.append(a_sk)
        indic_skel.append(i_sk)
        
        if len(examples) < 15:
            examples.append(f"RAW: {n1} || {n2}\nANY: {n2_any} (ts={a_ts}, sk={a_sk})\nIND: {n2_indic} (ts={i_ts}, sk={i_sk})\n")

    print("\n--- Mean Scores ---")
    print(f"Anyascii Token Sort: {sum(anyascii_scores)/len(anyascii_scores):.2f}")
    print(f"Indic Token Sort: {sum(indic_scores)/len(indic_scores):.2f}")
    print(f"Anyascii Skeleton: {sum(anyascii_skel)/len(anyascii_skel):.2f}")
    print(f"Indic Skeleton: {sum(indic_skel)/len(indic_skel):.2f}")
    
    print("\n--- Examples ---")
    for ex in examples:
        print(ex)

if __name__ == "__main__":
    run_test()

import os
import pandas as pd

# Root path detection
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(os.path.dirname(CODE_DIR))

DATA_DIR = os.path.join(ROOT_DIR, "dataset")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

# Ensure cache and output dirs exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample configuration
AMLC_SAMPLE = os.environ.get("AMLC_SAMPLE", "50000").lower()
SAMPLE = None if AMLC_SAMPLE == "none" else int(AMLC_SAMPLE)
SAMPLE_BLOCK_TEST = 5000

SEED = 42

# Blocking and Decision parameters (from DECISIONS.md)
K_PER_SOURCE = 10
MAX_ADDR_CANDS = 20
MAX_SKEL_CANDS = 20
MAX_RARE_CANDS = 20
MAX_BLOCK_SIZE = 50
CAND_CAP = 30
USE_EXCLUSIVITY = True
BLOCK_BY_COUNTRY = True
EMB_DIM = 256
EMB_MODEL = "Qwen/Qwen3-Embedding-0.6B"

T_TOP1 = None
T_EXTRA = None
EXCL_MARGIN = 0.0
FEATURES = []

def load_tsv(path):
    """
    Loads TSV file with correct quoting and dtype parameters.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=3
    )

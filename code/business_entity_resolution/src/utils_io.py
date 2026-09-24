import os
import time
import contextlib
import pandas as pd
import numpy as np
from config import CACHE_DIR

def save_parquet(df, filename):
    """Saves dataframe to cache/ as parquet."""
    path = os.path.join(CACHE_DIR, filename)
    df.to_parquet(path, index=False)

def load_parquet(filename):
    """Loads dataframe from cache/ as parquet."""
    path = os.path.join(CACHE_DIR, filename)
    return pd.read_parquet(path)

def save_npy(arr, filename):
    """Saves numpy array to cache/."""
    path = os.path.join(CACHE_DIR, filename)
    np.save(path, arr)

def load_npy(filename):
    """Loads numpy array from cache/."""
    path = os.path.join(CACHE_DIR, filename)
    return np.load(path)

@contextlib.contextmanager
def timer(name):
    """Timer context manager for logging elapsed time."""
    start = time.time()
    print(f"[{name}] started...")
    yield
    elapsed = time.time() - start
    print(f"[{name}] done in {elapsed:.2f} s")

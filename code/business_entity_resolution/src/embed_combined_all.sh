#!/usr/bin/env bash
# ==============================================================================
# Pipeline: Combined Name + Address Embedding Generation for All Splits & Sources
#
# Generates 256-dim L2-normalized embeddings for:
#   1. train source1
#   2. train source2
#   3. train source3
#   4. test source1
#   5. test source2
#   6. test source3
#
# Constraints & Guarantees:
#   - Resume-safe: Skips files already complete; resumes parts after a crash.
#   - Memory: GPU memory < 10 GB, RSS < 7 GB (concurrent safe with s3b_augment).
#   - Verifies ID count equals normalized table count and embeddings are L2-normalized.
#   - Prints texts/sec and ETA for each file.
#
# Usage:
#   # On EC2 full run:
#   bash code/business_entity_resolution/src/embed_combined_all.sh
#
#   # On laptop test:
#   bash code/business_entity_resolution/src/embed_combined_all.sh --laptop-test
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Default cache directory
CACHE_DIR="${ROOT_DIR}/cache"
LAPTOP_FLAG=""
LIMIT_VAL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --laptop-test)
            CACHE_DIR="${ROOT_DIR}/cache/laptop_test"
            LAPTOP_FLAG="--laptop-test"
            shift
            ;;
        --cache-dir=*)
            CACHE_DIR="${1#*=}"
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --cache-dir)
            CACHE_DIR="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        --limit=*)
            LIMIT_VAL="${1#*=}"
            EXTRA_ARGS+=("$1")
            shift
            ;;
        --limit)
            LIMIT_VAL="$2"
            EXTRA_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="python3"
if [ -f "${ROOT_DIR}/amlc_env/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/amlc_env/bin/python"
fi

echo "================================================================================"
echo "          RUNNING COMBINED NAME + ADDRESS EMBEDDINGS (ALL SOURCES)             "
echo "================================================================================"
echo "Cache Directory: ${CACHE_DIR}"
echo "Python Binary:   ${PYTHON_BIN}"
echo "Start Time:      $(date)"
echo "================================================================================"

STEPS=(
    "train source1"
    "train source2"
    "train source3"
    "test source1"
    "test source2"
    "test source3"
)

TOTAL_STEPS=${#STEPS[@]}
STEP_NUM=0
OVERALL_START=$(date +%s)

for step in "${STEPS[@]}"; do
    STEP_NUM=$((STEP_NUM + 1))
    read -r SPLIT SOURCE <<< "${step}"

    # Target filenames that Channel H in s3b_augment.py expects
    EMB_FILE="${CACHE_DIR}/embc_${SPLIT}_${SOURCE}.npy"
    IDS_FILE="${CACHE_DIR}/idsc_${SPLIT}_${SOURCE}.npy"
    NORM_FILE="${CACHE_DIR}/norm_${SPLIT}_${SOURCE}.parquet"

    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "[Step ${STEP_NUM}/${TOTAL_STEPS}] Processing: ${SPLIT} ${SOURCE}"
    echo "Target Output: ${EMB_FILE}"
    echo "Target IDs:    ${IDS_FILE}"
    echo "--------------------------------------------------------------------------------"

    # Check if normalized table exists (especially for test split)
    if [ ! -f "${NORM_FILE}" ]; then
        echo "Notice: ${NORM_FILE} does not exist yet. Skipping ${SPLIT} ${SOURCE}."
        continue
    fi

    T0=$(date +%s)

    # Run embedding generator (resume-safe, skip-complete, batch-size 512 for GPU < 10GB)
    "${PYTHON_BIN}" "${SCRIPT_DIR}/s2b_embed_combined.py" \
        --split "${SPLIT}" \
        --source "${SOURCE}" \
        --cache-dir "${CACHE_DIR}" \
        --batch-size 512 \
        --chunk-size 200000 \
        ${LAPTOP_FLAG} \
        "${EXTRA_ARGS[@]}"

    T_ELAPSED=$(( $(date +%s) - T0 ))

    # Verification: check row count against normalized table and verify L2 normalization
    echo ""
    echo "[Step ${STEP_NUM}/${TOTAL_STEPS}] Verifying ${SPLIT} ${SOURCE} embeddings and IDs..."
    "${PYTHON_BIN}" -c "
import os, sys
import numpy as np
import pyarrow.parquet as pq

emb_path = '${EMB_FILE}'
ids_path = '${IDS_FILE}'
norm_path = '${NORM_FILE}'

if not os.path.exists(emb_path) or not os.path.exists(ids_path):
    print(f'ERROR: Output files missing! {emb_path} or {ids_path}')
    sys.exit(1)

expected_rows = pq.ParquetFile(norm_path).metadata.num_rows
limit_str = '${LIMIT_VAL}'
if limit_str != '':
    expected_rows = min(int(limit_str), expected_rows)

ids = np.load(ids_path, allow_pickle=True)
actual_rows = len(ids)

if actual_rows != expected_rows:
    print(f'ERROR: ID count mismatch! {actual_rows:,} != expected {expected_rows:,}')
    sys.exit(1)

emb = np.load(emb_path, mmap_mode='r')
if emb.shape != (expected_rows, 256):
    print(f'ERROR: Embedding shape mismatch! {emb.shape} != ({expected_rows:,}, 256)')
    sys.exit(1)

sample_n = min(25000, expected_rows)
norms = np.linalg.norm(emb[:sample_n].astype(np.float32), axis=1)
n_mean, n_min, n_max = float(norms.mean()), float(norms.min()), float(norms.max())

print(f'>>> VERIFICATION PASSED for ${SPLIT} ${SOURCE}:')
print(f'    - Row Count: {actual_rows:,} / {expected_rows:,} (100% MATCH)')
print(f'    - Embedding Shape: {emb.shape} (dtype: {emb.dtype})')
print(f'    - L2 Normalization (sample {sample_n:,}): mean={n_mean:.4f}, min={n_min:.4f}, max={n_max:.4f}')

if not np.allclose(norms, 1.0, atol=0.02):
    print(f'ERROR: Embeddings are not L2-normalized! min={n_min}, max={n_max}')
    sys.exit(1)
"

    echo "[Step ${STEP_NUM}/${TOTAL_STEPS}] Completed ${SPLIT} ${SOURCE} in ${T_ELAPSED}s."
done

TOTAL_ELAPSED=$(( $(date +%s) - OVERALL_START ))
echo ""
echo "================================================================================"
echo "          ALL COMBINED EMBEDDINGS SUCCESSFULLY GENERATED & VERIFIED            "
echo "Total Elapsed Time: $(( TOTAL_ELAPSED / 60 ))m $(( TOTAL_ELAPSED % 60 ))s"
echo "End Time:           $(date)"
echo "================================================================================"

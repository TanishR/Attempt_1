#!/usr/bin/env bash
# ==============================================================================
# EC2 Full-Data Pipeline Orchestration Script
# Runs Stages 1 through 10 with done-markers in cache/markers/ and logs in logs/
# Stops on first error (set -e) and reports the failed stage.
# Usage:
#   bash ec2_full_pipeline.sh               # Run all stages (resume from markers)
#   bash ec2_full_pipeline.sh --stage 5     # Run stage 5 only
#   bash ec2_full_pipeline.sh --from-stage 3 # Run from stage 3 to 10
#   bash ec2_full_pipeline.sh --force       # Re-run all stages (ignore markers)
# ==============================================================================

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

# Setup Python environment and PYTHONPATH
PYTHON_BIN="python3"
if [ -d "amlc_env/bin" ]; then
    PYTHON_BIN="amlc_env/bin/python"
fi
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

MARKER_DIR="cache/markers"
LOG_DIR="logs"
mkdir -p "${MARKER_DIR}" "${LOG_DIR}"

TARGET_SINGLE_STAGE=""
FROM_STAGE=1
FORCE_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            TARGET_SINGLE_STAGE="$2"
            shift 2
            ;;
        --from-stage)
            FROM_STAGE="$2"
            shift 2
            ;;
        --force)
            FORCE_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--stage N] [--from-stage N] [--force]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

CURRENT_STAGE=0

# Trap to report which stage failed on error
trap 'echo ""; echo "============================================================"; echo "ERROR: Pipeline execution FAILED at Stage ${CURRENT_STAGE}!"; echo "Check log file: ${LOG_DIR}/stage_${CURRENT_STAGE}.log"; echo "============================================================"; exit 1' ERR

run_stage() {
    local stage_num="$1"
    local stage_desc="$2"
    local stage_cmd="$3"

    CURRENT_STAGE="${stage_num}"
    local marker_file="${MARKER_DIR}/stage_${stage_num}.done"
    local log_file="${LOG_DIR}/stage_${stage_num}.log"

    # Stage filter logic
    if [ -n "${TARGET_SINGLE_STAGE}" ]; then
        if [ "${TARGET_SINGLE_STAGE}" != "${stage_num}" ]; then
            return 0
        fi
    else
        if [ "${stage_num}" -lt "${FROM_STAGE}" ]; then
            echo "[SKIPPED] Stage ${stage_num}: ${stage_desc} (before from-stage ${FROM_STAGE})"
            return 0
        fi
        if [ -f "${marker_file}" ] && [ "${FORCE_RUN}" = false ]; then
            echo "[DONE-MARKER] Stage ${stage_num}: ${stage_desc} already completed. (Skipping, marker exists: ${marker_file})"
            return 0
        fi
    fi

    echo ""
    echo "================================================================================"
    echo "STAGE ${stage_num}: ${stage_desc}"
    echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Logging to: ${log_file}"
    echo "================================================================================"

    local start_ts
    start_ts=$(date +%s)

    # Execute stage command logging to file and teeing to console
    {
        echo "=== STAGE ${stage_num} START: $(date '+%Y-%m-%d %H:%M:%S') ==="
        eval "${stage_cmd}"
        echo "=== STAGE ${stage_num} END: $(date '+%Y-%m-%d %H:%M:%S') ==="
    } 2>&1 | tee "${log_file}"

    local end_ts
    end_ts=$(date +%s)
    local duration=$((end_ts - start_ts))

    touch "${marker_file}"
    echo "--------------------------------------------------------------------------------"
    echo "STAGE ${stage_num} COMPLETED SUCCESSFULLY in ${duration}s ($(date '+%Y-%m-%d %H:%M:%S'))"
    echo "--------------------------------------------------------------------------------"
}

echo "=== ML Challenge 2026 EC2 Full Pipeline ==="
echo "Python: ${PYTHON_BIN}"
echo "Project Root: ${PROJECT_ROOT}"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"

# ------------------------------------------------------------------------------
# Stage 1: Normalization (with safety check on name_full and raw_name)
# ------------------------------------------------------------------------------
run_stage 1 "Normalize all entities (Train & Test) with safety validation" "
    echo 'Running pre-normalization safety snapshot...'
    ${PYTHON_BIN} ${SCRIPT_DIR}/safety_check_norm.py --snapshot --cache-dir cache
    
    echo 'Normalizing train split...'
    ${PYTHON_BIN} ${SCRIPT_DIR}/s1_normalize.py --split train
    
    echo 'Normalizing test split...'
    ${PYTHON_BIN} ${SCRIPT_DIR}/s1_normalize.py --split test
    
    echo 'Running post-normalization safety equivalence check...'
    ${PYTHON_BIN} ${SCRIPT_DIR}/safety_check_norm.py --verify --cache-dir cache
"

# ------------------------------------------------------------------------------
# Stage 2: Train S1 Embeddings (ALL Train S1 via --all-s1)
# ------------------------------------------------------------------------------
run_stage 2 "Generate embeddings for ALL train S1 entities (--all-s1)" "
    echo 'Cleaning old train S1 embedding files...'
    rm -f cache/emb_train_source1*.npy cache/ids_train_source1*.npy cache/parts/train_source1_*
    
    echo 'Computing full Train S1 embeddings (main + alt transliterated)...'
    ${PYTHON_BIN} ${SCRIPT_DIR}/s2_embed.py --split train --source s1 --all-s1
"

# ------------------------------------------------------------------------------
# Stage 3: Train Blocking (All Train S1, recall report on val sample)
# ------------------------------------------------------------------------------
run_stage 3 "Train candidate generation & blocking for all Train S1" "
    ${PYTHON_BIN} ${SCRIPT_DIR}/s3_block.py --split train
"

# ------------------------------------------------------------------------------
# Stage 4: Train Feature Engineering (Sampled S1 + competitors, global reverse_rank)
# ------------------------------------------------------------------------------
run_stage 4 "Feature extraction for sampled S1 + competitors" "
    ${PYTHON_BIN} ${SCRIPT_DIR}/s4_features.py --split train
"

# ------------------------------------------------------------------------------
# Stage 5: Model Training (LightGBM on full train sample)
# ------------------------------------------------------------------------------
run_stage 5 "LightGBM model training with validation early stopping" "
    ${PYTHON_BIN} ${SCRIPT_DIR}/s5_train.py --version v1
"

# ------------------------------------------------------------------------------
# Stage 6: Decision Layer Tuning (Exclusivity & threshold grid search)
# ------------------------------------------------------------------------------
run_stage 6 "Tune decision thresholds (val + competitors) and update config.py" "
    ${PYTHON_BIN} ${SCRIPT_DIR}/s6_tune.py --version v1
"

# ------------------------------------------------------------------------------
# Stage 7: Test Blocking (All Test S1 in chunks with resume)
# ------------------------------------------------------------------------------
run_stage 7 "Test candidate generation & blocking in chunks" "
    ${PYTHON_BIN} ${SCRIPT_DIR}/s3_block.py --split test
"

# ------------------------------------------------------------------------------
# Stage 8: Test Feature Engineering (Global reverse_rank & chunked features)
# ------------------------------------------------------------------------------
run_stage 8 "Feature extraction for all test candidates" "
    ${PYTHON_BIN} ${SCRIPT_DIR}/s4_features.py --split test
"

# ------------------------------------------------------------------------------
# Stage 9: Test Inference and Submission Generation
# ------------------------------------------------------------------------------
run_stage 9 "Test probability prediction and output TSV generation" "
    echo 'Running test inference with global exclusivity...'
    ${PYTHON_BIN} ${SCRIPT_DIR}/s7_predict.py --split test --version v1
    
    echo 'Formatting submission TSVs and performing sanity checks...'
    ${PYTHON_BIN} ${SCRIPT_DIR}/s8_write.py --split test --version v1
"

# ------------------------------------------------------------------------------
# Stage 10: Official Submission Validation
# ------------------------------------------------------------------------------
run_stage 10 "Validate final submission files using utils/validate_submission.py" "
    ${PYTHON_BIN} utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test
"

echo ""
echo "================================================================================"
echo "ALL REQUESTED STAGES COMPLETED SUCCESSFULLY!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "================================================================================"

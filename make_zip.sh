#!/usr/bin/env bash
# ==============================================================================
# ML Challenge 2026 — Packaging Script
# Usage:
#   ./make_zip.sh <team_name>             # Packages code + output TSVs + Documentation
#   ./make_zip.sh <team_name> --code-only # Packages code + Documentation only (no TSVs)
# ==============================================================================

set -eo pipefail

TEAM_NAME=""
CODE_ONLY=false

for arg in "$@"; do
    case "${arg}" in
        --code-only)
            CODE_ONLY=true
            ;;
        -h|--help)
            echo "Usage: $0 <team_name> [--code-only]"
            exit 0
            ;;
        *)
            if [ -z "${TEAM_NAME}" ]; then
                TEAM_NAME="${arg}"
            else
                echo "Unknown argument: ${arg}"
                echo "Usage: $0 <team_name> [--code-only]"
                exit 1
            fi
            ;;
    esac
done

if [ -z "${TEAM_NAME}" ]; then
    echo "Error: Team name required."
    echo "Usage: $0 <team_name> [--code-only]"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

PKG_DIR="pkg"
rm -rf "${PKG_DIR}"
mkdir -p "${PKG_DIR}"

if [ "${CODE_ONLY}" = true ]; then
    ZIP_NAME="${TEAM_NAME}_code.zip"
    echo "Creating code-only submission package: ${ZIP_NAME}"
else
    ZIP_NAME="${TEAM_NAME}_submission.zip"
    echo "Creating full submission package: ${ZIP_NAME}"
fi

# 1. Copy code directory (excluding __pycache__, .git, cache, etc.)
mkdir -p "${PKG_DIR}/code"
cp -r "code/business_entity_resolution" "${PKG_DIR}/code/"

# Clean any pycache or temporary files inside package
find "${PKG_DIR}/code" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PKG_DIR}/code" -type f -name "*.pyc" -delete 2>/dev/null || true
find "${PKG_DIR}/code" -type f -name ".DS_Store" -delete 2>/dev/null || true

# 2. Copy documentation template if present
if [ -f "Documentation_template.md" ]; then
    cp "Documentation_template.md" "${PKG_DIR}/"
fi

# 3. Copy output TSVs if not code-only
if [ "${CODE_ONLY}" = false ]; then
    mkdir -p "${PKG_DIR}/output"
    if [ -f "output/matching_results.tsv" ]; then
        cp "output/matching_results.tsv" "${PKG_DIR}/output/"
    else
        echo "Warning: output/matching_results.tsv not found!"
    fi

    if [ -f "output/candidate_pairs.tsv" ]; then
        cp "output/candidate_pairs.tsv" "${PKG_DIR}/output/"
    else
        echo "Warning: output/candidate_pairs.tsv not found!"
    fi
fi

# 4. Create zip archive
rm -f "${ZIP_NAME}"
cd "${PKG_DIR}"
zip -r "../${ZIP_NAME}" . -x "*.DS_Store" "*__pycache__*" "*.pyc" "*.git*" "*cache*" "*dataset*" "*logs*" "*submissions*" "*amlc_env*" "*.tar.gz" > /dev/null
cd ..

rm -rf "${PKG_DIR}"

# 5. Display zip size and contents
echo ""
echo "================================================================================"
echo "SUBMISSION ZIP CREATED: ${ZIP_NAME}"
echo "================================================================================"
ls -lh "${ZIP_NAME}"
echo ""
echo "Archive Contents:"
unzip -l "${ZIP_NAME}"
echo "================================================================================"

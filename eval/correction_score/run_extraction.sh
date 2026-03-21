#!/bin/bash
# Final-Layer Hidden-State Extraction Pipeline
# ===========================================
#
# Usage:
#   ./run_extraction.sh
#
# This script:
# 1. Extracts cached final-layer hidden states and yes/no logits
# 2. Saves the cache for downstream correction experiments

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load shared environment variables (HF_TOKEN, etc.) from ProbMed repo root
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

# ============================================
# CONFIGURATION
# ============================================
MARGIN_SCORES_FILE="${REPO_ROOT}/eval/vcd/results/vcd_analysis/margin_scores.json"
TEST_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
MODEL_NAME="chaoyinshe/llava-med-v1.5-mistral-7b-hf"

OUTPUT_ROOT="${SCRIPT_DIR}/results/correction_score"
CACHE_DIR="${OUTPUT_ROOT}/cache"

# ============================================
# Install dependencies
# ============================================
echo "Installing dependencies..."
pip install -q transformers accelerate bitsandbytes pillow tqdm matplotlib scikit-learn

# ============================================
# Create output directories
# ============================================
mkdir -p "${CACHE_DIR}"

# ============================================
# Step 1: Extract hidden states and logits
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Extracting hidden states"
echo "=========================================="

python "${SCRIPT_DIR}/extract_hidden_states.py" \
    --margin-scores-file "${MARGIN_SCORES_FILE}" \
    --test-file "${TEST_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${CACHE_DIR}" \
    --model-name "${MODEL_NAME}" \
    --load-8bit

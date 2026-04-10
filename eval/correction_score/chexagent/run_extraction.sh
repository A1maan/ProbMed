#!/bin/bash
# Final-Layer Hidden-State Extraction Pipeline for CheXagent-2-3b
# ================================================================
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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

# ============================================
# CONFIGURATION
# ============================================
INFERENCE_FILE="${REPO_ROOT}/eval/response_file/chexagent.json"
TEST_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
MODEL_NAME="StanfordAIMI/CheXagent-2-3b"
PYTHON=/venv/main/bin/python3

OUTPUT_ROOT="${SCRIPT_DIR}/../results/chexagent"
CACHE_DIR="${OUTPUT_ROOT}/hidden_states"

# ============================================
# Install dependencies
# ============================================
echo "Installing dependencies..."
pip install -q transformers accelerate pillow tqdm matplotlib scikit-learn

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

$PYTHON "${SCRIPT_DIR}/run_extract_hidden_states_batch.py" \
    --inference-file "${INFERENCE_FILE}" \
    --test-file "${TEST_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${CACHE_DIR}" \
    --model-name "${MODEL_NAME}" \
    --num-chunks 4

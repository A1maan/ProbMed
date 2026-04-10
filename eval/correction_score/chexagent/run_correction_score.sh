#!/bin/bash
# Final-Layer Correction Training Pipeline for CheXagent-2-3b
# ============================================================
#
# Usage:
#   ./run_correction_score.sh
#
# This script:
# 1. Loads a previously extracted hidden-state cache
# 2. Trains and evaluates raw model, bias shift, logistic regression,
#    and learned correction baselines

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
CACHE_DIR="${SCRIPT_DIR}/../results/chexagent/hidden_states"
RUN_DIR="${SCRIPT_DIR}/../results/chexagent/correction_model"
PYTHON=/venv/main/bin/python3

# ============================================
# Install dependencies
# ============================================
echo "Installing dependencies..."
pip install -q transformers accelerate pillow tqdm matplotlib scikit-learn

# ============================================
# Create output directories
# ============================================
mkdir -p "${RUN_DIR}"

# ============================================
# Step 1: Train and evaluate correction models
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Training correction baselines"
echo "=========================================="

$PYTHON "${SCRIPT_DIR}/train_lm_correction.py" \
    --cache-dir "${CACHE_DIR}" \
    --output-dir "${RUN_DIR}"

echo ""
echo "=========================================="
echo "DONE!"
echo "Cache loaded from: ${CACHE_DIR}"
echo "Results saved to: ${RUN_DIR}"
echo "=========================================="

#!/bin/bash
# Attention Analysis & Representation Probing Pipeline (Multi-GPU)
# =================================================================
#
# Usage:
#   ./run_attention_probing.sh
#
# Prerequisites:
#   - VCD margin scores must be computed first (run_vcd_analysis.sh)

set -e

# ============================================
# CONFIGURATION
# ============================================
MARGIN_SCORES_FILE="../vcd/results/vcd_analysis/margin_scores.json"
TEST_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
OUTPUT_DIR_ATTENTION="./results/attention_analysis"
OUTPUT_DIR_PROBING="./results/representation_probing"
NUM_SAMPLES=500     # Number of samples for representation probing
NUM_GPUS=4          # Number of GPUs to use
# Note: Attention analysis processes ALL valid pairs by default

# ============================================
# Install dependencies
# ============================================
echo "Installing dependencies..."
pip install -q transformers accelerate bitsandbytes pillow tqdm matplotlib scikit-learn

# ============================================
# Create output directories
# ============================================
mkdir -p ${OUTPUT_DIR_ATTENTION}
mkdir -p ${OUTPUT_DIR_PROBING}

# ============================================
# Step 1: Attention Analysis (Multi-GPU, ALL pairs)
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Attention Analysis (ALL pairs, ${NUM_GPUS} GPUs)"
echo "=========================================="

python run_attention_batch.py \
    --margin-scores-file ${MARGIN_SCORES_FILE} \
    --test-file ${TEST_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --output-dir ${OUTPUT_DIR_ATTENTION} \
    --num-chunks ${NUM_GPUS} \
    --load-8bit

# ============================================
# Step 2: Representation Probing (Single GPU - fast enough)
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Representation Probing (${NUM_SAMPLES} samples)"
echo "=========================================="

python representation_probing.py \
    --margin-scores-file ${MARGIN_SCORES_FILE} \
    --test-file ${TEST_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --output-dir ${OUTPUT_DIR_PROBING} \
    --num-samples ${NUM_SAMPLES} \
    --load-8bit

echo ""
echo "=========================================="
echo "DONE!"
echo "Attention results: ${OUTPUT_DIR_ATTENTION}"
echo "Probing results: ${OUTPUT_DIR_PROBING}"
echo "=========================================="
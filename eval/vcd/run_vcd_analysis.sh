#!/bin/bash
# VCD Margin Analysis Pipeline (Multi-GPU)
# =========================================
#
# Usage:
#   ./run_vcd_analysis.sh
#
# This script:
# 1. Computes margin scores using multiple GPUs
# 2. Analyzes and plots the results

set -e

# ============================================
# CONFIGURATION
# ============================================
QUESTION_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
OUTPUT_DIR="./results/vcd_analysis"
SAMPLE_RATIO=1.0  # Use 30% of data for experimentation
DOWNSAMPLE_SCALE=0.5  # 50% downsampling
NUM_GPUS=4  # Number of GPUs to use

# ============================================
# Install dependencies
# ============================================
echo "Installing dependencies..."
pip install -q transformers accelerate bitsandbytes pillow tqdm matplotlib scikit-learn

# ============================================
# Create output directory
# ============================================
mkdir -p ${OUTPUT_DIR}

# ============================================
# Step 1: Compute margin scores (Multi-GPU)
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Computing VCD margin scores (${NUM_GPUS} GPUs)"
echo "=========================================="

python run_vcd_analysis_batch.py \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --output-file ${OUTPUT_DIR}/margin_scores.json \
    --sample-ratio ${SAMPLE_RATIO} \
    --downsample-scale ${DOWNSAMPLE_SCALE} \
    --num-chunks ${NUM_GPUS} \
    --load-8bit

# ============================================
# Step 2: Analyze and plot results
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Analyzing margin scores"
echo "=========================================="

python analyze_margin_scores.py \
    --input-file ${OUTPUT_DIR}/margin_scores.json \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "DONE!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="
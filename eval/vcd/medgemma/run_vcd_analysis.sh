#!/bin/bash
# VCD Margin Analysis Pipeline for MedGemma-1.5-4b-IT (Multi-GPU)
# ================================================================
#
# Usage:
#   ./run_vcd_analysis.sh
#
# This script:
# 1. Computes margin scores using multiple GPUs
# 2. Analyzes and plots the results

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load HF_TOKEN (required for MedGemma)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

# ============================================
# CONFIGURATION
# ============================================
MODEL_NAME="google/medgemma-1.5-4b-it"
QUESTION_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
OUTPUT_DIR="${SCRIPT_DIR}/../results/medgemma"
SAMPLE_RATIO=1.0
DOWNSAMPLE_SCALE=0.5
NUM_GPUS=4
PYTHON=/venv/main/bin/python3

# ============================================
# Create output directory
# ============================================
mkdir -p ${OUTPUT_DIR}

# ============================================
# Step 1: Compute margin scores (Multi-GPU)
# ============================================
echo ""
echo "=========================================="
echo "Step 1: Computing VCD margin scores for MedGemma (${NUM_GPUS} GPUs)"
echo "=========================================="

$PYTHON ${SCRIPT_DIR}/run_vcd_analysis_batch.py \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --output-file ${OUTPUT_DIR}/margin_scores.json \
    --sample-ratio ${SAMPLE_RATIO} \
    --downsample-scale ${DOWNSAMPLE_SCALE} \
    --num-chunks ${NUM_GPUS} \
    --model-name ${MODEL_NAME}

# ============================================
# Step 2: Analyze and plot results
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Analyzing margin scores"
echo "=========================================="

$PYTHON ${SCRIPT_DIR}/analyze_margin_scores.py \
    --input-file ${OUTPUT_DIR}/margin_scores.json \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "DONE!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

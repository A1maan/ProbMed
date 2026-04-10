#!/bin/bash
# VCD Margin Analysis Pipeline (Multi-GPU)
# =========================================
#
# Usage:
#   ./run_vcd_analysis.sh [model_name]
#
# Examples:
#   ./run_vcd_analysis.sh llavamed
#   ./run_vcd_analysis.sh chexagent
#
# This script:
# 1. Computes margin scores using multiple GPUs
# 2. Analyzes and plots the results

set -e

# ============================================
# CONFIGURATION
# ============================================
MODEL=${1:-"chexagent"}

if [ "${MODEL}" == "llavamed" ]; then
    MODEL_NAME="chaoyinshe/llava-med-v1.5-mistral-7b-hf"
elif [ "${MODEL}" == "chexagent" ]; then
    MODEL_NAME="StanfordAIMI/CheXagent-2-3b"
else
    echo "Unknown model: ${MODEL}"
    echo "Available models: llavamed, chexagent"
    exit 1
fi

QUESTION_FILE="/workspace/ProbMed-Dataset/test/test.json"
IMAGE_FOLDER="/workspace/ProbMed-Dataset/test/"
OUTPUT_DIR="./results/${MODEL}"
SAMPLE_RATIO=1.0
DOWNSAMPLE_SCALE=0.5
NUM_GPUS=4

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
echo "Step 1: Computing VCD margin scores for ${MODEL} (${NUM_GPUS} GPUs)"
echo "=========================================="

PYTHON=/venv/main/bin/python3

$PYTHON run_vcd_analysis_batch.py \
    --question-file ${QUESTION_FILE} \
    --image-folder ${IMAGE_FOLDER} \
    --output-file ${OUTPUT_DIR}/margin_scores.json \
    --sample-ratio ${SAMPLE_RATIO} \
    --downsample-scale ${DOWNSAMPLE_SCALE} \
    --num-chunks ${NUM_GPUS} \
    --model-name ${MODEL_NAME} \
    --load-8bit

# ============================================
# Step 2: Analyze and plot results
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Analyzing margin scores"
echo "=========================================="

$PYTHON analyze_margin_scores.py \
    --input-file ${OUTPUT_DIR}/margin_scores.json \
    --output-dir ${OUTPUT_DIR}

echo ""
echo "=========================================="
echo "DONE!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=========================================="

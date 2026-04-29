#!/usr/bin/env bash
# LLaVA-Med intermediate correction — full 3-step pipeline
#
# Step 1: Extract hidden states at the target intermediate layer
# Step 2: Train correction model on those hidden states → saves direction + test split
# Step 3: Forwarded eval using the trained direction
#
# Usage:
#   bash run_intermediate_correction.sh
#   LAYER=12 bash run_intermediate_correction.sh   # sweep a different layer
#   NUM_GPUS=2 bash run_intermediate_correction.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"

LAYER="${LAYER:-15}"
STRENGTH="${STRENGTH:-1.0}"
MODE="${MODE:-probe}"
NUM_GPUS="${NUM_GPUS:-4}"
RESULTS_FILE="${RESULTS_FILE:-../response_file/llavamed.json}"
TEST_FILE="${TEST_FILE:-../../../test/test.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-../../../test}"

CACHE_DIR="llavamed/results/hidden_states_layer${LAYER}"
TRAIN_DIR="llavamed/results/correction_layer${LAYER}"
EVAL_DIR="llavamed/results/forwarded_layer${LAYER}/${MODE}_strength_${STRENGTH}"

echo "============================================================"
echo "LLaVA-Med Intermediate Correction  (layer=${LAYER})"
echo "============================================================"

# --- Step 1: Extract intermediate hidden states ---
echo ""
echo "[Step 1] Extracting intermediate hidden states at layer ${LAYER}..."
"$PYTHON_BIN" llavamed/run_extract_batch.py \
  --margin-scores-file "$RESULTS_FILE" \
  --test-file "$TEST_FILE" \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir "$CACHE_DIR" \
  --target-layer "$LAYER" \
  --num-chunks "$NUM_GPUS" \
  --load-8bit

# --- Step 2: Train correction model on intermediate hidden states ---
echo ""
echo "[Step 2] Training correction model on layer ${LAYER} hidden states..."
"$PYTHON_BIN" train_intermediate_correction.py \
  --cache-file "${CACHE_DIR}/hidden_states_cache.npz" \
  --output-dir "$TRAIN_DIR"

# --- Step 3: Forwarded eval using trained LR direction ---
echo ""
echo "[Step 3] Forwarded eval: hook at layer ${LAYER}, forward through remaining layers..."
"$PYTHON_BIN" llavamed/intermediate_correction_llavamed.py \
  --trained-direction "${TRAIN_DIR}/trained_lr_direction.npy" \
  --test-image-ids "${TRAIN_DIR}/test_image_ids.json" \
  --layer "$LAYER" \
  --strength "$STRENGTH" \
  --mode "$MODE" \
  --output-dir "$EVAL_DIR" \
  --load-8bit

echo ""
echo "Done. Results in:"
echo "  Cache:   $CACHE_DIR"
echo "  Train:   $TRAIN_DIR"
echo "  Eval:    $EVAL_DIR"

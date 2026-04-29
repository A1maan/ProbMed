#!/usr/bin/env bash
# MedGemma intermediate correction — full 3-step pipeline
#
# NOTE: MedGemma's probing peak is at the final layer (26), so for the
# intermediate experiment pick a layer that still has remaining layers
# after it — e.g. 15-20.  Default here is 18; sweep with LAYER=<N>.
#
# Usage:
#   bash run_intermediate_correction.sh
#   LAYER=15 bash run_intermediate_correction.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON:-python3}"

LAYER="${LAYER:-18}"
STRENGTH="${STRENGTH:-1.0}"
MODE="${MODE:-probe}"
NUM_GPUS="${NUM_GPUS:-4}"
RESULTS_FILE="${RESULTS_FILE:-../response_file/medgemma.json}"
TEST_FILE="${TEST_FILE:-../../../test/test.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-../../../test}"

CACHE_DIR="medgemma/results/hidden_states_layer${LAYER}"
TRAIN_DIR="medgemma/results/correction_layer${LAYER}"
EVAL_DIR="medgemma/results/forwarded_layer${LAYER}/${MODE}_strength_${STRENGTH}"

echo "============================================================"
echo "MedGemma Intermediate Correction  (layer=${LAYER})"
echo "============================================================"

echo ""
echo "[Step 1] Extracting intermediate hidden states at layer ${LAYER}..."
"$PYTHON_BIN" medgemma/run_extract_batch.py \
  --inference-file "$RESULTS_FILE" \
  --test-file "$TEST_FILE" \
  --image-folder "$IMAGE_FOLDER" \
  --output-dir "$CACHE_DIR" \
  --target-layer "$LAYER" \
  --num-chunks "$NUM_GPUS"

echo ""
echo "[Step 2] Training correction model on layer ${LAYER} hidden states..."
"$PYTHON_BIN" train_intermediate_correction.py \
  --cache-file "${CACHE_DIR}/hidden_states_cache.npz" \
  --output-dir "$TRAIN_DIR"

echo ""
echo "[Step 3] Forwarded eval: hook at layer ${LAYER}, forward through remaining layers..."
"$PYTHON_BIN" medgemma/intermediate_correction_medgemma.py \
  --trained-direction "${TRAIN_DIR}/trained_lr_direction.npy" \
  --test-image-ids "${TRAIN_DIR}/test_image_ids.json" \
  --layer "$LAYER" \
  --strength "$STRENGTH" \
  --mode "$MODE" \
  --output-dir "$EVAL_DIR"

echo ""
echo "Done. Results in:"
echo "  Cache:   $CACHE_DIR"
echo "  Train:   $TRAIN_DIR"
echo "  Eval:    $EVAL_DIR"

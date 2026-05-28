#!/usr/bin/env bash
# LLaVA-Med probe-direction intervention — multi-GPU alpha sweep at layer 15
#
# Usage:
#   bash run_probe_direction_intervention.sh
#   NUM_GPUS=2 LAYER=12 bash run_probe_direction_intervention.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHON="${PYTHON:-python3}"
PYTHON_BIN="$PYTHON"

LAYER="${LAYER:-15}"
NUM_GPUS="${NUM_GPUS:-4}"
ALPHA_MIN="${ALPHA_MIN:--2.0}"
ALPHA_MAX="${ALPHA_MAX:-2.0}"
ALPHA_STEP="${ALPHA_STEP:-0.25}"

echo "Installing dependencies..."
"$PYTHON_BIN" -m pip install -q transformers accelerate bitsandbytes pillow tqdm einops scikit-learn matplotlib requests albumentations pyarrow

echo "============================================================"
echo "LLaVA-Med Probe Direction Intervention  (layer=${LAYER}, gpus=${NUM_GPUS})"
echo "============================================================"

"$PYTHON_BIN" probe_direction_intervention/run_batch.py \
  --model llavamed \
  --layer "$LAYER" \
  --num-chunks "$NUM_GPUS" \
  --alpha-min "$ALPHA_MIN" \
  --alpha-max "$ALPHA_MAX" \
  --alpha-step "$ALPHA_STEP"

echo "Done."

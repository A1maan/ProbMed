#!/usr/bin/env bash
# MedGemma probe-direction intervention — multi-GPU alpha sweep at layer 26
#
# Usage:
#   bash run_probe_direction_intervention.sh
#   NUM_GPUS=2 LAYER=18 bash run_probe_direction_intervention.sh
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHON="${PYTHON:-python3}"
PYTHON_BIN="$PYTHON"

# Load HF_TOKEN for MedGemma if present
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
    set -a; source "${REPO_ROOT}/.env"; set +a
fi

LAYER="${LAYER:-26}"
NUM_GPUS="${NUM_GPUS:-4}"
ALPHA_MIN="${ALPHA_MIN:--2.0}"
ALPHA_MAX="${ALPHA_MAX:-2.0}"
ALPHA_STEP="${ALPHA_STEP:-0.25}"

echo "Installing dependencies..."
"$PYTHON_BIN" -m pip install -q transformers accelerate bitsandbytes pillow tqdm einops scikit-learn matplotlib requests albumentations pyarrow

echo "============================================================"
echo "MedGemma Probe Direction Intervention  (layer=${LAYER}, gpus=${NUM_GPUS})"
echo "============================================================"

"$PYTHON_BIN" probe_direction_intervention/run_batch.py \
  --model medgemma \
  --layer "$LAYER" \
  --num-chunks "$NUM_GPUS" \
  --alpha-min "$ALPHA_MIN" \
  --alpha-max "$ALPHA_MAX" \
  --alpha-step "$ALPHA_STEP"

echo "Done."

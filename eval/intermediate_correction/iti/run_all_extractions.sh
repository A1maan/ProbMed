#!/usr/bin/env bash
# Full Phase-1 extraction for all three models, back-to-back, 4 GPUs each, 10k samples.
# Outputs -> iti/results/<model>/iti_head_activations/head_activations.npz
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
RESPONSE=/workspace/ProbMed-Dataset/ProbMed/eval/response_file
TEST=/workspace/ProbMed-Dataset/test/test.json
IMG=/workspace/ProbMed-Dataset/test
MAX_SAMPLES=10000
NUM_CHUNKS=4

declare -A EIGHTBIT=( [llavamed]="--load-8bit" [chexagent]="" [medgemma]="" )
declare -A VENV=( [llavamed]="/venv/main/bin/python3" \
                  [chexagent]="/venv/chexagent/bin/python3" \
                  [medgemma]="/venv/main/bin/python3" )

for MODEL in llavamed chexagent medgemma; do
  echo "############################################################"
  echo "# EXTRACT: $MODEL  (venv ${VENV[$MODEL]}, $NUM_CHUNKS GPUs, $MAX_SAMPLES samples)"
  echo "############################################################"
  OUT="$HERE/results/$MODEL/iti_head_activations"
  PYTHONUNBUFFERED=1 PYTHON="${VENV[$MODEL]}" "${VENV[$MODEL]}" "$HERE/run_extract_batch.py" \
    --model "$MODEL" \
    --results-file "$RESPONSE/$MODEL.json" \
    --test-file "$TEST" --image-folder "$IMG" \
    --num-chunks "$NUM_CHUNKS" --max-samples "$MAX_SAMPLES" \
    ${EIGHTBIT[$MODEL]}
  if [ -f "$OUT/head_activations.npz" ]; then
    echo "+++ [$MODEL] EXTRACTION OK -> $OUT/head_activations.npz"
  else
    echo "!!! [$MODEL] EXTRACTION FAILED — see $OUT/extract_chunk*.log"
  fi
done
echo "############################################################"
echo "# ALL EXTRACTIONS COMPLETE"
echo "############################################################"

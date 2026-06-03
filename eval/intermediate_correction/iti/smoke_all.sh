#!/usr/bin/env bash
# Smoke test: run all three models end-to-end on a small slice.
# Extraction: 1 GPU, 800 samples (~90 images @ ~9 q/img — enough for a train/val/test split).
# Phase 2: probes + ITI inference capped at 60 test questions.
# Outputs go to iti/results/_smoke/<model>/ so they show up in the repo tree.
# Models default to all three; pass model names as args to run a subset:
#   bash smoke_all.sh chexagent medgemma
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
RESPONSE=/workspace/ProbMed-Dataset/ProbMed/eval/response_file
TEST=/workspace/ProbMed-Dataset/test/test.json
IMG=/workspace/ProbMed-Dataset/test
SMOKE="$HERE/results/_smoke"
MAX_SAMPLES=800
LIMIT_TEST=60

MODELS=("$@"); [ ${#MODELS[@]} -eq 0 ] && MODELS=(llavamed chexagent medgemma)
mkdir -p "$SMOKE"

# llavamed needs 8-bit (matches its baseline); chexagent/medgemma ignore the flag (always bf16)
declare -A EIGHTBIT=( [llavamed]="--load-8bit" [chexagent]="" [medgemma]="" )
# CheXagent's bundled modeling code is incompatible with transformers 5.9 (rope_scaling["type"]).
# It must run under /venv/chexagent (transformers 4.40). Other models use /venv/main.
declare -A VENV=( [llavamed]="/venv/main/bin/python3" \
                  [chexagent]="/venv/chexagent/bin/python3" \
                  [medgemma]="/venv/main/bin/python3" )

for MODEL in "${MODELS[@]}"; do
  echo "============================================================"
  echo "SMOKE: $MODEL  (venv: ${VENV[$MODEL]})"
  echo "============================================================"
  PY="${VENV[$MODEL]}"
  ACTS="$SMOKE/$MODEL/acts"
  OUT="$SMOKE/$MODEL/out"
  LOGDIR="$SMOKE/$MODEL/logs"
  rm -rf "$SMOKE/$MODEL"; mkdir -p "$LOGDIR"

  echo "--- [$MODEL] Phase 1: extract $MAX_SAMPLES samples (1 GPU)  log=$LOGDIR/extract.log ---"
  # Full unfiltered output -> log file; filtered tail -> terminal
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$HERE/extract_head_activations.py" \
    --model "$MODEL" \
    --results-file "$RESPONSE/$MODEL.json" \
    --test-file "$TEST" --image-folder "$IMG" \
    --output-dir "$ACTS" \
    --num-chunks 1 --chunk-idx 0 --max-samples "$MAX_SAMPLES" \
    ${EIGHTBIT[$MODEL]} > "$LOGDIR/extract.log" 2>&1
  grep -vE "Loading weights|[0-9]+/[0-9]+ \[" "$LOGDIR/extract.log" | tail -8
  if [ ! -f "$ACTS/head_activations.npz" ]; then
    echo "!!! [$MODEL] EXTRACTION FAILED — see $LOGDIR/extract.log"; continue
  fi

  echo "--- [$MODEL] Phase 2: probes + ITI inference (30 test qs)  log=$LOGDIR/train.log ---"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "$PY" "$HERE/train_iti_probes.py" \
    --model "$MODEL" \
    --results-file "$RESPONSE/$MODEL.json" \
    --test-file "$TEST" --image-folder "$IMG" \
    --activations-dir "$ACTS" --output-dir "$OUT" \
    --num-heads 8 --alpha 15 --limit-test "$LIMIT_TEST" \
    ${EIGHTBIT[$MODEL]} > "$LOGDIR/train.log" 2>&1
  grep -vE "Loading weights|training probes:|ITI alpha|baseline|[0-9]+/[0-9]+ \[" "$LOGDIR/train.log" | tail -20
  if [ -f "$OUT/results.json" ]; then
    echo "+++ [$MODEL] SMOKE OK — results.json written"
  else
    echo "!!! [$MODEL] PHASE 2 FAILED — see $LOGDIR/train.log"
  fi
done
echo "============================================================"
echo "SMOKE COMPLETE"

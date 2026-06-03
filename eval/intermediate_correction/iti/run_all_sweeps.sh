#!/usr/bin/env bash
# Full Phase-2 sweep for all three models, back-to-back.
# Grid: K in {16,32,48,64,80,96} x alpha in {5,10,15,20,25,30}  = 36 configs/model.
# Reuses head_activations.npz from iti/results/<model>/iti_head_activations/.
# Outputs -> iti/results/<model>/sweep/sweep_summary.csv (+ per-config subdirs).
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
RESPONSE=/workspace/ProbMed-Dataset/ProbMed/eval/response_file
TEST=/workspace/ProbMed-Dataset/test/test.json
IMG=/workspace/ProbMed-Dataset/test
KGRID="16,32,48,64,80,96"
AGRID="5,10,15,20,25,30"

declare -A EIGHTBIT=( [llavamed]="--load-8bit" [chexagent]="" [medgemma]="" )
declare -A VENV=( [llavamed]="/venv/main/bin/python3" \
                  [chexagent]="/venv/chexagent/bin/python3" \
                  [medgemma]="/venv/main/bin/python3" )

for MODEL in llavamed chexagent medgemma; do
  echo "############################################################"
  echo "# SWEEP: $MODEL  (venv ${VENV[$MODEL]}, K=$KGRID, alpha=$AGRID)"
  echo "############################################################"
  SWEEP="$HERE/results/$MODEL/sweep"
  LOG="$HERE/results/$MODEL/sweep.log"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 "${VENV[$MODEL]}" "$HERE/train_iti_probes.py" \
    --model "$MODEL" \
    --results-file "$RESPONSE/$MODEL.json" \
    --test-file "$TEST" --image-folder "$IMG" \
    --num-heads "$KGRID" --alpha "$AGRID" \
    ${EIGHTBIT[$MODEL]} > "$LOG" 2>&1
  if [ -f "$SWEEP/sweep_summary.csv" ]; then
    echo "+++ [$MODEL] SWEEP OK -> $SWEEP/sweep_summary.csv"
    echo "--- best config (tail of log) ---"
    grep -E "Baseline adv_paired|Best ITI config" "$LOG"
  else
    echo "!!! [$MODEL] SWEEP FAILED — see $LOG"
    tail -5 "$LOG"
  fi
done
echo "############################################################"
echo "# ALL SWEEPS COMPLETE"
echo "############################################################"

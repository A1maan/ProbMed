#!/usr/bin/env bash
# Sharded Phase-2 sweep for ONE model: 4 workers (one per GPU), each handles 1/4 of
# the test images across the full K x alpha grid, then merge.
#
# Usage:
#   bash run_sharded_sweep.sh <model> [KGRID] [AGRID]
#   bash run_sharded_sweep.sh llavamed
#   bash run_sharded_sweep.sh chexagent 48,64,80 10,15,20,25
set -u

MODEL="${1:?usage: run_sharded_sweep.sh <model> [KGRID] [AGRID]}"
# Focused grid (the productive region from honest_llama fig 4: mid K, low-mid alpha)
KGRID="${2:-32,48,64,80}"
AGRID="${3:-10,15,20,25}"
NUM_CHUNKS=4

HERE="$(cd "$(dirname "$0")" && pwd)"
RESPONSE=/workspace/ProbMed-Dataset/ProbMed/eval/response_file
TEST=/workspace/ProbMed-Dataset/test/test.json
IMG=/workspace/ProbMed-Dataset/test
SWEEP="$HERE/results/$MODEL/sweep"

declare -A EIGHTBIT=( [llavamed]="--load-8bit" [chexagent]="" [medgemma]="" )
declare -A VENV=( [llavamed]="/venv/main/bin/python3" \
                  [chexagent]="/venv/chexagent/bin/python3" \
                  [medgemma]="/venv/main/bin/python3" )
PY="${VENV[$MODEL]}"

echo "############################################################"
echo "# SHARDED SWEEP: $MODEL  K=$KGRID  alpha=$AGRID  ($NUM_CHUNKS GPUs)"
echo "############################################################"
mkdir -p "$SWEEP"

pids=()
for c in $(seq 0 $((NUM_CHUNKS-1))); do
  LOG="$SWEEP/shard${c}.log"
  echo "[shard $c] GPU $c  log=$LOG"
  CUDA_VISIBLE_DEVICES=$c PYTHONUNBUFFERED=1 "$PY" "$HERE/train_iti_probes.py" \
    --model "$MODEL" \
    --results-file "$RESPONSE/$MODEL.json" \
    --test-file "$TEST" --image-folder "$IMG" \
    --num-heads "$KGRID" --alpha "$AGRID" \
    --num-chunks "$NUM_CHUNKS" --chunk-idx "$c" \
    ${EIGHTBIT[$MODEL]} > "$LOG" 2>&1 &
  pids+=($!)
done

echo "Waiting for $NUM_CHUNKS shards (PIDs: ${pids[*]})..."
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[shard $i] OK"; else echo "[shard $i] FAILED (see $SWEEP/shard${i}.log)"; fail=1; fi
done

if [ "$fail" -ne 0 ]; then
  echo "!!! Some shards failed — not merging. Inspect logs above."; exit 1
fi

echo "--- merging shards ---"
"$PY" "$HERE/merge_sweep.py" --sweep-dir "$SWEEP" --num-chunks "$NUM_CHUNKS"
echo "+++ [$MODEL] SHARDED SWEEP COMPLETE -> $SWEEP/sweep_summary.csv"

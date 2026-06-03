#!/usr/bin/env bash
# Confirmation: full test set, baseline + negative alphas at one K, 4-GPU sharded, then merge.
# Usage: bash run_confirm.sh <model> <K> <alphas>
#   bash run_confirm.sh llavamed 48 0,-5,-10,-15
set -u
MODEL="${1:?usage: run_confirm.sh <model> <K> <alphas>}"
K="${2:?need K}"
ALPHAS="${3:?need alphas e.g. 0,-5,-10,-15}"
NUM_CHUNKS=4
HERE="$(cd "$(dirname "$0")" && pwd)"
SWEEP="$HERE/results/$MODEL/sweep"

declare -A EIGHTBIT=( [llavamed]="--load-8bit" [chexagent]="" [medgemma]="" )
declare -A VENV=( [llavamed]="/venv/main/bin/python3" \
                  [chexagent]="/venv/chexagent/bin/python3" \
                  [medgemma]="/venv/main/bin/python3" )
PY="${VENV[$MODEL]}"

echo "=== CONFIRM: $MODEL K=$K alphas=$ALPHAS (4 GPUs) ==="
rm -f "$SWEEP"/records_confirm_chunk*.json
pids=()
for c in 0 1 2 3; do
  LOG="$SWEEP/confirm_shard${c}.log"
  CUDA_VISIBLE_DEVICES=$c PYTHONUNBUFFERED=1 "$PY" "$HERE/confirm_neg_alpha.py" \
    --model "$MODEL" --K "$K" --alphas "$ALPHAS" \
    --num-chunks $NUM_CHUNKS --chunk-idx $c ${EIGHTBIT[$MODEL]} > "$LOG" 2>&1 &
  pids+=($!)
done
echo "Waiting on PIDs: ${pids[*]}"
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[shard $i] OK"; else echo "[shard $i] FAILED ($SWEEP/confirm_shard${i}.log)"; fail=1; fi
done
[ "$fail" -ne 0 ] && { echo "Some shards failed."; exit 1; }

echo "--- merging confirm records ---"
"$PY" "$HERE/merge_confirm.py" --sweep-dir "$SWEEP"

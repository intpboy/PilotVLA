#!/usr/bin/env bash
# Swap-benchmark rollouts: all scenes x all distractors x EPS episodes,
# sharded over NGPU GPUs, for each condition in CONDS.
#
# Required:
#   OUT=...            output directory
# Optional:
#   CONDS="none noiseopt"   conditions (none | guide | noiseopt | bon)
#   NGPU=1                  number of GPUs to shard over
#   EPS=20                  episodes per cell
#   EXTRA="..."             extra flags, e.g. "--posswap" or
#                           "--release_replans 10" (fixed-horizon release)
#
# Example (paper setting, instruction swap, fixed-horizon release):
#   OUT=out/swaproll CONDS="none noiseopt" NGPU=4 \
#     EXTRA="--release_replans 10" bash scripts/run_swap.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${OUT:?set OUT to the output directory}"
CONDS=${CONDS:-"none noiseopt"}
NGPU=${NGPU:-1}
EPS=${EPS:-20}
EXTRA=${EXTRA:-}

for COND in $CONDS; do
  pids=()
  for g in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$g python "$REPO_ROOT/tano/rollout_swap.py" \
      --out "$OUT" --eps "$EPS" --cond "$COND" $EXTRA \
      --gpu "$g" --ngpu "$NGPU" &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  # Rollouts are resumable (meta.json skip): if a shard died, a single
  # sweep process picks up whatever is missing.
  if [ $fail -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/tano/rollout_swap.py" \
      --out "$OUT" --eps "$EPS" --cond "$COND" $EXTRA --gpu 0 --ngpu 1
  fi
  echo "[run_swap] $COND done"
done

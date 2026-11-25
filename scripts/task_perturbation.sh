#!/usr/bin/env bash
# Task perturbation (LIBERO-Object): the instruction asks for an in-scene
# distractor instead of the object the scene was trained on.
# 10 scenes x 5 distractors x EPS episodes; success = asked object in the basket.
#
#   bash scripts/task_perturbation.sh rollout   # baseline + PilotVLA rollouts
#   bash scripts/task_perturbation.sh stats     # paired statistics
#
# Env: OUT (default out/task_perturbation), NGPU (default 1), EPS (default 20).
# Paper setting: 10 Adam steps, lr 0.5, K=30, fixed-horizon release after 10 replans.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT=${OUT:-out/task_perturbation}
EPS=${EPS:-20}

case "${1:-}" in
  rollout)
    run_sharded rollout_libero.py --out "$OUT" --eps "$EPS" --cond none
    run_sharded rollout_libero.py --out "$OUT" --eps "$EPS" --cond noiseopt \
      --opt_steps 10 --opt_lr 0.5 --K 30 --release_replans 10 ;;
  stats)
    python "$REPO_ROOT/tools/paired_stats.py" --root "$OUT" --key asked_in_basket \
      --none-tag none --steer-tag noiseopt10x0.5-r10 --name "task perturbation" ;;
  *) echo "usage: $0 {rollout|stats}" >&2; exit 1 ;;
esac

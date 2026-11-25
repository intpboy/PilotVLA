#!/usr/bin/env bash
# Position perturbation (LIBERO-Object): the instruction is unchanged, but the
# target and an in-scene distractor exchange their initial positions.
# 10 scenes x 5 distractors x EPS episodes; success = the env's own signal.
#
#   bash scripts/position_perturbation.sh rollout   # baseline + PilotVLA rollouts
#   bash scripts/position_perturbation.sh stats     # paired statistics
#
# Env: OUT (default out/position_perturbation), NGPU (default 1), EPS (default 20).
# Paper setting: 10 Adam steps, lr 0.5, K=30, fixed-horizon release after 10 replans.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT=${OUT:-out/position_perturbation}
EPS=${EPS:-20}

case "${1:-}" in
  rollout)
    run_sharded rollout_libero.py --out "$OUT" --eps "$EPS" --posswap --cond none
    run_sharded rollout_libero.py --out "$OUT" --eps "$EPS" --posswap --cond noiseopt \
      --opt_steps 10 --opt_lr 0.5 --K 30 --release_replans 10 ;;
  stats)
    python "$REPO_ROOT/tools/paired_stats.py" --root "$OUT" --key env_success \
      --none-tag posnone --steer-tag posnoiseopt10x0.5-r10 --name "position perturbation" ;;
  *) echo "usage: $0 {rollout|stats}" >&2; exit 1 ;;
esac

#!/usr/bin/env bash
# LIBERO-Plus (libero_spatial): language, layout and robot-initial-pose axes.
# Task instances: annotations/liberoplus/. Success = the env's own signal.
#
#   LIBERO_PLUS_ROOT=/path/to/LIBERO-plus bash scripts/liberoplus.sh setup   # once
#   bash scripts/liberoplus.sh rollout   # baseline + PilotVLA rollouts, all AXES
#   bash scripts/liberoplus.sh stats     # paired statistics per axis
#
# Env: OUT (default out/liberoplus), NGPU (default 1), EPS (default 10),
#      AXES (default "language layout robotinit").
# Paper setting: 10 Adam steps, lr 0.5, K=30, proximity release (8 cm).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
OUT=${OUT:-out/liberoplus}
EPS=${EPS:-10}
AXES=${AXES:-"language layout robotinit"}

case "${1:-}" in
  setup)
    python "$REPO_ROOT/tools/setup_liberoplus.py" --root "${LIBERO_PLUS_ROOT:?set LIBERO_PLUS_ROOT}" ;;
  rollout)
    for AXIS in $AXES; do
      run_sharded rollout_liberoplus.py --axis "$AXIS" --out "$OUT" --eps "$EPS" --cond none
      run_sharded rollout_liberoplus.py --axis "$AXIS" --out "$OUT" --eps "$EPS" --cond noiseopt \
        --opt_steps 10 --opt_lr 0.5 --K 30
    done ;;
  stats)
    for AXIS in $AXES; do
      python "$REPO_ROOT/tools/paired_stats.py" --root "$OUT/$AXIS" --key success \
        --none-tag none --steer-tag noiseopt10x0.5 --name "LIBERO-Plus $AXIS"
    done ;;
  *) echo "usage: $0 {setup|rollout|stats}" >&2; exit 1 ;;
esac

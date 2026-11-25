# Shared helpers for the pipeline scripts (sourced, not executed).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NGPU=${NGPU:-1}

# run_sharded <tool.py> <args...>
# One process per GPU (--gpu g --ngpu NGPU). Rollouts are resumable (episodes
# with an existing meta.json are skipped), so if a shard dies, a single sweep
# process afterwards picks up whatever is missing.
run_sharded() {
  local tool=$1; shift
  local pids=() fail=0 g pid
  for g in $(seq 0 $((NGPU - 1))); do
    CUDA_VISIBLE_DEVICES=$g python "$REPO_ROOT/tools/$tool" "$@" \
      --gpu "$g" --ngpu "$NGPU" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [ $fail -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=0 python "$REPO_ROOT/tools/$tool" "$@" --gpu 0 --ngpu 1
  fi
}

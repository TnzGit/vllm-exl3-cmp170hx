#!/usr/bin/env bash
# R0 no-draft C1 context sweep: one fresh engine per context cell.
#
# usage: r0_run_context_sweep.sh [tag] [context...]
#
# Each cell starts a fresh engine at the given max_model_len, waits for health,
# runs one exact-token request, records the cell, then stops the engine and
# records the Xid delta.
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-cmp170hx}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PORT="${PORT:-8002}"
TAG="${1:-no-draft}"
shift || true
CONTEXTS=("$@")
if [[ ${#CONTEXTS[@]} -eq 0 ]]; then
  CONTEXTS=(4096 32768 65536 160000 250000)
fi

export PATH="$V/bin:$PATH"
export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export VIRTUAL_ENV="$V"
export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_EXL3_NGRAM_KERNEL=ext

mkdir -p "$R/results"

xid_count() {
  # Count GPU Xid events only: "NVRM: Xid (PCI:...)" lines from the kernel ring
  # buffer. A bare "xid" grep also matches unrelated driver messages (for
  # example the r8169 NIC prints "XID 541"), which would fake a nonzero delta.
  # grep -c exits non-zero on no match, so force success and keep one integer.
  local n
  n=$( { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9' )
  [ -n "$n" ] || n=0
  echo "$n"
}

wait_healthy() {
  # /health can answer while the old process is still shutting down, so also
  # require the model list to resolve: that only happens after weights are on
  # the device and the engine finished profiling/warmup.
  local port=$1 deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:${port}/health" >/dev/null 2>&1 \
       && curl -s -m 10 "http://127.0.0.1:${port}/v1/models" | grep -q '"id"'; then
      return 0
    fi
    sleep 5
  done
  return 1
}

stop_engine() {
  pkill -f "serve_cmp170hx_qwen_firstboot" 2>/dev/null || true
  pkill -f "r0_serve_env.sh" 2>/dev/null || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 8
}

for ctx in "${CONTEXTS[@]}"; do
  echo "=== cell context=${ctx} tag=${TAG} ==="
  stop_engine
  xid_before=$(xid_count)

  LOG="$R/results/serve_${TAG}_${ctx}.log"
  MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$ctx" PORT="$PORT" \
    MAX_NUM_SEQS=1 nohup bash "$REPO/tools/r0_serve_env.sh" > "$LOG" 2>&1 &
  SRV_PID=$!

  if ! wait_healthy "$PORT"; then
    echo "RESULT: capacity/startup failure at context=${ctx}; see $LOG"
    echo "{\"tag\":\"${TAG}\",\"context_target\":${ctx},\"status\":\"startup_failed\"}" \
      > "$R/results/cell_${TAG}_${ctx}.json"
    stop_engine
    continue
  fi

  "$V/bin/python" "$REPO/tools/r0_bench_context.py" \
    --context "$ctx" --max-tokens 128 --tag "${TAG}-${ctx}" \
    --out "$R/results/cell_${TAG}_${ctx}.json" || true

  xid_after=$(xid_count)
  delta=$(( xid_after - xid_before ))
  printf '{"context":%s,"xid_before":%s,"xid_after":%s,"xid_delta":%s}\n' \
    "$ctx" "$xid_before" "$xid_after" "$delta" > "$R/results/xid_${TAG}_${ctx}.json"
  echo "xid_delta=${delta}"
  stop_engine
done

echo "sweep complete"

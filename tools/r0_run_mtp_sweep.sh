#!/usr/bin/env bash
# R0 MTP sweep: one fresh engine per CONFIGURATION (k), reusing that engine
# across the context ladder. See docs for the reuse/sentinel contract.
#
# usage: r0_run_mtp_sweep.sh [k...]
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-cmp170hx}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PORT="${PORT:-8002}"
CONTEXTS="${CONTEXTS:-4096,32768,65536,160000,250000}"
KS=("$@")
if [[ ${#KS[@]} -eq 0 ]]; then
  KS=(1 2 3)
fi

export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_EXL3_NGRAM_KERNEL=ext

mkdir -p "$R/results"

wait_healthy() {
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
  pkill -f "r0_serve_mtp.sh" 2>/dev/null || true
  pkill -f "r0_serve_env.sh" 2>/dev/null || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 8
}

for k in "${KS[@]}"; do
  label=$([ "$k" = "0" ] && echo "nodraft" || echo "mtp-k${k}")
  echo "=== config ${label} (fresh engine) ==="
  stop_engine

  LOG="$R/results/serve_${label}.log"
  MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-250000}" MAX_NUM_SEQS=1 \
    NUM_SPEC_TOKENS="$k" PORT="$PORT" \
    nohup bash "$REPO/tools/r0_serve_mtp.sh" > "$LOG" 2>&1 &

  if ! wait_healthy "$PORT"; then
    echo "RESULT: startup failed for ${label}; see $LOG"
    printf '{"config":"%s","status":"startup_failed"}\n' "$label" \
      > "$R/results/sweep_${label}.json"
    stop_engine
    continue
  fi

  "$V/bin/python" "$REPO/tools/r0_sweep_reuse.py" \
    --port "$PORT" --tag "$label" --contexts "$CONTEXTS" \
    --out "$R/results/sweep_${label}.json" || true

  stop_engine
done

echo "mtp sweep complete"

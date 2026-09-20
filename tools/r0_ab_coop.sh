#!/usr/bin/env bash
# Coop-MoE A/B: fresh engine per config, profiler OFF, 4K C1.
#
# usage: r0_ab_coop.sh
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-amdahl}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
CONTEXT="${CONTEXT:-4096}"
OUT="${OUT:-$R/results/coop-ab}"
mkdir -p "$OUT"

export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"

xid_now() { { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9'; }

wait_healthy() {
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 http://127.0.0.1:8002/health >/dev/null 2>&1 \
       && curl -s -m 10 http://127.0.0.1:8002/v1/models | grep -q '"id"'; then
      return 0
    fi
    sleep 5
  done
  return 1
}

stop_engine() {
  pkill -9 -f serve_cmp170hx_qwen_firstboot 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
}

for coop in 0 1; do
  label="coop${coop}"
  echo "=== ${label} (fresh engine, profiler OFF) ==="
  stop_engine
  XID0=$(xid_now); XID0=${XID0:-0}
  LOG="$OUT/serve_${label}.log"

  # PATH/CUDA_HOME are already exported above and inherited by the child, so
  # pass only the per-config variables here.
  VLLM_EXL3_COOP="$coop" MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL=0.92 \
    MAX_MODEL_LEN="$CONTEXT" MAX_NUM_SEQS=1 PORT=8002 NUM_SPEC_TOKENS=0 \
    setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" > "$LOG" 2>&1 < /dev/null &

  if ! wait_healthy; then
    echo "startup FAILED for ${label}"; tail -5 "$LOG"; stop_engine; continue
  fi

  echo "markers: PRESCAN=$(grep -c 'PRESCAN ready' "$LOG") fallback=$(grep -c 'staging fallback (no arena plan)' "$LOG")"
  grep -m1 "Loading weights took" "$LOG" | grep -oE "[0-9.]+ seconds" || true

  "$V/bin/python" "$REPO/tools/r0_ab_cell.py" --context "$CONTEXT" --repeats 3 \
    --tag "$label" --out "$OUT/cell_${label}.json" | tail -12
  "$V/bin/python" "$REPO/tools/r0_smoke.py" --repeats 2 --max-tokens 8 \
    --out "$OUT/smoke_${label}.json" 2>&1 | tail -4

  XID1=$(xid_now); XID1=${XID1:-0}
  echo "xid_delta=$((XID1 - XID0))" | tee "$OUT/xid_${label}.txt"
  stop_engine
done
echo "coop A/B complete"

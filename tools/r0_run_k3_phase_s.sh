#!/usr/bin/env bash
# Phase S only: k=3 long-context sweep on ONE fresh engine.
# References from Phase R are reused; this does not regenerate them.
#
# usage: r0_run_k3_phase_s.sh [outdir]
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-k3}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/mtp-k3}"
MAXTOK="${MAXTOK:-256}"
CONTEXTS="${CONTEXTS:-32000 65000 126000 160000 200000 240000}"
# 256000 needs 7.24 GiB KV but only 7.03 GiB is available on this card at
# gpu_memory_utilization=0.92; the engine's own estimate is 246400.
MAXLEN="${MAXLEN:-246000}"

export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
mkdir -p "$OUT"

xid_now() { { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9'; }

stop_engine() {
  pkill -9 -f serve_cmp170hx_qwen_firstboot 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
}

# Wait for the engine to become idle instead of failing on the first sample:
# the tail of the previous cell's request can still be draining when the next
# guard runs. Idle is required before any measurement, but it is allowed to
# settle for up to 120 s first.
guard() {
  local deadline=$((SECONDS + 120))
  local st run wait_
  while (( SECONDS < deadline )); do
    st=$(curl -s -m 10 http://127.0.0.1:8002/metrics 2>/dev/null || true)
    run=$(echo "$st" | grep -E '^vllm:num_requests_running' | awk '{s+=$NF} END {print s+0}')
    wait_=$(echo "$st" | grep -E '^vllm:num_requests_waiting\{' | awk '{s+=$NF} END {print s+0}')
    if [[ "$run" == "0" && "$wait_" == "0" ]]; then
      return 0
    fi
    sleep 3
  done
  echo "GUARD FAIL: still running=$run waiting=$wait_ after 120s"
  return 1
}

stop_engine
XID0=$(xid_now); XID0=${XID0:-0}
LOG="$OUT/serve_k3.log"
VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL=0.92 \
  MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT=8002 NUM_SPEC_TOKENS=3 \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" > "$LOG" 2>&1 < /dev/null &

deadline=$((SECONDS + 2400))
ready=0
while (( SECONDS < deadline )); do
  if grep -q "Application startup complete" "$LOG" 2>/dev/null; then ready=1; break; fi
  if grep -q "Engine core initialization failed" "$LOG" 2>/dev/null; then break; fi
  sleep 5
done
if (( ready == 0 )); then
  echo "k3 ENGINE FAILED TO START (MAXLEN=$MAXLEN)"
  grep -iE "ValueError|RuntimeError" "$LOG" | grep -v "trellis staging" | tail -3
  stop_engine; exit 1
fi

APID=$(pgrep -f "vllm serve"|head -1); EPID=$(pgrep -f "VLLM::EngineCore"|head -1)
echo "k3 engine api=$APID engine=$EPID pwd=$(readlink /proc/$APID/cwd) maxlen=$MAXLEN"
echo "env: $(tr '\0' '\n' < /proc/$APID/environ | grep -E 'VLLM_EXL3_COOP' | head -1)"
echo "spec: $(tr '\0' ' ' < /proc/$APID/cmdline | grep -o 'speculative-config [^ ]*' | head -1)"
echo "markers: PRESCAN=$(grep -c 'PRESCAN ready' "$LOG") fallback=$(grep -c 'staging fallback (no arena plan)' "$LOG")"
echo "kv: $(grep -m1 'GPU KV cache size' "$LOG" | grep -oE '[0-9,]+ tokens' || true)"

guard || true
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context 4096 \
  --max-tokens "$MAXTOK" --repeats 3 --tag "k3-sentinel-in" \
  --out "$OUT/cell_k3_sentinel_in.json" > /dev/null 2>&1 || echo "sentinel_in failed"

for ctx in $CONTEXTS; do
  guard || { echo "guard failed before ctx=$ctx"; continue; }
  echo "--- cell ctx=$ctx ---"
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context "$ctx" \
    --max-tokens "$MAXTOK" --repeats 3 --tag "k3-ctx$ctx" \
    --out "$OUT/cell_k3_ctx${ctx}.json" > /dev/null 2>&1 \
    && echo "  ctx=$ctx done" || echo "  ctx=$ctx FAILED"
done

guard || true
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context 4096 \
  --max-tokens "$MAXTOK" --repeats 3 --tag "k3-sentinel-out" \
  --out "$OUT/cell_k3_sentinel_out.json" > /dev/null 2>&1 || echo "sentinel_out failed"

for ctx in 4096 160000 240000; do
  ref="$OUT/parity_ref_${ctx}.json"
  [ -f "$ref" ] || continue
  echo "--- parity ctx=$ctx ---"
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context "$ctx" \
    --max-tokens "$MAXTOK" --repeats 1 --tag "k3-parity-$ctx" \
    --parity-reference "$ref" --out "$OUT/parity_k3_${ctx}.json" > /dev/null 2>&1 \
    && echo "  parity $ctx done" || echo "  parity $ctx failed"
done

XID1=$(xid_now); XID1=${XID1:-0}
echo "{\"xid_before\":$XID0,\"xid_after\":$XID1,\"xid_delta\":$((XID1-XID0))}" > "$OUT/xid.json"
echo "xid_delta=$((XID1-XID0))"
stop_engine
echo "=== phase S complete ==="

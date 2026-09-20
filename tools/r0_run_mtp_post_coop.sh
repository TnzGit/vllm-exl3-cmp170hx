#!/usr/bin/env bash
# Post-COOP MTP requalification: fresh engine per (COOP, k) configuration.
#
# usage: r0_run_mtp_post_coop.sh <coop> <k> [outdir]
#   coop=0/1  k=0(no-draft)/1/2/3
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-mtp2}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
COOP="${1:?usage: r0_run_mtp_post_coop.sh <coop> <k> [outdir]}"
K="${2:?usage: r0_run_mtp_post_coop.sh <coop> <k> [outdir]}"
OUT="${3:-$R/results/mtp-post-coop}"
MAXLEN="${MAXLEN:-4096}"
MAXTOK="${MAXTOK:-128}"
CTX="${CTX:-4096}"

export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"

label="coop${COOP}-k${K}"
mkdir -p "$OUT"

xid_now() { { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9'; }

stop_engine() {
  pkill -9 -f serve_cmp170hx_qwen_firstboot 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
}

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

echo "=== ${label} (fresh engine) ==="
stop_engine
XID0=$(xid_now); XID0=${XID0:-0}
LOG="$OUT/serve_${label}.log"

VLLM_EXL3_COOP="$COOP" MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL=0.92 \
  MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT=8002 NUM_SPEC_TOKENS="$K" \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" > "$LOG" 2>&1 < /dev/null &

if ! wait_healthy; then
  echo "startup FAILED for ${label}"
  grep -iE "error|Traceback|ValueError" "$LOG" | grep -v "trellis staging" | tail -4
  stop_engine
  exit 1
fi

APID=$(pgrep -f "vllm serve" | head -1)
EPID=$(pgrep -f "VLLM::EngineCore" | head -1)
echo "api=$APID engine=$EPID pwd=$(readlink /proc/$APID/cwd)"
echo "env: $(tr '\0' '\n' < /proc/$APID/environ | grep -E 'VLLM_EXL3_COOP' | head -1)"
echo "cmdline spec: $(tr '\0' ' ' < /proc/$APID/cmdline | grep -o 'speculative-config [^ ]*' | head -1)"
echo "markers: PRESCAN=$(grep -c 'PRESCAN ready' "$LOG") direct_plan=$(grep -c 'mode=direct_plan' "$LOG") fallback=$(grep -c 'staging fallback (no arena plan)' "$LOG")"
grep -m1 "Model loading took" "$LOG" | grep -oE "[0-9.]+ seconds" || true

"$V/bin/python" "$REPO/tools/r0_mtp_post_coop_cell.py" --port 8002 \
  --context "$CTX" --max-tokens "$MAXTOK" --repeats 3 --tag "$label" \
  --out "$OUT/cell_${label}.json" | tail -30 || echo "cell failed"

XID1=$(xid_now); XID1=${XID1:-0}
printf '{"config":"%s","xid_before":%s,"xid_after":%s,"xid_delta":%s}\n' \
  "$label" "$XID0" "$XID1" "$((XID1 - XID0))" > "$OUT/xid_${label}.json"
echo "xid_delta=$((XID1 - XID0))"
stop_engine
echo "=== ${label} complete ==="

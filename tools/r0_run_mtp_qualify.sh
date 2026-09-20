#!/usr/bin/env bash
# MTP k qualification: fresh engine per k, contexts within a k reuse it.
#
# usage: r0_run_mtp_qualify.sh <k> [contexts...]
#
# Records acceptance counters per cell via tools/r0_mtp_cell.py, and a 4K
# sentinel at both ends so drift is measurable.
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-mtp}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
PORT="${PORT:-8002}"
MAXLEN="${MAXLEN:-250000}"

K="${1:?usage: r0_run_mtp_qualify.sh <k> [contexts...]}"
shift || true
CTX=("$@")
if [[ ${#CTX[@]} -eq 0 ]]; then CTX=(4096 126000); fi

export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"

label=$([ "$K" = "0" ] && echo "nodraft" || echo "mtp-k${K}")
mkdir -p "$R/results/mtp"

xid_now() { { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9'; }

wait_healthy() {
  local deadline=$((SECONDS + 1800))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 \
       && curl -s -m 10 "http://127.0.0.1:${PORT}/v1/models" | grep -q '"id"'; then
      return 0
    fi
    sleep 5
  done
  return 1
}

stop_engine() {
  pkill -9 -f r0_serve_mtp.sh 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
}

echo "=== config ${label} (fresh engine, max_model_len=${MAXLEN}) ==="
stop_engine
XID0=$(xid_now); XID0=${XID0:-0}

LOG="$R/results/mtp/serve_${label}.log"
MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAXLEN" \
  MAX_NUM_SEQS=1 NUM_SPEC_TOKENS="$K" PORT="$PORT" \
  nohup setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" > "$LOG" 2>&1 < /dev/null &

if ! wait_healthy; then
  echo "startup FAILED for ${label}; see $LOG"
  grep -iE "error|Traceback|ValueError" "$LOG" | grep -v "trellis staging" | tail -5
  stop_engine
  exit 1
fi

# --- MTP hidden-buffer device hard gate (upstream #56742) ---
# The probe script instruments the installed vLLM so the boot log carries the
# real device the engine uses; collected below from the server log.
if [ "$K" != "0" ]; then
  grep -m1 "EXL3_MTP_HIDDEN_BUFFER_DEVICE" "$LOG" \
    | tee "$R/results/mtp/hidden_buffer_${label}.txt" \
    || echo "WARN: no hidden-buffer device line found" \
    | tee "$R/results/mtp/hidden_buffer_${label}.txt"
fi

echo "=== markers ==="
for m in "EXL3 trellis PRESCAN ready" "mode=direct_plan" "staging fallback (no arena plan)"; do
  echo "$m = $(grep -c "$m" "$LOG" 2>/dev/null || echo 0)"
done
grep -m1 "Loading weights took" "$LOG" | grep -oE "[0-9.]+ seconds" || true
grep -m1 "Model loading took" "$LOG" | grep -oE "[0-9.]+ seconds" || true

run_cell() {
  local ctx=$1 tag=$2
  echo "--- cell ${tag} context=${ctx} ---"
  "$V/bin/python" "$REPO/tools/r0_mtp_cell.py" --port "$PORT" --context "$ctx" \
    --max-tokens "${MAXTOK:-128}" --tag "${label}-${tag}" \
    --out "$R/results/mtp/cell_${label}_${tag}.json" || echo "cell failed"
}

run_cell 4096 sentinel_in
for c in "${CTX[@]}"; do
  [ "$c" = "4096" ] && continue
  run_cell "$c" "ctx${c}"
done
run_cell 4096 sentinel_out

XID1=$(xid_now); XID1=${XID1:-0}
echo "{\"config\":\"${label}\",\"xid_before\":${XID0},\"xid_after\":${XID1},\"xid_delta\":$((XID1-XID0))}" \
  > "$R/results/mtp/xid_${label}.json"
echo "xid_delta=$((XID1-XID0))"

stop_engine
echo "=== ${label} complete ==="

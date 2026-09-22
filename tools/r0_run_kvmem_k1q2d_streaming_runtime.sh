#!/usr/bin/env bash
# K1-Q2D scheduler-owned, CPU-authoritative bounded QSA runtime qualification.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1q2d-streaming-runtime}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to exact Q2D streaming head}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-161000}"
MAXTOK="${K1Q2D_MAX_TOKENS:-512}"
MAX_BATCHED="${K1Q2D_MAX_NUM_BATCHED_TOKENS:-1024}"
Q2E_PROFILE="${K1Q2E_PROFILE:-0}"
Q2E_TRACE_MAX_BYTES="${K1Q2E_TRACE_MAX_BYTES:-268435456}"
if [[ "$GPU_MEM_UTIL" != "0.92" || "$MAXLEN" != "161000" || "$MAX_BATCHED" != "1024" ]]; then
  echo "REFUSE: Q2D requires gpu=0.92 maxlen=161000 chunk=1024" >&2
  exit 2
fi
if [[ "$Q2E_PROFILE" != "0" && "$Q2E_PROFILE" != "1" ]]; then
  echo "REFUSE: K1Q2E_PROFILE must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$Q2E_TRACE_MAX_BYTES" =~ ^[0-9]+$ ]] || (( Q2E_TRACE_MAX_BYTES < 1048576 )); then
  echo "REFUSE: K1Q2E_TRACE_MAX_BYTES must be an integer >= 1048576" >&2
  exit 2
fi
if [[ -e "$OUT" ]]; then
  echo "REFUSE: output path already exists: $OUT" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
SP="$("$V/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
LIB_DIRS=("$SP/torch/lib")
for d in "$SP"/nvidia/*/lib; do [[ -d "$d" ]] && LIB_DIRS+=("$d"); done
LIB_PATH="$(IFS=:; echo "${LIB_DIRS[*]}")"
export LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ACTUAL_SHA="$(git -C "$REPO" rev-parse HEAD)"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "REFUSE: expected head $EXPECTED_SHA, got $ACTUAL_SHA" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "REFUSE: worktree is dirty" >&2
  exit 2
fi

mkdir -p "$OUT/logs"
echo "$ACTUAL_SHA" > "$OUT/exact_sha.txt"
VLLM_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
QSA="$VLLM_ROOT/models/qwen4_exp/nvidia/qsa.py"
TURN_FILE="$K1A_DIR/turns/ctx160000/turn_04_ask_d_e.json"
K1B_SUMMARY="$K1B_DIR/k1b_sticky_summary.json"
Q2B_PLAN="$OUT/q2b_source_plan.json"
Q2C_PLAN="$OUT/q2c_source_plan.json"
PLAN="$OUT/q2d_streaming_plan.json"
QSA_BACKUP="$OUT/qsa.base.py"
WORKER_STATS="$OUT/q2d_worker_stats.jsonl"
SCHED_STATS="$OUT/q2d_scheduler_stats.jsonl"
RESPONSE="$OUT/q2d_response.json"
SUMMARY="$OUT/q2d_streaming_summary.json"
TRACE="$OUT/q2e_access_trace.bin"
TRACE_SUMMARY="$OUT/q2e_access_trace_summary.json"
LOG="$OUT/logs/serve_q2d_streaming.log"
LAUNCH_PID=""
PATCHED=0

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd "0-9"
}

stop_engine() {
  if [[ -n "$LAUNCH_PID" ]]; then
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then break; fi
      sleep 1
    done
    kill -KILL -- "-$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
    LAUNCH_PID=""
  fi
}

restore_qsa() {
  if [[ "$PATCHED" == "1" && -f "$QSA_BACKUP" ]]; then
    cp "$QSA_BACKUP" "$QSA"
    rm -f "$QSA.kvmem_qsa_q2d_streaming.orig"
    PATCHED=0
  fi
}

cleanup() {
  stop_engine || true
  restore_qsa || true
}
trap cleanup EXIT

wait_healthy() {
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 &&
       curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then
      return 0
    fi
    if [[ -n "$LAUNCH_PID" ]] && ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      local rc=0
      wait "$LAUNCH_PID" || rc=$?
      echo "ENGINE LAUNCHER EXITED BEFORE HEALTHY rc=$rc" >&2
      tail -n 160 "$LOG" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "ENGINE HEALTH TIMEOUT" >&2
  return 1
}

guard_idle() {
  local st run wait_
  st=$(curl -s -m 10 "http://127.0.0.1:$PORT/metrics")
  run=$(echo "$st" | grep -E '^vllm:num_requests_running' | awk '{s+=$NF} END {print s+0}')
  wait_=$(echo "$st" | grep -E '^vllm:num_requests_waiting\{' | awk '{s+=$NF} END {print s+0}')
  echo "guard running=$run waiting=$wait_"
  [[ "$run" == "0" && "$wait_" == "0" ]]
}

echo "=== provenance / idle ==="
echo "repo_head=$ACTUAL_SHA"
echo "vllm_root=$VLLM_ROOT"
test -f "$MODEL_DIR/config.json"
test -f "$TURN_FILE"
test -f "$K1B_SUMMARY"
test -f "$QSA"
if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "REFUSE: port $PORT already serving" >&2
  exit 2
fi
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
  tr -d " " | grep -E "^[0-9]+$" || true)
if [[ -n "$GPU_PIDS" ]]; then
  echo "REFUSE: GPU compute process already running: $GPU_PIDS" >&2
  exit 2
fi
for marker in \
  "# KVMEM_QSA_Q2D_STREAMING_RUNTIME_V1" \
  "# KVMEM_QSA_Q2D_RELOAD_SHADOW_V1" \
  "# KVMEM_QSA_Q2C_ATTRIBUTION_V1" \
  "# KVMEM_QSA_Q2C_RUNTIME_V1" \
  "# KVMEM_QSA_CPU_BACKED_V1"; do
  if grep -Fq "$marker" "$QSA"; then
    echo "REFUSE: installed QSA contains stale research marker: $marker" >&2
    exit 2
  fi
done

QSA_SHA_BEFORE=$(sha256sum "$QSA" | awk '{print $1}')
cp "$QSA" "$QSA_BACKUP"
echo "qsa_sha256_before=$QSA_SHA_BEFORE"

echo "=== CPU gates ==="
bash -n "$REPO/tools/r0_run_kvmem_k1q2d_streaming_runtime.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/kvmem_q2d_scheduler_runtime.py" \
  "$REPO/src/vllm_exl3/kvmem_q2d_streaming_worker.py" \
  "$REPO/src/vllm_exl3/kvmem_q2e_trace.py" \
  "$REPO/src/vllm_exl3/kvmem_vllm_offload.py" \
  "$REPO/tools/kvmem_qsa_make_q2d_runtime_plan.py" \
  "$REPO/tools/kvmem_q2d_streaming_summarize.py" \
  "$REPO/tools/kvmem_q2e_trace_summarize.py" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2d_streaming_runtime.py"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2d_streaming_runtime.py" \
  "$VLLM_ROOT" --check-only
cmp -s "$QSA_BACKUP" "$QSA"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_kvmem_q2d_streaming_runtime.py" \
  "$REPO/tests/test_qsa_q2d_streaming_runtime_patch.py" \
  "$REPO/tests/test_kvmem_q2d_streaming_summary.py" \
  "$REPO/tests/test_kvmem_q2e_trace.py" \
  "$REPO/tests/test_kvmem_vllm_offload_adapter.py"

echo "=== build exact frozen plans ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_make_cpu_backed_plan.py" \
  --k1b-summary "$K1B_SUMMARY" --turn-file "$TURN_FILE" \
  --context 160000 --turn ask_d_e --page-tokens 16 \
  --active-reserve-tokens 1024 --publication-staging-pages 128 \
  --out "$Q2B_PLAN" > "$OUT/q2b_source_plan.stdout.json"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_qsa_make_q2c_runtime_plan.py" \
  --q2b-plan "$Q2B_PLAN" --out "$Q2C_PLAN" \
  > "$OUT/q2c_source_plan.stdout.json"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_qsa_make_q2d_runtime_plan.py" \
  --q2c-plan "$Q2C_PLAN" --out "$PLAN" | tee "$OUT/q2d_streaming_plan.stdout.json"

XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"
echo "=== apply Q2D streaming runtime patch ==="
PATCHED=1
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2d_streaming_runtime.py" \
  "$VLLM_ROOT"
grep -Fq "# KVMEM_QSA_Q2D_STREAMING_RUNTIME_V1" "$QSA"

echo "=== boot and run 160K Q2D streaming runtime ==="
if [[ "$Q2E_PROFILE" == "1" ]]; then
  export VLLM_QWEN_KVMEM_Q2E_TRACE_PATH="$TRACE"
  export VLLM_QWEN_KVMEM_Q2E_TRACE_MAX_BYTES="$Q2E_TRACE_MAX_BYTES"
else
  unset VLLM_QWEN_KVMEM_Q2E_TRACE_PATH
  unset VLLM_QWEN_KVMEM_Q2E_TRACE_MAX_BYTES
fi
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
VLLM_QWEN_KVMEM_Q2D_RUNTIME_PLAN="$PLAN" \
VLLM_QWEN_KVMEM_Q2D_WORKER_STATS_PATH="$WORKER_STATS" \
VLLM_QWEN_KVMEM_Q2D_SCHED_STATS_PATH="$SCHED_STATS" \
VLLM_KV_CACHE_LAYOUT=BLHNC \
ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 \
MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED" PORT="$PORT" \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
  > "$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!
wait_healthy
guard_idle
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
  > "$OUT/q2e_boot_gpu_processes.csv" 2> "$OUT/q2e_boot_gpu_processes.stderr" || true
: > "$WORKER_STATS"
: > "$SCHED_STATS"
REQUEST_START=$("$V/bin/python" -c 'import time; print(time.perf_counter())')
"$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_probe.py" \
  --port "$PORT" --case "$TURN_FILE" --max-tokens "$MAXTOK" \
  --out "$RESPONSE" | tee "$OUT/q2d_response.stdout.json"
REQUEST_END=$("$V/bin/python" -c 'import time; print(time.perf_counter())')
"$V/bin/python" -c \
  'import sys; print(float(sys.argv[2]) - float(sys.argv[1]))' \
  "$REQUEST_START" "$REQUEST_END" > "$OUT/q2e_request_wall_seconds.txt"
guard_idle
stop_engine
test -s "$WORKER_STATS"
test -s "$SCHED_STATS"

TRACE_ARGS=()
if [[ "$Q2E_PROFILE" == "1" ]]; then
  test -s "$TRACE"
  PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "$V/bin/python" "$REPO/tools/kvmem_q2e_trace_summarize.py" \
    --trace "$TRACE" --out "$TRACE_SUMMARY"
  TRACE_ARGS=(--trace-summary "$TRACE_SUMMARY")
fi

echo "=== summarize ==="
set +e
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_q2d_streaming_summarize.py" \
  --response "$RESPONSE" --worker-stats "$WORKER_STATS" \
  --scheduler-stats "$SCHED_STATS" --plan "$PLAN" --out "$SUMMARY" \
  "${TRACE_ARGS[@]}" \
  | tee "$OUT/q2d_streaming_summary.stdout.json"
SUMMARY_RC=$?
set -e

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"
restore_qsa
QSA_SHA_AFTER=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_after=$QSA_SHA_AFTER"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" ]]; then exit 3; fi
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader || true
echo "gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+' || true)"
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
if (( XID_DELTA != 0 )); then exit 3; fi
exit "$SUMMARY_RC"

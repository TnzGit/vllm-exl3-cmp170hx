#!/usr/bin/env bash
# Exact-context Q2E performance benchmark; not a replacement qualification.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
VENV="${R0_VENV:-$ROOT/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$ROOT/results/kvmem-k1q2e-benchmark}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact benchmark head}"
PORT="${PORT:-8002}"
CONTEXTS=(16000 80000 160000 240000)
MAX_MODEL_LEN=246000
MAX_BATCHED=1024
MAX_TOKENS=256
GPU_MEM_UTIL=0.92

if [[ -e "$OUT" ]]; then
  echo "REFUSE: output path already exists: $OUT" >&2
  exit 2
fi
if [[ "$(git -C "$REPO" rev-parse HEAD)" != "$EXPECTED_SHA" ]]; then
  echo "REFUSE: repo does not match EXPECTED_SHA" >&2
  exit 2
fi
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  echo "REFUSE: benchmark worktree is dirty" >&2
  exit 2
fi

mkdir -p "$OUT/logs"
echo "$EXPECTED_SHA" > "$OUT/exact_sha.txt"

export CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
SITE_PACKAGES="$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
LIB_DIRS=("$SITE_PACKAGES/torch/lib")
for directory in "$SITE_PACKAGES"/nvidia/*/lib; do
  [[ -d "$directory" ]] && LIB_DIRS+=("$directory")
done
LIB_PATH="$(IFS=:; echo "${LIB_DIRS[*]}")"
export LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VLLM_ROOT="$("$VENV/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
QSA="$VLLM_ROOT/models/qwen4_exp/nvidia/qsa.py"
QSA_BACKUP="$OUT/qsa.base.py"
PLAN="$OUT/q2e_benchmark_plan.json"
WORKER_STATS="$OUT/worker.current.jsonl"
SCHED_STATS="$OUT/scheduler.current.jsonl"
LOG="$OUT/logs/serve.log"
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
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if [[ -n "$LAUNCH_PID" ]] && ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      tail -n 160 "$LOG" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "ENGINE HEALTH TIMEOUT" >&2
  return 1
}

guard_idle() {
  local metrics running waiting
  metrics="$(curl -s -m 10 "http://127.0.0.1:$PORT/metrics")"
  running="$(echo "$metrics" | awk '/^vllm:num_requests_running/{s+=$NF} END{print s+0}')"
  waiting="$(echo "$metrics" | awk '/^vllm:num_requests_waiting\{/{s+=$NF} END{print s+0}')"
  [[ "$running" == "0" && "$waiting" == "0" ]]
}

echo "=== provenance and idle guard ==="
test -f "$QSA"
if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "REFUSE: port $PORT is already serving" >&2
  exit 2
fi
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
  tr -d " " | grep -E "^[0-9]+$" || true)"
if [[ -n "$GPU_PIDS" ]]; then
  echo "REFUSE: GPU compute process already running: $GPU_PIDS" >&2
  exit 2
fi
QSA_SHA_BEFORE="$(sha256sum "$QSA" | awk '{print $1}')"
cp "$QSA" "$QSA_BACKUP"
echo "$QSA_SHA_BEFORE" > "$OUT/qsa_sha256_before.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader \
  > "$OUT/gpu_identity.txt"
"$VENV/bin/python" - <<'PY' > "$OUT/runtime_identity.json"
import json, platform, torch, vllm
print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "vllm": vllm.__version__,
}, indent=2))
PY

echo "=== CPU/static gates ==="
bash -n "$REPO/tools/r0_run_kvmem_k1q2e_benchmark.sh"
"$VENV/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/kvmem_q2d_scheduler_runtime.py" \
  "$REPO/src/vllm_exl3/kvmem_q2d_streaming_worker.py" \
  "$REPO/tools/kvmem_qsa_make_q2d_runtime_plan.py" \
  "$REPO/tools/kvmem_q2e_benchmark_client.py" \
  "$REPO/tools/kvmem_q2e_benchmark_summarize.py" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2d_streaming_runtime.py"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2d_streaming_runtime.py" \
  "$VLLM_ROOT" --check-only
cmp -s "$QSA_BACKUP" "$QSA"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV/bin/python" -m pytest -q \
  "$REPO/tests/test_kvmem_q2d_reload_shadow.py" \
  "$REPO/tests/test_kvmem_q2d_streaming_runtime.py" \
  "$REPO/tests/test_qsa_q2d_streaming_runtime_patch.py" \
  "$REPO/tests/test_kvmem_q2e_benchmark_client.py" \
  "$REPO/tests/test_kvmem_q2e_benchmark_summarize.py" \
  "$REPO/tests/test_kvmem_vllm_offload_adapter.py"

echo "=== exact-token cases and benchmark-only plan ==="
"$VENV/bin/python" "$REPO/tools/kvmem_qsa_make_turns.py" \
  --model-dir "$MODEL_DIR" --out-dir "$OUT/turns" \
  --contexts 4096 "${CONTEXTS[@]}"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV/bin/python" "$REPO/tools/kvmem_qsa_make_q2d_runtime_plan.py" \
  --benchmark-case "$OUT/turns/ctx240000/turn_04_ask_d_e.json" \
  --max-model-len "$MAX_MODEL_LEN" \
  --out "$PLAN" > "$OUT/q2e_benchmark_plan.stdout.json"

XID_BEFORE="$(xid_now)"; XID_BEFORE="${XID_BEFORE:-0}"
echo "$XID_BEFORE" > "$OUT/xid_before.txt"
PATCHED=1
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2d_streaming_runtime.py" \
  "$VLLM_ROOT"

echo "=== boot Q2E direct path ==="
unset VLLM_QWEN_KVMEM_Q2E_TRACE_PATH VLLM_QWEN_KVMEM_Q2E_TRACE_MAX_BYTES
export VLLM_QWEN_KVMEM_Q2E_DIRECT_IO=1
export VLLM_QWEN_KVMEM_Q2E_VERIFY_MAX_LOGICAL_PAGE=-1
export VLLM_QWEN_KVMEM_Q2E_DIRECT_CONSUMER_SYNC=1
export VLLM_QWEN_KVMEM_Q2E_VERIFY_DYNAMIC_TABLE=0
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
VLLM_QWEN_KVMEM_Q2D_RUNTIME_PLAN="$PLAN" \
VLLM_QWEN_KVMEM_Q2D_WORKER_STATS_PATH="$WORKER_STATS" \
VLLM_QWEN_KVMEM_Q2D_SCHED_STATS_PATH="$SCHED_STATS" \
VLLM_KV_CACHE_LAYOUT=BLHNC \
ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 \
MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED" PORT="$PORT" \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
  > "$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!
wait_healthy
guard_idle
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
  > "$OUT/boot_gpu_processes.csv" 2> "$OUT/boot_gpu_processes.stderr" || true

echo "=== unmeasured 4K warmup ==="
: > "$WORKER_STATS"; : > "$SCHED_STATS"
"$VENV/bin/python" "$REPO/tools/kvmem_q2e_benchmark_client.py" \
  --port "$PORT" --case "$OUT/turns/ctx4096/turn_04_ask_d_e.json" \
  --max-tokens 16 --out "$OUT/warmup.json" >/dev/null
guard_idle

echo "=== measured cells ==="
for context in "${CONTEXTS[@]}"; do
  cell_dir="$OUT/ctx$context"
  mkdir -p "$cell_dir"
  : > "$WORKER_STATS"; : > "$SCHED_STATS"
  set +e
  "$VENV/bin/python" "$REPO/tools/kvmem_q2e_benchmark_client.py" \
    --port "$PORT" --case "$OUT/turns/ctx$context/turn_04_ask_d_e.json" \
    --max-tokens "$MAX_TOKENS" --out "$cell_dir/cell.json" \
    | tee "$cell_dir/cell.stdout.json"
  cell_rc=${PIPESTATUS[0]}
  set -e
  guard_idle || true
  cp "$WORKER_STATS" "$cell_dir/worker.jsonl"
  cp "$SCHED_STATS" "$cell_dir/scheduler.jsonl"
  if (( cell_rc != 0 )); then
    echo "benchmark cell ctx$context failed rc=$cell_rc" >&2
    exit "$cell_rc"
  fi
done

stop_engine
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$VENV/bin/python" "$REPO/tools/kvmem_q2e_benchmark_summarize.py" \
  --run-dir "$OUT" --contexts "${CONTEXTS[@]}" \
  --out "$OUT/benchmark_summary.json" | tee "$OUT/benchmark_summary.stdout.json"

XID_AFTER="$(xid_now)"; XID_AFTER="${XID_AFTER:-0}"
echo "$XID_AFTER" > "$OUT/xid_after.txt"
if [[ "$XID_AFTER" != "$XID_BEFORE" ]]; then
  echo "FAIL: Xid count changed from $XID_BEFORE to $XID_AFTER" >&2
  exit 3
fi
restore_qsa
QSA_SHA_AFTER="$(sha256sum "$QSA" | awk '{print $1}')"
echo "$QSA_SHA_AFTER" > "$OUT/qsa_sha256_after.txt"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" ]]; then
  echo "FAIL: installed QSA restore differs" >&2
  exit 3
fi
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader | tee "$OUT/gpu_after.txt"
echo "gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+' || true)" \
  | tee "$OUT/process_after.txt"
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | grep -v pgrep | wc -l)" \
  | tee -a "$OUT/process_after.txt"
if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "FAIL: benchmark port still open" >&2
  exit 3
fi
echo "Q2E_BENCHMARK_VALID"

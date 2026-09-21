#!/usr/bin/env bash
# EXL3 main-model loader attribution. Stops once the first/main checkpoint
# timing and EXL3 all-weights-loaded boundary are both proven.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/startup-exl3-loader-attribution}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${LOADER_MAX_MODEL_LEN:-4096}"
NUM_SPEC="${NUM_SPEC_TOKENS:-3}"
REFERENCE_H2D_GIB_S="${LOADER_REFERENCE_H2D_GIB_S:-6.3494}"

export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"

SP="$("$V/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
LIB_DIRS=("$SP/torch/lib")
for d in "$SP"/nvidia/*/lib; do
  [[ -d "$d" ]] && LIB_DIRS+=("$d")
done
LIB_PATH="$(IFS=:; echo "${LIB_DIRS[*]}")"
export LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUT/logs"
LOG="$OUT/logs/serve_loader.log"
TRACE="$OUT/exl3_load_trace.jsonl"
PROC="$OUT/proc_io_watch.json"
STOP="$OUT/.proc_watch_stop"
WALL="$OUT/wall_loader.json"
STARTUP="$OUT/startup_loader.json"
SUMMARY="$OUT/loader_attribution_summary.json"
rm -f "$TRACE" "$PROC" "$STOP"

LAUNCH_PID=""
WATCH_PID=""

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9'
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

cleanup() {
  touch "$STOP" 2>/dev/null || true
  stop_engine || true
  if [[ -n "$WATCH_PID" ]]; then wait "$WATCH_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

idle_gate() {
  if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "REFUSE: port $PORT already serving" >&2
    exit 2
  fi
  local gpu_pids
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)
  if [[ -n "$gpu_pids" ]]; then
    echo "REFUSE: GPU compute process already running: $gpu_pids" >&2
    exit 2
  fi
}

echo "=== provenance ==="
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "model=$MODEL_DIR"
echo "max_model_len=$MAXLEN"
echo "num_spec_tokens=$NUM_SPEC"
echo "gpu_mem_util=$GPU_MEM_UTIL"
echo "reference_h2d_gib_s=$REFERENCE_H2D_GIB_S"
echo "NOTE: diagnostic source is loaded through PYTHONPATH; installed package is not modified."

test -f "$MODEL_DIR/config.json"
idle_gate

echo "=== CPU gates ==="
bash -n "$REPO/tools/r0_run_exl3_loader_attribution.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/exl3.py" \
  "$REPO/tools/r0_startup_attribution.py" \
  "$REPO/tools/r0_proc_io_watch.py" \
  "$REPO/tools/r0_exl3_loader_attribution.py"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_startup_attribution.py" \
  "$REPO/tests/test_exl3_loader_trace_contract.py" \
  "$REPO/tests/test_exl3_loader_attribution.py" \
  "$REPO/tests/test_exl3_loader_attribution_runner.py"

XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

"$V/bin/python" "$REPO/tools/r0_proc_io_watch.py" \
  --match "VLLM::EngineCore" --log "$LOG" --stop-file "$STOP" --out "$PROC" \
  > "$OUT/proc_io_watch.stdout.json" 2>&1 &
WATCH_PID=$!

START="$("$V/bin/python" - <<'PY'
import time
print(time.monotonic())
PY
)"

PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
VLLM_EXL3_LOAD_TRACE_PATH="$TRACE" \
NUM_SPEC_TOKENS="$NUM_SPEC" VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" \
GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT" \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
  > "$LOG" 2>&1 < /dev/null &
LAUNCH_PID=$!

echo "=== wait for main-model loader evidence only ==="
DEADLINE=$((SECONDS + 1200))
while (( SECONDS < DEADLINE )); do
  LOG_OK=0; TRACE_OK=0
  grep -q "Loading weights took" "$LOG" 2>/dev/null && LOG_OK=1 || true
  grep -q '"tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD"' "$TRACE" 2>/dev/null && TRACE_OK=1 || true
  if (( LOG_OK == 1 && TRACE_OK == 1 )); then break; fi
  if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "ERROR: launcher exited before loader evidence completed" >&2
    tail -200 "$LOG" >&2 || true
    exit 3
  fi
  sleep 0.1
done

grep -q "Loading weights took" "$LOG" 2>/dev/null || { echo "ERROR: main Loading weights timing not observed" >&2; exit 3; }
grep -q '"tag": "ALL_WEIGHTS_LOADED_BEFORE_POSTLOAD"' "$TRACE" 2>/dev/null || { echo "ERROR: EXL3 all-weights-loaded trace boundary missing" >&2; exit 3; }

END="$("$V/bin/python" - <<'PY'
import time
print(time.monotonic())
PY
)"

"$V/bin/python" - "$START" "$END" "$MAXLEN" "$WALL" <<'PY'
import json, sys
start=float(sys.argv[1]); end=float(sys.argv[2]); maxlen=int(sys.argv[3])
open(sys.argv[4],"w").write(json.dumps({
  "label":"main_loader_boundary",
  "max_model_len":maxlen,
  "start_monotonic_s":start,
  "health_monotonic_s":end,
  "total_to_health_s":end-start,
  "note":"end is main-loader evidence completion, not /health",
}, indent=2))
PY

touch "$STOP"
wait "$WATCH_PID"
WATCH_PID=""

echo "=== stop before draft/compile/warmup is required ==="
stop_engine

"$V/bin/python" "$REPO/tools/r0_startup_attribution.py" \
  --log "$LOG" --wall "$WALL" --out "$STARTUP" \
  | tee "$OUT/startup_loader.stdout.json"

"$V/bin/python" "$REPO/tools/r0_exl3_loader_attribution.py" \
  --startup "$STARTUP" --trace "$TRACE" --proc-watch "$PROC" \
  --reference-h2d-gib-s "$REFERENCE_H2D_GIB_S" --out "$SUMMARY" \
  | tee "$OUT/loader_attribution_summary.stdout.json"

echo "=== compact result ==="
"$V/bin/python" - "$SUMMARY" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); s=d["startup"]; e=d["exl3_trace"]; k=d["kernel_model_load_window"]
print(f'loader_attribution_valid={d["loader_attribution_valid"]}')
for key in ("main_weights_s","draft_weights_s","total_weights_s","checkpoint_gib"): print(f"{key}={s[key]}")
for key in (
  "elapsed_first_copy_to_all_weights_loaded_s","direct_fill_calls","direct_fill_bytes",
  "direct_fill_fallback_calls","direct_fill_fallback_bytes","generic_copy_calls",
  "generic_copy_bytes","generic_copy_wall_s","direct_trellis_prep_calls",
  "direct_trellis_prep_bytes","direct_trellis_prep_wall_s","direct_trellis_copy_calls",
  "direct_trellis_copy_bytes","direct_trellis_copy_wall_s","instrumented_copy_gib",
  "instrumented_copy_wall_s","instrumented_copy_effective_gib_s","reference_raw_h2d_gib_s",
  "raw_h2d_floor_for_instrumented_bytes_s","copy_wall_over_raw_h2d_floor",
  "copy_wall_fraction_of_main_weights","trace_ru_minflt_delta","trace_ru_majflt_delta",
  "trace_ru_inblock_delta",
): print(f"{key}={e[key]}")
for key in (
  "source_window","elapsed_s","read_bytes","read_gib","rchar_gib","syscr","major_faults",
  "minor_faults","user_cpu_s","system_cpu_s","kernel_read_gib_per_main_weight_s",
  "kernel_read_vs_checkpoint_ratio",
): print(f"{key}={k[key]}")
print(f'outside_instrumented_copy_calls_s={d["interpretation_contract"]["outside_instrumented_copy_calls_s"]}')
print(f'direct_fill_no_fallback={d["interpretation_contract"]["direct_fill_no_fallback"]}')
PY

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"

echo "=== final state ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
echo "gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+' || true)"
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
if command -v ss >/dev/null 2>&1; then
  if ss -ltnp 2>/dev/null | grep -q ":$PORT "; then echo "port_$PORT=busy"; else echo "port_$PORT=free"; fi
fi
echo "results=$OUT"
echo "NOTE: stopped at main-model loader boundary; no prompt and no compile/warmup conclusion from this run."

if (( XID_DELTA != 0 )); then exit 3; fi

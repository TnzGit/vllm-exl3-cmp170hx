#!/usr/bin/env bash
# 161K compile-cache reuse probe: reparse the previous 161K log with the fixed
# parser, then run one identical health-only 161K boot against the now-populated
# compile cache. No prompt, no page-cache drop, no source mutation.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
BASELINE_DIR="${STARTUP_BASELINE_DIR:-$R/results/startup-load-attribution}"
OUT="${1:-$R/results/startup-161k-cache-reuse}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${STARTUP_LONG_MAX_MODEL_LEN:-161000}"
NUM_SPEC="${NUM_SPEC_TOKENS:-3}"

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
LAUNCH_PID=""

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd '0-9'
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
trap 'stop_engine || true' EXIT

mono() {
  "$V/bin/python" - <<'PY'
import time
print(time.monotonic())
PY
}

wait_health() {
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 &&
       curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then
      return 0
    fi
    sleep 2
  done
  return 1
}

echo "=== provenance ==="
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "baseline_dir=$BASELINE_DIR"
echo "model=$MODEL_DIR"
echo "max_model_len=$MAXLEN"
echo "num_spec_tokens=$NUM_SPEC"
echo "gpu_mem_util=$GPU_MEM_UTIL"

test -f "$BASELINE_DIR/serve_161k_warm.log"
test -f "$BASELINE_DIR/wall_161k_warm.json"
test -f "$MODEL_DIR/config.json"

if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "REFUSE: port already serving" >&2
  exit 2
fi
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
  tr -d ' ' | grep -E '^[0-9]+$' || true)
if [[ -n "$GPU_PIDS" ]]; then
  echo "REFUSE: GPU compute process already running: $GPU_PIDS" >&2
  exit 2
fi

echo "=== CPU gates ==="
bash -n "$REPO/tools/r0_run_startup_161k_cache_reuse.sh"
"$V/bin/python" -m py_compile "$REPO/tools/r0_startup_attribution.py"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_startup_attribution.py" \
  "$REPO/tests/test_startup_attribution_runner.py" \
  "$REPO/tests/test_startup_161k_cache_reuse_runner.py"

echo "=== reparse previous 161K hardware log with fixed parser ==="
"$V/bin/python" "$REPO/tools/r0_startup_attribution.py" \
  --log "$BASELINE_DIR/serve_161k_warm.log" \
  --wall "$BASELINE_DIR/wall_161k_warm.json" \
  --out "$OUT/baseline_161k_reparsed.json" \
  | tee "$OUT/baseline_161k_reparsed.stdout.json"

XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

START=$(mono)
NUM_SPEC_TOKENS="$NUM_SPEC" VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" \
GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT" \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
  > "$OUT/logs/serve_161k_repeat.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

wait_health
HEALTH=$(mono)

"$V/bin/python" - "$START" "$HEALTH" "$MAXLEN" "$OUT/wall_161k_repeat.json" <<'PY'
import json, sys
start=float(sys.argv[1]); health=float(sys.argv[2]); maxlen=int(sys.argv[3])
open(sys.argv[4],"w").write(json.dumps({
    "label":"161k_repeat",
    "max_model_len":maxlen,
    "start_monotonic_s":start,
    "health_monotonic_s":health,
    "total_to_health_s":health-start,
}, indent=2))
PY

"$V/bin/python" "$REPO/tools/r0_startup_attribution.py" \
  --log "$OUT/logs/serve_161k_repeat.log" \
  --wall "$OUT/wall_161k_repeat.json" \
  --out "$OUT/repeat_161k.json" \
  | tee "$OUT/repeat_161k.stdout.json"

stop_engine

"$V/bin/python" - \
  "$OUT/baseline_161k_reparsed.json" \
  "$OUT/repeat_161k.json" \
  "$OUT/161k_cache_reuse_comparison.json" <<'PY'
import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
keys=[
    "main_weights_s","draft_weights_s","total_weights_s",
    "model_construct_postload_s","engine_init_s","engine_non_graph_s",
    "torch_compile_total_s","initial_profiling_warmup_s",
    "dynamo_bytecode_s","compile_graph_sum_s","graph_capture_s",
    "total_to_health_s",
]
comparison={}
for k in keys:
    x=a["timings"].get(k); y=b["timings"].get(k)
    comparison[k]={
        "baseline":x,
        "repeat":y,
        "repeat_minus_baseline": (y-x if x is not None and y is not None else None),
        "repeat_over_baseline": (y/x if x not in (None,0) and y is not None else None),
    }
out={
    "schema":1,
    "baseline_161k":a,
    "repeat_161k":b,
    "comparison":comparison,
    "cache_evidence":{
        "baseline":a.get("compile_cache"),
        "repeat":b.get("compile_cache"),
        "same_cache_dirs": a.get("compile_cache",{}).get("cache_dirs") == b.get("compile_cache",{}).get("cache_dirs"),
        "repeat_cache_hit_evidence": b.get("compile_cache",{}).get("cache_hit_evidence"),
    },
    "interpretation":{
        "compile_time_saved_s": (
            (a["timings"].get("torch_compile_total_s") or 0.0)
            - (b["timings"].get("torch_compile_total_s") or 0.0)
        ),
        "profiling_warmup_saved_s": (
            (a["timings"].get("initial_profiling_warmup_s") or 0.0)
            - (b["timings"].get("initial_profiling_warmup_s") or 0.0)
        ),
        "engine_init_saved_s": (
            (a["timings"].get("engine_init_s") or 0.0)
            - (b["timings"].get("engine_init_s") or 0.0)
        ),
        "total_to_health_saved_s": (
            (a["timings"].get("total_to_health_s") or 0.0)
            - (b["timings"].get("total_to_health_s") or 0.0)
        ),
    },
}
open(sys.argv[3],"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
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
echo "NOTE: one identical 161K health-only repeat; no prompt; no page-cache drop; no source modification."

if (( XID_DELTA != 0 )); then exit 3; fi

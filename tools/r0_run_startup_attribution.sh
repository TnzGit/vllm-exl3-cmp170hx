#!/usr/bin/env bash
# Startup attribution: two identical fresh boots, no prompt.
# Run 1 is current-cache-state; run 2 immediately follows and is warm-cache-biased.
# Never drops Linux page cache and never mutates model/runtime source.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/startup-load-attribution}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-4096}"
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

mkdir -p "$OUT"
LAUNCH_PID=""

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
  stop_engine || true
}
trap cleanup EXIT

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
echo "model=$MODEL_DIR"
echo "max_model_len=$MAXLEN"
echo "num_spec_tokens=$NUM_SPEC"
echo "gpu_mem_util=$GPU_MEM_UTIL"

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

"$V/bin/python" -m py_compile "$REPO/tools/r0_startup_attribution.py"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   "$V/bin/python" -m pytest -q "$REPO/tests/test_startup_attribution.py"

for RUN in 1 2; do
  LOG="$OUT/serve_run${RUN}.log"
  WALL="$OUT/wall_run${RUN}.json"
  SUMMARY="$OUT/startup_run${RUN}.json"

  START=$(mono)
  echo "=== boot $RUN start_monotonic=$START ==="

  NUM_SPEC_TOKENS="$NUM_SPEC"   VLLM_EXL3_COOP=1   MODEL_DIR="$MODEL_DIR"   GPU_MEM_UTIL="$GPU_MEM_UTIL"   MAX_MODEL_LEN="$MAXLEN"   MAX_NUM_SEQS=1   PORT="$PORT"     setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"     > "$LOG" 2>&1 < /dev/null &
  LAUNCH_PID=$!

  wait_health
  HEALTH=$(mono)
  echo "boot=$RUN health_monotonic=$HEALTH"

  "$V/bin/python" - "$START" "$HEALTH" "$WALL" <<'PY'
import json, sys
start=float(sys.argv[1]); health=float(sys.argv[2])
open(sys.argv[3],"w").write(json.dumps({
    "start_monotonic_s": start,
    "health_monotonic_s": health,
    "total_to_health_s": health-start,
}, indent=2))
PY

  "$V/bin/python" "$REPO/tools/r0_startup_attribution.py"     --log "$LOG" --wall "$WALL" --out "$SUMMARY"     | tee "$OUT/startup_run${RUN}.stdout.json"

  stop_engine
  sleep 5
done

"$V/bin/python" - "$OUT/startup_run1.json" "$OUT/startup_run2.json"   "$OUT/startup_comparison.json" <<'PY'
import json, sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
def get(d,k):
    return d["timings"].get(k)
keys=[
    "weights_s","model_total_s","model_construct_postload_s",
    "engine_init_s","graph_capture_s","engine_non_graph_s","total_to_health_s",
]
rows={}
for k in keys:
    x=get(a,k); y=get(b,k)
    rows[k]={
        "run1":x, "run2":y,
        "run2_minus_run1": (y-x if x is not None and y is not None else None),
        "run2_over_run1": (y/x if x and y is not None else None),
    }
out={
    "schema":1,
    "run1_current_cache_state":a,
    "run2_warm_cache_biased":b,
    "comparison":rows,
    "interpretation":{
        "large_weight_time_drop_on_run2":
            bool(rows["weights_s"]["run2_over_run1"] is not None
                 and rows["weights_s"]["run2_over_run1"] < 0.8),
        "note":
            "No page cache was dropped. A large run2 weight-time reduction is "
            "evidence that storage/page-cache faults materially contribute. "
            "Similar run1/run2 weight times point more strongly to tensor "
            "materialization, EXL3 loader work and blocking H2D."
    }
}
open(sys.argv[3],"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
PY

echo "=== final state ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu   --format=csv,noheader || true
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
echo "results=$OUT"
echo "NOTE: no prompt executed; no page-cache drop; no source modification."

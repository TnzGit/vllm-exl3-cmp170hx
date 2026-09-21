#!/usr/bin/env bash
# Startup attribution: 4K current-cache -> 4K warm -> 161K warm, no prompt.
# Never drops Linux page cache and never mutates model/runtime source.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/startup-load-attribution}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
SHORT_MAXLEN="${STARTUP_SHORT_MAX_MODEL_LEN:-4096}"
LONG_MAXLEN="${STARTUP_LONG_MAX_MODEL_LEN:-161000}"
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
echo "short_max_model_len=$SHORT_MAXLEN"
echo "long_max_model_len=$LONG_MAXLEN"
echo "num_spec_tokens=$NUM_SPEC"
echo "NOTE: page cache is never dropped."
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

bash -n "$REPO/tools/r0_run_startup_attribution.sh"
"$V/bin/python" -m py_compile "$REPO/tools/r0_startup_attribution.py"
PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"   "$V/bin/python" -m pytest -q "$REPO/tests/test_startup_attribution.py" "$REPO/tests/test_startup_attribution_runner.py"

run_boot() {
  local RUN="$1"
  local LABEL="$2"
  local MAXLEN="$3"
  local LOG="$OUT/serve_${LABEL}.log"
  local WALL="$OUT/wall_${LABEL}.json"
  local SUMMARY="$OUT/startup_${LABEL}.json"

  START=$(mono)
  echo "=== boot $RUN label=$LABEL max_model_len=$MAXLEN start_monotonic=$START ==="

  NUM_SPEC_TOKENS="$NUM_SPEC"   VLLM_EXL3_COOP=1   MODEL_DIR="$MODEL_DIR"   GPU_MEM_UTIL="$GPU_MEM_UTIL"   MAX_MODEL_LEN="$MAXLEN"   MAX_NUM_SEQS=1   PORT="$PORT"     setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"     > "$LOG" 2>&1 < /dev/null &
  LAUNCH_PID=$!

  wait_health
  HEALTH=$(mono)
  echo "boot=$RUN label=$LABEL health_monotonic=$HEALTH"

  "$V/bin/python" - "$START" "$HEALTH" "$MAXLEN" "$LABEL" "$WALL" <<'PY'
import json, sys
start=float(sys.argv[1]); health=float(sys.argv[2])
maxlen=int(sys.argv[3]); label=sys.argv[4]
open(sys.argv[5],"w").write(json.dumps({
    "label": label,
    "max_model_len": maxlen,
    "start_monotonic_s": start,
    "health_monotonic_s": health,
    "total_to_health_s": health-start,
}, indent=2))
PY

  "$V/bin/python" "$REPO/tools/r0_startup_attribution.py"     --log "$LOG" --wall "$WALL" --out "$SUMMARY"     | tee "$OUT/startup_${LABEL}.stdout.json"

  stop_engine
  sleep 5
}

run_boot 1 "4k_current" "$SHORT_MAXLEN"
run_boot 2 "4k_warm" "$SHORT_MAXLEN"
run_boot 3 "161k_warm" "$LONG_MAXLEN"
"$V/bin/python" - "$OUT/startup_4k_current.json" "$OUT/startup_4k_warm.json" "$OUT/startup_161k_warm.json" "$OUT/startup_comparison.json" <<'PY'
import json, sys
cold=json.load(open(sys.argv[1])); warm=json.load(open(sys.argv[2])); long=json.load(open(sys.argv[3]))
keys=[
    "weights_s","model_total_s","model_construct_postload_s",
    "engine_init_s","graph_capture_s","engine_non_graph_s","total_to_health_s",
]
def compare(a,b):
    rows={}
    for k in keys:
        x=a["timings"].get(k); y=b["timings"].get(k)
        rows[k]={
            "a":x,"b":y,
            "b_minus_a": (y-x if x is not None and y is not None else None),
            "b_over_a": (y/x if x not in (None,0) and y is not None else None),
        }
    return rows
cache=compare(cold,warm)
ctx=compare(warm,long)
weight_ratio=cache["weights_s"]["b_over_a"]
out={
    "schema":2,
    "boots":{
        "4k_current_cache_state":cold,
        "4k_warm_cache_biased":warm,
        "161k_warm_cache_biased":long,
    },
    "cache_effect_4k":cache,
    "long_context_effect_warm":ctx,
    "runtime_geometry":{
        "4k_current":cold.get("runtime_geometry"),
        "4k_warm":warm.get("runtime_geometry"),
        "161k_warm":long.get("runtime_geometry"),
    },
    "interpretation":{
        "large_4k_weight_time_drop_when_warm": bool(
            weight_ratio is not None and weight_ratio < 0.8
        ),
        "4k_warm_over_current_weights_ratio":weight_ratio,
        "161k_over_4k_warm_weights_ratio":ctx["weights_s"]["b_over_a"],
        "long_context_added_engine_init_s":ctx["engine_init_s"]["b_minus_a"],
        "long_context_added_engine_non_graph_s":ctx["engine_non_graph_s"]["b_minus_a"],
        "long_context_added_total_to_health_s":ctx["total_to_health_s"]["b_minus_a"],
        "note":(
            "Boot 1->2 measures warm-cache sensitivity for identical 4K starts. "
            "Boot 2->3 compares warm-cache-biased 4K vs 161K. Inspect actual "
            "KV memory/token capacity/block geometry instead of assuming 161K "
            "allocates more KV bytes. Stable warm weights_s plus larger engine "
            "time at 161K localizes the added cost downstream of weight loading."
        ),
    },
}
open(sys.argv[4],"w").write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
PY
echo "=== final state ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu   --format=csv,noheader || true
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
echo "results=$OUT"
echo "NOTE: no prompt executed; no page-cache drop; no source modification."

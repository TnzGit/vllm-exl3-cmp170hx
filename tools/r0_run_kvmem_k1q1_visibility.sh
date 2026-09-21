#!/usr/bin/env bash
# K1-Q1: real-Qwen target-only resident-visibility replay.
# Full physical KV stays allocated as a safe reference. Only QSA selected
# history at/after the query boundary is filtered through the frozen K1B
# 64K/5% sticky resident plan.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1q1-visibility}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-161000}"
MAXTOK="${K1Q1_MAX_TOKENS:-512}"

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

VLLM_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
QSA="$VLLM_ROOT/models/qwen4_exp/nvidia/qsa.py"
QSA_BACKUP="$OUT/qsa.base.py"
TURN_FILE="$K1A_DIR/turns/ctx160000/turn_04_ask_d_e.json"
K1B_SUMMARY="$K1B_DIR/k1b_sticky_summary.json"
PLAN="$OUT/resident_plan.json"
STATS="$OUT/visibility_stats.jsonl"
LAUNCH_PID=""
PATCHED=0

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd '0-9'
}

stop_engine() {
  if [[ -n "$LAUNCH_PID" ]]; then
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
    sleep 4
    kill -KILL -- "-$LAUNCH_PID" 2>/dev/null || true
    wait "$LAUNCH_PID" 2>/dev/null || true
    LAUNCH_PID=""
  fi
}

restore_qsa() {
  if [[ "$PATCHED" == "1" && -f "$QSA_BACKUP" ]]; then
    cp "$QSA_BACKUP" "$QSA"
    rm -f "$QSA.kvmem_qsa_visibility.orig"
    PATCHED=0
    echo "restored_qsa_sha256=$(sha256sum "$QSA" | awk '{print $1}')"
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
    sleep 5
  done
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

start_engine() {
  local mode=$1 log=$2
  stop_engine
  if [[ "$mode" == "baseline" ]]; then
    env -u VLLM_QWEN_KVMEM_RESIDENT_PLAN         -u VLLM_QWEN_KVMEM_VISIBILITY_STATS_PATH       ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1       MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL"       MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT"       setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"       > "$log" 2>&1 < /dev/null &
  else
    : > "$STATS"
    VLLM_QWEN_KVMEM_RESIDENT_PLAN="$PLAN"     VLLM_QWEN_KVMEM_VISIBILITY_STATS_PATH="$STATS"     ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1     MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL"     MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT"       setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"       > "$log" 2>&1 < /dev/null &
  fi
  LAUNCH_PID=$!
  wait_healthy

  local api engine
  api=$(pgrep -f "vllm serve.*--port $PORT" | head -1 || true)
  engine=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
  echo "mode=$mode api_pid=$api engine_pid=$engine launch_pid=$LAUNCH_PID"
  if [[ -n "$engine" ]]; then
    echo "engine_pwd=$(readlink /proc/$engine/cwd)"
    tr '\0' '\n' < "/proc/$engine/environ" |
      grep -E '^(ENFORCE_EAGER|NUM_SPEC_TOKENS|VLLM_QWEN_KVMEM_|VLLM_EXL3_COOP)=' || true
  fi
  if [[ -n "$api" ]]; then
    tr '\0' ' ' < "/proc/$api/cmdline" | grep -q -- '--enforce-eager' || {
      echo "REFUSE: K1-Q1 engine is not eager" >&2
      return 2
    }
  fi
}

echo "=== provenance ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "vllm_root=$VLLM_ROOT"
echo "qsa=$QSA"
echo "model=$MODEL_DIR"
echo "k1a=$K1A_DIR"
echo "k1b=$K1B_DIR"
echo "max_tokens=$MAXTOK"

test -f "$QSA"
test -f "$MODEL_DIR/config.json"
test -f "$TURN_FILE"
test -f "$K1B_SUMMARY"

if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "REFUSE: port $PORT already has a healthy service" >&2
  exit 2
fi
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
  tr -d ' ' | grep -E '^[0-9]+$' || true)
if [[ -n "$GPU_PIDS" ]]; then
  echo "REFUSE: GPU compute process is already running: $GPU_PIDS" >&2
  exit 2
fi

if grep -Fq '# KVMEM_QSA_RESIDENT_VISIBILITY_V1' "$QSA"; then
  echo "REFUSE: installed QSA is already K1-Q1 patched" >&2
  exit 2
fi
if grep -Fq '# KVMEM_QSA_SHADOW_V1' "$QSA"; then
  echo "REFUSE: installed QSA still contains the older shadow patch" >&2
  exit 2
fi
if [[ -e "$QSA.kvmem_qsa_visibility.orig" ]]; then
  echo "REFUSE: stale K1-Q1 QSA backup exists" >&2
  exit 2
fi

QSA_SHA_BEFORE=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_before=$QSA_SHA_BEFORE"
cp "$QSA" "$QSA_BACKUP"

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile   "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_visibility.py"   "$REPO/tools/kvmem_qsa_make_visibility_plan.py"   "$REPO/tools/kvmem_qsa_visibility_probe.py"   "$REPO/tools/kvmem_qsa_visibility_summarize.py"

"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_visibility.py"   "$VLLM_ROOT" --check-only
cmp -s "$QSA_BACKUP" "$QSA" || {
  echo "REFUSE: --check-only modified installed QSA" >&2
  exit 2
}

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_qsa_visibility_patch.py"   "$REPO/tests/test_kvmem_qsa_visibility_plan.py"   "$REPO/tests/test_kvmem_qsa_visibility_summary.py"   "$REPO/tests/test_kvmem_qsa_visibility_runner.py"

PAGE_TOKENS=$("$V/bin/python" - <<'PY'
from vllm.config.cache import CacheConfig
print(CacheConfig.DEFAULT_BLOCK_SIZE)
PY
)
echo "installed_page_tokens=$PAGE_TOKENS"

echo "=== build frozen K1B resident replay plan ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_make_visibility_plan.py"   --k1b-summary "$K1B_SUMMARY"   --turn-file "$TURN_FILE"   --context 160000   --turn ask_d_e   --page-tokens "$PAGE_TOKENS"   --out "$PLAN"   | tee "$OUT/resident_plan.stdout.json"

echo "=== apply default-off K1-Q1 QSA patch ==="
"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_visibility.py"   "$VLLM_ROOT"
PATCHED=1
grep -Fq '# KVMEM_QSA_RESIDENT_VISIBILITY_V1' "$QSA"
echo "qsa_sha256_patched=$(sha256sum "$QSA" | awk '{print $1}')"

XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

echo "=== baseline full-QSA replay ==="
start_engine baseline "$OUT/logs/serve_baseline.log"
guard_idle
"$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_probe.py"   --port "$PORT" --case "$TURN_FILE" --max-tokens "$MAXTOK"   --out "$OUT/baseline_response.json"   | tee "$OUT/baseline_response.stdout.json"
guard_idle
stop_engine

echo "=== masked 64K/5% sticky-resident replay ==="
start_engine masked "$OUT/logs/serve_masked.log"
guard_idle
"$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_probe.py"   --port "$PORT" --case "$TURN_FILE" --max-tokens "$MAXTOK"   --out "$OUT/masked_response.json"   | tee "$OUT/masked_response.stdout.json"
guard_idle
stop_engine

if [[ ! -s "$STATS" ]]; then
  echo "ERROR: visibility stats are empty; resident mask was not exercised" >&2
  exit 3
fi

echo "=== semantic summary ==="
set +e
"$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_summarize.py"   --baseline "$OUT/baseline_response.json"   --masked "$OUT/masked_response.json"   --stats "$STATS"   --plan "$PLAN"   --out "$OUT/k1q1_visibility_summary.json"   | tee "$OUT/k1q1_visibility_summary.stdout.json"
SUMMARY_RC=$?
set -e

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"

echo "=== restore QSA ==="
restore_qsa
QSA_SHA_AFTER=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_after=$QSA_SHA_AFTER"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" ]]; then
  echo "ERROR: installed QSA was not restored byte-identically" >&2
  exit 3
fi

ERROR_LINES=$(grep -RciE 'traceback|exception|AttributeError|(^| )error[: ]'   "$OUT/logs" 2>/dev/null | awk -F: '{s+=$2} END {print s+0}')
echo "serve_error_lines=$ERROR_LINES"

echo "=== compact result ==="
"$V/bin/python" - "$OUT/k1q1_visibility_summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("classification=", d["classification"])
print("semantic_go=", d["semantic_go"])
print("baseline_target_correct=", d["baseline_target_correct"])
print("masked_target_correct=", d["masked_target_correct"])
print("baseline_finish_reason=", d["baseline_finish_reason"])
print("masked_finish_reason=", d["masked_finish_reason"])
print("baseline_completion_tokens=", d["baseline_completion_tokens"])
print("masked_completion_tokens=", d["masked_completion_tokens"])
print("baseline_measurement_valid=", d["baseline_measurement_valid"])
print("evidence_valid=", d["evidence_valid"])
print("exact_text_parity=", d["exact_text_parity"])
print("exact_token_parity=", d["exact_token_parity"])
v = d["visibility"]
print("layer_count=", v["layer_count"])
print("expected_layer_count=", v["expected_layer_count"])
print("layer_coverage_ok=", v["layer_coverage_ok"])
print("accounting_ok=", v["accounting_ok"])
print("rows_applied=", v["rows_applied"])
print("historical_selected=", v["historical_selected"])
print("historical_resident_kept=", v["historical_resident_kept"])
print("historical_selected_dropped=", v["historical_selected_dropped"])
print("historical_resident_hit_rate=", v["historical_resident_hit_rate"])
print("overall_selected_visible_rate=", v["overall_selected_visible_rate"])
print("mask_exercised=", v["mask_exercised"])
PY

echo "=== final health ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu   --format=csv,noheader || true
echo "gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+' || true)"
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
if command -v ss >/dev/null 2>&1; then
  if ss -ltnp 2>/dev/null | grep -q ":$PORT "; then
    echo "port_$PORT=busy"
  else
    echo "port_$PORT=free"
  fi
fi
echo "results=$OUT"
echo "NOTE: eager diagnostic only; no performance conclusion."
echo "NOTE: full physical main KV remains allocated in K1-Q1."

if (( XID_DELTA != 0 )); then
  exit 3
fi
exit "$SUMMARY_RC"

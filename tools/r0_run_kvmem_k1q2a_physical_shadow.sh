#!/usr/bin/env bash
# K1-Q2A: real-Qwen physical-shadow resident-cache diagnostic.
# Full scheduler-owned QSA KV remains allocated as source/reference. A separate
# bounded 64K resident physical cache is bootstrapped D2D and used for affected
# query/decode attention rows. Same-forward full-cache vs resident attention
# must be bit-exact.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1q2a-physical-shadow-crosspage}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-161000}"
MAXTOK="${K1Q2A_MAX_TOKENS:-512}"
ACTIVE_RESERVE="${K1Q2A_ACTIVE_RESERVE_TOKENS:-1024}"

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
PLAN="$OUT/physical_plan.json"
STATS="$OUT/physical_stats.jsonl"
LAUNCH_PID=""
PATCHED=0

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd '0-9'
}

stop_engine() {
  if [[ -n "$LAUNCH_PID" ]]; then
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
        break
      fi
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
    rm -f "$QSA.kvmem_qsa_physical.orig"
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

echo "=== provenance / idle ==="
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "vllm_root=$VLLM_ROOT"
echo "qsa=$QSA"
echo "model=$MODEL_DIR"
echo "max_tokens=$MAXTOK active_reserve_tokens=$ACTIVE_RESERVE"

test -f "$QSA"
test -f "$MODEL_DIR/config.json"
test -f "$TURN_FILE"
test -f "$K1B_SUMMARY"

if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "REFUSE: port $PORT already serving" >&2
  exit 2
fi
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
  tr -d ' ' | grep -E '^[0-9]+$' || true)
if [[ -n "$GPU_PIDS" ]]; then
  echo "REFUSE: GPU compute process already running: $GPU_PIDS" >&2
  exit 2
fi

for marker in   '# KVMEM_QSA_PHYSICAL_SHADOW_V1'   '# KVMEM_QSA_RESIDENT_VISIBILITY_V1'   '# KVMEM_QSA_SHADOW_V1'; do
  if grep -Fq "$marker" "$QSA"; then
    echo "REFUSE: installed QSA contains stale research marker: $marker" >&2
    exit 2
  fi
done
if [[ -e "$QSA.kvmem_qsa_physical.orig" ]]; then
  echo "REFUSE: stale Q2A backup exists" >&2
  exit 2
fi

QSA_SHA_BEFORE=$(sha256sum "$QSA" | awk '{print $1}')
cp "$QSA" "$QSA_BACKUP"
echo "qsa_sha256_before=$QSA_SHA_BEFORE"

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile   "$REPO/tools/kvmem_qsa_make_physical_plan.py"   "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_physical_shadow.py"   "$REPO/tools/kvmem_qsa_visibility_probe.py"   "$REPO/tools/kvmem_qsa_physical_shadow_summarize.py"

"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_physical_shadow.py"   "$VLLM_ROOT" --check-only
cmp -s "$QSA_BACKUP" "$QSA" || {
  echo "REFUSE: check-only modified QSA" >&2
  exit 2
}

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_kvmem_qsa_physical_plan.py"   "$REPO/tests/test_qsa_physical_shadow_patch.py"   "$REPO/tests/test_kvmem_qsa_physical_shadow_summary.py"   "$REPO/tests/test_kvmem_qsa_physical_shadow_runner.py"

PAGE_TOKENS=$("$V/bin/python" - <<'PY'
from vllm.config.cache import CacheConfig
print(CacheConfig.DEFAULT_BLOCK_SIZE)
PY
)
echo "resident_page_tokens=$PAGE_TOKENS"
echo "NOTE: resident page size is independent of the hybrid scheduler full-cache block size."

"$V/bin/python" "$REPO/tools/kvmem_qsa_make_physical_plan.py"   --k1b-summary "$K1B_SUMMARY"   --turn-file "$TURN_FILE"   --context 160000   --turn ask_d_e   --page-tokens "$PAGE_TOKENS"   --active-reserve-tokens "$ACTIVE_RESERVE"   --out "$PLAN"   | tee "$OUT/physical_plan.stdout.json"

echo "=== apply Q2A patch ==="
"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_physical_shadow.py"   "$VLLM_ROOT"
PATCHED=1
grep -Fq '# KVMEM_QSA_PHYSICAL_SHADOW_V1' "$QSA"
QSA_SHA_PATCHED=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_patched=$QSA_SHA_PATCHED"

: > "$STATS"
XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

echo "=== start one physical-shadow engine ==="
VLLM_QWEN_KVMEM_PHYSICAL_PLAN="$PLAN" VLLM_QWEN_KVMEM_PHYSICAL_STATS_PATH="$STATS" ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT"   setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"   > "$OUT/logs/serve_physical.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

wait_healthy
guard_idle

API_PID=$(pgrep -f "vllm serve.*--port $PORT" | head -1 || true)
ENGINE_PID=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
echo "api_pid=$API_PID engine_pid=$ENGINE_PID launch_pid=$LAUNCH_PID"
if [[ -n "$ENGINE_PID" ]]; then
  echo "engine_pwd=$(readlink /proc/$ENGINE_PID/cwd)"
  tr '\0' '\n' < "/proc/$ENGINE_PID/environ" |
    grep -E '^(ENFORCE_EAGER|NUM_SPEC_TOKENS|VLLM_QWEN_KVMEM_PHYSICAL_|VLLM_EXL3_COOP)=' || true
fi
if [[ -n "$API_PID" ]]; then
  tr '\0' ' ' < "/proc/$API_PID/cmdline" | grep -q -- '--enforce-eager' || {
    echo "REFUSE: Q2A engine is not eager" >&2
    exit 2
  }
fi

echo "=== exact-token semantic request ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_probe.py"   --port "$PORT"   --case "$TURN_FILE"   --max-tokens "$MAXTOK"   --out "$OUT/physical_response.json"   | tee "$OUT/physical_response.stdout.json"

guard_idle

if [[ ! -s "$STATS" ]]; then
  echo "ERROR: physical stats are empty" >&2
  exit 3
fi

echo "=== summarize physical-shadow exactness ==="
set +e
"$V/bin/python" "$REPO/tools/kvmem_qsa_physical_shadow_summarize.py"   --response "$OUT/physical_response.json"   --stats "$STATS"   --plan "$PLAN"   --out "$OUT/k1q2a_physical_summary.json"   | tee "$OUT/k1q2a_physical_summary.stdout.json"
SUMMARY_RC=$?
set -e

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"

stop_engine

echo "=== restore QSA ==="
restore_qsa
QSA_SHA_AFTER=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_after=$QSA_SHA_AFTER"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" ]]; then
  echo "ERROR: QSA restore mismatch" >&2
  exit 3
fi

echo "=== compact result ==="
"$V/bin/python" - "$OUT/k1q2a_physical_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
e=d["evidence"]
for k in (
    "classification","physical_shadow_go","target_correct","finish_reason",
    "completion_tokens","resident_budget_tokens","resident_page_count",
    "active_reserve_pages","physical_page_count"
):
    print(f"{k}={d[k]}")
for k in (
    "records","layer_count","expected_layer_count","layer_coverage_ok",
    "attention_exact_all_records","attention_max_abs","bootstrap_ok",
    "physical_geometry_ok","resident_page_tokens","full_block_tokens",
    "resident_table_widths","cross_granularity_ratio","historical_selected",
    "historical_resident_kept","historical_selected_dropped",
    "historical_visibility_rate","accounting_ok","mask_exercised"
):
    print(f"{k}={e[k]}")
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
echo "NOTE: Q2A keeps full scheduler-owned QSA KV allocated; no memory reduction claim."

if (( XID_DELTA != 0 )); then
  exit 3
fi
exit "$SUMMARY_RC"

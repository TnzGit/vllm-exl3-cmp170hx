#!/usr/bin/env bash
# K1-Q2B: real-Qwen CPU-backed resident-cache diagnostic.
# Full scheduler-owned QSA KV remains allocated as source/reference. A separate
# bounded 64K resident physical cache is populated through generic CPU offload
# (chunked D2H publication + real H2D stage-in) and used for affected
# query/decode attention rows. Selected K/V payload mapping must be byte-exact.
# Attention exactness is measured separately because PAGE_SIZE specialization
# can change floating evaluation order even when addressed K/V bytes are equal.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1q2b-cpu-backed}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-161000}"
MAXTOK="${K1Q2B_MAX_TOKENS:-512}"
ACTIVE_RESERVE="${K1Q2B_ACTIVE_RESERVE_TOKENS:-1024}"

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
PLAN="$OUT/cpu_backed_plan.json"
STATS="$OUT/cpu_backed_stats.jsonl"
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
    rm -f "$QSA.kvmem_qsa_cpu_backed.orig"
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

for marker in   '# KVMEM_QSA_CPU_BACKED_V1'   '# KVMEM_QSA_PHYSICAL_SHADOW_V1'   '# KVMEM_QSA_RESIDENT_VISIBILITY_V1'   '# KVMEM_QSA_SHADOW_V1'; do
  if grep -Fq "$marker" "$QSA"; then
    echo "REFUSE: installed QSA contains stale research marker: $marker" >&2
    exit 2
  fi
done
if [[ -e "$QSA.kvmem_qsa_cpu_backed.orig" ]]; then
  echo "REFUSE: stale Q2B backup exists" >&2
  exit 2
fi

QSA_SHA_BEFORE=$(sha256sum "$QSA" | awk '{print $1}')
cp "$QSA" "$QSA_BACKUP"
echo "qsa_sha256_before=$QSA_SHA_BEFORE"

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile   "$REPO/src/vllm_exl3/kvmem_vllm_offload.py"   "$REPO/tools/kvmem_qsa_make_cpu_backed_plan.py"   "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_cpu_backed.py"   "$REPO/tools/kvmem_qsa_visibility_probe.py"   "$REPO/tools/kvmem_qsa_cpu_backed_summarize.py"

"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_cpu_backed.py"   "$VLLM_ROOT" --check-only
cmp -s "$QSA_BACKUP" "$QSA" || {
  echo "REFUSE: check-only modified QSA" >&2
  exit 2
}

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -c "from vllm_exl3.kvmem_vllm_offload import single_tensor_cpu_backing; print('single_tensor_cpu_backing=OK')" 

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_kvmem_vllm_offload_adapter.py"   "$REPO/tests/test_kvmem_qsa_cpu_backed_plan.py"   "$REPO/tests/test_qsa_cpu_backed_patch.py"   "$REPO/tests/test_kvmem_qsa_cpu_backed_summary.py"   "$REPO/tests/test_kvmem_qsa_cpu_backed_runner.py"

PAGE_TOKENS=$("$V/bin/python" - <<'PY'
from vllm.config.cache import CacheConfig
print(CacheConfig.DEFAULT_BLOCK_SIZE)
PY
)
echo "resident_page_tokens=$PAGE_TOKENS"
echo "NOTE: resident page size is independent of the hybrid scheduler full-cache block size."

"$V/bin/python" "$REPO/tools/kvmem_qsa_make_cpu_backed_plan.py"   --k1b-summary "$K1B_SUMMARY"   --turn-file "$TURN_FILE"   --context 160000   --turn ask_d_e   --page-tokens "$PAGE_TOKENS"   --active-reserve-tokens "$ACTIVE_RESERVE"   --publication-staging-pages 128   --out "$PLAN"   | tee "$OUT/cpu_backed_plan.stdout.json"

echo "=== apply Q2B patch ==="
"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_cpu_backed.py"   "$VLLM_ROOT"
PATCHED=1
grep -Fq '# KVMEM_QSA_CPU_BACKED_V1' "$QSA"
QSA_SHA_PATCHED=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_patched=$QSA_SHA_PATCHED"

: > "$STATS"
XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

echo "=== start one CPU-backed engine ==="
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" VLLM_QWEN_KVMEM_CPU_PLAN="$PLAN" VLLM_QWEN_KVMEM_CPU_STATS_PATH="$STATS" VLLM_QWEN_KVMEM_CPU_CONTINUE_INPUT_EXACT_NONEXACT=1 ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT"   setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"   > "$OUT/logs/serve_cpu_backed.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

wait_healthy
guard_idle

API_PID=$(pgrep -f "vllm serve.*--port $PORT" | head -1 || true)
ENGINE_PID=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
echo "api_pid=$API_PID engine_pid=$ENGINE_PID launch_pid=$LAUNCH_PID"
if [[ -n "$ENGINE_PID" ]]; then
  echo "engine_pwd=$(readlink /proc/$ENGINE_PID/cwd)"
  tr '\0' '\n' < "/proc/$ENGINE_PID/environ" |
    grep -E '^(ENFORCE_EAGER|NUM_SPEC_TOKENS|VLLM_QWEN_KVMEM_|VLLM_EXL3_COOP)=' || true
fi
if [[ -n "$API_PID" ]]; then
  tr '\0' ' ' < "/proc/$API_PID/cmdline" | grep -q -- '--enforce-eager' || {
    echo "REFUSE: Q2B engine is not eager" >&2
    exit 2
  }
fi

echo "=== exact-token semantic request ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_probe.py"   --port "$PORT"   --case "$TURN_FILE"   --max-tokens "$MAXTOK"   --out "$OUT/cpu_backed_response.json"   | tee "$OUT/cpu_backed_response.stdout.json"

guard_idle

if [[ ! -s "$STATS" ]]; then
  echo "ERROR: CPU-backed stats are empty" >&2
  exit 3
fi

echo "=== summarize CPU-backed exactness ==="
set +e
"$V/bin/python" "$REPO/tools/kvmem_qsa_cpu_backed_summarize.py"   --response "$OUT/cpu_backed_response.json"   --stats "$STATS"   --plan "$PLAN"   --out "$OUT/k1q2b_cpu_backed_summary.json"   | tee "$OUT/k1q2b_cpu_backed_summary.stdout.json"
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
"$V/bin/python" - "$OUT/k1q2b_cpu_backed_summary.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
e=d["evidence"]
t=d["transfer"]
for k in (
    "classification","cpu_backed_go","transfer_gate","physical_mapping_go",
    "semantic_go","target_correct","finish_reason","completion_tokens",
    "resident_budget_tokens","resident_page_count","active_reserve_pages",
    "physical_page_count","publication_staging_pages","transfer_tensor_page_count",
):
    print(f"{k}={d[k]}")
for k in (
    "cpu_backing_all_present_by_layer","bootstrap_all_pages_exact_by_layer",
    "bootstrap_pages_compared_by_layer","first_bad_bootstrap_pages",
    "page_size_bytes","expected_bytes_per_layer","d2h_publish_bytes_by_layer",
    "h2d_stage_in_bytes_by_layer","d2h_publish_jobs_by_layer",
    "expected_publish_jobs_per_layer","transfer_tensor_geometry_ok",
    "d2h_total_gib","h2d_total_gib","d2h_event_seconds_sum_layers",
    "d2h_wall_seconds_sum_layers","h2d_event_seconds_sum_layers",
    "h2d_wall_seconds_sum_layers","bootstrap_sources",
    "transfer_tensor_mib_per_layer",
):
    print(f"{k}={t[k]}")
for k in (
    "records","layer_count","expected_layer_count","layer_coverage_ok",
    "bootstrap_ok","physical_geometry_ok","input_mapping_exact_all_records",
    "input_tokens_compared","first_bad_input_tokens",
    "attention_exact_all_records","attention_max_abs",
    "attention_mean_abs_max_record","attention_mismatch_elements",
    "attention_elements","attention_mismatch_fraction",
    "resident_page_tokens","full_block_tokens","resident_table_widths",
    "resident_cache_mib_per_layer","cross_granularity_ratio",
    "historical_selected","historical_resident_kept",
    "historical_selected_dropped","historical_visibility_rate",
    "accounting_ok","mask_exercised",
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
echo "NOTE: Q2B keeps full scheduler-owned QSA KV allocated as a reference; no memory reduction claim."

if (( XID_DELTA != 0 )); then
  exit 3
fi
exit "$SUMMARY_RC"

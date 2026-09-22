#!/usr/bin/env bash
# K1-Q2C attribution: A full/original, B full/masked, C bounded/masked.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1q2c-attribution}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact attribution head}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-161000}"
MAXTOK="${K1Q2C_MAX_TOKENS:-512}"
MAX_BATCHED="${K1Q2C_MAX_NUM_BATCHED_TOKENS:-1024}"
A_ONLY="${K1Q2C_ATTRIB_A_ONLY:-0}"
WORKSET_ROW_BATCHES="${K1Q2C_WORKSET_ROW_BATCHES:-}"
if [[ "$MAX_BATCHED" != "1024" ]]; then
  echo "REFUSE: Q2C attribution requires max-num-batched-tokens=1024" >&2
  exit 2
fi
if [[ "$A_ONLY" != "0" && "$A_ONLY" != "1" ]]; then
  echo "REFUSE: K1Q2C_ATTRIB_A_ONLY must be 0 or 1" >&2
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
PLAN="$OUT/q2c_runtime_plan.json"
BOOT_SIZING="$OUT/q2c_boot_sizing_contract.json"
QSA_BACKUP="$OUT/qsa.base.py"
A_STATS="$OUT/a_full_original_stats.jsonl"
B_STATS="$OUT/b_full_masked_stats.jsonl"
C_STATS="$OUT/c_bounded_stats.jsonl"
A_RESPONSE="$OUT/a_full_original_response.json"
B_RESPONSE="$OUT/b_full_masked_response.json"
C_RESPONSE="$OUT/c_bounded_response.json"
SCHED_STATS="$OUT/c_scheduler_stats.jsonl"
WORKER_STATS="$OUT/c_worker_stats.jsonl"
C_SUMMARY="$OUT/c_q2c_runtime_summary.json"
SUMMARY="$OUT/q2c_attribution_summary.json"
WORKING_SET_SUMMARY="$OUT/q2c_working_set_summary.json"
LAUNCH_PID=""
ACTIVE_LOG=""
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
    rm -f "$QSA.kvmem_qsa_q2c_attribution.orig"
    rm -f "$QSA.kvmem_qsa_q2c_runtime.orig"
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
      tail -n 160 "$ACTIVE_LOG" >&2 || true
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

launch_full_control() {
  local mode="$1" stats="$2" log="$3"
  ACTIVE_LOG="$log"
  PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  VLLM_QWEN_KVMEM_Q2C_PLAN="$PLAN" \
  VLLM_QWEN_KVMEM_Q2C_ATTRIB_MODE="$mode" \
  VLLM_QWEN_KVMEM_Q2C_ATTRIB_STATS_PATH="$stats" \
  VLLM_QWEN_KVMEM_Q2C_WORKSET_ROW_BATCHES="$WORKSET_ROW_BATCHES" \
  VLLM_KV_CACHE_LAYOUT=BLHNC \
  ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 \
  MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED" PORT="$PORT" \
    setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
    > "$log" 2>&1 < /dev/null &
  LAUNCH_PID=$!
}

launch_bounded() {
  ACTIVE_LOG="$OUT/logs/serve_c_bounded.log"
  PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  VLLM_QWEN_KVMEM_Q2C_PLAN="$PLAN" \
  VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH="$SCHED_STATS" \
  VLLM_QWEN_KVMEM_Q2C_WORKER_STATS_PATH="$WORKER_STATS" \
  VLLM_QWEN_KVMEM_Q2C_ATTRIB_STATS_PATH="$C_STATS" \
  VLLM_KV_CACHE_LAYOUT=BLHNC \
  ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 \
  MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED" PORT="$PORT" \
    setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
    > "$ACTIVE_LOG" 2>&1 < /dev/null &
  LAUNCH_PID=$!
}

run_request() {
  local out="$1" stdout="$2"
  wait_healthy
  guard_idle
  "$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_probe.py" \
    --port "$PORT" --case "$TURN_FILE" --max-tokens "$MAXTOK" \
    --out "$out" | tee "$stdout"
  guard_idle
  stop_engine
}

echo "=== provenance / idle ==="
echo "repo_head=$ACTUAL_SHA"
echo "vllm_root=$VLLM_ROOT"
echo "qsa=$QSA"
echo "model=$MODEL_DIR"
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
  "# KVMEM_QSA_Q2C_ATTRIBUTION_V1" \
  "# KVMEM_QSA_Q2C_RUNTIME_V1" \
  "# KVMEM_QSA_CPU_BACKED_V1" \
  "# KVMEM_QSA_PHYSICAL_SHADOW_V1" \
  "# KVMEM_QSA_RESIDENT_VISIBILITY_V1"; do
  if grep -Fq "$marker" "$QSA"; then
    echo "REFUSE: installed QSA contains stale research marker: $marker" >&2
    exit 2
  fi
done

QSA_SHA_BEFORE=$(sha256sum "$QSA" | awk '{print $1}')
cp "$QSA" "$QSA_BACKUP"
echo "qsa_sha256_before=$QSA_SHA_BEFORE"

echo "=== CPU gates ==="
bash -n "$REPO/tools/r0_run_kvmem_k1q2c_attribution.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/kvmem_q2c_attribution.py" \
  "$REPO/src/vllm_exl3/kvmem_qsa_scheduler_runtime.py" \
  "$REPO/src/vllm_exl3/kvmem_vllm_offload.py" \
  "$REPO/tools/kvmem_qsa_make_q2c_runtime_plan.py" \
  "$REPO/tools/kvmem_q2c_boot_sizing_contract.py" \
  "$REPO/tools/kvmem_q2c_attribution_summarize.py" \
  "$REPO/tools/kvmem_q2c_working_set_summarize.py" \
  "$REPO/tools/kvmem_q2c_runtime_summarize.py" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_attribution.py" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py"
for patcher in \
  patch_vllm_qsa_q2c_attribution.py \
  patch_vllm_qsa_q2c_runtime.py; do
  PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/$patcher" \
    "$VLLM_ROOT" --check-only
done
cmp -s "$QSA_BACKUP" "$QSA"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_kvmem_q2c_attribution.py" \
  "$REPO/tests/test_kvmem_q2c_attribution_summary.py" \
  "$REPO/tests/test_kvmem_q2c_working_set_summary.py" \
  "$REPO/tests/test_qsa_q2c_attribution_patch.py" \
  "$REPO/tests/test_kvmem_q2c_scheduler_contract.py" \
  "$REPO/tests/test_kvmem_q2c_boot_sizing.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_manager.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_plan.py" \
  "$REPO/tests/test_qsa_q2c_runtime_patch.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_summary.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_runner.py" \
  "$REPO/tests/test_kvmem_q2c_attribution_runner.py"

echo "=== build exact frozen plan ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_make_cpu_backed_plan.py" \
  --k1b-summary "$K1B_SUMMARY" --turn-file "$TURN_FILE" \
  --context 160000 --turn ask_d_e --page-tokens 16 \
  --active-reserve-tokens 1024 --publication-staging-pages 128 \
  --out "$Q2B_PLAN" > "$OUT/q2b_source_plan.stdout.json"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_qsa_make_q2c_runtime_plan.py" \
  --q2b-plan "$Q2B_PLAN" --out "$PLAN" \
  | tee "$OUT/q2c_runtime_plan.stdout.json"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_q2c_boot_sizing_contract.py" \
  --plan "$PLAN" --out "$BOOT_SIZING" \
  | tee "$OUT/q2c_boot_sizing_contract.stdout.json"

XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

echo "=== apply full-KV attribution patch ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_attribution.py" \
  "$VLLM_ROOT"
PATCHED=1
grep -Fq "# KVMEM_QSA_Q2C_ATTRIBUTION_V1" "$QSA"

: > "$A_STATS"
echo "=== A: full KV, original QSA selection ==="
launch_full_control baseline "$A_STATS" "$OUT/logs/serve_a_full_original.log"
run_request "$A_RESPONSE" "$OUT/a_full_original_response.stdout.json"
test -s "$A_STATS"

if [[ "$A_ONLY" == "1" ]]; then
  echo "=== summarize original-selection working set ==="
  set +e
  PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
    "$V/bin/python" "$REPO/tools/kvmem_q2c_working_set_summarize.py" \
    --stats "$A_STATS" --plan "$PLAN" --out "$WORKING_SET_SUMMARY" \
    | tee "$OUT/q2c_working_set_summary.stdout.json"
  WORKING_SET_RC=$?
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
  exit "$WORKING_SET_RC"
fi

: > "$B_STATS"
echo "=== B: full KV, shared progressive mask ==="
launch_full_control progressive_mask "$B_STATS" "$OUT/logs/serve_b_full_masked.log"
run_request "$B_RESPONSE" "$OUT/b_full_masked_response.stdout.json"
test -s "$B_STATS"

restore_qsa
cmp -s "$QSA_BACKUP" "$QSA"

echo "=== apply bounded Q2C runtime patch ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py" \
  "$VLLM_ROOT"
PATCHED=1
grep -Fq "# KVMEM_QSA_Q2C_RUNTIME_V1" "$QSA"
: > "$C_STATS"
: > "$SCHED_STATS"
: > "$WORKER_STATS"

echo "=== C: bounded Q2C, shared progressive mask ==="
launch_bounded
run_request "$C_RESPONSE" "$OUT/c_bounded_response.stdout.json"
test -s "$C_STATS"
test -s "$SCHED_STATS"
test -s "$WORKER_STATS"

echo "=== summarize bounded ownership ==="
set +e
"$V/bin/python" "$REPO/tools/kvmem_q2c_runtime_summarize.py" \
  --response "$C_RESPONSE" --scheduler-stats "$SCHED_STATS" \
  --worker-stats "$WORKER_STATS" --plan "$PLAN" --out "$C_SUMMARY" \
  | tee "$OUT/c_q2c_runtime_summary.stdout.json"
C_SUMMARY_RC=$?
set -e
echo "c_runtime_summary_rc=$C_SUMMARY_RC"

echo "=== summarize A/B/C attribution ==="
set +e
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_q2c_attribution_summarize.py" \
  --response-a "$A_RESPONSE" --stats-a "$A_STATS" \
  --response-b "$B_RESPONSE" --stats-b "$B_STATS" \
  --response-c "$C_RESPONSE" --stats-c "$C_STATS" \
  --scheduler-stats "$SCHED_STATS" --q2c-summary "$C_SUMMARY" \
  --plan "$PLAN" --out "$SUMMARY" \
  | tee "$OUT/q2c_attribution_summary.stdout.json"
SUMMARY_RC=$?
set -e

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"
restore_qsa
QSA_SHA_AFTER=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_after=$QSA_SHA_AFTER"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" ]]; then
  echo "ERROR: QSA restore mismatch" >&2
  exit 3
fi

echo "=== compact result ==="
"$V/bin/python" - "$SUMMARY" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
for key in ("classification", "conclusive", "semantic", "evidence_gate"):
    print(f"{key}={d[key]}")
print(f"b_c_selection_exact={d['b_c_selection']['exact']}")
print(f"b_c_attention_output_exact={d['b_c_attention_output']['exact']}")
print(f"virtual_lifecycle_exact={d['virtual_lifecycle']['exact']}")
PY

echo "=== final health ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader || true
echo "gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+' || true)"
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
if command -v ss >/dev/null 2>&1; then
  if ss -ltnp 2>/dev/null | grep -q ":$PORT "; then
    echo "port_$PORT=busy"
  else
    echo "port_$PORT=free"
  fi
fi
if (( XID_DELTA != 0 )); then exit 3; fi
exit "$SUMMARY_RC"

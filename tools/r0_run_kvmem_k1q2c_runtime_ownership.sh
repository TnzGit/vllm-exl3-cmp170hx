#!/usr/bin/env bash
# K1-Q2C-transition: live scheduler-visible QSA ownership shrink.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1q2c-runtime-ownership}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact Q2C runtime head}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-161000}"
MAXTOK="${K1Q2C_MAX_TOKENS:-512}"
MAX_BATCHED="${K1Q2C_MAX_NUM_BATCHED_TOKENS:-1024}"
if [[ "$MAX_BATCHED" != "1024" ]]; then
  echo "REFUSE: Q2C runtime proof requires max-num-batched-tokens=1024" >&2
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
test -f "$QSA"
TURN_FILE="$K1A_DIR/turns/ctx160000/turn_04_ask_d_e.json"
K1B_SUMMARY="$K1B_DIR/k1b_sticky_summary.json"
Q2B_PLAN="$OUT/q2b_source_plan.json"
PLAN="$OUT/q2c_runtime_plan.json"
BOOT_SIZING="$OUT/q2c_boot_sizing_contract.json"
SCHED_STATS="$OUT/q2c_scheduler_stats.jsonl"
WORKER_STATS="$OUT/q2c_worker_stats.jsonl"
RESPONSE="$OUT/q2c_response.json"
SUMMARY="$OUT/q2c_runtime_summary.json"
QSA_BACKUP="$OUT/qsa.base.py"
LAUNCH_PID=""
PATCHED=0

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd "0-9"
}

stop_engine() {
  if [[ -n "$LAUNCH_PID" ]]; then
    kill -TERM -- "-$LAUNCH_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
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
      tail -n 120 "$OUT/logs/serve_q2c.log" >&2 || true
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
echo "qsa=$QSA"
echo "model=$MODEL_DIR"
test -f "$MODEL_DIR/config.json"
test -f "$TURN_FILE"
test -f "$K1B_SUMMARY"

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
bash -n "$REPO/tools/r0_run_kvmem_k1q2c_runtime_ownership.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/kvmem_qsa_scheduler_runtime.py" \
  "$REPO/src/vllm_exl3/kvmem_vllm_offload.py" \
  "$REPO/tools/kvmem_qsa_make_q2c_runtime_plan.py" \
  "$REPO/tools/kvmem_q2c_boot_sizing_contract.py" \
  "$REPO/tools/kvmem_q2c_runtime_summarize.py" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py"

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py" \
  "$VLLM_ROOT" --check-only
cmp -s "$QSA_BACKUP" "$QSA"

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_kvmem_q2c_scheduler_contract.py" \
  "$REPO/tests/test_kvmem_q2c_boot_sizing.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_manager.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_plan.py" \
  "$REPO/tests/test_qsa_q2c_runtime_patch.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_summary.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_runner.py"

echo "=== build exact frozen Q2C plan ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_make_cpu_backed_plan.py" \
  --k1b-summary "$K1B_SUMMARY" \
  --turn-file "$TURN_FILE" \
  --context 160000 --turn ask_d_e \
  --page-tokens 16 --active-reserve-tokens 1024 \
  --publication-staging-pages 128 \
  --out "$Q2B_PLAN" > "$OUT/q2b_source_plan.stdout.json"

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_qsa_make_q2c_runtime_plan.py" \
  --q2b-plan "$Q2B_PLAN" --out "$PLAN" \
  | tee "$OUT/q2c_runtime_plan.stdout.json"

echo "=== validate Q2C boot sizing contract ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_q2c_boot_sizing_contract.py" \
  --plan "$PLAN" --out "$BOOT_SIZING" \
  | tee "$OUT/q2c_boot_sizing_contract.stdout.json"

echo "=== apply Q2C QSA patch ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py" \
  "$VLLM_ROOT"
PATCHED=1
grep -Fq "# KVMEM_QSA_Q2C_RUNTIME_V1" "$QSA"
QSA_SHA_PATCHED=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_patched=$QSA_SHA_PATCHED"

: > "$SCHED_STATS"
: > "$WORKER_STATS"
XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

echo "=== start one Q2C-transition engine ==="
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
VLLM_QWEN_KVMEM_Q2C_PLAN="$PLAN" \
VLLM_QWEN_KVMEM_Q2C_SCHED_STATS_PATH="$SCHED_STATS" \
VLLM_QWEN_KVMEM_Q2C_WORKER_STATS_PATH="$WORKER_STATS" \
VLLM_KV_CACHE_LAYOUT=BLHNC \
ENFORCE_EAGER=1 NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 \
MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS="$MAX_BATCHED" PORT="$PORT" \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
  > "$OUT/logs/serve_q2c.log" 2>&1 < /dev/null &
LAUNCH_PID=$!

wait_healthy
guard_idle

echo "=== exact-token semantic request ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_visibility_probe.py" \
  --port "$PORT" --case "$TURN_FILE" --max-tokens "$MAXTOK" \
  --out "$RESPONSE" | tee "$OUT/q2c_response.stdout.json"
guard_idle

if [[ ! -s "$SCHED_STATS" ]]; then
  echo "ERROR: Q2C scheduler stats are empty" >&2
  exit 3
fi
if [[ ! -s "$WORKER_STATS" ]]; then
  echo "ERROR: Q2C worker stats are empty" >&2
  exit 3
fi

echo "=== summarize Q2C runtime ownership ==="
set +e
"$V/bin/python" "$REPO/tools/kvmem_q2c_runtime_summarize.py" \
  --response "$RESPONSE" \
  --scheduler-stats "$SCHED_STATS" \
  --worker-stats "$WORKER_STATS" \
  --plan "$PLAN" \
  --out "$SUMMARY" | tee "$OUT/q2c_runtime_summary.stdout.json"
SUMMARY_RC=$?
set -e

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"

stop_engine
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
for k in (
    "classification","q2c_runtime_go","scheduler_shrink_gate",
    "worker_block_table_gate","cpu_authority_gate","visibility_gate",
    "semantic_gate","target_correct","finish_reason","completion_tokens",
):
    print(f"{k}={d[k]}")
for section in ("scheduler","worker"):
    for k,v in d[section].items():
        print(f"{section}.{k}={v}")
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
echo "STOP HERE: do not start 240K, MTP, multi-request, or dynamic replacement."
exit "$SUMMARY_RC"

#!/usr/bin/env bash
# K1-Q2C dual-pool bootfix + frozen-plan live ownership proof.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1q2c-dual-pool}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact Q2C dual-pool head}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-161000}"
MAXTOK="${K1Q2C_MAX_TOKENS:-512}"
MAX_BATCHED="${K1Q2C_MAX_NUM_BATCHED_TOKENS:-1024}"

if [[ "$MAX_BATCHED" != "1024" ]]; then
  echo "REFUSE: Q2C dual-pool proof requires max-num-batched-tokens=1024" >&2
  exit 2
fi
if [[ "$MAXLEN" != "161000" ]]; then
  echo "REFUSE: Q2C dual-pool proof requires MAX_MODEL_LEN=161000" >&2
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
  echo "REFUSE: Q2C execution worktree is dirty" >&2
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
PLATFORM="$VLLM_ROOT/platforms/interface.py"
CORE="$VLLM_ROOT/v1/core/kv_cache_utils.py"
WORKER="$VLLM_ROOT/v1/worker/utils.py"
for f in "$QSA" "$PLATFORM" "$CORE" "$WORKER"; do test -f "$f"; done

TURN_FILE="$K1A_DIR/turns/ctx160000/turn_04_ask_d_e.json"
K1B_SUMMARY="$K1B_DIR/k1b_sticky_summary.json"
test -f "$TURN_FILE"
test -f "$K1B_SUMMARY"

Q2B_PLAN="$OUT/q2b_source_plan.json"
PLAN="$OUT/q2c_runtime_plan.json"
BOOT="$OUT/q2c_dual_pool_boot.json"
SCHED_STATS="$OUT/q2c_scheduler_stats.jsonl"
WORKER_STATS="$OUT/q2c_worker_stats.jsonl"
RESPONSE="$OUT/q2c_response.json"
SUMMARY="$OUT/q2c_runtime_summary.json"

QSA_BACKUP="$OUT/qsa.base.py"
PLATFORM_BACKUP="$OUT/platform_interface.base.py"
CORE_BACKUP="$OUT/kv_cache_utils.base.py"
WORKER_BACKUP="$OUT/worker_utils.base.py"
LAUNCH_PID=""
QSA_PATCHED=0
DUAL_PATCHED=0

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

restore_sources() {
  if [[ "$QSA_PATCHED" == "1" && -f "$QSA_BACKUP" ]]; then
    cp "$QSA_BACKUP" "$QSA"
    rm -f "$QSA.kvmem_qsa_q2c_runtime.orig"
    QSA_PATCHED=0
  fi
  if [[ "$DUAL_PATCHED" == "1" ]]; then
    cp "$PLATFORM_BACKUP" "$PLATFORM"
    cp "$CORE_BACKUP" "$CORE"
    cp "$WORKER_BACKUP" "$WORKER"
    DUAL_PATCHED=0
  fi
}

cleanup() {
  stop_engine || true
  restore_sources || true
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
      tail -n 160 "$OUT/logs/serve_q2c.log" >&2 || true
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

echo "=== provenance / clean guards ==="
echo "repo_head=$ACTUAL_SHA"
echo "repo=$REPO"
echo "vllm_root=$VLLM_ROOT"
echo "model=$MODEL_DIR"
test -f "$MODEL_DIR/config.json"

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

for pair in \
  "$QSA:# KVMEM_QSA_Q2C_RUNTIME_V1" \
  "$QSA:# KVMEM_QSA_CPU_BACKED_V1" \
  "$QSA:# KVMEM_QSA_PHYSICAL_SHADOW_V1" \
  "$QSA:# KVMEM_QSA_RESIDENT_VISIBILITY_V1" \
  "$PLATFORM:# KVMEM_Q2C_DUAL_POOL_PLATFORM_V1" \
  "$CORE:# KVMEM_Q2C_DUAL_POOL_CORE_V1" \
  "$WORKER:# KVMEM_Q2C_DUAL_POOL_WORKER_V1"; do
  file="${pair%%:*}"
  marker="${pair#*:}"
  if grep -Fq "$marker" "$file"; then
    echo "REFUSE: stale installed research marker: $marker in $file" >&2
    exit 2
  fi
done

cp "$QSA" "$QSA_BACKUP"
cp "$PLATFORM" "$PLATFORM_BACKUP"
cp "$CORE" "$CORE_BACKUP"
cp "$WORKER" "$WORKER_BACKUP"
QSA_SHA_BEFORE=$(sha256sum "$QSA" | awk '{print $1}')
PLATFORM_SHA_BEFORE=$(sha256sum "$PLATFORM" | awk '{print $1}')
CORE_SHA_BEFORE=$(sha256sum "$CORE" | awk '{print $1}')
WORKER_SHA_BEFORE=$(sha256sum "$WORKER" | awk '{print $1}')
echo "qsa_sha256_before=$QSA_SHA_BEFORE"
echo "platform_sha256_before=$PLATFORM_SHA_BEFORE"
echo "core_sha256_before=$CORE_SHA_BEFORE"
echo "worker_sha256_before=$WORKER_SHA_BEFORE"

echo "=== CPU/source gates before installed mutation ==="
bash -n "$REPO/tools/r0_run_kvmem_k1q2c_dual_pool.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/kvmem_qsa_scheduler_runtime.py" \
  "$REPO/src/vllm_exl3/kvmem_vllm_offload.py" \
  "$REPO/tools/kvmem_qsa_make_q2c_runtime_plan.py" \
  "$REPO/tools/kvmem_q2c_dual_pool_boot_summarize.py" \
  "$REPO/tools/kvmem_q2c_runtime_summarize.py" \
  "$REPO/tools/patch_vllm_q2c_dual_pool.py" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py"

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_q2c_dual_pool.py" \
  "$VLLM_ROOT" --check-only
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py" \
  "$VLLM_ROOT" --check-only
cmp -s "$PLATFORM_BACKUP" "$PLATFORM"
cmp -s "$CORE_BACKUP" "$CORE"
cmp -s "$WORKER_BACKUP" "$WORKER"
cmp -s "$QSA_BACKUP" "$QSA"

echo "=== apply dual-pool installed patch for CPU integration gates ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_q2c_dual_pool.py" "$VLLM_ROOT"
DUAL_PATCHED=1
grep -Fq "# KVMEM_Q2C_DUAL_POOL_PLATFORM_V1" "$PLATFORM"
grep -Fq "# KVMEM_Q2C_DUAL_POOL_CORE_V1" "$CORE"
grep -Fq "# KVMEM_Q2C_DUAL_POOL_WORKER_V1" "$WORKER"

echo "=== CPU/unit gates against patched installed vLLM ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_kvmem_q2c_scheduler_contract.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_manager.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_plan.py" \
  "$REPO/tests/test_qsa_q2c_runtime_patch.py" \
  "$REPO/tests/test_kvmem_q2c_dual_pool_installed.py" \
  "$REPO/tests/test_kvmem_q2c_dual_pool_boot_summary.py" \
  "$REPO/tests/test_kvmem_q2c_runtime_summary.py" \
  "$REPO/tests/test_kvmem_q2c_dual_pool_runner.py"

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

echo "=== apply QSA runtime patch ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_q2c_runtime.py" \
  "$VLLM_ROOT"
QSA_PATCHED=1
grep -Fq "# KVMEM_QSA_Q2C_RUNTIME_V1" "$QSA"
QSA_SHA_PATCHED=$(sha256sum "$QSA" | awk '{print $1}')
PLATFORM_SHA_PATCHED=$(sha256sum "$PLATFORM" | awk '{print $1}')
CORE_SHA_PATCHED=$(sha256sum "$CORE" | awk '{print $1}')
WORKER_SHA_PATCHED=$(sha256sum "$WORKER" | awk '{print $1}')
echo "qsa_sha256_patched=$QSA_SHA_PATCHED"
echo "platform_sha256_patched=$PLATFORM_SHA_PATCHED"
echo "core_sha256_patched=$CORE_SHA_PATCHED"
echo "worker_sha256_patched=$WORKER_SHA_PATCHED"

: > "$SCHED_STATS"
: > "$WORKER_STATS"
XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

echo "=== start one Q2C dual-pool engine ==="
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

if ! wait_healthy; then
  echo "BOOT_INVALID: Q2C runtime gates remain unproven" >&2
  exit 3
fi

echo "=== dual-pool boot evidence gate ==="
"$V/bin/python" "$REPO/tools/kvmem_q2c_dual_pool_boot_summarize.py" \
  --log "$OUT/logs/serve_q2c.log" --out "$BOOT" \
  | tee "$OUT/q2c_dual_pool_boot.stdout.json"
guard_idle

echo "=== exact frozen semantic request ==="
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
  --plan "$PLAN" --boot "$BOOT" \
  --out "$SUMMARY" | tee "$OUT/q2c_runtime_summary.stdout.json"
SUMMARY_RC=$?
set -e

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"

stop_engine
restore_sources

QSA_SHA_AFTER=$(sha256sum "$QSA" | awk '{print $1}')
PLATFORM_SHA_AFTER=$(sha256sum "$PLATFORM" | awk '{print $1}')
CORE_SHA_AFTER=$(sha256sum "$CORE" | awk '{print $1}')
WORKER_SHA_AFTER=$(sha256sum "$WORKER" | awk '{print $1}')
echo "qsa_sha256_after=$QSA_SHA_AFTER"
echo "platform_sha256_after=$PLATFORM_SHA_AFTER"
echo "core_sha256_after=$CORE_SHA_AFTER"
echo "worker_sha256_after=$WORKER_SHA_AFTER"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" || \
      "$PLATFORM_SHA_AFTER" != "$PLATFORM_SHA_BEFORE" || \
      "$CORE_SHA_AFTER" != "$CORE_SHA_BEFORE" || \
      "$WORKER_SHA_AFTER" != "$WORKER_SHA_BEFORE" ]]; then
  echo "ERROR: installed-source restore mismatch" >&2
  exit 3
fi

echo "=== compact result ==="
"$V/bin/python" - "$BOOT" "$SUMMARY" <<'PY'
import json, sys
boot=json.load(open(sys.argv[1]))
d=json.load(open(sys.argv[2]))
print("boot.classification="+boot["classification"])
print("boot.dual_pool_boot_gate="+str(boot["dual_pool_boot_gate"]))
for k in (
    "classification","q2c_runtime_go","dual_pool_boot_gate",
    "scheduler_shrink_gate","worker_block_table_gate","cpu_authority_gate",
    "visibility_gate","semantic_gate","target_correct","finish_reason",
    "completion_tokens",
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
echo "STOP HERE: no 240K, MTP, multi-request, causal planner, or merge."
exit "$SUMMARY_RC"

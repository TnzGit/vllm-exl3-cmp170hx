#!/usr/bin/env bash
# K1A: QSA shadow resident-set churn across fixed-history multi-query turns.
# Diagnostic-only: no KV eviction/offload, no attention change, no MTP.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/kvmem-k1a-churn}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-246000}"

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

mkdir -p "$OUT"/{turns,shadows,responses,logs}
LIVE_SHADOW="$OUT/shadow_live.jsonl"

VLLM_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
QSA="$VLLM_ROOT/models/qwen4_exp/nvidia/qsa.py"
QSA_BACKUP="$OUT/qsa.base.py"
PATCHED=0

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd '0-9'
}

stop_engine() {
  pkill -9 -f "serve_cmp170hx_qwen_firstboot" 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  pkill -9 -f "vllm serve" 2>/dev/null || true
  sleep 6
}

restore_qsa() {
  if [[ "$PATCHED" == "1" && -f "$QSA_BACKUP" ]]; then
    cp "$QSA_BACKUP" "$QSA"
    rm -f "$QSA.kvmem_qsa_shadow.orig"
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
  local ctx=$1 history_tokens=$2 log=$3
  local min_pos=$((history_tokens - 512))
  : > "$LIVE_SHADOW"

  ENFORCE_EAGER=1   NUM_SPEC_TOKENS=0   VLLM_QWEN_KVMEM_SHADOW_PATH="$LIVE_SHADOW"   VLLM_QWEN_KVMEM_SHADOW_ROWS=128   VLLM_QWEN_KVMEM_SHADOW_MIN_POS="$min_pos"   VLLM_EXL3_COOP=1   MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL"   MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT"     setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"     > "$log" 2>&1 < /dev/null &
  wait_healthy

  local api engine
  api=$(pgrep -f "vllm serve" | head -1 || true)
  engine=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
  echo "ctx=$ctx history_tokens=$history_tokens api_pid=$api engine_pid=$engine"
  if [[ -n "$engine" ]]; then
    echo "engine_pwd=$(readlink /proc/$engine/cwd)"
    tr '\0' '\n' < "/proc/$engine/environ" |
      grep -E '^(ENFORCE_EAGER|NUM_SPEC_TOKENS|VLLM_QWEN_KVMEM_SHADOW_|VLLM_EXL3_COOP)=' || true
  fi
  if [[ -n "$api" ]]; then
    tr '\0' ' ' < "/proc/$api/cmdline" | grep -q -- '--enforce-eager' || {
      echo "REFUSE: diagnostic server did not start with --enforce-eager" >&2
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
test -f "$QSA"
test -f "$MODEL_DIR/config.json"

if grep -Fq '# KVMEM_QSA_SHADOW_V1' "$QSA"; then
  echo "REFUSE: installed QSA source is already shadow-patched" >&2
  exit 2
fi
if [[ -e "$QSA.kvmem_qsa_shadow.orig" ]]; then
  echo "REFUSE: stale QSA shadow backup exists beside installed source" >&2
  exit 2
fi

QSA_SHA_BEFORE=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_before=$QSA_SHA_BEFORE"
cp "$QSA" "$QSA_BACKUP"

echo "=== CPU/read-only gates ==="
"$V/bin/python" -m py_compile   "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_shadow.py"   "$REPO/tools/kvmem_qsa_make_turns.py"   "$REPO/tools/kvmem_qsa_shadow_probe.py"   "$REPO/tools/kvmem_qsa_churn_analyze.py"

"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_shadow.py"   "$VLLM_ROOT" --check-only
cmp -s "$QSA_BACKUP" "$QSA" || {
  echo "REFUSE: QSA --check-only modified installed source" >&2
  exit 2
}

PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_qsa_shadow_patch.py"   "$REPO/tests/test_kvmem_qsa_turns.py"   "$REPO/tests/test_kvmem_qsa_probe_schema.py"   "$REPO/tests/test_kvmem_qsa_churn_analyze.py"   "$REPO/tests/test_kvmem_qsa_churn_runner.py"

echo "=== generate fixed-history turn suites ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_make_turns.py"   --model-dir "$MODEL_DIR" --out-dir "$OUT/turns" --contexts 160000 240000

echo "=== apply research-only QSA shadow patch ==="
"$V/bin/python" "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_qsa_shadow.py" "$VLLM_ROOT"
PATCHED=1
grep -Fq '# KVMEM_QSA_SHADOW_V1' "$QSA"
echo "qsa_sha256_patched=$(sha256sum "$QSA" | awk '{print $1}')"

XID0=$(xid_now); XID0=${XID0:-0}

for ctx in 160000 240000; do
  suite="$OUT/turns/ctx$ctx/suite.json"
  history_tokens=$("$V/bin/python" - "$suite" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["history_tokens"])
PY
)
  echo "=== K1A churn context=$ctx history=$history_tokens ==="
  stop_engine
  start_engine "$ctx" "$history_tokens" "$OUT/logs/serve_ctx$ctx.log"
  guard_idle

  for turnfile in "$OUT/turns/ctx$ctx"/turn_*.json; do
    name=$(basename "$turnfile" .json)
    echo "--- turn=$name ---"
    : > "$LIVE_SHADOW"
    "$V/bin/python" "$REPO/tools/kvmem_qsa_shadow_probe.py"       --port "$PORT" --case "$turnfile"       --out "$OUT/responses/ctx${ctx}_${name}.json" >/dev/null
    guard_idle
    if [[ ! -s "$LIVE_SHADOW" ]]; then
      echo "ERROR: shadow export is empty for ctx=$ctx turn=$name" >&2
      exit 2
    fi
    cp "$LIVE_SHADOW" "$OUT/shadows/ctx${ctx}_${name}.jsonl"
  done

  guard_idle
  stop_engine
done

echo "=== churn analysis ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_churn_analyze.py"   --manifest "$OUT/turns/manifest.json"   --shadow-dir "$OUT/shadows"   --model-config "$MODEL_DIR/config.json"   --out "$OUT/k1a_churn_summary.json"   | tee "$OUT/k1a_churn_summary.stdout.json"

XID1=$(xid_now); XID1=${XID1:-0}
echo "xid_before=$XID0 xid_after=$XID1 xid_delta=$((XID1-XID0))"

ERROR_LINES=$(grep -RciE 'traceback|exception|AttributeError|(^| )error[: ]' "$OUT/logs" 2>/dev/null |
  awk -F: '{s+=$2} END {print s+0}')
echo "serve_error_lines=$ERROR_LINES"

echo "=== restore installed vLLM QSA source ==="
restore_qsa
QSA_SHA_AFTER=$(sha256sum "$QSA" | awk '{print $1}')
echo "qsa_sha256_after=$QSA_SHA_AFTER"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" ]]; then
  echo "ERROR: installed QSA source was not restored byte-identically" >&2
  exit 2
fi

echo "=== final ==="
echo "results=$OUT"
echo "NOTE: K1A is planner economics only; eager wall time is not production latency."

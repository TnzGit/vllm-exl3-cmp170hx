#!/usr/bin/env bash
# Production-shaped all-expert metadata bulk qualification.
# Measured order: control A -> full bulk -> control B.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/startup-exl3-metadata-full-bulk-qual}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact qualification head}"
PORT="${PORT:-8002}"
MAXLEN="${MAXLEN:-4096}"
MAXTOK="${MAXTOK:-128}"

# Match the established R0 hardware-runner environment.  The first-boot
# launcher intentionally invokes bare `python` and `vllm`, so the owning
# runner must put the selected venv (and CUDA toolchain) on PATH.
export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"

mkdir -p "$OUT"
ACTUAL_SHA="$(git -C "$REPO" rev-parse HEAD)"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "REFUSE: expected head $EXPECTED_SHA, got $ACTUAL_SHA" >&2
  exit 2
fi
echo "$ACTUAL_SHA" > "$OUT/exact_sha.txt"

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd '0-9'
}

stop_engine() {
  pkill -9 -f serve_cmp170hx_qwen_firstboot 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
}

wait_healthy() {
  local launcher_pid=$1
  local log=$2
  local tag=$3
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
       && curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then
      return 0
    fi
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      local rc=0
      wait "$launcher_pid" || rc=$?
      echo "ENGINE LAUNCHER EXITED BEFORE HEALTHY: tag=$tag pid=$launcher_pid rc=$rc" >&2
      echo "=== launcher tail: $log ===" >&2
      tail -n 80 "$log" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "ENGINE HEALTH TIMEOUT: tag=$tag pid=$launcher_pid" >&2
  return 1
}

snapshot_hashes() {
  "$V/bin/python" - <<'PY'
import hashlib
import importlib.util
import json
from pathlib import Path

mods = {
    "weight_utils": "vllm.model_executor.model_loader.weight_utils",
    "qsa": "vllm.models.qwen4_exp.nvidia.qsa",
}
out = {}
for key, mod in mods.items():
    spec = importlib.util.find_spec(mod)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot resolve {mod}")
    path = Path(spec.origin)
    out[key + "_path"] = str(path)
    out[key] = hashlib.sha256(path.read_bytes()).hexdigest()
print(json.dumps(out))
PY
}

run_case() {
  local tag=$1
  local full=$2
  local log="$OUT/serve_${tag}.log"
  local cell="$OUT/cell_${tag}.json"
  local trace=""

  stop_engine
  if [[ "$full" == "1" ]]; then
    trace="$OUT/full_bulk_trace.jsonl"
    rm -f "$trace"
  fi

  local t0 t1
  t0="$("$V/bin/python" -c 'import time; print(time.time())')"
  PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  VLLM_EXL3_METADATA_BULK_AB=0 \
  VLLM_EXL3_METADATA_FULL_BULK="$full" \
  VLLM_EXL3_LOAD_TRACE_PATH="$trace" \
  VLLM_EXL3_COOP=1 \
  MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL=0.92 \
  MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT" NUM_SPEC_TOKENS=3 \
    setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" >"$log" 2>&1 < /dev/null &
  local launcher_pid=$!

  if ! wait_healthy "$launcher_pid" "$log" "$tag"; then
    echo "ENGINE FAILED TO BECOME HEALTHY: $tag" >&2
    return 1
  fi
  t1="$("$V/bin/python" -c 'import time; print(time.time())')"
  "$V/bin/python" - "$t0" "$t1" "$OUT/timing_${tag}.json" <<'PY'
import json, sys
a, b = float(sys.argv[1]), float(sys.argv[2])
open(sys.argv[3], "w").write(json.dumps({"start_to_health_s": b-a}, indent=2))
PY

  curl -s -m 10 "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics_${tag}.txt" || true
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" \
    --port "$PORT" --context 4096 --max-tokens "$MAXTOK" --repeats 1 \
    --tag "metadata-full-bulk-${tag}" --out "$cell"

  if [[ "$full" == "1" && ! -s "$trace" ]]; then
    echo "FULL BULK TRACE MISSING" >&2
    return 1
  fi
  stop_engine
}

trap stop_engine EXIT

echo "=== exact CPU gates ==="
bash -n "$REPO/tools/r0_run_exl3_metadata_full_bulk_qual.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/exl3.py" \
  "$REPO/tools/r0_metadata_full_bulk_qual_summary.py"
PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_metadata_bulk_ab.py" \
  "$REPO/tests/test_metadata_bulk_ab_summary.py" \
  "$REPO/tests/test_metadata_full_bulk.py" \
  "$REPO/tests/test_metadata_full_bulk_summary.py" \
  "$REPO/tests/test_metadata_full_bulk_runner.py"

snapshot_hashes > "$OUT/hashes_before.json"
XID0="$(xid_now)"; XID0="${XID0:-0}"

echo "=== control A ==="
run_case control_a 0
echo "=== full bulk ==="
run_case full 1
echo "=== control B ==="
run_case control_b 0

XID1="$(xid_now)"; XID1="${XID1:-0}"
printf '{"xid_before":%s,"xid_after":%s,"xid_delta":%s}\n' \
  "$XID0" "$XID1" "$((XID1-XID0))" > "$OUT/xid.json"

snapshot_hashes > "$OUT/hashes_after.json"
"$V/bin/python" - "$OUT/hashes_before.json" "$OUT/hashes_after.json" "$OUT/restore.json" <<'PY'
import json, sys
b=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2]))
out={
  "weight_utils_before":b["weight_utils"], "weight_utils_after":a["weight_utils"],
  "weight_utils_path":b["weight_utils_path"],
  "qsa_before":b["qsa"], "qsa_after":a["qsa"], "qsa_path":b["qsa_path"],
}
open(sys.argv[3],"w").write(json.dumps(out,indent=2))
if out["weight_utils_before"] != out["weight_utils_after"] or out["qsa_before"] != out["qsa_after"]:
    raise SystemExit("installed-source restore gate failed")
PY

"$V/bin/python" "$REPO/tools/r0_metadata_full_bulk_qual_summary.py" \
  --control-a-log "$OUT/serve_control_a.log" \
  --full-log "$OUT/serve_full.log" \
  --control-b-log "$OUT/serve_control_b.log" \
  --control-a-cell "$OUT/cell_control_a.json" \
  --full-cell "$OUT/cell_full.json" \
  --control-b-cell "$OUT/cell_control_b.json" \
  --full-trace "$OUT/full_bulk_trace.jsonl" \
  --xid-json "$OUT/xid.json" \
  --restore-json "$OUT/restore.json" \
  --out "$OUT/full_bulk_qualification.json" \
  | tee "$OUT/full_bulk_qualification.stdout.json"

echo "=== final cleanup ==="
stop_engine
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || true
echo "results=$OUT"
echo "STOP HERE: report the qualification artifact; do not merge or start iterator coalescing."

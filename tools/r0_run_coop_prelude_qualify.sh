#!/usr/bin/env bash
# Production-like qualification for the EARLY cooperative-MoE prelude only.
#
# Candidate is pure Python. The runner requires the installed plugin to match
# frozen f2c6a71, temporarily installs this branch's exl3.py, compares flag
# OFF/ON under the same branch code, and restores the original plugin on EXIT.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/mtp-k3-coop-prelude-qualify}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-246000}"
MAXTOK="${MAXTOK:-256}"
BASE_SHA="f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8"

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

INSTALLED_PLUGIN=""
PLUGIN_BACKUP=""
SO_PATH=""
SO_SHA_BEFORE=""

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

cleanup() {
  stop_engine || true
  if [[ -n "$PLUGIN_BACKUP" && -f "$PLUGIN_BACKUP" && -n "$INSTALLED_PLUGIN" ]]; then
    cp "$PLUGIN_BACKUP" "$INSTALLED_PLUGIN"
    echo "restored_installed_plugin=1"
  fi
}
trap cleanup EXIT

wait_healthy() {
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
       && curl -s -m 10 "http://127.0.0.1:$PORT/v1/models" | grep -q '"id"'; then
      return 0
    fi
    sleep 5
  done
  return 1
}

guard_idle() {
  local st run wait_
  st=$(curl -s -m 10 "http://127.0.0.1:$PORT/metrics")
  run=$(echo "$st" | grep -E '^vllm:num_requests_running' |
    awk '{s+=$NF} END {print s+0}')
  wait_=$(echo "$st" | grep -E '^vllm:num_requests_waiting\{' |
    awk '{s+=$NF} END {print s+0}')
  echo "guard running=$run waiting=$wait_"
  [[ "$run" == "0" && "$wait_" == "0" ]]
}

start_engine() {
  local mode=$1 log=$2
  local early
  case "$mode" in
    base) early=0 ;;
    early) early=1 ;;
    *) echo "bad mode: $mode" >&2; return 2 ;;
  esac

  local -a env_unset=(
    -u VLLM_EXL3_COOP_OUT_EMPTY
    -u TORCH_PROFILER_DIR
    -u TORCH_PROFILER_RECORD_SHAPES
  )
  local -a env_vars=(
    VLLM_EXL3_COOP_EARLY_PRELUDE="$early"
  )

  VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT" NUM_SPEC_TOKENS=3 \
    env "${env_unset[@]}" "${env_vars[@]}" \
      setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
      > "$log" 2>&1 < /dev/null &
  wait_healthy

  local api engine
  api=$(pgrep -f "vllm serve" | head -1 || true)
  engine=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
  echo "mode=$mode api_pid=$api engine_pid=$engine"
  if [[ -n "$engine" ]]; then
    echo "engine_pwd=$(readlink /proc/$engine/cwd)"
    tr '\0' '\n' < "/proc/$engine/environ" |
      grep -E '^(VLLM_EXL3_COOP|VLLM_EXL3_COOP_EARLY_PRELUDE|VLLM_EXL3_COOP_OUT_EMPTY|TORCH_PROFILER_DIR)=' || true
  fi
}

echo "=== provenance / install guard ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "base_sha=$BASE_SHA"
echo "model=$MODEL_DIR"

if ! git -C "$REPO" diff --quiet "$BASE_SHA" HEAD -- csrc; then
  echo "REFUSE: qualification branch changes csrc." >&2
  exit 2
fi

SRC_DELTA="$(git -C "$REPO" diff --name-only "$BASE_SHA" HEAD -- src/vllm_exl3)"
if [[ "$SRC_DELTA" != "src/vllm_exl3/exl3.py" ]]; then
  echo "REFUSE: qualification source delta is not isolated to src/vllm_exl3/exl3.py" >&2
  echo "$SRC_DELTA" >&2
  exit 2
fi

if grep -q 'VLLM_EXL3_COOP_OUT_EMPTY' "$REPO/src/vllm_exl3/exl3.py"; then
  echo "REFUSE: rejected OUT_EMPTY probe is still present in candidate source." >&2
  exit 2
fi

INSTALLED_PLUGIN="$("$V/bin/python" - <<'PY'
import vllm_exl3.exl3 as m
print(m.__file__)
PY
)"
BASE_EXPECTED="$OUT/exl3_base_expected.py"
git -C "$REPO" show "$BASE_SHA:src/vllm_exl3/exl3.py" > "$BASE_EXPECTED"

echo "installed_plugin=$INSTALLED_PLUGIN"
echo "base_plugin_sha256=$(sha256sum "$BASE_EXPECTED" | awk '{print $1}')"
echo "installed_before_sha256=$(sha256sum "$INSTALLED_PLUGIN" | awk '{print $1}')"
echo "candidate_plugin_sha256=$(sha256sum "$REPO/src/vllm_exl3/exl3.py" | awk '{print $1}')"

if ! cmp -s "$BASE_EXPECTED" "$INSTALLED_PLUGIN"; then
  echo "REFUSE: installed plugin is not frozen production baseline $BASE_SHA." >&2
  exit 2
fi

PLUGIN_BACKUP="$OUT/exl3_installed_backup.py"
cp "$INSTALLED_PLUGIN" "$PLUGIN_BACKUP"

SO_PATH="$("$V/bin/python" - <<'PY'
import torch
import vllm_exl3_c
print(vllm_exl3_c.__file__)
PY
)"
SO_SHA_BEFORE=$(sha256sum "$SO_PATH" | awk '{print $1}')
echo "extension=$SO_PATH"
echo "extension_sha256_before=$SO_SHA_BEFORE"

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile \
  "$REPO/tools/r0_k3_cell.py" \
  "$REPO/tools/r0_k3_parity_recheck.py" \
  "$REPO/tools/r0_coop_prelude_qualify_summary.py"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
"$V/bin/python" -m pytest -q \
  "$REPO/tests/test_coop_early_prelude.py" \
  "$REPO/tests/test_mtp_denominator_contract.py" \
  "$REPO/tests/test_moe_coop_expert_range.py" \
  "$REPO/tests/test_coop_prelude_qualify_runner.py" \
  "$REPO/tests/test_mtp_inline_parity_contract.py"

XID0=$(xid_now); XID0=${XID0:-0}

run_config() {
  local mode=$1 prefix=$2 log=$3
  stop_engine
  start_engine "$mode" "$log"
  guard_idle

  echo "=== $mode 4K sentinel in ==="
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
    --max-tokens "$MAXTOK" --repeats 5 --tag "$mode-4k-sentinel-in" \
    --out "$OUT/${prefix}_4096_sentinel_in.json" >/dev/null

  echo "=== $mode contexts ==="
  local ctx repeats outname parity_arg
  for ctx in 4096 160000 240000; do
    if [[ "$ctx" == "4096" ]]; then repeats=7; else repeats=3; fi
    if [[ "$mode" == "base" ]]; then
      outname="$OUT/ref_${ctx}.json"
      parity_arg=()
    else
      outname="$OUT/parity_k3_${ctx}.json"
      parity_arg=(--parity-reference "$OUT/ref_${ctx}.json")
    fi
    guard_idle
    "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context "$ctx" \
      --max-tokens "$MAXTOK" --repeats "$repeats" --tag "$mode-ctx$ctx" \
      "${parity_arg[@]}" --out "$outname" >/dev/null
  done

  echo "=== $mode 4K sentinel out ==="
  guard_idle
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
    --max-tokens "$MAXTOK" --repeats 5 --tag "$mode-4k-sentinel-out" \
    --out "$OUT/${prefix}_4096_sentinel_out.json" >/dev/null

  guard_idle
  stop_engine
}

echo "=== BASE qualification cells: exact frozen production plugin ==="
cmp -s "$BASE_EXPECTED" "$INSTALLED_PLUGIN"
run_config base base "$OUT/serve_base.log"

echo "=== install EARLY-only candidate plugin ==="
cp "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"
"$V/bin/python" -m py_compile "$INSTALLED_PLUGIN"
cmp -s "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"
echo "installed_candidate_sha256=$(sha256sum "$INSTALLED_PLUGIN" | awk '{print $1}')"

echo "=== EARLY qualification cells ==="
run_config early early "$OUT/serve_early.log"

echo "=== flattened parity recheck ==="
"$V/bin/python" "$REPO/tools/r0_k3_parity_recheck.py" --dir "$OUT" |
  tee "$OUT/parity.txt"

echo "=== qualification summary ==="
set +e
"$V/bin/python" "$REPO/tools/r0_coop_prelude_qualify_summary.py" --dir "$OUT" |
  tee "$OUT/qualification_summary.json"
QUAL_RC=${PIPESTATUS[0]}
set -e
echo "qualification_summary_rc=$QUAL_RC"

echo "=== health summary ==="
"$V/bin/python" - "$OUT" <<'PY' | tee "$OUT/health_summary.json"
import glob, json, os, sys
d=sys.argv[1]
files=sorted(glob.glob(os.path.join(d,"*.json")))
cells=[]
for p in files:
    try:
        j=json.load(open(p))
    except Exception:
        continue
    if isinstance(j,dict) and isinstance(j.get("cells"),list):
        for c in j["cells"]:
            cells.append((os.path.basename(p),c))
pre=sum(float(c.get("preemptions_delta_total") or 0) for _,c in cells)
invalid=[p for p,c in cells if c.get("denominator_consistent") is False]
print(json.dumps({
    "preemptions_delta_total": pre,
    "denominator_inconsistent_cells": invalid,
    "json_cells_seen": len(cells),
}, indent=2))
PY

BASE_FALLBACK=$(grep -c 'DIRECT_FILL_FALLBACK_CALLS: [1-9]' "$OUT/serve_base.log" 2>/dev/null || true)
EARLY_FALLBACK=$(grep -c 'DIRECT_FILL_FALLBACK_CALLS: [1-9]' "$OUT/serve_early.log" 2>/dev/null || true)
BASE_ERRORS=$(grep -ciE 'traceback|exception|(^| )error[: ]' "$OUT/serve_base.log" 2>/dev/null || true)
EARLY_ERRORS=$(grep -ciE 'traceback|exception|(^| )error[: ]' "$OUT/serve_early.log" 2>/dev/null || true)
echo "base_nonzero_fallback_lines=$BASE_FALLBACK"
echo "early_nonzero_fallback_lines=$EARLY_FALLBACK"
echo "base_error_lines=$BASE_ERRORS"
echo "early_error_lines=$EARLY_ERRORS"

XID1=$(xid_now); XID1=${XID1:-0}
SO_SHA_AFTER=$(sha256sum "$SO_PATH" | awk '{print $1}')
echo "xid_before=$XID0 xid_after=$XID1 xid_delta=$((XID1-XID0))"
echo "extension_sha256_after=$SO_SHA_AFTER"

if [[ "$SO_SHA_AFTER" != "$SO_SHA_BEFORE" ]]; then
  echo "ERROR: compiled extension changed unexpectedly" >&2
  exit 2
fi

echo "=== final ==="
echo "qualification_summary_rc=$QUAL_RC (0=qualified, 3=performance gate not met)"
echo "results=$OUT"
echo "EXIT trap will restore the frozen production plugin."

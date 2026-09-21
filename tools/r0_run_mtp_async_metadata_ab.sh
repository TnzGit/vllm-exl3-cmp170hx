#!/usr/bin/env bash
# Execute the prepared vLLM #55054 async-MTP-metadata A/B on CMP170HX.
#
# Repository-side design is already fixed. This script only:
#   1) runs CPU gates,
#   2) measures BASE,
#   3) applies the prepared selective backport,
#   4) measures ASYNC,
#   5) conditionally captures a diagnostic trace when the candidate clears 3%.
#
# It never merges or productionizes the candidate.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/mtp-async-metadata}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-246000}"
MAXTOK="${MAXTOK:-256}"

export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
mkdir -p "$OUT"

VLLM_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
SHORT_CONV="$VLLM_ROOT/v1/attention/backends/short_conv_attn.py"
BASE_COPY="$OUT/short_conv_attn.base.py"

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

start_engine() {
  local log=$1
  shift
  VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT" NUM_SPEC_TOKENS=3 \
    "$@" setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
      > "$log" 2>&1 < /dev/null &
  wait_healthy
  local api engine
  api=$(pgrep -f "vllm serve" | head -1 || true)
  engine=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
  echo "api_pid=$api engine_pid=$engine"
  if [[ -n "$engine" ]]; then
    echo "engine_pwd=$(readlink /proc/$engine/cwd)"
    tr '\0' '\n' < "/proc/$engine/environ" |
      grep -E '^(VLLM_EXL3_COOP|TORCH_PROFILER_DIR)=' || true
  fi
}

guard_idle() {
  local st run wait_
  st=$(curl -s -m 10 "http://127.0.0.1:$PORT/metrics")
  run=$(echo "$st" | grep -E '^vllm:num_requests_running' |
    awk '{s+=$NF} END {print s+0}')
  wait_=$(echo "$st" | grep -E '^vllm:num_requests_waiting\{' |
    awk '{s+=$NF} END {print s+0}')
  [[ "$run" == "0" && "$wait_" == "0" ]]
}

echo "=== provenance ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "vllm_root=$VLLM_ROOT"
echo "short_conv_sha256=$(sha256sum "$SHORT_CONV" | awk '{print $1}')"

if grep -q 'spec_req_idx = async_tensor_h2d(spec_req_idx_cpu' "$SHORT_CONV"; then
  echo "REFUSE: installed vLLM is already async-metadata patched; BASE is contaminated." >&2
  echo "Restore exact v0.29.0 short_conv_attn.py before running this A/B." >&2
  exit 2
fi
cp "$SHORT_CONV" "$BASE_COPY"

cleanup() {
  # Always leave the machine on the exact pre-experiment short-conv source,
  # even if a later benchmark/profiler step fails.
  stop_engine || true
  if [[ -f "$BASE_COPY" ]]; then
    cp "$BASE_COPY" "$SHORT_CONV"
  fi
}
trap cleanup EXIT

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile \
  "$REPO/tools/apply_qwen4_exp_patches.py" \
  "$REPO/tools/patch_vllm_qwen4_exp/patch_vllm_mtp_async_metadata.py" \
  "$REPO/tools/r0_k3_cell.py" \
  "$REPO/tools/r0_k3_parity_recheck.py" \
  "$REPO/tools/r0_trace_sync_summary.py"
"$V/bin/python" -m pytest -q \
  "$REPO/tests/test_mtp_async_metadata_patch.py" \
  "$REPO/tests/test_mtp_async_metadata_wrapper.py" \
  "$REPO/tests/test_qwen4_exp_patch_stack.py" \
  "$REPO/tests/test_mtp_denominator_contract.py"

echo "=== prepare BASE runtime ==="
"$V/bin/python" "$REPO/tools/apply_qwen4_exp_patches.py" "$VLLM_ROOT" \
  --profile text-mtp
"$V/bin/python" "$REPO/tools/cmp170hx_qwen_preflight.py" \
  --profile text-mtp

# The BASE wrapper must not have touched short_conv_attn.py.
if ! cmp -s "$BASE_COPY" "$SHORT_CONV"; then
  echo "REFUSE: BASE preparation unexpectedly changed short_conv_attn.py" >&2
  exit 2
fi

stop_engine
XID0=$(xid_now); XID0=${XID0:-0}

echo "=== BASE k3 ==="
start_engine "$OUT/serve_base.log" env
guard_idle
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
  --max-tokens "$MAXTOK" --repeats 3 --tag "base-4k" \
  --out "$OUT/ref_4096.json" >/dev/null
guard_idle
# Capture a same-engine 160K reference now, before mutating the vLLM source.
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 160000 \
  --max-tokens "$MAXTOK" --repeats 3 --tag "base-160k" \
  --out "$OUT/ref_160000.json" >/dev/null
stop_engine

echo "=== apply ASYNC candidate ==="
"$V/bin/python" "$REPO/tools/apply_qwen4_exp_patches.py" "$VLLM_ROOT" \
  --profile text-mtp --mtp-async-metadata
"$V/bin/python" "$REPO/tools/cmp170hx_qwen_preflight.py" \
  --profile text-mtp

if ! grep -q 'spec_req_idx = async_tensor_h2d(spec_req_idx_cpu' "$SHORT_CONV"; then
  echo "REFUSE: ASYNC postcondition missing in installed vLLM" >&2
  exit 2
fi

stop_engine
echo "=== ASYNC k3 ==="
start_engine "$OUT/serve_async.log" env
guard_idle
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
  --max-tokens "$MAXTOK" --repeats 3 --tag "async-4k" \
  --out "$OUT/parity_k3_4096.json" >/dev/null

GAIN="$("$V/bin/python" - "$OUT/ref_4096.json" "$OUT/parity_k3_4096.json" <<'PY'
import json, sys
b=json.load(open(sys.argv[1]))["ms_per_output_token_median"]
a=json.load(open(sys.argv[2]))["ms_per_output_token_median"]
print(f"{(b-a)/b*100:.4f}")
PY
)"
echo "4k_latency_gain_pct=$GAIN"

RUN_LONG="$("$V/bin/python" - "$GAIN" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= 3.0 else 0)
PY
)"
if [[ "$RUN_LONG" == "1" ]]; then
  guard_idle
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 160000 \
    --max-tokens "$MAXTOK" --repeats 3 --tag "async-160k" \
    --out "$OUT/parity_k3_160000.json" >/dev/null
fi

"$V/bin/python" "$REPO/tools/r0_k3_parity_recheck.py" --dir "$OUT" |
  tee "$OUT/parity.txt"
stop_engine

XID1=$(xid_now); XID1=${XID1:-0}

"$V/bin/python" - "$OUT" "$GAIN" "$XID0" "$XID1" <<'PY'
import json, os, sys
d, gain, xid0, xid1 = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
def load(name):
    p=os.path.join(d,name)
    return json.load(open(p)) if os.path.exists(p) else None
out={
  "base_4k": load("ref_4096.json"),
  "async_4k": load("parity_k3_4096.json"),
  "base_160k": load("ref_160000.json"),
  "async_160k": load("parity_k3_160000.json"),
  "latency_gain_pct_4k": gain,
  "xid_before": xid0,
  "xid_after": xid1,
  "xid_delta": xid1-xid0,
}
json.dump(out, open(os.path.join(d,"ab_summary.json"),"w"), indent=2)
print(json.dumps({
  "base_4k_ms": out["base_4k"]["ms_per_output_token_median"],
  "async_4k_ms": out["async_4k"]["ms_per_output_token_median"],
  "gain_pct": gain,
  "base_4k_tok_s": out["base_4k"]["output_tok_s_median"],
  "async_4k_tok_s": out["async_4k"]["output_tok_s_median"],
  "xid_delta": xid1-xid0,
}, indent=2))
PY

# Only a candidate that clears 3% gets a profiler diagnostic. The profiler is
# never used for the formal A/B latency above.
if [[ "$RUN_LONG" == "1" ]]; then
  echo "=== diagnostic trace for ASYNC (profiler ON; not a perf number) ==="
  TRACE_DIR="$OUT/traces"
  mkdir -p "$TRACE_DIR"
  stop_engine
  start_engine "$OUT/serve_async_prof.log" \
    env TORCH_PROFILER_DIR="$TRACE_DIR" TORCH_PROFILER_RECORD_SHAPES=0
  "$V/bin/python" "$REPO/tools/r0_k3_capture_passes.py" --port "$PORT" \
    --context 4096 --max-tokens "$MAXTOK" --target-passes 14 \
    --out "$OUT/async_trace_window.json"
  stop_engine

  TRACE=$(find "$TRACE_DIR" -type f \( -name '*.pt.trace.json.gz' -o -name '*.json.gz' -o -name '*.json' \) \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
  if [[ -n "$TRACE" ]]; then
    "$V/bin/python" "$REPO/tools/r0_trace_sync_summary.py" "$TRACE" --passes 14 |
      tee "$OUT/async_sync_summary.json"
  else
    echo "WARNING: profiler trace not found under $TRACE_DIR" >&2
  fi
fi

echo "=== final ==="
echo "results=$OUT"
echo "installed_vllm_candidate_will_be_restored_by_exit_trap=1"
echo "baseline_source_copy=$BASE_COPY"
echo "No PR was merged and no production default was changed."

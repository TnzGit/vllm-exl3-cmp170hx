#!/usr/bin/env bash
# Diagnostic A/B for the post-MTP dense EXL3 output boundary.
#
# BASE keeps the historical FP32 EXL3 epilogue.
# PROBE uses the native ExLlamaV3 FP16 epilogue only for eligible BF16-input,
# pure-EXL3, non-lm_head dense linears, then still casts back to BF16.
#
# PROBE changes internal rounding and is NOT a production candidate. Its purpose
# is to measure whether the output side is valuable enough to justify a later
# BF16-native epilogue that preserves FP32 arithmetic.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/mtp-k3-dense-output-probe}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-246000}"
MAXTOK="${MAXTOK:-256}"

export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
mkdir -p "$OUT"

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
  local probe=$1 log=$2 profiler_dir=${3:-}
  local -a extra=()
  if [[ -n "$profiler_dir" ]]; then
    mkdir -p "$profiler_dir"
    extra+=(env TORCH_PROFILER_DIR="$profiler_dir" TORCH_PROFILER_RECORD_SHAPES=0)
  else
    extra+=(env -u TORCH_PROFILER_DIR -u TORCH_PROFILER_RECORD_SHAPES)
  fi

  VLLM_EXL3_COOP=1 \
  VLLM_EXL3_DENSE_FP16_OUT_PROBE="$probe" \
  MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT="$PORT" NUM_SPEC_TOKENS=3 \
    "${extra[@]}" setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
    > "$log" 2>&1 < /dev/null &
  wait_healthy

  local api engine
  api=$(pgrep -f "vllm serve" | head -1 || true)
  engine=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
  echo "api_pid=$api engine_pid=$engine"
  if [[ -n "$engine" ]]; then
    echo "engine_pwd=$(readlink /proc/$engine/cwd)"
    tr '\0' '\n' < "/proc/$engine/environ" |
      grep -E '^(VLLM_EXL3_COOP|VLLM_EXL3_DENSE_FP16_OUT_PROBE|TORCH_PROFILER_DIR)=' || true
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

cleanup() {
  stop_engine || true
}
trap cleanup EXIT

echo "=== provenance ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "model=$MODEL_DIR"

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile \
  "$REPO/tools/r0_k3_cell.py" \
  "$REPO/tools/r0_k3_capture_passes.py" \
  "$REPO/tools/r0_k3_amdahl.py" \
  "$REPO/tools/r0_dense_output_probe_report.py"
"$V/bin/python" -m pytest -q \
  "$REPO/tests/test_dense_output_probe.py" \
  "$REPO/tests/test_mtp_denominator_contract.py"

XID0=$(xid_now); XID0=${XID0:-0}

echo "=== BASE: FP32 epilogue ==="
stop_engine
start_engine 0 "$OUT/serve_base.log"
guard_idle
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
  --max-tokens "$MAXTOK" --repeats 5 --tag "dense-base-4k" \
  --out "$OUT/ref_4096.json" >/dev/null
guard_idle
stop_engine

echo "=== PROBE: FP16 epilogue, final BF16 return preserved ==="
start_engine 1 "$OUT/serve_probe.log"
guard_idle
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
  --max-tokens "$MAXTOK" --repeats 5 --tag "dense-probe-4k" \
  --out "$OUT/parity_k3_4096.json" >/dev/null
guard_idle
stop_engine

"$V/bin/python" "$REPO/tools/r0_k3_parity_recheck.py" --dir "$OUT" |
  tee "$OUT/parity.txt"

read -r GAIN PARITY <<EOF
$("$V/bin/python" - "$OUT/ref_4096.json" "$OUT/parity_k3_4096.json" <<'PY'
import json, sys
def flat(path):
    j=json.load(open(path))
    seq=(j.get("cells") or [{}])[0].get("token_pieces") or []
    out=[]
    for x in seq:
        out.extend(x if isinstance(x,list) else [x])
    return out
b=json.load(open(sys.argv[1]))
p=json.load(open(sys.argv[2]))
bm=b["ms_per_output_token_median"]
pm=p["ms_per_output_token_median"]
gain=(bm-pm)/bm*100
print(f"{gain:.6f}", "PASS" if flat(sys.argv[1]) == flat(sys.argv[2]) else "MISMATCH")
PY
)
EOF
echo "4k_latency_gain_pct=$GAIN parity=$PARITY"

RUN_LONG=$("$V/bin/python" - "$GAIN" "$PARITY" <<'PY'
import sys
print(1 if float(sys.argv[1]) >= 3.0 and sys.argv[2] == "PASS" else 0)
PY
)

if [[ "$RUN_LONG" == "1" ]]; then
  echo "=== PROBE 160K: only because 4K >=3% and parity PASS ==="
  start_engine 1 "$OUT/serve_probe_160k.log"
  guard_idle
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 160000 \
    --max-tokens "$MAXTOK" --repeats 3 --tag "dense-probe-160k" \
    --out "$OUT/probe_160000.json" >/dev/null
  guard_idle
  stop_engine
fi

# Diagnostic trace is always taken: this branch is a measurement probe, and
# per-pass bucket deltas are needed even if end-to-end gain is hidden by
# acceptance noise or the FP16 rounding changes greedy tokens.
echo "=== PROBE diagnostic 14-pass trace (profiler ON; never an e2e number) ==="
TRACE_DIR="$OUT/traces"
stop_engine
start_engine 1 "$OUT/serve_probe_prof.log" "$TRACE_DIR"
"$V/bin/python" "$REPO/tools/r0_k3_capture_passes.py" --port "$PORT" \
  --context 4096 --max-tokens "$MAXTOK" --target-passes 14 \
  --out "$OUT/probe_trace_window.json"
stop_engine

TRACE=$(find "$TRACE_DIR" -type f \( -name '*.pt.trace.json.gz' -o -name '*.json.gz' \) \
  -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
if [[ -z "$TRACE" ]]; then
  echo "ERROR: profiler trace not found under $TRACE_DIR" >&2
  exit 2
fi

read -r PASSES EMITTED <<EOF
$("$V/bin/python" - "$OUT/probe_trace_window.json" <<'PY'
import json, sys
j=json.load(open(sys.argv[1]))
p=int(j["verification_passes_in_window"])
a=int(j["accepted_in_window"])
print(p, p+a)
PY
)
EOF

"$V/bin/python" "$REPO/tools/r0_k3_amdahl.py" "$TRACE" \
  --passes "$PASSES" --emitted "$EMITTED" \
  --json-out "$OUT/probe_amdahl.json" \
  > "$OUT/probe_amdahl.txt"

"$V/bin/python" "$REPO/tools/r0_dense_output_probe_report.py" \
  "$OUT/probe_amdahl.json" | tee "$OUT/probe_component_delta.json"

XID1=$(xid_now); XID1=${XID1:-0}

"$V/bin/python" - "$OUT" "$GAIN" "$PARITY" "$XID0" "$XID1" <<'PY'
import json, os, sys
d,gain,parity,x0,x1=sys.argv[1],float(sys.argv[2]),sys.argv[3],int(sys.argv[4]),int(sys.argv[5])
def load(name):
    p=os.path.join(d,name)
    return json.load(open(p)) if os.path.exists(p) else None
out={
  "base_4k": load("ref_4096.json"),
  "probe_4k": load("parity_k3_4096.json"),
  "probe_160k": load("probe_160000.json"),
  "gain_pct_4k": gain,
  "parity": parity,
  "probe_components": load("probe_component_delta.json"),
  "xid_delta": x1-x0,
}
json.dump(out, open(os.path.join(d,"dense_output_probe_summary.json"),"w"), indent=2)
print(json.dumps({
  "base_ms": out["base_4k"]["ms_per_output_token_median"],
  "probe_ms": out["probe_4k"]["ms_per_output_token_median"],
  "gain_pct": gain,
  "parity": parity,
  "xid_delta": x1-x0,
}, indent=2))
PY

echo "=== done ==="
echo "This was a diagnostic precision probe only. Do not productionize it."
echo "results=$OUT"

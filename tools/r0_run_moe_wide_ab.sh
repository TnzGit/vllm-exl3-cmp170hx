#!/usr/bin/env bash
# Compare EXL3 cooperative MoE wide/wide (Ampere default) against narrow/narrow.
#
# This is a measurement experiment only.  It uses the already-compiled
# EXL3_MOE_COOP_WIDE knob; no CUDA rebuild is performed.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/mtp-k3-moe-wide-ab}"
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
  local mode=$1 log=$2 profiler_dir=${3:-}
  local -a env_unset=()
  local -a env_vars=()

  if [[ "$mode" == "auto" ]]; then
    env_unset+=(-u EXL3_MOE_COOP_WIDE)
  elif [[ "$mode" == "narrow" ]]; then
    env_vars+=(EXL3_MOE_COOP_WIDE=0)
  else
    echo "bad mode: $mode" >&2
    return 2
  fi

  if [[ -n "$profiler_dir" ]]; then
    mkdir -p "$profiler_dir"
    env_vars+=(TORCH_PROFILER_DIR="$profiler_dir" TORCH_PROFILER_RECORD_SHAPES=0)
  else
    env_unset+=(-u TORCH_PROFILER_DIR -u TORCH_PROFILER_RECORD_SHAPES)
  fi

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
      grep -E '^(VLLM_EXL3_COOP|EXL3_MOE_COOP_WIDE|TORCH_PROFILER_DIR)=' || true
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

echo "=== provenance / source attestation ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"

INSTALLED_PLUGIN="$("$V/bin/python" - <<'PY'
import vllm_exl3.exl3 as m
print(m.__file__)
PY
)"
echo "installed_plugin=$INSTALLED_PLUGIN"
echo "repo_plugin_sha256=$(sha256sum "$REPO/src/vllm_exl3/exl3.py" | awk '{print $1}')"
echo "installed_plugin_sha256=$(sha256sum "$INSTALLED_PLUGIN" | awk '{print $1}')"
if ! cmp -s "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"; then
  echo "REFUSE: installed vllm_exl3/exl3.py does not match this branch." >&2
  echo "Restore/sync the accepted production plugin source before GPU work." >&2
  echo "Do not silently benchmark a stale or probe plugin." >&2
  exit 2
fi

EXL3_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import exllamav3
print(Path(exllamav3.__file__).resolve().parent)
PY
)"
MOE_CU="$EXL3_ROOT/exllamav3_ext/quant/exl3_moe_coop.cu"
echo "exllamav3_root=$EXL3_ROOT"
echo "exllamav3_moe_cu_sha256=$(sha256sum "$MOE_CU" | awk '{print $1}')"
"$V/bin/python" - <<'PY'
from exllamav3.version import __version__
print("exllamav3_version=" + __version__)
PY

for marker in \
  'EXL3_MOE_COOP_WIDE' \
  'if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true' \
  'const bool wide_a = moe_coop_pick_wide(p.Hi / 16, slots, device)' \
  'const bool wide_b = moe_coop_pick_wide(p.I / 16, slots, device)'
do
  grep -Fq "$marker" "$MOE_CU" || {
    echo "REFUSE: expected ExLlamaV3 coop-wide source marker missing: $marker" >&2
    exit 2
  }
done

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile \
  "$REPO/tools/r0_k3_cell.py" \
  "$REPO/tools/r0_k3_capture_passes.py" \
  "$REPO/tools/r0_k3_amdahl.py" \
  "$REPO/tools/r0_moe_wide_compare.py"
"$V/bin/python" -m pytest -q \
  "$REPO/tests/test_mtp_denominator_contract.py" \
  "$REPO/tests/test_moe_coop_expert_range.py"

XID0=$(xid_now); XID0=${XID0:-0}

echo "=== AUTO: Ampere policy = wide/wide ==="
stop_engine
start_engine auto "$OUT/serve_auto.log"
guard_idle
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
  --max-tokens "$MAXTOK" --repeats 5 --tag "moe-auto-4k" \
  --out "$OUT/ref_4096.json" >/dev/null
guard_idle
stop_engine

echo "=== NARROW: forced narrow/narrow ==="
start_engine narrow "$OUT/serve_narrow.log"
guard_idle
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
  --max-tokens "$MAXTOK" --repeats 5 --tag "moe-narrow-4k" \
  --out "$OUT/parity_k3_4096.json" >/dev/null
guard_idle
stop_engine

"$V/bin/python" "$REPO/tools/r0_k3_parity_recheck.py" --dir "$OUT" |
  tee "$OUT/parity.txt"

"$V/bin/python" - "$OUT/ref_4096.json" "$OUT/parity_k3_4096.json" <<'PY' | tee "$OUT/perf_summary.json"
import json, sys
a=json.load(open(sys.argv[1]))
b=json.load(open(sys.argv[2]))
am=a["ms_per_output_token_median"]; bm=b["ms_per_output_token_median"]
at=a["output_tok_s_median"]; bt=b["output_tok_s_median"]
print(json.dumps({
  "auto_ms_per_token": am,
  "narrow_ms_per_token": bm,
  "latency_gain_pct_narrow": (am-bm)/am*100,
  "auto_tok_s": at,
  "narrow_tok_s": bt,
  "throughput_gain_pct_narrow": (bt-at)/at*100,
  "auto_accepted_per_pass": a.get("accepted_per_pass"),
  "narrow_accepted_per_pass": b.get("accepted_per_pass"),
  "auto_emitted_per_pass": a.get("emitted_per_pass"),
  "narrow_emitted_per_pass": b.get("emitted_per_pass"),
  "auto_status": a.get("status"),
  "narrow_status": b.get("status"),
}, indent=2))
PY

capture_trace() {
  local mode=$1
  local dir="$OUT/traces_$mode"
  local log="$OUT/serve_${mode}_prof.log"
  local cap="$OUT/${mode}_trace_window.json"
  local amd="$OUT/${mode}_amdahl.json"
  local txt="$OUT/${mode}_amdahl.txt"

  stop_engine
  start_engine "$mode" "$log" "$dir"
  "$V/bin/python" "$REPO/tools/r0_k3_capture_passes.py" --port "$PORT" \
    --context 4096 --max-tokens "$MAXTOK" --target-passes 14 --out "$cap"
  stop_engine

  local trace
  trace=$(find "$dir" -type f \( -name '*.pt.trace.json.gz' -o -name '*.json.gz' \) \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- || true)
  if [[ -z "$trace" ]]; then
    echo "ERROR: no profiler trace for $mode" >&2
    exit 2
  fi
  echo "$trace" > "$OUT/${mode}_trace_path.txt"

  read -r passes emitted <<EOF
$("$V/bin/python" - "$cap" <<'PY'
import json, sys
j=json.load(open(sys.argv[1]))
p=int(j["verification_passes_in_window"])
a=int(j["accepted_in_window"])
print(p, p+a)
PY
)
EOF
  "$V/bin/python" "$REPO/tools/r0_k3_amdahl.py" "$trace" \
    --passes "$passes" --emitted "$emitted" --json-out "$amd" > "$txt"
}

echo "=== diagnostic trace AUTO wide/wide ==="
capture_trace auto

echo "=== diagnostic trace NARROW narrow/narrow ==="
capture_trace narrow

"$V/bin/python" "$REPO/tools/r0_moe_wide_compare.py" \
  "$OUT/auto_amdahl.json" "$OUT/narrow_amdahl.json" |
  tee "$OUT/moe_wide_compare.json"

XID1=$(xid_now); XID1=${XID1:-0}
echo "xid_before=$XID0 xid_after=$XID1 xid_delta=$((XID1-XID0))"

echo "=== final ==="
echo "results=$OUT"
echo "No production source or CUDA extension was modified."

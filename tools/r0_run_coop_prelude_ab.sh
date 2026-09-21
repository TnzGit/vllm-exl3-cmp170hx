#!/usr/bin/env bash
# BASE vs EARLY cooperative-MoE prelude vs EARLY+EMPTY output.
#
# Pure-Python experiment. The runner temporarily installs this branch's exl3.py
# only after proving installed plugin == frozen production baseline and csrc has
# no branch delta. EXIT trap always restores the original installed source.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/mtp-k3-coop-prelude}"
PORT="${PORT:-8002}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAXLEN="${MAX_MODEL_LEN:-246000}"
MAXTOK="${MAXTOK:-256}"
BASE_SHA="f2c6a719a1709ba40d08b97a4cc8d64b3c2256d8"

export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
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

start_engine() {
  local mode=$1 log=$2 profiler_dir=${3:-}
  local early=0 empty=0
  case "$mode" in
    base) early=0; empty=0 ;;
    early) early=1; empty=0 ;;
    empty) early=1; empty=1 ;;
    *) echo "bad mode: $mode" >&2; return 2 ;;
  esac

  local -a env_unset=()
  local -a env_vars=(
    VLLM_EXL3_COOP_EARLY_PRELUDE="$early"
    VLLM_EXL3_COOP_OUT_EMPTY="$empty"
  )
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

  local engine
  engine=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
  echo "mode=$mode engine_pid=$engine"
  if [[ -n "$engine" ]]; then
    echo "engine_pwd=$(readlink /proc/$engine/cwd)"
    tr '\0' '\n' < "/proc/$engine/environ" |
      grep -E '^(VLLM_EXL3_COOP|VLLM_EXL3_COOP_EARLY_PRELUDE|VLLM_EXL3_COOP_OUT_EMPTY|TORCH_PROFILER_DIR)=' || true
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

echo "=== provenance / plugin installation guard ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "base_sha=$BASE_SHA"

if ! git -C "$REPO" diff --quiet "$BASE_SHA" HEAD -- csrc; then
  echo "REFUSE: branch changes csrc; pure-Python temporary install is unsafe." >&2
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
echo "baseline_expected_sha256=$(sha256sum "$BASE_EXPECTED" | awk '{print $1}')"
echo "installed_before_sha256=$(sha256sum "$INSTALLED_PLUGIN" | awk '{print $1}')"
echo "branch_plugin_sha256=$(sha256sum "$REPO/src/vllm_exl3/exl3.py" | awk '{print $1}')"

if ! cmp -s "$BASE_EXPECTED" "$INSTALLED_PLUGIN"; then
  echo "REFUSE: installed plugin is not the frozen production baseline $BASE_SHA." >&2
  echo "Restore the accepted baseline plugin before running this experiment." >&2
  exit 2
fi

PLUGIN_BACKUP="$OUT/exl3_installed_backup.py"
cp "$INSTALLED_PLUGIN" "$PLUGIN_BACKUP"
cp "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"
"$V/bin/python" -m py_compile "$INSTALLED_PLUGIN"
cmp -s "$REPO/src/vllm_exl3/exl3.py" "$INSTALLED_PLUGIN"

SO_PATH="$("$V/bin/python" - <<'PY'
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
  "$REPO/tools/r0_k3_capture_passes.py" \
  "$REPO/tools/r0_k3_amdahl.py" \
  "$REPO/tools/r0_coop_prelude_compare.py"
"$V/bin/python" -m pytest -q \
  "$REPO/tests/test_coop_early_prelude.py" \
  "$REPO/tests/test_mtp_denominator_contract.py" \
  "$REPO/tests/test_moe_coop_expert_range.py"

XID0=$(xid_now); XID0=${XID0:-0}

run_cell() {
  local mode=$1 outfile=$2 log=$3
  stop_engine
  start_engine "$mode" "$log"
  guard_idle
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 4096 \
    --max-tokens "$MAXTOK" --repeats 5 --tag "coop-prelude-$mode" \
    --out "$outfile" >/dev/null
  guard_idle
  stop_engine
}

echo "=== formal BASE ==="
run_cell base "$OUT/base_4096.json" "$OUT/serve_base.log"

echo "=== formal EARLY ==="
run_cell early "$OUT/early_4096.json" "$OUT/serve_early.log"

echo "=== formal EARLY+EMPTY ==="
run_cell empty "$OUT/empty_4096.json" "$OUT/serve_empty.log"

"$V/bin/python" - "$OUT/base_4096.json" "$OUT/early_4096.json" "$OUT/empty_4096.json" > "$OUT/perf_and_parity.json" <<'PY'
import json, sys

paths={"base":sys.argv[1],"early":sys.argv[2],"early_empty":sys.argv[3]}
js={k:json.load(open(v)) for k,v in paths.items()}

def flat(j):
    out=[]
    for cell in j.get("cells") or []:
        for piece in cell.get("token_pieces") or []:
            out.extend(piece if isinstance(piece,list) else [piece])
        if out:
            break
    return out

base_tok=flat(js["base"])
base_ms=js["base"]["ms_per_output_token_median"]
base_ts=js["base"]["output_tok_s_median"]
out={}
for name,j in js.items():
    ms=j["ms_per_output_token_median"]
    ts=j["output_tok_s_median"]
    tok=flat(j)
    out[name]={
        "ms_per_output_token":ms,
        "output_tok_s":ts,
        "latency_gain_pct_vs_base":(base_ms-ms)/base_ms*100,
        "throughput_gain_pct_vs_base":(ts-base_ts)/base_ts*100,
        "accepted_per_pass":j.get("accepted_per_pass"),
        "emitted_per_pass":j.get("emitted_per_pass"),
        "status":j.get("status"),
        "parity_vs_base":"PASS" if tok == base_tok else "MISMATCH",
        "first_mismatch": next(
            (i for i,(a,b) in enumerate(zip(base_tok,tok)) if a != b),
            None,
        ) if tok != base_tok else None,
    }
print(json.dumps(out, indent=2))
PY
cat "$OUT/perf_and_parity.json"

capture_trace() {
  local mode=$1
  local dir="$OUT/traces_$mode"
  local cap="$OUT/${mode}_trace_window.json"
  local amd="$OUT/${mode}_amdahl.json"
  local txt="$OUT/${mode}_amdahl.txt"

  stop_engine
  start_engine "$mode" "$OUT/serve_${mode}_prof.log" "$dir"
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

echo "=== diagnostic traces ==="
capture_trace base
capture_trace early
capture_trace empty

"$V/bin/python" "$REPO/tools/r0_coop_prelude_compare.py" \
  "$OUT/base_amdahl.json" "$OUT/early_amdahl.json" "$OUT/empty_amdahl.json" \
  > "$OUT/coop_prelude_compare.json"
cat "$OUT/coop_prelude_compare.json"

# Long-context sentinel only for a >=3% parity-clean winner.
BEST_MODE="$("$V/bin/python" - "$OUT/perf_and_parity.json" <<'PY'
import json, sys
j=json.load(open(sys.argv[1]))
c=[(v["latency_gain_pct_vs_base"],k) for k,v in j.items()
   if k != "base" and v["parity_vs_base"] == "PASS" and v["status"] == "VALID"]
gain,name=max(c, default=(-999,"none"))
print(name if gain >= 3.0 else "none")
PY
)"
if [[ "$BEST_MODE" != "none" ]]; then
  echo "=== 160K sentinel for $BEST_MODE ==="
  RUN_MODE="$BEST_MODE"
  [[ "$RUN_MODE" == "early_empty" ]] && RUN_MODE="empty"
  stop_engine
  start_engine "$RUN_MODE" "$OUT/serve_${BEST_MODE}_160k.log"
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port "$PORT" --context 160000 \
    --max-tokens "$MAXTOK" --repeats 3 --tag "coop-prelude-${BEST_MODE}-160k" \
    --out "$OUT/${BEST_MODE}_160000.json" >/dev/null
  stop_engine
fi

XID1=$(xid_now); XID1=${XID1:-0}
SO_SHA_AFTER=$(sha256sum "$SO_PATH" | awk '{print $1}')
echo "xid_before=$XID0 xid_after=$XID1 xid_delta=$((XID1-XID0))"
echo "extension_sha256_after=$SO_SHA_AFTER"
if [[ "$SO_SHA_AFTER" != "$SO_SHA_BEFORE" ]]; then
  echo "ERROR: compiled extension changed unexpectedly" >&2
  exit 2
fi

echo "=== done ==="
echo "results=$OUT"
echo "EXIT trap will restore the original installed plugin."

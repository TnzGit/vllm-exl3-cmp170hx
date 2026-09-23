#!/usr/bin/env bash
# Matched stock-KV eager/graph decode controls for the Q2E 16K benchmark.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
VENV="${R0_VENV:-$ROOT/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
CASE="${K1Q2E_CONTROL_CASE:-$ROOT/results/kvmem-k1q2e-benchmark-61a8eb2-live3/turns/ctx16000/turn_04_ask_d_e.json}"
OUT="${1:?pass a new output directory}"
EXPECTED_SHA="${EXPECTED_SHA:?set exact source SHA}"
PORT="${PORT:-8002}"
QSA_BASE_SHA="c501c2d2b647bdc93be3c0102f36a84446767b20cbff123314173e04757a026d"

[[ ! -e "$OUT" ]] || { echo "REFUSE: output exists: $OUT" >&2; exit 2; }
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$EXPECTED_SHA" ]] || { echo "REFUSE: SHA mismatch" >&2; exit 2; }
[[ -z "$(git -C "$REPO" status --porcelain)" ]] || { echo "REFUSE: dirty worktree" >&2; exit 2; }
[[ -f "$CASE" ]] || { echo "REFUSE: missing exact case" >&2; exit 2; }
mkdir -p "$OUT"
echo "$EXPECTED_SHA" > "$OUT/exact_sha.txt"
sha256sum "$CASE" > "$OUT/case.sha256"

export CUDA_HOME="${CUDA_HOME:-$VENV/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"
SITE_PACKAGES="$("$VENV/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
LIB_DIRS=("$SITE_PACKAGES/torch/lib")
for directory in "$SITE_PACKAGES"/nvidia/*/lib; do
  [[ -d "$directory" ]] && LIB_DIRS+=("$directory")
done
LIB_PATH="$(IFS=:; echo "${LIB_DIRS[*]}")"
export LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

QSA="$("$VENV/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent / 'models/qwen4_exp/nvidia/qsa.py')
PY
)"
[[ "$(sha256sum "$QSA" | awk '{print $1}')" == "$QSA_BASE_SHA" ]] || { echo "REFUSE: QSA is not stock baseline" >&2; exit 2; }
if curl -s -m 2 "http://127.0.0.1:$PORT/health" >/dev/null; then
  echo "REFUSE: port busy" >&2; exit 2
fi
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null |
  tr -d ' ' | grep -E '^[0-9]+$' || true)"
[[ -z "$GPU_PIDS" ]] || { echo "REFUSE: GPU busy" >&2; exit 2; }

XID_BEFORE="$(journalctl -k 2>/dev/null | grep -ciE 'NVRM: Xid' || true)"
LAUNCH_PID=""
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
trap stop_engine EXIT

wait_healthy() {
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then return 0; fi
    if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
      echo "ENGINE EXITED BEFORE HEALTHY" >&2
      tail -c 9000 "$OUT/$mode.serve.log" >&2
      return 1
    fi
    sleep 5
  done
  echo "HEALTH TIMEOUT" >&2
  return 1
}

for mode in eager graph; do
  echo "=== stock $mode ==="
  if [[ "$mode" == eager ]]; then EAGER=1; else EAGER=0; fi
  env -u VLLM_QWEN_KVMEM_Q2D_RUNTIME_PLAN \
    -u VLLM_QWEN_KVMEM_Q2D_WORKER_STATS_PATH \
    -u VLLM_QWEN_KVMEM_Q2D_SCHED_STATS_PATH \
    ENFORCE_EAGER="$EAGER" NUM_SPEC_TOKENS=0 VLLM_EXL3_COOP=1 \
    MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL=0.92 \
    MAX_MODEL_LEN=246000 MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=1024 PORT="$PORT" \
    setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
    > "$OUT/$mode.serve.log" 2>&1 < /dev/null &
  LAUNCH_PID=$!
  wait_healthy
  "$VENV/bin/python" "$REPO/tools/kvmem_q2e_benchmark_client.py" \
    --port "$PORT" --case "$CASE" --max-tokens 16 \
    --allow-semantic-failure --out "$OUT/$mode.warmup.json" > /dev/null
  "$VENV/bin/python" "$REPO/tools/kvmem_q2e_benchmark_client.py" \
    --port "$PORT" --case "$CASE" --max-tokens 256 \
    --out "$OUT/$mode.cell.json" > "$OUT/$mode.cell.stdout.json"
  stop_engine
  [[ "$(sha256sum "$QSA" | awk '{print $1}')" == "$QSA_BASE_SHA" ]] || { echo "FAIL: QSA drift" >&2; exit 3; }
  XID_AFTER="$(journalctl -k 2>/dev/null | grep -ciE 'NVRM: Xid' || true)"
  [[ "$XID_AFTER" == "$XID_BEFORE" ]] || { echo "FAIL: Xid changed" >&2; exit 3; }
done
"$VENV/bin/python" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
result = {}
for mode in ('eager', 'graph'):
    cell = json.loads((out / f'{mode}.cell.json').read_text())
    result[mode] = {k: cell[k] for k in (
        'prompt_tokens', 'completion_tokens', 'prefill_s', 'prefill_tok_s',
        'decode_s', 'decode_tok_s', 'decode_ms_per_token',
        'target_codes_in_order', 'count_exact')}
(out / 'summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PY
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader
echo Q2E_DECODE_CONTROLS_DONE

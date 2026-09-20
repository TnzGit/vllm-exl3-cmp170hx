#!/usr/bin/env bash
# R0 MTP launcher: same contract as r0_serve_env.sh plus a speculative config.
#
# usage: NUM_SPEC_TOKENS=1 MODEL_DIR=... GPU_MEM_UTIL=... tools/r0_serve_mtp.sh
#
# Prefix caching stays explicitly off for the first MTP sweep so open upstream
# hybrid-cache annotation bugs are not mixed into the throughput baseline.
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-cmp170hx}"

CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export VLLM_EXL3_NGRAM_TABLE="${VLLM_EXL3_NGRAM_TABLE:-disk}"
export VLLM_EXL3_NGRAM_KERNEL="${VLLM_EXL3_NGRAM_KERNEL:-ext}"

: "${MODEL_DIR:?set MODEL_DIR to the prepared EXL3 pack}"
: "${GPU_MEM_UTIL:?set GPU_MEM_UTIL explicitly for the target card}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:?set NUM_SPEC_TOKENS (use 0 for no-draft)}"

if [[ "$PORT" == "8000" && "${ALLOW_PORT_8000:-0}" != "1" ]]; then
  echo "REFUSE: port 8000 is reserved for the existing production service." >&2
  exit 2
fi

SPLIT_OPS="$(
  "$V/bin/python" - <<'PY'
import json
from vllm.config import CompilationConfig
print(json.dumps(list(CompilationConfig._attention_ops) + ["vllm::exl3_ngram_lookup_out"]))
PY
)"
COMPILATION_CONFIG="$(
  "$V/bin/python" - "$SPLIT_OPS" <<'PY'
import json, sys
print(json.dumps({"cudagraph_mode": "PIECEWISE", "splitting_ops": json.loads(sys.argv[1])}))
PY
)"

ARGS=(
  serve "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --quantization exl3
  --language-model-only
  --no-enable-prefix-caching
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --compilation-config "$COMPILATION_CONFIG"
)
if [[ "$NUM_SPEC_TOKENS" != "0" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}")
fi

echo "[runtime] NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS} MAX_MODEL_LEN=${MAX_MODEL_LEN} PORT=${PORT}"
echo "[runtime] NGRAM_TABLE=${VLLM_EXL3_NGRAM_TABLE} CUDA_HOME=${CUDA_HOME}"
exec vllm "${ARGS[@]}"

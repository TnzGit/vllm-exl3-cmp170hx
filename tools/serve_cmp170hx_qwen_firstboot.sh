#!/usr/bin/env bash
# Conservative first-boot launcher for Qwen3.8-Flash-Next EXL3 on one CMP170HX.
#
# Required:
#   MODEL_DIR=/path/to/prepared/exl3-pack
#   GPU_MEM_UTIL=<explicit vLLM fraction>
#
# Optional:
#   PORT=8002 MAX_MODEL_LEN=4096 MAX_NUM_SEQS=1 HOST=127.0.0.1
#
# This script does not patch vLLM. Run apply_qwen4_exp_patches.py separately.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_DIR:?set MODEL_DIR to the prepared EXL3 pack}"
: "${GPU_MEM_UTIL:?set GPU_MEM_UTIL explicitly for the target card}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

if [[ "$PORT" == "8000" && "${ALLOW_PORT_8000:-0}" != "1" ]]; then
  echo "REFUSE: port 8000 is reserved for the existing production service." >&2
  echo "Set ALLOW_PORT_8000=1 only when that reservation is intentionally lifted." >&2
  exit 2
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "ERROR: missing $MODEL_DIR/config.json" >&2
  exit 2
fi

export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_EXL3_NGRAM_KERNEL="${VLLM_EXL3_NGRAM_KERNEL:-ext}"

echo "[preflight]"
python "$ROOT/tools/cmp170hx_qwen_preflight.py"

SPLIT_OPS="$(
  python - <<'PY'
import json
from vllm.config import CompilationConfig

print(
    json.dumps(
        list(CompilationConfig._attention_ops)
        + ["vllm::exl3_ngram_lookup_out"]
    )
)
PY
)"

COMPILATION_CONFIG="$(
  python - "$SPLIT_OPS" <<'PY'
import json
import sys

ops = json.loads(sys.argv[1])
print(json.dumps({"cudagraph_mode": "PIECEWISE", "splitting_ops": ops}))
PY
)"

ARGS=(
  serve "$MODEL_DIR"
  --host "$HOST"
  --port "$PORT"
  --quantization exl3
  --language-model-only
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --compilation-config "$COMPILATION_CONFIG"
)

echo "[runtime]"
printf '  %-28s %s\n' \
  "MODEL_DIR" "$MODEL_DIR" \
  "GPU_MEM_UTIL" "$GPU_MEM_UTIL" \
  "MAX_MODEL_LEN" "$MAX_MODEL_LEN" \
  "MAX_NUM_SEQS" "$MAX_NUM_SEQS" \
  "PORT" "$PORT" \
  "VLLM_EXL3_NGRAM_TABLE" "$VLLM_EXL3_NGRAM_TABLE" \
  "VLLM_EXL3_NGRAM_KERNEL" "$VLLM_EXL3_NGRAM_KERNEL"
echo "  speculation                  disabled"
echo "  service profile              text-only"
echo "  CUDA graph mode              PIECEWISE"

echo "[exec] vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"

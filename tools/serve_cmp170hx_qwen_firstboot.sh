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
# This script does not patch vLLM. For R0 first boot run:\n#   python tools/apply_qwen4_exp_patches.py <site-packages/vllm> --profile text-no-draft
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

export VLLM_EXL3_MODEL_DIR="$MODEL_DIR"
export VLLM_EXL3_TRELLIS_ARENA="${VLLM_EXL3_TRELLIS_ARENA:-1}"
export VLLM_EXL3_ARENA_PRESCAN="${VLLM_EXL3_ARENA_PRESCAN:-1}"
# CMP170HX is a discrete GPU: keep file-backed page cache reclaim under the
# kernel rather than issuing one MADV_DONTNEED per routed tensor. These are
# launcher defaults only; library defaults remain backward-compatible.
export VLLM_EXL3_MADV_AFTER_H2D="${VLLM_EXL3_MADV_AFTER_H2D:-0}"
export VLLM_EXL3_EXPERT_MATCH_CACHE="${VLLM_EXL3_EXPERT_MATCH_CACHE:-1}"
export VLLM_EXL3_GC_AFTER_MOE_LAYER="${VLLM_EXL3_GC_AFTER_MOE_LAYER:-0}"
export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_EXL3_NGRAM_KERNEL="${VLLM_EXL3_NGRAM_KERNEL:-ext}"

NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-0}"
if [[ "$NUM_SPEC_TOKENS" == "0" ]]; then
  PREFLIGHT_PROFILE="text-no-draft"
else
  PREFLIGHT_PROFILE="text-mtp"
fi

echo "[preflight] profile=$PREFLIGHT_PROFILE"
python "$ROOT/tools/cmp170hx_qwen_preflight.py" --profile "$PREFLIGHT_PROFILE"

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
  --no-enable-prefix-caching
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --compilation-config "$COMPILATION_CONFIG"
)

# NUM_SPEC_TOKENS=0 keeps the no-draft profile; any other value enables the
# Qwen4Exp MTP draft. Prefix caching stays off for the whole MTP baseline so
# the open hybrid/MTP cache-annotation issues are not a confound.
if [[ "$NUM_SPEC_TOKENS" != "0" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}")
fi

echo "[runtime]"
printf '  %-28s %s\n' \
  "MODEL_DIR" "$MODEL_DIR" \
  "GPU_MEM_UTIL" "$GPU_MEM_UTIL" \
  "MAX_MODEL_LEN" "$MAX_MODEL_LEN" \
  "MAX_NUM_SEQS" "$MAX_NUM_SEQS" \
  "PORT" "$PORT" \
  "NUM_SPEC_TOKENS" "$NUM_SPEC_TOKENS" \
  "VLLM_EXL3_MODEL_DIR" "$VLLM_EXL3_MODEL_DIR" \
  "VLLM_EXL3_TRELLIS_ARENA" "$VLLM_EXL3_TRELLIS_ARENA" \
  "VLLM_EXL3_ARENA_PRESCAN" "$VLLM_EXL3_ARENA_PRESCAN" \
  "VLLM_EXL3_MADV_AFTER_H2D" "$VLLM_EXL3_MADV_AFTER_H2D" \
  "VLLM_EXL3_EXPERT_MATCH_CACHE" "$VLLM_EXL3_EXPERT_MATCH_CACHE" \
  "VLLM_EXL3_GC_AFTER_MOE_LAYER" "$VLLM_EXL3_GC_AFTER_MOE_LAYER" \
  "VLLM_EXL3_NGRAM_TABLE" "$VLLM_EXL3_NGRAM_TABLE" \
  "VLLM_EXL3_NGRAM_KERNEL" "$VLLM_EXL3_NGRAM_KERNEL"
echo "  speculation                  ${NUM_SPEC_TOKENS} draft token(s) (0 = disabled)"
echo "  prefix caching               disabled explicitly"
echo "  service profile              text-only"
echo "  CUDA graph mode              PIECEWISE"

echo "[exec] vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"

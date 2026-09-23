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
# The Qwen3.8-Flash-Next routed MoE geometry is eligible for the ExLlamaV3
# cooperative decode kernel on one CMP170HX. Hardware A/B measured a
# context-independent 30.19 -> 19.05 ms/output-token win with parity/Xid clean.
export VLLM_EXL3_COOP="${VLLM_EXL3_COOP:-1}"
export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_EXL3_NGRAM_KERNEL="${VLLM_EXL3_NGRAM_KERNEL:-ext}"

echo "[preflight]"
python "$ROOT/tools/cmp170hx_qwen_preflight.py" --profile text-no-draft

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

# Optional explicit prefill/decode scheduling budget. An empty value retains
# vLLM's installed default, so existing launch profiles are unchanged.
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi

# NUM_SPEC_TOKENS=0 keeps the no-draft profile; any other value enables the
# Qwen4Exp MTP draft. Requires the text-mtp patch stack. Prefix caching stays
# off for every MTP cell so the open hybrid/MTP cache issues are not a confound.
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-0}"
if [[ "$NUM_SPEC_TOKENS" != "0" ]]; then
  ARGS+=(--speculative-config "{\"method\":\"qwen4_exp_mtp\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}")
fi

# Diagnostic-only server-side profiling. OFF unless TORCH_PROFILER_DIR is set;
# it adds --profiler-config and therefore must never be used for performance
# numbers. ENFORCE_EAGER additionally disables torch.compile + CUDA graphs so
# the profiler can correlate CPU ops to kernels (attribution only).
if [[ -n "${TORCH_PROFILER_DIR:-}" ]]; then
  mkdir -p "$TORCH_PROFILER_DIR"
  PROFILER_CONFIG="$(
    python - "$TORCH_PROFILER_DIR" <<'PYEOF'
import json, os, sys
print(json.dumps({
    "profiler": "torch",
    "torch_profiler_dir": os.path.abspath(sys.argv[1]),
    "torch_profiler_with_stack": True,
    "torch_profiler_record_shapes": os.environ.get("TORCH_PROFILER_RECORD_SHAPES", "0") == "1",
    "torch_profiler_with_memory": False,
    "torch_profiler_with_flops": False,
    "torch_profiler_use_gzip": True,
    "ignore_frontend": True,
}))
PYEOF
  )"
  ARGS+=(--profiler-config "$PROFILER_CONFIG")
fi
if [[ "${ENFORCE_EAGER:-0}" == "1" ]]; then
  ARGS+=(--enforce-eager)
fi

echo "[runtime]"
printf '  %-28s %s\n' \
  "MODEL_DIR" "$MODEL_DIR" \
  "GPU_MEM_UTIL" "$GPU_MEM_UTIL" \
  "MAX_MODEL_LEN" "$MAX_MODEL_LEN" \
  "MAX_NUM_SEQS" "$MAX_NUM_SEQS" \
  "MAX_NUM_BATCHED_TOKENS" "${MAX_NUM_BATCHED_TOKENS:-<default>}" \
  "PORT" "$PORT" \
  "NUM_SPEC_TOKENS" "$NUM_SPEC_TOKENS" \
  "VLLM_EXL3_MODEL_DIR" "$VLLM_EXL3_MODEL_DIR" \
  "VLLM_EXL3_TRELLIS_ARENA" "$VLLM_EXL3_TRELLIS_ARENA" \
  "VLLM_EXL3_ARENA_PRESCAN" "$VLLM_EXL3_ARENA_PRESCAN" \
  "VLLM_EXL3_MADV_AFTER_H2D" "$VLLM_EXL3_MADV_AFTER_H2D" \
  "VLLM_EXL3_EXPERT_MATCH_CACHE" "$VLLM_EXL3_EXPERT_MATCH_CACHE" \
  "VLLM_EXL3_GC_AFTER_MOE_LAYER" "$VLLM_EXL3_GC_AFTER_MOE_LAYER" \
  "VLLM_EXL3_COOP" "$VLLM_EXL3_COOP" \
  "VLLM_EXL3_NGRAM_TABLE" "$VLLM_EXL3_NGRAM_TABLE" \
  "VLLM_EXL3_NGRAM_KERNEL" "$VLLM_EXL3_NGRAM_KERNEL" \
  "TORCH_PROFILER_DIR" "${TORCH_PROFILER_DIR:-<off>}" \
  "ENFORCE_EAGER" "${ENFORCE_EAGER:-0}"
if [[ -n "${TORCH_PROFILER_DIR:-}" || "${ENFORCE_EAGER:-0}" == "1" ]]; then
  echo "  !! PROFILING/DIAGNOSTIC MODE: latency from this run is NOT a production number"
fi
echo "  speculation                  ${NUM_SPEC_TOKENS} draft token(s) (0 = disabled)"
echo "  prefix caching               disabled explicitly"
echo "  service profile              text-only"
echo "  CUDA graph mode              PIECEWISE"

echo "[exec] vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"

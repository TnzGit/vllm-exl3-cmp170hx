#!/usr/bin/env bash
# Profiling-only launcher for post-COOP eager attribution.
#
# This deliberately disables vLLM torch.compile + CUDA graphs via
# --enforce-eager so torch.profiler can correlate CPU launch ops with CUDA
# kernels. Do not use profiler-on/eager latency as a production performance
# measurement.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_DIR:?set MODEL_DIR}"
: "${GPU_MEM_UTIL:?set GPU_MEM_UTIL}"
: "${TORCH_PROFILER_DIR:?set TORCH_PROFILER_DIR}"

export ENFORCE_EAGER=1
export VLLM_EXL3_COOP="${VLLM_EXL3_COOP:-1}"
export TORCH_PROFILER_RECORD_SHAPES="${TORCH_PROFILER_RECORD_SHAPES:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export PORT="${PORT:-8002}"

echo "[eager-attribution]"
echo "  ENFORCE_EAGER=$ENFORCE_EAGER"
echo "  VLLM_EXL3_COOP=$VLLM_EXL3_COOP"
echo "  TORCH_PROFILER_RECORD_SHAPES=$TORCH_PROFILER_RECORD_SHAPES"
echo "  TORCH_PROFILER_DIR=$TORCH_PROFILER_DIR"
echo "  NOTE: this mode is attribution-only; do not compare its latency to production."

exec "$ROOT/tools/serve_cmp170hx_qwen_firstboot.sh"

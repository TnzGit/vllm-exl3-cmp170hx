#!/usr/bin/env bash
# R0 serve launcher wrapper: exports the isolated venv, CUDA_HOME and the
# disk n-gram environment before delegating to the first-boot script.
#
# usage: MODEL_DIR=... GPU_MEM_UTIL=... [MAX_MODEL_LEN=4096] [PORT=8002] \
#          tools/r0_serve_env.sh
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"

CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export VLLM_EXL3_NGRAM_TABLE="${VLLM_EXL3_NGRAM_TABLE:-disk}"
export VLLM_EXL3_NGRAM_KERNEL="${VLLM_EXL3_NGRAM_KERNEL:-ext}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_DIR:?set MODEL_DIR to the prepared EXL3 pack}"
: "${GPU_MEM_UTIL:?set GPU_MEM_UTIL explicitly for the target card}"

echo "[env] python=$V/bin/python CUDA_HOME=$CUDA_HOME"
echo "[env] VLLM_EXL3_NGRAM_TABLE=$VLLM_EXL3_NGRAM_TABLE"

exec env \
  MODEL_DIR="$MODEL_DIR" \
  GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
  PORT="${PORT:-8002}" \
  HOST="${HOST:-127.0.0.1}" \
  bash "$ROOT/tools/serve_cmp170hx_qwen_firstboot.sh"

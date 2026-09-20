#!/usr/bin/env bash
# COOP=1 + server-side torch profiler launcher (4K, no draft).
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-coopamd}"
PROFILE_ROOT="${PROFILE_ROOT:-$R/coop-amdahl}"
export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
mkdir -p "$PROFILE_ROOT/traces"
export TORCH_PROFILER_DIR="$PROFILE_ROOT/traces"
export VLLM_EXL3_COOP="${VLLM_EXL3_COOP:-1}"
: "${MODEL_DIR:?set MODEL_DIR}"
: "${GPU_MEM_UTIL:?set GPU_MEM_UTIL}"
export MODEL_DIR GPU_MEM_UTIL
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export PORT="${PORT:-8002}"
export NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-0}"
exec bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"

#!/usr/bin/env bash
# R0 loadfix A/B launcher: same contract as r0_serve_env.sh, but delegates to
# the loadfix worktree's first-boot script so VLLM_EXL3_MODEL_DIR /
# TRELLIS_ARENA / ARENA_PRESCAN are exported, and stamps T0..T3.
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-loadfix}"
STAMP="${STAMP:-$R/results/loadfix_timing.txt}"

CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export VLLM_EXL3_NGRAM_TABLE="${VLLM_EXL3_NGRAM_TABLE:-disk}"
export VLLM_EXL3_NGRAM_KERNEL="${VLLM_EXL3_NGRAM_KERNEL:-ext}"
export VLLM_EXL3_MODEL_DIR="${MODEL_DIR:?set MODEL_DIR}"
export VLLM_EXL3_TRELLIS_ARENA="${VLLM_EXL3_TRELLIS_ARENA:-1}"
export VLLM_EXL3_ARENA_PRESCAN="${VLLM_EXL3_ARENA_PRESCAN:-1}"

: "${GPU_MEM_UTIL:?set GPU_MEM_UTIL explicitly for the target card}"

mkdir -p "$(dirname "$STAMP")"
date +%s > "$STAMP.t0"
echo "T0=$(cat "$STAMP.t0")"

MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}" \
  PORT="${PORT:-8002}" HOST="${HOST:-127.0.0.1}" \
  bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"

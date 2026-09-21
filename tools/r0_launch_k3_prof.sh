#!/usr/bin/env bash
# Diagnostic-only: k=3 production graph trace (4K or 160K).
# Profiler ON; the resulting wall latency is NOT a production number.
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-k3a}"
export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export TORCH_PROFILER_DIR="${TORCH_PROFILER_DIR:-$R/results/mtp-k3-amdahl/traces}"
export TORCH_PROFILER_RECORD_SHAPES="${TORCH_PROFILER_RECORD_SHAPES:-1}"
mkdir -p "$TORCH_PROFILER_DIR"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
export VLLM_EXL3_COOP=1
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-246000}"
export MAX_NUM_SEQS=1 PORT=8002 NUM_SPEC_TOKENS=3
export ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
exec bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"

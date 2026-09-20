#!/usr/bin/env bash
# Diagnostic-only: k=2 (m=3) with the server-side profiler, to prove the coop
# MoE path and identify the dense EXL3 kernel family at each verify width.
# Never use this run's latency as a performance number.
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-mtp2}"
export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export TORCH_PROFILER_DIR="${PROFILE_ROOT:-$R/mtp-dispatch}/traces"
mkdir -p "$TORCH_PROFILER_DIR"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
export VLLM_EXL3_COOP=1
export MAX_MODEL_LEN=4096 MAX_NUM_SEQS=1 PORT=8002 NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-2}"
exec bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh"

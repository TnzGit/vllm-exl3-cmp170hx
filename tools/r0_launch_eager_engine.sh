#!/usr/bin/env bash
# Self-contained eager-attribution engine launcher (run this on the GPU node).
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-eager}"
PROFILE_ROOT="$R/eager-attribution"

export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"

export TORCH_PROFILER_DIR="$PROFILE_ROOT/traces"
export MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"

mkdir -p "$PROFILE_ROOT/traces" "$PROFILE_ROOT/result"

exec bash "$REPO/tools/r0_serve_coop_eager_profiler.sh"

#!/usr/bin/env bash
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
export R0_REPO="$HOME/vllm-exl3-amdahl"
CONTEXT=160000 OUT="$R/results/coop-ab-160k" bash "$R0_REPO/tools/r0_ab_coop.sh"

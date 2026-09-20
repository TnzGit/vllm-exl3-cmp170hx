#!/usr/bin/env bash
# Wrapper: run Phase S (k=3 sweep) with a strict pre-launch cleanup.
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
REPO="${R0_REPO:-$HOME/vllm-exl3-k3}"
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f r0_run_k3 2>/dev/null || true
sleep 6
rm -f "$R/results/mtp-k3/cell_k3_"*.json "$R/results/mtp-k3/parity_k3_"*.json
bash "$REPO/tools/r0_run_k3_phase_s.sh" "$R/results/mtp-k3"

#!/usr/bin/env bash
# Phase 2: COOP=1 matrix, no-draft + k=1/2/3, fresh engine per config.
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
REPO="${R0_REPO:-$HOME/vllm-exl3-mtp2}"
OUT="$R/results/mtp-post-coop2"
mkdir -p "$OUT"
for k in 0 1 2 3; do
  bash "$REPO/tools/r0_run_mtp_post_coop.sh" 1 "$k" "$OUT" || echo "config k=$k failed"
done
echo "phase2 complete"

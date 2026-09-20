#!/usr/bin/env bash
# One-shot CMP170HX low-risk load-policy qualification.
#
# Runs the existing load profiler with:
#   - MADV-after-H2D disabled (discrete GPU policy candidate)
#   - expert-name match memoization enabled
#   - per-MoE-layer full gc.collect disabled
#   - EXL3 load section counters enabled
#
# Usage:
#   MODEL_DIR=... GPU_MEM_UTIL=... tools/r0_profile_lowrisk.sh [outdir]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$HOME/.codex_tasks/qwen38-flashnext-r0/results/load-lowrisk}"

: "${MODEL_DIR:?set MODEL_DIR}"
: "${GPU_MEM_UTIL:?set GPU_MEM_UTIL}"

export VLLM_EXL3_LOAD_PROFILE=1
export VLLM_EXL3_MADV_AFTER_H2D=0
export VLLM_EXL3_EXPERT_MATCH_CACHE=1
export VLLM_EXL3_GC_AFTER_MOE_LAYER=0

mkdir -p "$OUT"

{
  echo "timestamp=$(date -Is)"
  echo "VLLM_EXL3_MADV_AFTER_H2D=$VLLM_EXL3_MADV_AFTER_H2D"
  echo "VLLM_EXL3_EXPERT_MATCH_CACHE=$VLLM_EXL3_EXPERT_MATCH_CACHE"
  echo "VLLM_EXL3_GC_AFTER_MOE_LAYER=$VLLM_EXL3_GC_AFTER_MOE_LAYER"
  grep -E '^(MemTotal|MemAvailable|Cached|SwapTotal|SwapFree):' /proc/meminfo || true
  nvidia-smi --query-gpu=memory.used,memory.free,power.draw,clocks.sm --format=csv,noheader || true
} > "$OUT/before.txt"

MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL"   bash "$ROOT/tools/r0_profile_load.sh" "$OUT"

{
  echo "timestamp=$(date -Is)"
  grep -E '^(MemTotal|MemAvailable|Cached|SwapTotal|SwapFree):' /proc/meminfo || true
  nvidia-smi --query-gpu=memory.used,memory.free,power.draw,clocks.sm --format=csv,noheader || true
} > "$OUT/after.txt"

echo "=== low-risk policy ==="
cat "$OUT/before.txt"
echo "=== after ==="
cat "$OUT/after.txt"
echo "=== load profile summary ==="
grep -A20 "EXL3 LOAD PROFILE" "$OUT/server.log" | tail -80 || true

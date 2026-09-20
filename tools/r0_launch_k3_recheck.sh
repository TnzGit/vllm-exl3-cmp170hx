#!/usr/bin/env bash
# Fresh-engine sentinel recheck for k=3 (sentinel drift exceeded the 3% band).
set -euo pipefail
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-k3}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="$R/results/mtp-k3"
export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
mkdir -p "$OUT"
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 6
LOG="$OUT/serve_k3_recheck.log"
VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL=0.92 \
  MAX_MODEL_LEN=246000 MAX_NUM_SEQS=1 PORT=8002 NUM_SPEC_TOKENS=3 \
  setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" > "$LOG" 2>&1 < /dev/null &
deadline=$((SECONDS + 2400))
while (( SECONDS < deadline )); do
  grep -q "Application startup complete" "$LOG" 2>/dev/null && break
  grep -q "Engine core initialization failed" "$LOG" 2>/dev/null && break
  sleep 5
done
echo "startup=$(grep -c 'Application startup complete' "$LOG")"
echo "markers: PRESCAN=$(grep -c 'PRESCAN ready' "$LOG") fallback=$(grep -c 'staging fallback (no arena plan)' "$LOG")"
# 5 repeats so a warm-up outlier cannot dominate the median
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context 4096 \
  --max-tokens 256 --repeats 5 --tag "k3-recheck-4k" \
  --out "$OUT/cell_k3_recheck_4k.json" > /dev/null 2>&1 || echo "recheck failed"
pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
echo "=== recheck complete ==="

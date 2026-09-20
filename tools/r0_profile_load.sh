#!/usr/bin/env bash
# Profile ONE no-draft R0 startup: server launch -> model load -> init -> healthy.
#
# nsys is not installed on this host, so CPU attribution comes from py-spy
# (Python stack sampling of the live EngineCore) plus perf for native/kernel
# context. No long generation is profiled; the server is stopped as soon as it
# reports healthy.
#
# usage: r0_profile_load.sh [outdir]
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-loadprofile}"
OUT="${1:-$R/results/loadprofile}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"

CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_EXL3_NGRAM_KERNEL=ext
export VLLM_EXL3_MODEL_DIR="$MODEL_DIR"
export VLLM_EXL3_TRELLIS_ARENA=1
export VLLM_EXL3_ARENA_PRESCAN=1

mkdir -p "$OUT"
LOG="$OUT/server.log"

pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
sleep 3

T0=$(date +%s)

# Start the server detached so this script can attach profilers to the child.
setsid bash -c "cd '$REPO' && MODEL_DIR='$MODEL_DIR' GPU_MEM_UTIL=0.92 \
  MAX_MODEL_LEN=4096 MAX_NUM_SEQS=1 PORT=8002 \
  bash tools/r0_serve_loadfix.sh" > "$LOG" 2>&1 < /dev/null &
SRV=$!
echo "T0=$T0 server_pid=$SRV"

# Wait for the EngineCore process to exist, then sample it while it loads.
EPID=""
for _ in $(seq 1 120); do
  EPID=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
  [ -n "$EPID" ] && break
  sleep 1
done
echo "engine_pid=$EPID"

# py-spy: native stacks off, idle threads out, 100 Hz over the whole startup.
if [ -n "$EPID" ] && [ -x "$V/bin/py-spy" ]; then
  setsid sudo -n "$V/bin/py-spy" record --pid "$EPID" \
    --rate 100 --nonblocking --idle --format speedscope \
    --output "$OUT/pyspy_speedscope.json" > "$OUT/pyspy.err" 2>&1 < /dev/null &
  echo "py-spy attached to $EPID (sudo)"
fi

# perf: CPU-clock sampling of the same process for native attribution.
if [ -n "$EPID" ] && command -v perf >/dev/null 2>&1; then
  setsid sudo -n perf record -F 99 -g --pid "$EPID" \
    --output "$OUT/perf.data" > "$OUT/perf.err" 2>&1 < /dev/null &
  echo "perf attached to $EPID (sudo)"
fi

# Wait until healthy.
DEADLINE=$((SECONDS + 1500))
while (( SECONDS < DEADLINE )); do
  if curl -s -m 5 http://127.0.0.1:8002/health >/dev/null 2>&1 \
     && curl -s -m 10 http://127.0.0.1:8002/v1/models | grep -q '"id"'; then
    break
  fi
  sleep 2
done
T3=$(date +%s)
echo "T3=$T3 total_startup_s=$((T3 - T0))"

# Stop profilers first so their outputs are flushed and complete.
sudo -n pkill -INT -f "py-spy record" 2>/dev/null || true
sudo -n pkill -INT -f "perf record" 2>/dev/null || true
sleep 10
sudo -n chown "$(id -u):$(id -g)" "$OUT"/perf.data "$OUT"/pyspy_speedscope.json 2>/dev/null || true

echo "=== markers ==="
for m in "EXL3 trellis PRESCAN ready" "mode=direct_plan" "mode=post_stage_pack" \
         "staging fallback summary" "staging fallback (no arena plan)"; do
  echo "$m = $(grep -c "$m" "$LOG" 2>/dev/null || echo 0)"
done
grep -m1 "Model loading took" "$LOG" | grep -oE "[0-9.]+ seconds" || true
grep -m1 "init engine" "$LOG" | grep -oE "took [0-9.]+ s" || true

pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f "r0_serve_loadfix" 2>/dev/null || true
sleep 5
echo "profile outputs in $OUT"
ls -la "$OUT"

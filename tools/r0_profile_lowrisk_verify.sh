#!/usr/bin/env bash
# Low-risk policy run 2: profile the load AND run the correctness gate while the
# server is healthy, so one startup serves both.
#
# usage: r0_profile_lowrisk_verify.sh [outdir]
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-lrq}"
OUT="${1:-$R/results/load-lowrisk2}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"

CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
export VLLM_EXL3_NGRAM_TABLE=disk
export VLLM_EXL3_NGRAM_KERNEL=ext
export VLLM_EXL3_LOAD_PROFILE=1
export VLLM_EXL3_MADV_AFTER_H2D=0
export VLLM_EXL3_EXPERT_MATCH_CACHE=1
export VLLM_EXL3_GC_AFTER_MOE_LAYER=0

mkdir -p "$OUT"
LOG="$OUT/server.log"

xid_now() { { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9'; }

pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f r0_serve_loadfix 2>/dev/null || true
sleep 4

XID0=$(xid_now); XID0=${XID0:-0}
MEM0=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
echo "xid_before=$XID0 memavailable_kb_before=$MEM0"

setsid bash -c "cd '$REPO' && MODEL_DIR='$MODEL_DIR' GPU_MEM_UTIL=0.92 \
  MAX_MODEL_LEN=4096 MAX_NUM_SEQS=1 PORT=8002 \
  bash tools/r0_serve_loadfix.sh" > "$LOG" 2>&1 < /dev/null &

# Track the lowest MemAvailable the host sees during the load.
MINMEM=$MEM0
for _ in $(seq 1 150); do
  cur=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$cur" -lt "$MINMEM" ] && MINMEM=$cur
  if curl -s -m 5 http://127.0.0.1:8002/health >/dev/null 2>&1 \
     && curl -s -m 10 http://127.0.0.1:8002/v1/models | grep -q '"id"'; then
    break
  fi
  sleep 2
done

EPID=$(pgrep -f "VLLM::EngineCore" | head -1 || true)
RSS=""
[ -n "$EPID" ] && RSS=$(grep -E "VmHWM|VmRSS" /proc/$EPID/status | tr '\n' ' ')

echo "=== markers ==="
for m in "EXL3 trellis PRESCAN ready" "mode=direct_plan" "mode=post_stage_pack" \
         "staging fallback summary" "staging fallback (no arena plan)"; do
  echo "$m = $(grep -c "$m" "$LOG" 2>/dev/null || echo 0)"
done

echo "=== timings ==="
grep -m1 "Loading weights took" "$LOG" | grep -oE "[0-9.]+ seconds" || true
grep -m1 "Model loading took" "$LOG" | grep -oE "[0-9.]+ seconds" || true

echo "=== correctness ==="
"$V/bin/python" - <<'PY'
import json, urllib.request
base = json.load(urllib.request.urlopen("http://127.0.0.1:8002/v1/models", timeout=30))["data"][0]["id"]
p = {"model": base, "prompt": "The capital of France is", "max_tokens": 8,
     "temperature": 0.0, "seed": 0, "logprobs": 1}
outs = []
for _ in range(4):
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8002/v1/completions",
        data=json.dumps(p).encode(),
        headers={"Content-Type": "application/json"}), timeout=180))
    c = r["choices"][0]
    outs.append((c["text"], c["logprobs"]["tokens"]))
print("text:", repr(outs[0][0]))
print("tokens:", outs[0][1][:5])
print("4/4 identical:", len({str(o) for o in outs}) == 1)
print("usage:", r["usage"])
PY

MEM1=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
XID1=$(xid_now); XID1=${XID1:-0}
echo "xid_after=$XID1 xid_delta=$((XID1 - XID0))"
echo "memavailable_kb_after=$MEM1 min_during_load=$MINMEM"
echo "engine_rss: $RSS"

{
  echo "min_memavailable_kb=$MINMEM"
  echo "memavailable_end_kb=$MEM1"
  echo "engine_rss='$RSS'"
} > "$OUT/memory.txt"

cp "$LOG" "$OUT/../load-lowrisk2-server.log" 2>/dev/null || true

pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
pkill -9 -f r0_serve_loadfix 2>/dev/null || true
sleep 5
echo "run2 complete"

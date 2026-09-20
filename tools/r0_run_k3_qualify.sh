#!/usr/bin/env bash
# MTP k=3 production qualification.
#
# Phase R: no-draft reference greedy token sequences (parity baseline)
# Phase S: k=3 long-context sweep on ONE fresh engine (fresh engine per
#          configuration, not per context)
#
# usage: r0_run_k3_qualify.sh [outdir]
set -euo pipefail

R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="$R/venv"
REPO="${R0_REPO:-$HOME/vllm-exl3-k3}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
OUT="${1:-$R/results/mtp-k3}"
MAXTOK="${MAXTOK:-256}"
CONTEXTS="${CONTEXTS:-4096 32000 65000 126000 160000 200000 240000 250000}"
MAXLEN="${MAXLEN:-256000}"

export CUDA_HOME="$V/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"
mkdir -p "$OUT"

xid_now() { { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } | head -1 | tr -cd '0-9'; }

stop_engine() {
  pkill -9 -f serve_cmp170hx_qwen_firstboot 2>/dev/null || true
  pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
}

wait_healthy() {
  local deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if curl -s -m 5 http://127.0.0.1:8002/health >/dev/null 2>&1 \
       && curl -s -m 10 http://127.0.0.1:8002/v1/models | grep -q '"id"'; then
      return 0
    fi
    sleep 5
  done
  return 1
}

start_engine() {
  local k=$1 log=$2
  VLLM_EXL3_COOP=1 MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL=0.92 \
    MAX_MODEL_LEN="$MAXLEN" MAX_NUM_SEQS=1 PORT=8002 NUM_SPEC_TOKENS="$k" \
    setsid bash "$REPO/tools/serve_cmp170hx_qwen_firstboot.sh" > "$log" 2>&1 < /dev/null &
  wait_healthy
}

guard() {
  local tag=$1
  local st
  st=$(curl -s -m 10 http://127.0.0.1:8002/metrics 2>/dev/null || true)
  local run wait_ pre
  run=$(echo "$st" | grep -E '^vllm:num_requests_running' | awk '{s+=$NF} END {print s+0}')
  wait_=$(echo "$st" | grep -E '^vllm:num_requests_waiting\{' | awk '{s+=$NF} END {print s+0}')
  pre=$(echo "$st" | grep -E '^vllm:num_preemptions_total' | awk '{s+=$NF} END {print s+0}')
  echo "[guard $tag] running=$run waiting=$wait_ preemptions=$pre"
  if [[ "$run" != "0" || "$wait_" != "0" ]]; then
    echo "GUARD FAIL: engine not idle at $tag"
    return 1
  fi
  return 0
}

echo "=== Phase R: no-draft greedy reference ==="
stop_engine
XID0=$(xid_now); XID0=${XID0:-0}
if start_engine 0 "$OUT/serve_ref.log"; then
  APID=$(pgrep -f "vllm serve"|head -1)
  echo "ref engine pwd=$(readlink /proc/$APID/cwd) coop=$(tr '\0' '\n' < /proc/$APID/environ | grep -o 'VLLM_EXL3_COOP=.*')"
  for ctx in 4096 160000 240000; do
    "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context "$ctx" \
      --max-tokens "$MAXTOK" --repeats 1 --tag "nodraft-ref-$ctx" \
      --out "$OUT/ref_${ctx}.json" > /dev/null 2>&1 || echo "ref $ctx failed"
    echo "ref $ctx done"
  done
  cp "$OUT/ref_4096.json"  "$OUT/parity_ref_4096.json" 2>/dev/null || true
  cp "$OUT/ref_160000.json" "$OUT/parity_ref_160000.json" 2>/dev/null || true
  cp "$OUT/ref_240000.json" "$OUT/parity_ref_240000.json" 2>/dev/null || true
else
  echo "REFERENCE ENGINE FAILED TO START"
fi
stop_engine

echo "=== Phase S: k=3 sweep on one fresh engine ==="
if ! start_engine 3 "$OUT/serve_k3.log"; then
  echo "k3 ENGINE FAILED TO START"
  exit 1
fi
APID=$(pgrep -f "vllm serve"|head -1)
EPID=$(pgrep -f "VLLM::EngineCore"|head -1)
echo "k3 engine api=$APID engine=$EPID pwd=$(readlink /proc/$APID/cwd)"
echo "env: $(tr '\0' '\n' < /proc/$APID/environ | grep -E 'VLLM_EXL3_COOP' | head -1)"
echo "spec: $(tr '\0' ' ' < /proc/$APID/cmdline | grep -o 'speculative-config [^ ]*' | head -1)"
echo "markers: PRESCAN=$(grep -c 'PRESCAN ready' "$OUT/serve_k3.log") fallback=$(grep -c 'staging fallback (no arena plan)' "$OUT/serve_k3.log")"

# 4K sentinel in
guard "sentinel-in" || true
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context 4096 \
  --max-tokens "$MAXTOK" --repeats 3 --tag "k3-sentinel-in" \
  --out "$OUT/cell_k3_sentinel_in.json" > /dev/null 2>&1 || echo "sentinel_in failed"
python3 -c "
import json;d=json.load(open('$OUT/cell_k3_sentinel_in.json'));print('sentinel_in ms/tok=',d['ms_per_output_token_median'],'tok/s=',d['output_tok_s_median'],'acc/pass=',d['accepted_per_pass'],'status=',d['status'])" 2>/dev/null || true

for ctx in $CONTEXTS; do
  [ "$ctx" = "4096" ] && continue
  guard "ctx$ctx" || { echo "guard failed at $ctx"; continue; }
  echo "--- cell ctx=$ctx ---"
  if "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context "$ctx" \
       --max-tokens "$MAXTOK" --repeats 3 --tag "k3-ctx$ctx" \
       --out "$OUT/cell_k3_ctx${ctx}.json" > /dev/null 2>&1; then
    python3 -c "
import json;d=json.load(open('$OUT/cell_k3_ctx${ctx}.json'))
print(f\"  ptok={d['prompt_tokens']} ms/tok={d['ms_per_output_token_median']} tok/s={d['output_tok_s_median']} acc/pass={d['accepted_per_pass']} emit/pass={d['emitted_per_pass']} status={d['status']}\")"
  else
    echo "  ctx=$ctx FAILED (capacity or crash)"
  fi
done

# 4K sentinel out
guard "sentinel-out" || true
"$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context 4096 \
  --max-tokens "$MAXTOK" --repeats 3 --tag "k3-sentinel-out" \
  --out "$OUT/cell_k3_sentinel_out.json" > /dev/null 2>&1 || echo "sentinel_out failed"
python3 -c "
import json;d=json.load(open('$OUT/cell_k3_sentinel_out.json'));print('sentinel_out ms/tok=',d['ms_per_output_token_median'],'tok/s=',d['output_tok_s_median'],'status=',d['status'])" 2>/dev/null || true

# Parity against the no-draft reference for the same contexts.
for ctx in 4096 160000 240000; do
  ref="$OUT/parity_ref_${ctx}.json"
  [ -f "$ref" ] || continue
  echo "--- parity ctx=$ctx ---"
  "$V/bin/python" "$REPO/tools/r0_k3_cell.py" --port 8002 --context "$ctx" \
    --max-tokens "$MAXTOK" --repeats 1 --tag "k3-parity-$ctx" \
    --parity-reference "$ref" --out "$OUT/parity_k3_${ctx}.json" > /dev/null 2>&1 || true
  python3 -c "
import json;d=json.load(open('$OUT/parity_k3_${ctx}.json'))
p=d.get('parity',{});print('  parity ctx=$ctx',p.get('status'),'first_mismatch=',p.get('first_mismatch_index'),'reflen=',p.get('reference_len'),'mtplen=',p.get('mtp_len'))" 2>/dev/null || echo "  parity $ctx unavailable"
done

XID1=$(xid_now); XID1=${XID1:-0}
echo "{\"xid_before\":$XID0,\"xid_after\":$XID1,\"xid_delta\":$((XID1-XID0))}" > "$OUT/xid.json"
echo "xid_delta=$((XID1-XID0))"
stop_engine
echo "=== k3 qualification complete ==="

#!/usr/bin/env bash
# K1-T real transfer-plane probe.
# Uses vLLM generic CPU offload manager/worker with synthetic Qwen main-KV
# geometry. No model engine and no installed source modification.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
OUT="${1:-$R/results/kvmem-k1t-transfer}"

export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"

SP="$("$V/bin/python" - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)"
LIB_DIRS=("$SP/torch/lib")
for d in "$SP"/nvidia/*/lib; do
  [[ -d "$d" ]] && LIB_DIRS+=("$d")
done
LIB_PATH="$(IFS=:; echo "${LIB_DIRS[*]}")"
export LD_LIBRARY_PATH="$LIB_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$OUT"

xid_now() {
  { journalctl -k 2>/dev/null | grep -ciE "NVRM: Xid" || true; } |
    head -1 | tr -cd '0-9'
}

echo "=== provenance ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "venv=$V"

echo "=== refuse if model/GPU is busy ==="
if pgrep -af 'VLLM::EngineCore|vllm serve' >/dev/null 2>&1; then
  echo "REFUSE: vLLM process is running" >&2
  pgrep -af 'VLLM::EngineCore|vllm serve' >&2 || true
  exit 2
fi

GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' || true)
if [[ -n "$GPU_PIDS" ]]; then
  echo "REFUSE: GPU compute process is running: $GPU_PIDS" >&2
  exit 2
fi

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile   "$REPO/src/vllm_exl3/kvmem_resident.py"   "$REPO/src/vllm_exl3/kvmem_transfer.py"   "$REPO/src/vllm_exl3/kvmem_vllm_offload.py"   "$REPO/tools/kvmem_k1t_transfer_probe.py"   "$REPO/tools/kvmem_k1t_summarize.py"

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_kvmem_resident_runtime.py"   "$REPO/tests/test_kvmem_transfer_plan.py"   "$REPO/tests/test_kvmem_vllm_offload_adapter.py"   "$REPO/tests/test_kvmem_k1t_summary.py"   "$REPO/tests/test_kvmem_k1t_runner.py"

echo "=== installed vLLM transfer contract ==="
PAGE_TOKENS=$("$V/bin/python" - <<'PY'
from vllm.config.cache import CacheConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.base import GPULoadStoreSpec, CanonicalKVCaches
print(CacheConfig.DEFAULT_BLOCK_SIZE)
PY
)
echo "installed_default_page_tokens=$PAGE_TOKENS"

if (( 256 % PAGE_TOKENS != 0 )); then
  echo "REFUSE: 256-token planner region is not divisible by installed KV page size $PAGE_TOKENS" >&2
  exit 2
fi

echo "=== GPU state before ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu   --format=csv,noheader

XID0=$(xid_now); XID0=${XID0:-0}
echo "xid_before=$XID0"

echo "=== K1-T real CPU backing + H2D stage-in probe ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" "$REPO/tools/kvmem_k1t_transfer_probe.py"   --logical-tokens 240000   --capacity-tokens 65536   --region-tokens 256   --page-tokens "$PAGE_TOKENS"   --replacement-fraction 0.05   --qsa-layers 12   --num-kv-heads 2   --head-dim 256   --dtype-bytes 2   --load-repeats 5   --out "$OUT/k1t_transfer_probe.json"   | tee "$OUT/k1t_transfer_probe.stdout.json"

echo "=== K1-T summary ==="
"$V/bin/python" "$REPO/tools/kvmem_k1t_summarize.py"   --probe "$OUT/k1t_transfer_probe.json"   --reference-gib-s 6.3494   --out "$OUT/k1t_transfer_summary.json"   | tee "$OUT/k1t_transfer_summary.stdout.json"

XID1=$(xid_now); XID1=${XID1:-0}
XID_DELTA=$((XID1-XID0))
echo "xid_after=$XID1 xid_delta=$XID_DELTA"
if (( XID_DELTA != 0 )); then
  echo "REFUSE: new NVIDIA Xid observed" >&2
  exit 3
fi

echo "=== compact result ==="
"$V/bin/python" - "$OUT/k1t_transfer_probe.json" "$OUT/k1t_transfer_summary.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
s = json.load(open(sys.argv[2]))
print("transfer_correctness_go=", s["transfer_correctness_go"])
print("performance_class=", s["performance_class"])
print("stage_in_regions=", p["transition"]["query_replacements_regions"])
print("stage_in_pages=", p["transition"]["stage_in_pages"])
print("stage_in_mib=", p["transition"]["stage_in_mib"])
print("raw_copy_floor_ms=", s["raw_copy_floor_ms"])
print("median_worker_event_ms=", s["median_worker_event_ms"])
print("median_submit_wait_wall_ms=", s["median_submit_wait_wall_ms"])
print("event_over_raw_floor=", s["event_over_raw_floor"])
print("wall_over_raw_floor=", s["wall_over_raw_floor"])
print("all_repeats_byte_exact=", p["byte_verification"]["all_repeats_exact"])
print("qsa_stage_in_mapping_exact=", p["qsa_page_table"]["all_stage_in_mappings_exact"])
print("qsa_evicted_pages_negative=", p["qsa_page_table"]["all_evicted_pages_are_negative"])
PY

echo "=== final state ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu   --format=csv,noheader || true
echo "gpu_processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -cE '^[[:space:]]*[0-9]+' || true)"
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
if command -v ss >/dev/null 2>&1; then
  if ss -ltnp 2>/dev/null | grep -q ':8002 '; then
    echo "port_8002=busy"
  else
    echo "port_8002=free"
  fi
fi
echo "results=$OUT"
echo "NOTE: no model engine was started and no installed source was modified."

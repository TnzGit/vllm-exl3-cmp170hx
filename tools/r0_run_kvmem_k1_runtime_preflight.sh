#!/usr/bin/env bash
# K1 runtime preflight:
# - rerun only the fixed persistent-topk microdiagnostic
# - inspect installed vLLM ownership/offload interfaces
# No model engine is launched and no installed source is modified.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
OUT="${1:-$R/results/kvmem-k1-runtime-preflight}"

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

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile   "$REPO/tools/kvmem_persistent_topk_diagnose.py"   "$REPO/tools/kvmem_k1_runtime_preflight.py"

PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_kvmem_persistent_topk_diagnose.py"   "$REPO/tests/test_kvmem_k1_runtime_preflight.py"   "$REPO/tests/test_kvmem_k1_runtime_preflight_runner.py"

echo "=== installed vLLM K1 capability preflight ==="
"$V/bin/python" "$REPO/tools/kvmem_k1_runtime_preflight.py"   --out "$OUT/k1_runtime_preflight.json"   | tee "$OUT/k1_runtime_preflight.stdout.json"

echo "=== fixed persistent-topk microdiagnostic ==="
XID0=$(xid_now); XID0=${XID0:-0}
"$V/bin/python" "$REPO/tools/kvmem_persistent_topk_diagnose.py"   --rows 8 --width 60000 --k 512 --repeats 30   --out "$OUT/persistent_topk_diagnosis.json"   | tee "$OUT/persistent_topk_diagnosis.stdout.json"
XID1=$(xid_now); XID1=${XID1:-0}

echo "xid_before=$XID0 xid_after=$XID1 xid_delta=$((XID1-XID0))"

echo "=== final state ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu   --format=csv,noheader || true
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
echo "results=$OUT"
echo "NOTE: no model engine started and no installed source was modified."

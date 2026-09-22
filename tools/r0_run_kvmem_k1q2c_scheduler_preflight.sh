#!/usr/bin/env bash
# Q2C scheduler-visible shrink preflight. CPU/read-only only.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
OUT="${1:-$R/results/kvmem-k1q2c-scheduler-shrink-preflight}"
EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact preflight head}"

export CUDA_HOME="${CUDA_HOME:-$V/lib/python3.12/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$V/bin:$PATH"
export VIRTUAL_ENV="$V"

mkdir -p "$OUT"
ACTUAL_SHA="$(git -C "$REPO" rev-parse HEAD)"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  echo "REFUSE: expected head $EXPECTED_SHA, got $ACTUAL_SHA" >&2
  exit 2
fi
echo "$ACTUAL_SHA" > "$OUT/exact_sha.txt"

GPU_PIDS_BEFORE="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d " " | grep -E "^[0-9]+$" || true)"
if [[ -n "$GPU_PIDS_BEFORE" ]]; then
  echo "REFUSE: GPU compute process already running: $GPU_PIDS_BEFORE" >&2
  exit 2
fi

VLLM_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
QSA="$VLLM_ROOT/models/qwen4_exp/nvidia/qsa.py"
if [[ ! -f "$QSA" ]]; then
  QSA="$VLLM_ROOT/model_executor/models/qwen4_exp/nvidia/qsa.py"
fi
test -f "$QSA"
QSA_SHA_BEFORE="$(sha256sum "$QSA" | awk '{print $1}')"

echo "=== identity ==="
echo "repo_head=$ACTUAL_SHA"
echo "vllm_root=$VLLM_ROOT"
echo "qsa=$QSA"
echo "qsa_sha_before=$QSA_SHA_BEFORE"

echo "=== CPU/static gates ==="
bash -n "$REPO/tools/r0_run_kvmem_k1q2c_scheduler_preflight.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/kvmem_qsa_scheduler_contract.py" \
  "$REPO/tools/kvmem_q2c_scheduler_preflight.py"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_kvmem_q2c_scheduler_contract.py" \
  "$REPO/tests/test_kvmem_q2c_scheduler_preflight_runner.py"

echo "=== installed-vLLM Q2C preflight ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" "$REPO/tools/kvmem_q2c_scheduler_preflight.py" \
  --observed-full-block-tokens 1568 \
  --out "$OUT/q2c_scheduler_preflight.json" \
  | tee "$OUT/q2c_scheduler_preflight.stdout.json"

"$V/bin/python" - "$OUT/q2c_scheduler_preflight.json" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
if d["classification"] != "Q2C_SCHEDULER_SHRINK_PREFLIGHT_GO" or not d["go"]:
    raise SystemExit("Q2C preflight did not GO")
g=d["geometry_161k"]
assert g["resident_page_tokens"] == 16
assert g["page_size_bytes"] == 32768
assert g["logical_table_pages"] == 10063
assert g["physical_page_cap"] == 4160
assert g["bounded_mib_per_layer"] == 130.0
assert d["prior_q2b_observation"]["cross_granularity_ratio"] == 98
assert all(d["checks"].values())
print("classification="+d["classification"])
print("logical_table_pages="+str(g["logical_table_pages"]))
print("physical_page_cap="+str(g["physical_page_cap"]))
print("bounded_mib_per_layer="+str(g["bounded_mib_per_layer"]))
print("packed_layout="+d["layout"]["packed_probe"]["resolved_layout"])
PY

QSA_SHA_AFTER="$(sha256sum "$QSA" | awk '{print $1}')"
echo "qsa_sha_after=$QSA_SHA_AFTER"
if [[ "$QSA_SHA_AFTER" != "$QSA_SHA_BEFORE" ]]; then
  echo "ERROR: read-only preflight modified installed QSA" >&2
  exit 3
fi

GPU_PIDS_AFTER="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d " " | grep -E "^[0-9]+$" || true)"
if [[ -n "$GPU_PIDS_AFTER" ]]; then
  echo "ERROR: CPU/read-only preflight left GPU compute process: $GPU_PIDS_AFTER" >&2
  exit 3
fi

echo "=== final ==="
echo "qsa_restore_ok=true"
echo "gpu_processes=0"
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
echo "result=$OUT/q2c_scheduler_preflight.json"
echo "STOP HERE: no engine boot, no installed patch, no Q2C runtime implementation."

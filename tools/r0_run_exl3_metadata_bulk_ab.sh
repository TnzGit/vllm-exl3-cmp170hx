#!/usr/bin/env bash
# Single-boot interleaved routed-expert metadata bulk-commit A/B.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
OUT="${1:-$R/results/startup-exl3-metadata-bulk-ab}"

mkdir -p "$OUT"
SUMMARY="$OUT/metadata_bulk_ab_summary.json"

echo "=== CPU gates ==="
bash -n "$REPO/tools/r0_run_exl3_metadata_bulk_ab.sh"
"$V/bin/python" -m py_compile   "$REPO/src/vllm_exl3/exl3.py"   "$REPO/tools/r0_metadata_bulk_ab_summary.py"   "$REPO/tools/r0_run_tensor_consumer_attribution.sh"   "$REPO/tools/r0_tensor_consumer_attribution.py"
PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}"   "$V/bin/python" -m pytest -q   "$REPO/tests/test_metadata_bulk_ab.py"   "$REPO/tests/test_metadata_bulk_ab_summary.py"   "$REPO/tests/test_tensor_consumer_attr_patch.py"   "$REPO/tests/test_tensor_consumer_attribution.py"   "$REPO/tests/test_tensor_consumer_attr_runner.py"

echo "=== one target-loader boot; even=control odd=deferred/bulk ==="
VLLM_EXL3_METADATA_BULK_AB=1 R0_REPO="$REPO"   bash "$REPO/tools/r0_run_tensor_consumer_attribution.sh" "$OUT"

TRACE="$OUT/exl3_load_trace.jsonl"
TENSOR="$OUT/tensor_consumer_summary.json"
LOADER="$OUT/loader_attribution_summary.json"

"$V/bin/python" "$REPO/tools/r0_metadata_bulk_ab_summary.py"   --trace "$TRACE"   --tensor-summary "$TENSOR"   --loader-summary "$LOADER"   --out "$SUMMARY"   | tee "$OUT/metadata_bulk_ab_summary.stdout.json"

echo "=== compact result ==="
"$V/bin/python" - "$SUMMARY" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(f'metadata_bulk_ab_valid={d["metadata_bulk_ab_valid"]}')
print(f'classification={d["classification"]}')
print(f'main_weights_s={d["main_weights_s"]}')
for sec in ("control","deferred","comparison"):
    for k,v in d[sec].items():
        print(f"{sec}.{k}={v}")
print("=== by suffix ===")
for suffix,row in d["by_suffix"].items():
    print(suffix, json.dumps(row, sort_keys=True))
print("=== mixed-arm tensor suffix totals ===")
for suffix,row in sorted(
    (d["tensor_consumer_mixed_arm"].get("by_suffix") or {}).items(),
    key=lambda kv: float(kv[1].get("consumer_wall_s",0.0)),
    reverse=True,
):
    print(suffix, json.dumps(row, sort_keys=True))
PY

echo "results=$OUT"
echo "NOTE: interleaved mechanism A/B only; do not call this production qualification."

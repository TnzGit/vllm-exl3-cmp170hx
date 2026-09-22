#!/usr/bin/env bash
# One-boot, shape-stratified direct-trellis pinned staging A/B.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
OUT="${1:-$R/results/startup-exl3-pinned-stage-ab}"
REFERENCE_H2D_GIB_S="${LOADER_REFERENCE_H2D_GIB_S:-6.3494}"

echo "=== pinned-stage A/B provenance ==="
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "out=$OUT"
echo "reference_h2d_gib_s=$REFERENCE_H2D_GIB_S"

echo "=== A/B CPU gates ==="
bash -n "$REPO/tools/r0_run_exl3_pinned_stage_ab.sh"
"$V/bin/python" -m py_compile \
  "$REPO/src/vllm_exl3/exl3.py" \
  "$REPO/tools/r0_proc_io_watch.py" \
  "$REPO/tools/r0_exl3_pinned_stage_ab.py"
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_exl3_pinned_stage_ab_contract.py" \
  "$REPO/tests/test_exl3_pinned_stage_ab.py" \
  "$REPO/tests/test_exl3_pinned_stage_ab_runner.py"

echo "=== single loader-boundary boot with interleaved A/B ==="
VLLM_EXL3_PINNED_STAGE_AB=1 \
R0_REPO="$REPO" \
LOADER_REFERENCE_H2D_GIB_S="$REFERENCE_H2D_GIB_S" \
  bash "$REPO/tools/r0_run_exl3_loader_attribution.sh" "$OUT"

BASE="$OUT/loader_attribution_summary.json"
STARTUP="$OUT/startup_loader.json"
TRACE="$OUT/exl3_load_trace.jsonl"
AB="$OUT/pinned_stage_ab_summary.json"

"$V/bin/python" - "$BASE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
assert d["loader_attribution_valid"] is True, d
assert d["kernel_model_load_window"]["source_window"] == "model_start_to_main_weights_done", d["kernel_model_load_window"]
assert d["evidence_identity"]["trace_same_pid"] is True
assert d["evidence_identity"]["proc_same_pid"] is True
assert d["evidence_identity"]["proc_window_present"] is True
print("ideal_model_load_window=true")
PY

"$V/bin/python" "$REPO/tools/r0_exl3_pinned_stage_ab.py" \
  --startup "$STARTUP" \
  --trace "$TRACE" \
  --reference-h2d-gib-s "$REFERENCE_H2D_GIB_S" \
  --out "$AB" \
  | tee "$OUT/pinned_stage_ab_summary.stdout.json"

echo "=== compact A/B result ==="
"$V/bin/python" - "$AB" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
print(f'pinned_stage_ab_valid={d["pinned_stage_ab_valid"]}')
print(f'main_weights_s={d["main_weights_s"]}')
for section in ("direct_total","control","pinned","balance","comparison"):
    for key, value in d[section].items():
        print(f"{section}.{key}={value}")
for key, value in d["interpretation_contract"].items():
    print(f"interpretation_contract.{key}={value}")
PY

echo "results=$OUT"
echo "NOTE: this is an interleaved micro A/B; projected full-pinned numbers are not qualification."

#!/usr/bin/env bash
# Single-boot lazy-safetensors tensor-consumer attribution.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
OUT="${1:-$R/results/startup-tensor-consumer-attribution}"

mkdir -p "$OUT"
TENSOR_STATS="$OUT/tensor_consumer_stats.json"
SUMMARY="$OUT/tensor_consumer_summary.json"
rm -f "$TENSOR_STATS" "$SUMMARY"

VLLM_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
WEIGHT_UTILS="$VLLM_ROOT/model_executor/model_loader/weight_utils.py"
BACKUP="$WEIGHT_UTILS.exl3_tensor_attr.orig"
MARKER="# EXL3_TENSOR_CONSUMER_ATTR_V1"

if [[ ! -f "$WEIGHT_UTILS" ]]; then echo "ERROR: missing $WEIGHT_UTILS" >&2; exit 2; fi
if grep -qF "$MARKER" "$WEIGHT_UTILS"; then echo "REFUSE: tensor-attribution marker already present" >&2; exit 2; fi
if [[ -e "$BACKUP" ]]; then echo "REFUSE: stale backup exists: $BACKUP" >&2; exit 2; fi

SHA_BEFORE=$(sha256sum "$WEIGHT_UTILS" | awk '{print $1}')
PATCHED=0

restore_weight_utils() {
  if (( PATCHED == 1 )) && [[ -f "$BACKUP" ]]; then
    cp "$BACKUP" "$WEIGHT_UTILS"
    rm -f "$BACKUP"
    PATCHED=0
  fi
}
cleanup() { restore_weight_utils || true; }
trap cleanup EXIT

echo "=== CPU gates ==="
bash -n "$REPO/tools/r0_run_tensor_consumer_attribution.sh"
"$V/bin/python" -m py_compile \
  "$REPO/tools/patch_vllm_weight_utils_tensor_attr.py" \
  "$REPO/tools/r0_tensor_consumer_attribution.py" \
  "$REPO/tools/r0_proc_io_watch.py"
"$V/bin/python" "$REPO/tools/patch_vllm_weight_utils_tensor_attr.py" "$VLLM_ROOT" --check-only
PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_tensor_consumer_attr_patch.py" \
  "$REPO/tests/test_tensor_consumer_attribution.py" \
  "$REPO/tests/test_tensor_consumer_attr_runner.py"

echo "=== apply temporary installed-vLLM attribution patch ==="
"$V/bin/python" "$REPO/tools/patch_vllm_weight_utils_tensor_attr.py" "$VLLM_ROOT"
PATCHED=1
SHA_PATCHED=$(sha256sum "$WEIGHT_UTILS" | awk '{print $1}')
grep -qF "$MARKER" "$WEIGHT_UTILS"
echo "weight_utils_sha_before=$SHA_BEFORE"
echo "weight_utils_sha_patched=$SHA_PATCHED"

echo "=== single target-loader boundary boot ==="
VLLM_EXL3_TENSOR_ATTR_PATH="$TENSOR_STATS" \
R0_REPO="$REPO" \
  bash "$REPO/tools/r0_run_exl3_loader_attribution.sh" "$OUT"

BASE="$OUT/loader_attribution_summary.json"
STARTUP="$OUT/startup_loader.json"

"$V/bin/python" - "$BASE" "$TENSOR_STATS" <<'PY'
import json, sys
base=json.load(open(sys.argv[1]))
stats=json.load(open(sys.argv[2]))
assert base["loader_attribution_valid"] is True, base
assert base["kernel_model_load_window"]["source_window"] == "model_start_to_main_weights_done", base["kernel_model_load_window"]
assert stats["mode"] == "lazy_safetensors_tensor_consumer", stats
assert stats["rows"], stats
assert sum(int(r["tensor_count"]) for r in stats["rows"]) > 0
print(f'ideal_model_load_window=true tensor_shards={len(stats["rows"])}')
PY

set +e
"$V/bin/python" "$REPO/tools/r0_tensor_consumer_attribution.py" \
  --tensor-stats "$TENSOR_STATS" \
  --loader-summary "$BASE" \
  --out "$SUMMARY" \
  | tee "$OUT/tensor_consumer_summary.stdout.json"
SUMMARY_RC=${PIPESTATUS[0]}
set -e

echo "=== restore installed vLLM ==="
restore_weight_utils
SHA_AFTER=$(sha256sum "$WEIGHT_UTILS" | awk '{print $1}')
echo "weight_utils_sha_after=$SHA_AFTER"
if [[ "$SHA_AFTER" != "$SHA_BEFORE" ]]; then echo "ERROR: installed weight_utils restore mismatch" >&2; exit 4; fi
if grep -qF "$MARKER" "$WEIGHT_UTILS"; then echo "ERROR: marker remains after restore" >&2; exit 4; fi
if [[ -e "$BACKUP" ]]; then echo "ERROR: backup remains after restore" >&2; exit 4; fi
echo "installed_weight_utils_restored=true"
echo "summary_exit_code=$SUMMARY_RC"

if (( SUMMARY_RC != 0 )); then
  echo "tensor attribution invalid; restore proof completed before exit" >&2
  exit "$SUMMARY_RC"
fi

echo "=== compact result ==="
"$V/bin/python" - "$SUMMARY" "$BASE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
print(f'tensor_consumer_attribution_valid={d["tensor_consumer_attribution_valid"]}')
print(f'main_weights_s={d["main_weights_s"]}')
for k,v in d["totals"].items(): print(f"totals.{k}={v}")
print("=== by_suffix consumer wall descending ===")
for name,row in sorted(d["by_suffix"].items(), key=lambda kv: float(kv[1]["consumer_wall_s"]), reverse=True):
    print(name, json.dumps(row, sort_keys=True))
print("=== by_scope consumer wall descending ===")
for name,row in sorted(d["by_scope"].items(), key=lambda kv: float(kv[1]["consumer_wall_s"]), reverse=True):
    print(name, json.dumps(row, sort_keys=True))
print("=== by_size_bin consumer wall descending ===")
for name,row in sorted(d["by_size_bin"].items(), key=lambda kv: float(kv[1]["consumer_wall_s"]), reverse=True):
    print(name, json.dumps(row, sort_keys=True))
print("=== copy reconciliation ===")
for k,v in d["exl3_copy_reconciliation"].items(): print(f"{k}={v}")
print("=== top 20 consumers ===")
for row in d["top_consumers"][:20]: print(json.dumps(row, sort_keys=True))
km=b["kernel_model_load_window"]
for k in ("elapsed_s","read_gib","major_faults","minor_faults","user_cpu_s","system_cpu_s","kernel_read_vs_checkpoint_ratio"):
    print(f"whole_load.{k}={km[k]}")
PY

echo "results=$OUT"
echo "NOTE: attribution only; no loader policy, ordering, prefetch, pinning, or async behavior changed."

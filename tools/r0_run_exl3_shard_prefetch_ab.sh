#!/usr/bin/env bash
# One-boot bounded safetensors shard-prefetch A/B.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
OUT="${1:-$R/results/startup-exl3-shard-prefetch-ab}"

mkdir -p "$OUT"
STATS="$OUT/shard_prefetch_stats.jsonl"
SUMMARY="$OUT/shard_prefetch_ab_summary.json"
rm -f "$STATS"

VLLM_ROOT="$("$V/bin/python" - <<'PY'
from pathlib import Path
import vllm
print(Path(vllm.__file__).resolve().parent)
PY
)"
WEIGHT_UTILS="$VLLM_ROOT/model_executor/model_loader/weight_utils.py"
BACKUP="$WEIGHT_UTILS.exl3_shard_prefetch_ab.orig"
MARKER="# EXL3_SHARD_PREFETCH_AB_V1"

if [[ ! -f "$WEIGHT_UTILS" ]]; then echo "ERROR: missing $WEIGHT_UTILS" >&2; exit 2; fi
if grep -qF "$MARKER" "$WEIGHT_UTILS"; then echo "REFUSE: shard-prefetch marker already present" >&2; exit 2; fi
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
bash -n "$REPO/tools/r0_run_exl3_shard_prefetch_ab.sh"
"$V/bin/python" -m py_compile \
  "$REPO/tools/patch_vllm_weight_utils_shard_prefetch_ab.py" \
  "$REPO/tools/r0_shard_prefetch_ab_summary.py" \
  "$REPO/tools/r0_proc_io_watch.py"
"$V/bin/python" "$REPO/tools/patch_vllm_weight_utils_shard_prefetch_ab.py" "$VLLM_ROOT" --check-only
PYTHONPATH="$REPO:$REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$V/bin/python" -m pytest -q \
  "$REPO/tests/test_shard_prefetch_ab_patch.py" \
  "$REPO/tests/test_shard_prefetch_ab_summary.py" \
  "$REPO/tests/test_shard_prefetch_ab_runner.py"

echo "=== apply temporary installed-vLLM diagnostic patch ==="
"$V/bin/python" "$REPO/tools/patch_vllm_weight_utils_shard_prefetch_ab.py" "$VLLM_ROOT"
PATCHED=1
SHA_PATCHED=$(sha256sum "$WEIGHT_UTILS" | awk '{print $1}')
grep -qF "$MARKER" "$WEIGHT_UTILS"
echo "weight_utils_sha_before=$SHA_BEFORE"
echo "weight_utils_sha_patched=$SHA_PATCHED"

echo "=== single target-loader boundary boot ==="
VLLM_EXL3_SHARD_PREFETCH_AB=1 \
VLLM_EXL3_SHARD_PREFETCH_AB_STATS_PATH="$STATS" \
R0_REPO="$REPO" \
  bash "$REPO/tools/r0_run_exl3_loader_attribution.sh" "$OUT"

BASE="$OUT/loader_attribution_summary.json"
STARTUP="$OUT/startup_loader.json"

"$V/bin/python" - "$BASE" "$STATS" <<'PY'
import json, sys
base=json.load(open(sys.argv[1]))
assert base["loader_attribution_valid"] is True, base
assert base["kernel_model_load_window"]["source_window"] == "model_start_to_main_weights_done", base["kernel_model_load_window"]
rows=[json.loads(x) for x in open(sys.argv[2]) if x.strip()]
assert rows, "empty shard-prefetch stats"
assert any(r["arm"]=="control" for r in rows)
assert any(r["arm"]=="prefetch" for r in rows)
print(f"ideal_model_load_window=true shard_records={len(rows)}")
PY

set +e
"$V/bin/python" "$REPO/tools/r0_shard_prefetch_ab_summary.py" \
  --stats "$STATS" --startup "$STARTUP" --out "$SUMMARY" \
  | tee "$OUT/shard_prefetch_ab_summary.stdout.json"
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
  echo "A/B summary invalid; restore proof completed before exit" >&2
  exit "$SUMMARY_RC"
fi

echo "=== compact result ==="
"$V/bin/python" - "$SUMMARY" "$BASE" <<'PY'
import json, sys
d=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2]))
print(f'shard_prefetch_ab_valid={d["shard_prefetch_ab_valid"]}')
print(f'main_weights_s={d["main_weights_s"]}')
for section in ("control","prefetch","excluded","balance","comparison"):
    for k,v in d[section].items(): print(f"{section}.{k}={v}")
km=b["kernel_model_load_window"]
for k in ("elapsed_s","read_gib","major_faults","minor_faults","user_cpu_s","system_cpu_s","kernel_read_vs_checkpoint_ratio"):
    print(f"whole_load.{k}={km[k]}")
PY

echo "results=$OUT"
echo "NOTE: one boot only; per-shard prefetch thread is joined before advancing to next shard."

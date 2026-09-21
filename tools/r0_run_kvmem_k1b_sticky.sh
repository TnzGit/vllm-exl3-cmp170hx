#!/usr/bin/env bash
# K1B: diagnose persistent-topk tie stability and replay existing K1A shadows
# through a stateful sticky-residency planner. No model engine is launched.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/Lygodactylus-Qwen3.8-Flash-Next-Uncensored-exl3-3bpw}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-k1a-churn}"
OUT="${1:-$R/results/kvmem-k1b-sticky}"

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
echo "k1a_dir=$K1A_DIR"
echo "model=$MODEL_DIR"

test -f "$K1A_DIR/k1a_churn_summary.json"
test -f "$K1A_DIR/turns/manifest.json"
test -d "$K1A_DIR/shadows"
test -f "$MODEL_DIR/config.json"

SHADOW_COUNT=$(find "$K1A_DIR/shadows" -maxdepth 1 -type f -name 'ctx*_turn_*.jsonl' | wc -l)
echo "k1a_shadow_count=$SHADOW_COUNT"
if [[ "$SHADOW_COUNT" != "12" ]]; then
  echo "REFUSE: expected exactly 12 K1A shadow files" >&2
  exit 2
fi

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile   "$REPO/tools/kvmem_persistent_topk_diagnose.py"   "$REPO/tools/kvmem_qsa_sticky_replay.py"

PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_kvmem_qsa_sticky_replay.py"   "$REPO/tests/test_kvmem_k1b_runner.py"

echo "=== sticky replay over existing K1A shadows ==="
"$V/bin/python" "$REPO/tools/kvmem_qsa_sticky_replay.py"   --manifest "$K1A_DIR/turns/manifest.json"   --shadow-dir "$K1A_DIR/shadows"   --model-config "$MODEL_DIR/config.json"   --out "$OUT/k1b_sticky_summary.json"   | tee "$OUT/k1b_sticky_summary.stdout.json"

echo "=== SM80 persistent-topk tie diagnostic ==="
XID0=$(xid_now); XID0=${XID0:-0}
"$V/bin/python" "$REPO/tools/kvmem_persistent_topk_diagnose.py"   --rows 8 --width 60000 --k 512 --repeats 30   --out "$OUT/persistent_topk_diagnosis.json"   | tee "$OUT/persistent_topk_diagnosis.stdout.json"
XID1=$(xid_now); XID1=${XID1:-0}
echo "xid_before=$XID0 xid_after=$XID1 xid_delta=$((XID1-XID0))"

echo "=== final ==="
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu   --format=csv,noheader || true
echo "results=$OUT"
echo "NOTE: no model engine was launched; this run is offline planner replay plus a top-k kernel microdiagnostic."

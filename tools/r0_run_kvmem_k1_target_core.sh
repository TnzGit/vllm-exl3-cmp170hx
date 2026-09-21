#!/usr/bin/env bash
# K1 target runtime core gate:
# 1) generate the fixed official K1 capability artifact;
# 2) prove runtime sticky coordinator exactly reproduces K1B primary evidence.
# No model engine, CUDA kernel, or installed source modification.
set -euo pipefail

REPO="${R0_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
R="${R0_ROOT:-$HOME/.codex_tasks/qwen38-flashnext-r0}"
V="${R0_VENV:-$R/venv}"
K1A_DIR="${K1A_DIR:-$R/results/kvmem-qsa-churn}"
K1B_DIR="${K1B_DIR:-$R/results/kvmem-k1b-sticky}"
OUT="${1:-$R/results/kvmem-k1-target-core}"

mkdir -p "$OUT"

echo "=== provenance ==="
echo "repo=$REPO"
echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
echo "venv=$V"
echo "k1a_dir=$K1A_DIR"
echo "k1b_dir=$K1B_DIR"

test -f "$K1A_DIR/turns/manifest.json"
test -d "$K1A_DIR/shadows"
test -f "$K1B_DIR/k1b_sticky_summary.json"

SHADOW_COUNT=$(find "$K1A_DIR/shadows" -maxdepth 1 -type f -name 'ctx*_turn_*.jsonl' | wc -l)
echo "k1a_shadow_count=$SHADOW_COUNT"
if [[ "$SHADOW_COUNT" != "12" ]]; then
  echo "REFUSE: expected exactly 12 K1A shadow files" >&2
  exit 2
fi

echo "=== CPU gates ==="
"$V/bin/python" -m py_compile   "$REPO/tools/kvmem_k1_runtime_preflight.py"   "$REPO/src/vllm_exl3/kvmem_resident.py"   "$REPO/tools/kvmem_runtime_replay_check.py"

PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" -m pytest -q   "$REPO/tests/test_kvmem_k1_runtime_preflight.py"   "$REPO/tests/test_kvmem_resident_runtime.py"   "$REPO/tests/test_kvmem_k1_target_core_runner.py"

echo "=== official installed-vLLM preflight artifact ==="
"$V/bin/python" "$REPO/tools/kvmem_k1_runtime_preflight.py"   --out "$OUT/k1_runtime_preflight.json"   | tee "$OUT/k1_runtime_preflight.stdout.json"

"$V/bin/python" - "$OUT/k1_runtime_preflight.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["decision_support"]
print("hisparse_core_available=", s["hisparse_core_available"])
print("generic_cpu_offloading_available=", s["generic_cpu_offloading_available"])
print("qwen4_exp_mentions_hisparse=", s["qwen4_exp_mentions_hisparse"])
print("recommended_integration_route=", s["recommended_integration_route"])
PY

echo "=== research -> runtime planner exact replay contract ==="
PYTHONPATH="$REPO/src:$REPO${PYTHONPATH:+:$PYTHONPATH}" "$V/bin/python" "$REPO/tools/kvmem_runtime_replay_check.py"   --k1a-dir "$K1A_DIR"   --k1b-summary "$K1B_DIR/k1b_sticky_summary.json"   --out "$OUT/k1_runtime_replay_check.json"   | tee "$OUT/k1_runtime_replay_check.stdout.json"

echo "=== final state ==="
echo "vllm_processes=$(pgrep -af 'VLLM::EngineCore|vllm serve' | wc -l)"
if command -v ss >/dev/null 2>&1; then
  if ss -ltnp 2>/dev/null | grep -q ':8002 '; then
    echo "port_8002=busy"
  else
    echo "port_8002=free"
  fi
fi
echo "results=$OUT"
echo "NOTE: CPU/read-only gate only; no model engine or installed source modification."

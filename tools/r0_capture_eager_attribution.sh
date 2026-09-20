#!/usr/bin/env bash
# Capture one short eager decode trace and summarize residual BF16/framework callers.
#
# Server prerequisite:
#   launched with tools/r0_serve_coop_eager_profiler.sh
#
# Required:
#   TORCH_PROFILER_DIR=<same directory used by server>
#
# Optional:
#   BASE_URL=http://127.0.0.1:8002
#   CONTEXT=4096
#   PROFILE_CHUNKS=16
#   MAX_TOKENS=48
#   OUT_DIR=<directory for capture metadata/summary>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${TORCH_PROFILER_DIR:?set TORCH_PROFILER_DIR to the server trace directory}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8002}"
CONTEXT="${CONTEXT:-4096}"
PROFILE_CHUNKS="${PROFILE_CHUNKS:-16}"
MAX_TOKENS="${MAX_TOKENS:-48}"
OUT_DIR="${OUT_DIR:-$HOME/.codex_tasks/qwen38-flashnext-r0/eager-attribution}"
mkdir -p "$OUT_DIR"

before="$OUT_DIR/traces.before.txt"
after="$OUT_DIR/traces.after.txt"
capture="$OUT_DIR/capture.txt"
summary="$OUT_DIR/summary.txt"

find "$TORCH_PROFILER_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.json.gz' \) -printf '%p\n' 2>/dev/null | sort > "$before" || true

python "$ROOT/tools/r0_profile_decode_window.py" \
  --base-url "$BASE_URL" \
  --context "$CONTEXT" \
  --max-tokens "$MAX_TOKENS" \
  --start-after-chunks 3 \
  --profile-chunks "$PROFILE_CHUNKS" \
  | tee "$capture"

find "$TORCH_PROFILER_DIR" -maxdepth 1 -type f \( -name '*.json' -o -name '*.json.gz' \) -printf '%p\n' 2>/dev/null | sort > "$after" || true

trace="$(
  comm -13 "$before" "$after" | tail -n 1
)"
if [[ -z "$trace" || ! -f "$trace" ]]; then
  echo "ERROR: could not identify the new profiler trace" >&2
  exit 2
fi

{
  echo "trace=$trace"
  stat --printf='trace_size=%s\n' "$trace"
  sha256sum "$trace"
  echo
  python "$ROOT/tools/r0_analyze_eager_attribution.py" "$trace" --top 100
} | tee "$summary"

echo
echo "capture=$capture"
echo "summary=$summary"

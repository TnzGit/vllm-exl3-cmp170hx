#!/usr/bin/env bash
# Own and run exactly one C2/C4, MTP k=2/3 cell on a fresh local engine.
# Required env: EXPECTED_SHA R0_REPO MODEL_DIR MANIFEST OUT. Args: POINT K.
# This script is for the experiment host; --dry-run is CPU/static-only.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: r0_run_c2_c4_mtp_cell.sh [--dry-run] POINT K
POINT: c2_16k | c2_80k | c4_16k | c4_32k; K: 2 | 3
Required environment: EXPECTED_SHA, R0_REPO, MODEL_DIR, MANIFEST, OUT.
Optional: R0_ROOT (defaults to /home/base-node/.codex_tasks/qwen38-flashnext-r0),
          LAUNCH_TIMEOUT (defaults to 2400 seconds).

OUT must not already exist. The manifest must contain exact token IDs and
frozen hashes; model_pack_sha256 and tokenizer_sha256 use the canonical,
streamed tree-hash format implemented by this runner. This command starts one
fresh port-8002 engine, runs one cell, and stops only its proven process group.
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then DRY_RUN=1; shift; fi
if (($# != 2)); then usage >&2; exit 2; fi
POINT="$1" K="$2"
case "$POINT" in c2_16k|c2_80k|c4_16k|c4_32k) ;; *) usage >&2; exit 2 ;; esac
[[ "$K" == 2 || "$K" == 3 ]] || { usage >&2; exit 2; }

EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the full isolated R0_REPO commit}"
R0_REPO="${R0_REPO:?set R0_REPO to the clean isolated runtime checkout}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the prepared EXL3 pack}"
MANIFEST="${MANIFEST:?set MANIFEST to the frozen exact-token manifest}"
OUT="${OUT:?set OUT to a new per-cell artifact directory}"
R0_ROOT="${R0_ROOT:-/home/base-node/.codex_tasks/qwen38-flashnext-r0}"
VENV="$R0_ROOT/venv"
HOST=127.0.0.1 PORT=8002 GPU_MEM_UTIL=0.92 MAX_MODEL_LEN=246000 MAX_NUM_SEQS=4
LAUNCH_TIMEOUT="${LAUNCH_TIMEOUT:-2400}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$SCRIPT_DIR/r0_c2_c4_mtp_matched_load.py"
LAUNCHER="$R0_REPO/tools/serve_cmp170hx_qwen_firstboot.sh"

[[ "$EXPECTED_SHA" =~ ^[0-9a-f]{40}$ ]] || { echo 'BLOCKED: EXPECTED_SHA must be 40 lowercase hex chars' >&2; exit 2; }
[[ "$LAUNCH_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || { echo 'BLOCKED: invalid LAUNCH_TIMEOUT' >&2; exit 2; }
[[ -x "$VENV/bin/python" && -f "$LAUNCHER" && -f "$HELPER" ]] || { echo 'BLOCKED: missing R0 runtime, launcher, or matched-load helper' >&2; exit 2; }
[[ -f "$MODEL_DIR/config.json" && -f "$MANIFEST" ]] || { echo 'BLOCKED: missing model config or frozen manifest' >&2; exit 2; }
[[ "$HOST" == 127.0.0.1 && "$PORT" == 8002 ]] || exit 2

if (( DRY_RUN )); then
  "$VENV/bin/python" "$HELPER" --manifest "$MANIFEST" --dry-run >/dev/null
  printf 'DRY_RUN point=%s k=%s port=%s gpu_memory_utilization=%s max_model_len=%s max_num_seqs=%s max_num_batched_tokens=auto coop=1 graph=PIECEWISE text_only=1 disk_ngram=1 profiler=off prefix_caching=off\n' \
    "$POINT" "$K" "$PORT" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS"
  exit 0
fi

[[ ! -e "$OUT" && ! -L "$OUT" ]] || { echo "BLOCKED: OUT must not exist: $OUT" >&2; exit 2; }
mkdir -m 700 -- "$OUT"
OUT="$(cd "$OUT" && pwd)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EVENTS="$OUT/events.log"
ACTIVE_PID="" ACTIVE_PGID="" GROUP_OWNED=0 OWNERSHIP_UNPROVEN=0
XID_INITIAL=""
FINISHING=0

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
event() { printf '%s run=%s %s\n' "$(timestamp)" "$RUN_ID" "$*" | tee -a "$EVENTS"; }
die() { event "BLOCKED $*"; exit 2; }

port_listener_count() { ss -H -ltn "sport = :$PORT" | wc -l | tr -d ' '; }
port_is_open() {
  python3 - "$HOST" "$PORT" <<'PY'
import socket,sys
s=socket.socket(); s.settimeout(1)
try: s.connect((sys.argv[1],int(sys.argv[2])))
except OSError: raise SystemExit(1)
else: raise SystemExit(0)
finally: s.close()
PY
}
gpu_snapshot() {
  local file="$1"
  { printf 'timestamp_utc=%s\n' "$(timestamp)"
    nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    printf 'compute_processes:\n'
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  } > "$file" 2>&1
}
gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/,""); print}'
}
xid_count() {
  local log
  log="$(journalctl -k -b -n 50000 --no-pager -o cat 2>/dev/null)" || return 1
  printf '%s\n' "$log" | tail -c 4000000 | grep -ciE 'NVRM: Xid' || true
}
record_processes() {
  local file="$1" pgid="${2:-}"
  { printf 'timestamp_utc=%s\n' "$(timestamp)"
    if [[ -n "$pgid" ]]; then ps -eo pid,ppid,pgid,lstart,args | awk -v p="$pgid" 'NR==1 || $3==p'
    else ps -eo pid,ppid,pgid,lstart,args | head -n 100; fi
  } > "$file" 2>&1 || true
}
record_port() {
  local file="$1" listeners state=CLOSED
  listeners="$(port_listener_count)"
  if (( listeners > 0 )) || port_is_open; then state=OPEN; fi
  printf 'timestamp_utc=%s\nhost=%s\nport=%s\nlisteners=%s\nstate=%s\n' \
    "$(timestamp)" "$HOST" "$PORT" "$listeners" "$state" > "$file"
}
stop_owned() {
  [[ -n "$ACTIVE_PID" ]] || return 0
  local rc=0 deadline leader_pgid
  event "STOP_REQUEST pid=$ACTIVE_PID pgid=${ACTIVE_PGID:-unknown} group_owned=$GROUP_OWNED"
  if (( GROUP_OWNED )) && kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; then
    leader_pgid="$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "$leader_pgid" == "$ACTIVE_PGID" ]]; then
      kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
      deadline=$((SECONDS+25))
      while (( SECONDS < deadline )) && kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; do sleep 1; done
      if kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; then
        leader_pgid="$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d '[:space:]' || true)"
        if [[ "$leader_pgid" == "$ACTIVE_PGID" ]]; then
          kill -KILL -- "-$ACTIVE_PGID" 2>/dev/null || true
          deadline=$((SECONDS+5))
          while (( SECONDS < deadline )) && kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; do sleep 1; done
        else
          event 'CLEANUP_FAIL group leader identity unavailable before SIGKILL'
          OWNERSHIP_UNPROVEN=1
        fi
      fi
    else
      event 'CLEANUP_FAIL group leader identity unavailable before SIGTERM'
      OWNERSHIP_UNPROVEN=1
    fi
  elif kill -0 "$ACTIVE_PID" 2>/dev/null; then
    # If set-session ownership was not proven, signal only the direct child.
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
    deadline=$((SECONDS+25))
    while (( SECONDS < deadline )) && kill -0 "$ACTIVE_PID" 2>/dev/null; do sleep 1; done
    if kill -0 "$ACTIVE_PID" 2>/dev/null; then kill -KILL "$ACTIVE_PID" 2>/dev/null || true; fi
  fi
  if wait "$ACTIVE_PID" 2>/dev/null; then rc=0; else rc=$?; fi
  printf 'server_exit_status=%s\nserver_exit_utc=%s\n' "$rc" "$(timestamp)" > "$OUT/server_exit.txt"
  record_processes "$OUT/processes_after_stop.txt" "$ACTIVE_PGID"
  event "STOPPED pid=$ACTIVE_PID pgid=${ACTIVE_PGID:-unknown} rc=$rc"
  ACTIVE_PID="" ACTIVE_PGID="" GROUP_OWNED=0
}
finish() {
  local rc=$?
  local stopped_pgid="${ACTIVE_PGID:-}" stopped_group="$GROUP_OWNED"
  (( FINISHING )) && return
  FINISHING=1
  trap - EXIT INT TERM HUP
  stop_owned || rc=2
  if command -v ss >/dev/null 2>&1; then
    record_port "$OUT/port_after.txt" || rc=2
    (( $(port_listener_count) == 0 )) || { event 'CLEANUP_FAIL port listener remains'; rc=2; }
    port_is_open && { event 'CLEANUP_FAIL port still accepts connections'; rc=2; }
  fi
  record_processes "$OUT/processes_final.txt"
  (( OWNERSHIP_UNPROVEN == 0 )) || { event 'CLEANUP_FAIL process-group ownership was not proven'; rc=2; }
  if (( stopped_group )) && [[ -n "$stopped_pgid" ]] && ps -eo pgid= | awk -v p="$stopped_pgid" '$1==p {found=1} END{exit !found}'; then
    event "CLEANUP_FAIL processes remain in owned group $stopped_pgid"; rc=2
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_snapshot "$OUT/gpu_after.csv" || rc=2
    local remaining
    if ! remaining="$(gpu_compute_pids)"; then
      printf 'compute_pid_inventory_unavailable=true\n' > "$OUT/gpu_cleanup_failure.txt"
      rc=2
    elif [[ -n "$remaining" ]]; then
      printf 'remaining_compute_pids=%s\n' "$remaining" > "$OUT/gpu_cleanup_failure.txt"
      rc=2
    fi
  else
    printf 'nvidia_smi_unavailable=true\n' > "$OUT/gpu_cleanup_failure.txt"
    rc=2
  fi
  if [[ -n "$XID_INITIAL" ]]; then
    local after
    if after="$(xid_count)"; then
      printf 'xid_count_after=%s\nxid_delta=%s\n' "$after" "$((after-XID_INITIAL))" > "$OUT/xid_after.txt"
      (( after == XID_INITIAL )) || { event 'CLEANUP_FAIL Xid count changed'; rc=2; }
    else
      printf 'status=UNAVAILABLE\n' > "$OUT/xid_after.txt"
      event 'CLEANUP_FAIL could not read post-run Xid count'; rc=2
    fi
  fi
  printf 'run_id=%s\nfinished_utc=%s\nexit_status=%s\n' "$RUN_ID" "$(timestamp)" "$rc" > "$OUT/run_exit.txt"
  event "RUN_EXIT status=$rc"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

for cmd in nvidia-smi curl setsid ss journalctl sha256sum git rg grep; do command -v "$cmd" >/dev/null 2>&1 || die "required command missing: $cmd"; done
ACTUAL_SHA="$(git -C "$R0_REPO" rev-parse HEAD 2>/dev/null)" || die 'R0_REPO is not a git checkout'
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || die "R0_REPO HEAD $ACTUAL_SHA != EXPECTED_SHA $EXPECTED_SHA"
[[ -z "$(git -C "$R0_REPO" status --porcelain=v1 --untracked-files=all)" ]] || die 'R0_REPO is dirty'
[[ "$(git -C "$R0_REPO" rev-parse --show-toplevel)" == "$(cd "$R0_REPO" && pwd)" ]] || die 'R0_REPO path is not its Git root'
[[ -f "$R0_REPO/tools/serve_cmp170hx_qwen_firstboot.sh" ]] || die 'expected launcher missing'
[[ -f "$R0_REPO/src/vllm_exl3/exl3.py" ]] || die 'repo EXL3 source missing'
if rg -n -- '--max-num-scheduled-tokens|MAX_NUM_SCHEDULED_TOKENS' "$LAUNCHER" >/dev/null; then
  die 'launcher contains a scheduled-token override; source inference is invalid'
fi
INSTALLED_PLUGIN="$VENV/lib/python3.12/site-packages/vllm_exl3/exl3.py"
[[ -f "$INSTALLED_PLUGIN" ]] || die 'installed EXL3 plugin is missing'
REPO_PLUGIN_SHA="$(sha256sum "$R0_REPO/src/vllm_exl3/exl3.py" | awk '{print $1}')"
INSTALLED_PLUGIN_SHA="$(sha256sum "$INSTALLED_PLUGIN" | awk '{print $1}')"
[[ "$REPO_PLUGIN_SHA" == "$INSTALLED_PLUGIN_SHA" ]] || die 'installed EXL3 source differs from exact R0 source'

export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH" VIRTUAL_ENV="$VENV"
unset MAX_NUM_SCHEDULED_TOKENS VLLM_MAX_NUM_SCHEDULED_TOKENS
unset VLLM_CONFIG

# Stream-hash the model tree and tokenizer assets; reject symlinks and compare
# against the frozen manifest before creating an engine. No mmap/model load.
"$VENV/bin/python" - "$HELPER" "$MANIFEST" "$MODEL_DIR" "$OUT/input_hashes.json" <<'PY'
import hashlib, importlib.util, json, pathlib, sys
helper, manifest_path, model_text, output = map(pathlib.Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("r0_c2_helper", helper)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
manifest, raw = module.load_prompt_manifest(manifest_path)
model_arg=pathlib.Path(model_text)
if model_arg.is_symlink(): raise SystemExit("BLOCKED: model dir cannot be a symlink")
root = model_arg.resolve()
if not root.is_dir(): raise SystemExit("BLOCKED: model dir must be a directory")

def tree_hash(paths):
    digest = hashlib.sha256(); count = 0; total = 0
    for path in sorted(paths, key=lambda p: p.relative_to(root).as_posix()):
        if path.is_symlink() or not path.is_file(): raise SystemExit(f"BLOCKED: unsafe model asset: {path}")
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(8, "big")); digest.update(rel)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024): digest.update(chunk); total += len(chunk)
        count += 1
    return digest.hexdigest(), count, total

all_files = []
for directory, names, files in __import__("os").walk(root, followlinks=False):
    base = pathlib.Path(directory)
    if any((base / name).is_symlink() for name in names): raise SystemExit("BLOCKED: symlink directory in model tree")
    all_files.extend(base / name for name in files)
model_hash, model_count, model_bytes = tree_hash(all_files)
token_names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt", "merges.txt", "chat_template.jinja"}
token_files = [p for p in all_files if p.name in token_names]
if not token_files: raise SystemExit("BLOCKED: no recognized tokenizer assets in model tree")
token_hash, token_count, token_bytes = tree_hash(token_files)
config = root / "config.json"
if config.is_symlink() or not config.is_file(): raise SystemExit("BLOCKED: config.json is not a regular file")
config_digest = hashlib.sha256()
with config.open("rb") as stream:
    while chunk := stream.read(1024 * 1024): config_digest.update(chunk)
config_hash = config_digest.hexdigest()
expected = manifest["provenance"]
for name, actual in (("model_pack_sha256", model_hash), ("tokenizer_sha256", token_hash), ("model_config_sha256", config_hash)):
    if expected.get(name) != actual: raise SystemExit(f"BLOCKED: actual {name} differs from frozen manifest")
with open(output, "x", encoding="utf-8") as stream:
    json.dump({"model_tree_sha256": model_hash, "model_file_count": model_count, "model_bytes_hashed": model_bytes,
               "model_config_sha256": config_hash, "tokenizer_tree_sha256": token_hash,
               "tokenizer_file_count": token_count, "tokenizer_bytes_hashed": token_bytes,
               "manifest_sha256": module.sha256_bytes(raw)}, stream, indent=2, sort_keys=True); stream.write("\n")
PY

(( $(port_listener_count) == 0 )) || die 'port 8002 has a listener before launch'
port_is_open && die 'port 8002 accepts connections before launch'
record_port "$OUT/port_before.txt"
gpu_snapshot "$OUT/gpu_before.csv" || die 'cannot capture pre-launch GPU inventory'
GPU_PIDS="$(gpu_compute_pids)" || die 'cannot query pre-launch GPU compute processes'
[[ -z "$GPU_PIDS" ]] || die "GPU is not idle; compute PIDs: $GPU_PIDS"
XID_INITIAL="$(xid_count)" || die 'cannot capture pre-launch Xid baseline'
printf 'xid_count_before=%s\n' "$XID_INITIAL" > "$OUT/xid_before.txt"
record_processes "$OUT/processes_before.txt"
printf 'status=STARTUP_PROOF_INCOMPLETE\nreason=EngineCore startup evidence not yet verified\n' > "$OUT/startup_proof_status.txt"

cat > "$OUT/cell_identity.txt" <<EOF
run_id=$RUN_ID
point=$POINT
k=$K
expected_sha=$EXPECTED_SHA
repo_head=$ACTUAL_SHA
repo_status=clean
model_dir=$MODEL_DIR
model_config_sha256=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["model_config_sha256"])' "$OUT/input_hashes.json")
installed_exl3_source_sha256=$INSTALLED_PLUGIN_SHA
point_prompt_tokens=$($VENV/bin/python -c "import importlib.util,pathlib; p=pathlib.Path('$HELPER'); s=importlib.util.spec_from_file_location('h',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(m.POINTS['$POINT']['prompt_tokens'])")
gpu_mem_util=$GPU_MEM_UTIL
max_model_len=$MAX_MODEL_LEN
max_num_seqs=$MAX_NUM_SEQS
max_num_batched_tokens=auto
mtp_k=$K
EOF

printf '[%s] launch point=%s k=%s\n' "$(timestamp)" "$POINT" "$K" > "$OUT/server.log"
env MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS="$MAX_NUM_SEQS" PORT="$PORT" HOST="$HOST" PREFIX_CACHING=0 \
  NUM_SPEC_TOKENS="$K" MAX_NUM_BATCHED_TOKENS= TORCH_PROFILER_DIR= ENFORCE_EAGER=0 \
  VLLM_EXL3_COOP=1 VLLM_EXL3_NGRAM_TABLE=disk VLLM_EXL3_NGRAM_KERNEL=ext \
  setsid bash "$LAUNCHER" >> "$OUT/server.log" 2>&1 < /dev/null &
ACTIVE_PID=$!
observed_pgid="$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ' || true)"
if [[ "$observed_pgid" == "$ACTIVE_PID" ]]; then ACTIVE_PGID="$observed_pgid"; GROUP_OWNED=1; else
  ACTIVE_PGID="$observed_pgid"; GROUP_OWNED=0
  OWNERSHIP_UNPROVEN=1
  die "setsid ownership not proven (pid=$ACTIVE_PID pgid=${observed_pgid:-missing}); direct child only will be signaled"
fi
printf 'launch_pid=%s\nlaunch_pgid=%s\nlaunch_utc=%s\n' "$ACTIVE_PID" "$ACTIVE_PGID" "$(timestamp)" > "$OUT/startup.txt"
record_processes "$OUT/processes_launch.txt" "$ACTIVE_PGID"

healthy=0 deadline=$((SECONDS+LAUNCH_TIMEOUT))
while (( SECONDS < deadline )); do
  if curl -fsS --max-time 5 "http://$HOST:$PORT/health" >/dev/null 2>&1 \
     && curl -fsS --max-time 10 "http://$HOST:$PORT/v1/models" | grep -q '"id"'; then healthy=1; break; fi
  kill -0 "$ACTIVE_PID" 2>/dev/null || break
  sleep 5
done
(( healthy == 1 )) || die 'engine failed health gate; startup artifacts retained'
printf 'healthy_utc=%s\n' "$(timestamp)" >> "$OUT/startup.txt"
record_port "$OUT/port_healthy.txt"
record_processes "$OUT/processes_healthy.txt" "$ACTIVE_PGID"
gpu_snapshot "$OUT/gpu_healthy.csv"

# Confirm the actual exec arguments, not merely launcher environment. In this
# launch profile the scheduled-token option must be absent and batched tokens
# must remain on the launcher default (auto).
"$VENV/bin/python" - "$ACTIVE_PID" "$K" <<'PY'
import json,pathlib,sys
raw=pathlib.Path(f"/proc/{sys.argv[1]}/cmdline").read_bytes()
args=[x.decode(errors="replace") for x in raw.split(b"\0") if x]
if not args or "vllm" not in pathlib.Path(args[0]).name:
    raise SystemExit(f"BLOCKED: owned PID is not the vLLM exec: {args[:4]}")
for forbidden in ("--max-num-scheduled-tokens", "--max-num-batched-tokens", "--config"):
    if forbidden in args: raise SystemExit(f"BLOCKED: unexpected explicit CLI option {forbidden}")
if "--max-num-seqs" not in args or args[args.index("--max-num-seqs")+1] != "4":
    raise SystemExit("BLOCKED: live argv does not show max-num-seqs=4")
if "--max-model-len" not in args or args[args.index("--max-model-len")+1] != "246000":
    raise SystemExit("BLOCKED: live argv does not show max-model-len=246000")
if "--no-enable-prefix-caching" not in args:
    raise SystemExit("BLOCKED: live argv does not explicitly disable prefix caching")
if "--language-model-only" not in args:
    raise SystemExit("BLOCKED: live argv is not text-only")
if "--quantization" not in args or args[args.index("--quantization")+1] != "exl3":
    raise SystemExit("BLOCKED: live argv quantization is not EXL3")
if "--gpu-memory-utilization" not in args or args[args.index("--gpu-memory-utilization")+1] != "0.92":
    raise SystemExit("BLOCKED: live argv GPU memory utilization is not 0.92")
try: graph=json.loads(args[args.index("--compilation-config")+1])
except (ValueError,IndexError,json.JSONDecodeError): raise SystemExit("BLOCKED: cannot parse live compilation config")
if graph.get("cudagraph_mode") != "PIECEWISE":
    raise SystemExit("BLOCKED: live compilation config is not PIECEWISE")
try: spec=args[args.index("--speculative-config")+1]
except (ValueError,IndexError): raise SystemExit("BLOCKED: live argv lacks speculative config")
try: value=json.loads(spec)
except json.JSONDecodeError: raise SystemExit("BLOCKED: malformed live speculative config")
if value.get("method") != "qwen4_exp_mtp" or value.get("num_speculative_tokens") != int(sys.argv[2]):
    raise SystemExit("BLOCKED: live argv speculative k/method mismatch")
envraw=pathlib.Path(f"/proc/{sys.argv[1]}/environ").read_bytes()
env={item.split(b"=",1)[0].decode():item.split(b"=",1)[1].decode(errors="replace") for item in envraw.split(b"\0") if b"=" in item}
expected={"GPU_MEM_UTIL":"0.92","MAX_MODEL_LEN":"246000","MAX_NUM_SEQS":"4","PORT":"8002",
          "HOST":"127.0.0.1","PREFIX_CACHING":"0","NUM_SPEC_TOKENS":sys.argv[2],
          "MAX_NUM_BATCHED_TOKENS":"","VLLM_EXL3_COOP":"1","VLLM_EXL3_NGRAM_TABLE":"disk",
          "VLLM_EXL3_NGRAM_KERNEL":"ext",
          "TORCH_PROFILER_DIR":"","ENFORCE_EAGER":"0"}
for name, wanted in expected.items():
    if env.get(name) != wanted: raise SystemExit(f"BLOCKED: live environment mismatch for {name}")
if "MAX_NUM_SCHEDULED_TOKENS" in env or "VLLM_MAX_NUM_SCHEDULED_TOKENS" in env or "VLLM_CONFIG" in env:
    raise SystemExit("BLOCKED: environment could override the default scheduler budget")
PY

# Gather observed versions and engine-core log evidence, validate every field,
# then pass the generated proof into the CPU/client helper before first request.
if "$VENV/bin/python" - "$HELPER" "$MANIFEST" "$OUT/server.log" "$OUT/input_hashes.json" \
  "$POINT" "$K" "$ACTIVE_PID" "$ACTIVE_PGID" "$EXPECTED_SHA" "$INSTALLED_PLUGIN_SHA" \
  "$OUT/runtime_proof.json" > "$OUT/runtime_proof_build.log" 2>&1 <<'PY'
import importlib.metadata as md, importlib.util, json, pathlib, subprocess, sys
helper, manifest_path, log_path, hashes_path = map(pathlib.Path, sys.argv[1:5])
point, k, pid, pgid, r0_sha, plugin_sha, proof_path = sys.argv[5:]
spec=importlib.util.spec_from_file_location("r0_c2_helper",helper)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
manifest, _=module.load_prompt_manifest(manifest_path)
hashes=json.loads(hashes_path.read_text())
def version(name):
    try: return md.version(name)
    except md.PackageNotFoundError: return ""
try:
    import torch
    torch_version=torch.__version__
    cuda_version=torch.version.cuda or ""
except Exception as exc:
    raise SystemExit(f"BLOCKED: cannot observe installed torch/CUDA versions: {exc}")
driver=subprocess.check_output(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader"],text=True).splitlines()
drivers=sorted(set(x.strip() for x in driver if x.strip()))
if len(drivers)!=1: raise SystemExit("BLOCKED: driver version is missing or ambiguous")
exllama_revision=""
try:
    dist=md.distribution("exllamav3")
    direct=dist.read_text("direct_url.json")
    if direct:
        obj=json.loads(direct); exllama_revision=(obj.get("vcs_info") or {}).get("commit_id", "")
        if not exllama_revision and obj.get("dir_info") is not None:
            from urllib.parse import unquote, urlsplit
            source_url=urlsplit(obj.get("url", ""))
            if source_url.scheme == "file" and not source_url.netloc:
                source_dir=pathlib.Path(unquote(source_url.path)).resolve(strict=True)
                if (source_dir / ".git").exists():
                    exllama_revision=subprocess.check_output(
                        ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
                    ).strip()
                    dirty=subprocess.check_output(
                        ["git", "-C", str(source_dir), "status", "--porcelain=v1"], text=True
                    )
                    if dirty:
                        raise SystemExit("BLOCKED: installed ExLlamaV3 source checkout is dirty")
except Exception: pass
if not exllama_revision:
    try:
        spec_ex=importlib.util.find_spec("exllamav3_ext")
        path=pathlib.Path(spec_ex.origin).resolve()
        for parent in path.parents:
            if (parent/".git").exists():
                exllama_revision=subprocess.check_output(["git","-C",str(parent),"rev-parse","HEAD"],text=True).strip(); break
    except Exception: pass
if len(exllama_revision)!=40 or any(c not in "0123456789abcdef" for c in exllama_revision):
    raise SystemExit("BLOCKED: installed ExLlamaV3 revision cannot be proven from package metadata/source")
vllm_spec=importlib.util.find_spec("vllm.config.vllm")
if not vllm_spec or not vllm_spec.origin: raise SystemExit("BLOCKED: cannot locate installed vLLM config source")
source=pathlib.Path(vllm_spec.origin).read_text(encoding="utf-8")
vllm_source_sha256=__import__("hashlib").sha256(source.encode("utf-8")).hexdigest()
import re
default_pattern=(r"if\s+self\.scheduler_config\.max_num_scheduled_tokens\s+is\s+None\s*:\s*"
                r"self\.scheduler_config\.max_num_scheduled_tokens\s*=\s*\(?\s*max_num_batched_tokens")
if not re.search(default_pattern,source):
    raise SystemExit("BLOCKED: installed vLLM source does not establish scheduled-token default equals batched-token budget")
if "max_num_scheduled_tokens is set to" not in source or "based on" not in source:
    raise SystemExit("BLOCKED: installed vLLM source lacks expected speculative-budget warning")
arg_spec=importlib.util.find_spec("vllm.engine.arg_utils")
if not arg_spec or not arg_spec.origin: raise SystemExit("BLOCKED: cannot locate installed vLLM EngineArgs source")
arg_source=pathlib.Path(arg_spec.origin).read_text(encoding="utf-8")
if not re.search(r"max_num_scheduled_tokens\s*:\s*int\s*\|\s*None\s*=\s*None",arg_source):
    raise SystemExit("BLOCKED: installed EngineArgs source does not default max_num_scheduled_tokens to None")
arg_source_sha256=__import__("hashlib").sha256(arg_source.encode("utf-8")).hexdigest()
observed={"engine_id":f"pid-{pid}-pgid-{pgid}-{pathlib.Path(log_path).stat().st_mtime_ns}",
 "vllm_version":version("vllm"),"exllamav3_revision":exllama_revision,
 "driver_version":drivers[0],"cuda_version":cuda_version,"torch_version":torch_version,
 "model_pack_sha256":hashes["model_tree_sha256"],"tokenizer_sha256":hashes["tokenizer_tree_sha256"],
 "model_config_sha256":hashes["model_config_sha256"],"installed_vllm_config_source_sha256":vllm_source_sha256,
 "installed_vllm_arg_utils_source_sha256":arg_source_sha256,
 "installed_exl3_source_sha256":plugin_sha,"r0_source_sha256":r0_sha,
 "scheduler_override_absent":True}
for key in ("vllm_version","exllamav3_revision","driver_version","cuda_version","torch_version"):
    if observed[key] != manifest["runtime_expectations"].get(key):
        raise SystemExit(f"BLOCKED: observed {key} differs from frozen manifest")
if hashes["model_config_sha256"] != manifest["provenance"]["model_config_sha256"]:
    raise SystemExit("BLOCKED: model config hash differs from frozen manifest")
if log_path.stat().st_size > 64 * 1024 * 1024:
    raise SystemExit("BLOCKED: startup log exceeds 64 MiB proof parsing limit")
log=log_path.read_text(encoding="utf-8",errors="replace")
proof=module.make_runtime_proof(manifest,log,point,int(k),observed,manifest["runtime_expectations"]["effective_max_num_batched_tokens"])
with open(proof_path,"x",encoding="utf-8") as f: json.dump(proof,f,indent=2,sort_keys=True); f.write("\n")
PY
then
  printf 'status=PASS\nsource_derived_auto_budget=%s\n' \
    "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["config"]["effective_max_num_batched_tokens"])' "$OUT/runtime_proof.json")" \
    > "$OUT/startup_proof_status.txt"
else
  proof_rc=$?
  proof_reason="$(tail -c 2000 "$OUT/runtime_proof_build.log" | tr '\n' ' ')"
  printf 'status=STARTUP_PROOF_INCOMPLETE\nexit_status=%s\nreason=%s\n' \
    "$proof_rc" "$proof_reason" > "$OUT/startup_proof_status.txt"
  die "STARTUP_PROOF_INCOMPLETE: $proof_reason"
fi

event "PROOF_PASS point=$POINT k=$K budget_source=EngineCore_warning+vllm_0.29.0_default kv=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["kv_pool_tokens"])' "$OUT/runtime_proof.json")"
if "$VENV/bin/python" "$HELPER" --manifest "$MANIFEST" --execute --point "$POINT" --k "$K" \
     --runtime-proof "$OUT/runtime_proof.json" --out-dir "$OUT/cell" --base-url "http://$HOST:$PORT" \
     > "$OUT/helper.stdout.log" 2>&1; then
  event 'CELL_RESULT PASS'
else
  rc=$?
  event "CELL_RESULT FAIL helper_rc=$rc"
  exit "$rc"
fi

XID_AFTER="$(xid_count)" || die 'cannot capture post-cell Xid count'
printf 'xid_count_after_cell=%s\nxid_delta=%s\n' "$XID_AFTER" "$((XID_AFTER-XID_INITIAL))" > "$OUT/xid_after_cell.txt"
(( XID_AFTER == XID_INITIAL )) || die 'Xid count changed during cell'
event "CELL_COMPLETE point=$POINT k=$K"

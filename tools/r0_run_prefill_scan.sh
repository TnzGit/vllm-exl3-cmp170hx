#!/usr/bin/env bash
# Repeatable non-KVMEM MTP k=3 prefill scan over
# MAX_NUM_BATCHED_TOKENS=auto/1024/4096.
#
# Required environment:
#   EXPECTED_SHA  exact HEAD expected in R0_REPO
#   R0_REPO       isolated runtime checkout containing the serving/cell scripts
#   OUT           artifact directory (created if absent)
#   MODEL_DIR     prepared EXL3 model pack
#   GPU_MEM_UTIL  explicit vLLM GPU memory fraction
#   PROMPT_CASE_DIR directory containing ctx16000/ctx80000/ctx160000
#                   turn_04_ask_d_e.json exact-token cases
#
# Optional positional arguments select one or more candidates from
# auto 1024 2048 4096. With no arguments, auto/1024/4096 run; 2048 is
# available for an explicit control after verifying the installed auto value.
# Defaults/assumptions: port 8002, MAX_MODEL_LEN=246000, one sequence,
# max output 256, three measured repeats per context. The cell runner is expected
# to issue exactly one unmeasured warmup and then --repeats measured requests;
# it must consume the same exact token-ID file for a given context across
# candidates. Each context is invoked separately so raw output is retained.
# This runner never patches installed sources or edits another checkout.

set -Eeuo pipefail

EXPECTED_SHA="${EXPECTED_SHA:?set EXPECTED_SHA to the exact R0_REPO HEAD}"
R0_REPO="${R0_REPO:?set R0_REPO to the isolated runtime checkout}"
OUT="${OUT:?set OUT to an artifact directory}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the prepared EXL3 pack}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:?set GPU_MEM_UTIL explicitly}"
PROMPT_CASE_DIR="${PROMPT_CASE_DIR:?set PROMPT_CASE_DIR to exact-token cases}"

PORT="${PORT:-8002}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-246000}"
MAXTOK="${MAXTOK:-256}"
REPEATS="${REPEATS:-3}"
HOST="${HOST:-127.0.0.1}"
R0_ROOT="${R0_ROOT:-/home/base-node/.codex_tasks/qwen38-flashnext-r0}"
VENV="$R0_ROOT/venv"
CONTEXT_SPECS=("15533:ctx16000" "79533:ctx80000" "159533:ctx160000")
DEFAULT_CANDIDATES=(auto 1024 4096)

[[ -x "$VENV/bin/python" ]] || { echo "ERROR: missing R0 venv: $VENV" >&2; exit 2; }
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
export VIRTUAL_ENV="$VENV"

if (($#)); then
  CANDIDATES=("$@")
else
  CANDIDATES=("${DEFAULT_CANDIDATES[@]}")
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
EVENTS="$OUT/events.log"
ACTIVE_PID=""
ACTIVE_PGID=""
ACTIVE_CONFIG=""
ACTIVE_LOG=""

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

event() {
  printf '%s run=%s %s\n' "$(timestamp)" "$RUN_ID" "$*" | tee -a "$EVENTS"
}

die() {
  event "ERROR $*"
  exit 2
}

shopt -s nullglob dotglob
existing_outputs=("$OUT"/*)
if ((${#existing_outputs[@]} != 0)); then
  printf 'ERROR: OUT must be empty so prior artifacts cannot be overwritten: %s\n' "$OUT" >&2
  exit 2
fi

declare -A seen_candidates=()
for candidate in "${CANDIDATES[@]}"; do
  case "$candidate" in
    auto|1024|2048|4096) ;;
    *) die "invalid MAX_NUM_BATCHED_TOKENS candidate: $candidate" ;;
  esac
  [[ -z "${seen_candidates[$candidate]+present}" ]] || die "duplicate candidate: $candidate"
  seen_candidates[$candidate]=1
done

[[ -d "$R0_REPO/.git" || -f "$R0_REPO/.git" ]] || die "R0_REPO is not a git checkout: $R0_REPO"
[[ -f "$R0_REPO/tools/serve_cmp170hx_qwen_firstboot.sh" ]] || die "missing serving script under R0_REPO"
[[ -f "$R0_REPO/tools/r0_prefill_scan_cell.py" ]] || die "missing delegated cell runner: $R0_REPO/tools/r0_prefill_scan_cell.py"
[[ -f "$MODEL_DIR/config.json" ]] || die "missing model config: $MODEL_DIR/config.json"
for spec in "${CONTEXT_SPECS[@]}"; do
  case_dir="${spec#*:}"
  [[ -f "$PROMPT_CASE_DIR/$case_dir/turn_04_ask_d_e.json" ]] || \
    die "missing exact prompt case: $PROMPT_CASE_DIR/$case_dir/turn_04_ask_d_e.json"
done
[[ "$EXPECTED_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || die "EXPECTED_SHA must be a full 40-character git SHA"
[[ "$PORT" =~ ^[0-9]+$ && "$MAXTOK" =~ ^[1-9][0-9]*$ && "$REPEATS" =~ ^[1-9][0-9]*$ ]] || die "PORT/MAXTOK/REPEATS must be positive integers"
[[ "$HOST" == 127.0.0.1 ]] || die "the delegated cell runner targets 127.0.0.1; HOST must remain 127.0.0.1"

ACTUAL_SHA="$(git -C "$R0_REPO" rev-parse HEAD)"
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || die "R0_REPO HEAD $ACTUAL_SHA does not match EXPECTED_SHA $EXPECTED_SHA"
[[ -z "$(git -C "$R0_REPO" status --porcelain=v1)" ]] || die "R0_REPO is dirty"
INSTALLED_PLUGIN="$VENV/lib/python3.12/site-packages/vllm_exl3/exl3.py"
[[ -f "$INSTALLED_PLUGIN" ]] || die "missing installed EXL3 plugin: $INSTALLED_PLUGIN"
REPO_PLUGIN_SHA="$(sha256sum "$R0_REPO/src/vllm_exl3/exl3.py" | awk '{print $1}')"
INSTALLED_PLUGIN_SHA="$(sha256sum "$INSTALLED_PLUGIN" | awk '{print $1}')"
[[ "$REPO_PLUGIN_SHA" == "$INSTALLED_PLUGIN_SHA" ]] || \
  die "installed EXL3 plugin differs from exact source ($INSTALLED_PLUGIN_SHA != $REPO_PLUGIN_SHA)"

mkdir -p "$OUT/prompt_cases"
for spec in "${CONTEXT_SPECS[@]}"; do
  context="${spec%%:*}"
  case_dir="${spec#*:}"
  cp "$PROMPT_CASE_DIR/$case_dir/turn_04_ask_d_e.json" \
    "$OUT/prompt_cases/prompt_${context}.json"
done
sha256sum "$OUT"/prompt_cases/*.json > "$OUT/prompt_cases/sha256.txt"

cat > "$OUT/run_identity.txt" <<EOF
run_id=$RUN_ID
started_utc=$(timestamp)
expected_sha=$EXPECTED_SHA
r0_repo=$R0_REPO
repo_head=$ACTUAL_SHA
repo_status_begin
$(git -C "$R0_REPO" status --short)
repo_status_end
model_dir=$MODEL_DIR
prompt_case_dir=$PROMPT_CASE_DIR
installed_plugin_sha256=$INSTALLED_PLUGIN_SHA
venv=$VENV
port=$PORT
host=$HOST
gpu_mem_util=$GPU_MEM_UTIL
max_model_len=$MAX_MODEL_LEN
mtp_num_speculative_tokens=3
contexts=${CONTEXT_SPECS[*]}
candidates=${CANDIDATES[*]}
repeats=$REPEATS
max_tokens=$MAXTOK
cell_contract=one_warmup_then_repeats_measured_requests_per_context
EOF

gpu_snapshot() {
  local path="$1"
  {
    echo "timestamp_utc=$(timestamp)"
    nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader
    echo "compute_processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader || true
  } > "$path" 2>&1
}

xid_count() {
  journalctl -k -b -n 50000 --no-pager -o cat 2>/dev/null \
    | grep -ciE 'NVRM: Xid' || true
}

port_is_open() {
  python - "$HOST" "$PORT" <<'PY'
import socket, sys
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect((sys.argv[1], int(sys.argv[2])))
except OSError:
    raise SystemExit(1)
else:
    raise SystemExit(0)
finally:
    sock.close()
PY
}

record_port() {
  local path="$1" state=CLOSED
  if port_is_open; then state=OPEN; fi
  printf 'timestamp_utc=%s\nhost=%s\nport=%s\nstate=%s\n' \
    "$(timestamp)" "$HOST" "$PORT" "$state" > "$path"
}

record_processes() {
  local path="$1" pgid="${2:-}"
  {
    echo "timestamp_utc=$(timestamp)"
    if [[ -n "$pgid" ]]; then
      ps -eo pid,ppid,pgid,lstart,args | awk -v pgid="$pgid" 'NR == 1 || $3 == pgid'
    else
      ps -eo pid,ppid,pgid,lstart,args | head -n 80
    fi
  } > "$path" 2>&1 || true
}

installed_source_fingerprint() {
  local destination="$1"
  python - "$destination" <<'PY'
import hashlib, importlib.util, json, pathlib, sys

roots = {}
versions = {}
for name in ("vllm", "vllm_exl3"):
    spec = importlib.util.find_spec(name)
    locations = list(spec.submodule_search_locations or []) if spec else []
    if not locations and spec and spec.origin:
        locations = [str(pathlib.Path(spec.origin).parent)]
    roots[name] = [str(pathlib.Path(p).resolve()) for p in locations]
    try:
        module = __import__(name)
        versions[name] = getattr(module, "__version__", "unknown")
    except Exception as exc:
        versions[name] = f"import-error: {type(exc).__name__}: {exc}"

digest = hashlib.sha256()
file_count = 0
for name in sorted(roots):
    for root_text in roots[name]:
        root = pathlib.Path(root_text)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            relative = f"{name}/{path.relative_to(root).as_posix()}".encode()
            digest.update(len(relative).to_bytes(8, "big")); digest.update(relative)
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
            file_count += 1

with open(sys.argv[1], "w") as out:
    json.dump({"sha256_python_sources": digest.hexdigest(), "python_file_count": file_count,
               "roots": roots, "versions": versions}, out, indent=2)
    out.write("\n")
PY
}

stop_owned_process() {
  [[ -n "$ACTIVE_PID" ]] || return 0
  local process_rc=0
  event "STOP_REQUEST config=$ACTIVE_CONFIG pid=$ACTIVE_PID pgid=${ACTIVE_PGID:-unknown}"
  if [[ -n "$ACTIVE_PGID" ]] && kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
    local deadline=$((SECONDS + 20))
    while (( SECONDS < deadline )) && kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; do
      sleep 1
    done
    kill -0 -- "-$ACTIVE_PGID" 2>/dev/null && kill -KILL -- "-$ACTIVE_PGID" 2>/dev/null || true
  elif kill -0 "$ACTIVE_PID" 2>/dev/null; then
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
  fi
  if wait "$ACTIVE_PID" 2>/dev/null; then
    process_rc=0
  else
    process_rc=$?
  fi
  printf 'server_exit_status=%s\nserver_exit_utc=%s\n' "$process_rc" "$(timestamp)" \
    >> "$OUT/$ACTIVE_CONFIG/startup.txt"
  record_processes "$OUT/processes_${ACTIVE_CONFIG}_after.txt" "${ACTIVE_PGID:-}"
  event "STOPPED config=$ACTIVE_CONFIG pid=$ACTIVE_PID pgid=${ACTIVE_PGID:-unknown}"
  ACTIVE_PID=""
  ACTIVE_PGID=""
  ACTIVE_CONFIG=""
  ACTIVE_LOG=""
}

finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  stop_owned_process || true
  gpu_snapshot "$OUT/gpu_final.csv" || true
  printf 'run_id=%s\nfinished_utc=%s\nexit_status=%s\n' \
    "$RUN_ID" "$(timestamp)" "$rc" > "$OUT/run_exit.txt"
  event "RUN_EXIT status=$rc"
  exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required on the experiment host"
command -v curl >/dev/null 2>&1 || die "curl is required on the experiment host"
command -v setsid >/dev/null 2>&1 || die "setsid is required to scope cleanup to this runner's process group"
port_is_open && die "port $HOST:$PORT is already accepting connections"
record_port "$OUT/port_before.txt"
gpu_snapshot "$OUT/gpu_before.csv" || die "failed to capture GPU identity"
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | grep -E '^[[:space:]]*[0-9]+' || true)"
[[ -z "$GPU_PIDS" ]] || die "GPU already has compute processes; refusing to interfere: $GPU_PIDS"
XID_INITIAL="$(xid_count)"
printf 'xid_count_before=%s\n' "$XID_INITIAL" > "$OUT/xid_before.txt"
record_processes "$OUT/processes_before.txt"

event "RUN_START repo=$R0_REPO sha=$ACTUAL_SHA candidates=${CANDIDATES[*]} contexts=${CONTEXT_SPECS[*]}"
installed_source_fingerprint "$OUT/installed_source_before.json"

for candidate in "${CANDIDATES[@]}"; do
  [[ "$candidate" == auto ]] && config="auto" || config="$candidate"
  config_dir="$OUT/batched_tokens_${config}"
  mkdir -p "$config_dir"
  ACTIVE_CONFIG="batched_tokens_${config}"
  ACTIVE_LOG="$config_dir/server.log"
  : > "$ACTIVE_LOG"
  event "CONFIG_START config=$config dir=$config_dir"
  printf 'config=%s\nstarted_utc=%s\n' "$config" "$(timestamp)" > "$config_dir/startup.txt"
  XID_CONFIG_BEFORE="$(xid_count)"
  printf 'xid_count_before=%s\n' "$XID_CONFIG_BEFORE" > "$config_dir/xid.txt"

  if [[ "$candidate" == auto ]]; then
    BATCHED_TOKENS=""
  else
    BATCHED_TOKENS="$candidate"
  fi

  MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" MAX_NUM_SEQS=1 PORT="$PORT" HOST="$HOST" \
    NUM_SPEC_TOKENS=3 MAX_NUM_BATCHED_TOKENS="$BATCHED_TOKENS" \
    TORCH_PROFILER_DIR= \
    setsid bash "$R0_REPO/tools/serve_cmp170hx_qwen_firstboot.sh" \
    > "$ACTIVE_LOG" 2>&1 < /dev/null &
  ACTIVE_PID=$!
  ACTIVE_PGID="$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ' || true)"
  [[ "$ACTIVE_PGID" == "$ACTIVE_PID" ]] || \
    die "setsid process group does not match owned launcher PID; refusing scoped cleanup"
  printf 'launch_pid=%s\nlaunch_pgid=%s\nlaunch_utc=%s\n' \
    "$ACTIVE_PID" "${ACTIVE_PGID:-unknown}" "$(timestamp)" >> "$config_dir/startup.txt"
  record_processes "$config_dir/processes_start.txt" "$ACTIVE_PGID"

  healthy=0
  deadline=$((SECONDS + 2400))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 "http://$HOST:$PORT/health" >/dev/null 2>&1 \
      && curl -fsS --max-time 10 "http://$HOST:$PORT/v1/models" \
        | grep -q '"id"'; then
      healthy=1
      break
    fi
    if ! kill -0 "$ACTIVE_PID" 2>/dev/null; then
      break
    fi
    sleep 5
  done

  if (( healthy == 0 )); then
    failed_log="$ACTIVE_LOG"
    printf 'startup_failed_utc=%s\n' "$(timestamp)" >> "$config_dir/startup.txt"
    event "STARTUP_FAILED config=$config pid=$ACTIVE_PID"
    stop_owned_process
    record_port "$config_dir/port_after.txt"
    xid_now="$(xid_count)"
    printf 'xid_count_after=%s\nxid_delta=%s\n' "$xid_now" "$((xid_now - XID_CONFIG_BEFORE))" >> "$config_dir/xid.txt"
    die "engine failed to become healthy for MAX_NUM_BATCHED_TOKENS=$config; see $failed_log"
  fi

  printf 'healthy_utc=%s\n' "$(timestamp)" >> "$config_dir/startup.txt"
  event "ENGINE_HEALTHY config=$config pid=$ACTIVE_PID pgid=${ACTIVE_PGID:-unknown}"
  record_port "$config_dir/port_healthy.txt"
  gpu_snapshot "$config_dir/gpu_healthy.csv"
  installed_source_fingerprint "$config_dir/installed_source.json"
  record_processes "$config_dir/processes_healthy.txt" "$ACTIVE_PGID"
  curl -fsS --max-time 10 "http://$HOST:$PORT/metrics" > "$config_dir/metrics_healthy.prom"

  for spec in "${CONTEXT_SPECS[@]}"; do
    context="${spec%%:*}"
    cell_json="$config_dir/cell_${context}.json"
    cell_log="$config_dir/cell_${context}.stdout.log"
    event "CELL_START config=$config context=$context repeats=$REPEATS"
    if python "$R0_REPO/tools/r0_prefill_scan_cell.py" \
      --port "$PORT" --context "$context" --max-tokens "$MAXTOK" \
      --repeats "$REPEATS" --prompt-token-ids "$OUT/prompt_cases/prompt_${context}.json" \
      --out "$cell_json" > "$cell_log" 2>&1; then
      :
    else
      cell_rc=$?
      printf 'cell_exit_status=%s\nfailed_utc=%s\n' "$cell_rc" "$(timestamp)" \
        > "$config_dir/cell_${context}.status.txt"
      event "CELL_FAILED config=$config context=$context status=$cell_rc"
      stop_owned_process
      die "cell failed for MAX_NUM_BATCHED_TOKENS=$config context=$context; artifacts retained"
    fi
    printf 'cell_exit_status=0\nfinished_utc=%s\n' "$(timestamp)" \
      > "$config_dir/cell_${context}.status.txt"
    event "CELL_DONE config=$config context=$context"
  done

  curl -fsS --max-time 10 "http://$HOST:$PORT/metrics" > "$config_dir/metrics_final.prom" || true
  XID_CONFIG_AFTER="$(xid_count)"
  printf 'xid_count_after=%s\nxid_delta=%s\n' "$XID_CONFIG_AFTER" \
    "$((XID_CONFIG_AFTER - XID_CONFIG_BEFORE))" >> "$config_dir/xid.txt"
  gpu_snapshot "$config_dir/gpu_before_shutdown.csv"
  printf 'completed_utc=%s\n' "$(timestamp)" >> "$config_dir/startup.txt"
  event "CONFIG_DONE config=$config"
  stop_owned_process
  installed_source_fingerprint "$config_dir/installed_source_after.json"
  cmp -s "$OUT/installed_source_before.json" "$config_dir/installed_source_after.json" || \
    die "installed Python source fingerprint changed during config=$config"
  record_port "$config_dir/port_after.txt"
  if port_is_open; then
    die "port $HOST:$PORT remains open after stopping owned engine; artifacts retained"
  fi
done

XID_FINAL="$(xid_count)"
printf 'xid_count_after=%s\nxid_delta=%s\n' "$XID_FINAL" \
  "$((XID_FINAL - XID_INITIAL))" > "$OUT/xid_after.txt"
event "RUN_COMPLETE xid_delta=$((XID_FINAL - XID_INITIAL))"
(( XID_FINAL == XID_INITIAL )) || die "Xid count changed during scan"

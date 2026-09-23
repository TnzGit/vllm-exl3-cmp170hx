#!/usr/bin/env bash
# Isolated non-KVMEM APC probe: exact ctx16000 prompt, MTP k=3, two requests.
# Requires EXPECTED_SHA, clean R0_REPO, empty OUT, MODEL_DIR, PROMPT_CASE_DIR.
set -Eeuo pipefail
EXPECTED_SHA="${EXPECTED_SHA:?set exact clean R0_REPO SHA}"
R0_REPO="${R0_REPO:?set isolated runtime checkout}"
OUT="${OUT:?set empty artifact directory}"
MODEL_DIR="${MODEL_DIR:?set prepared EXL3 model pack}"
PROMPT_CASE_DIR="${PROMPT_CASE_DIR:?set exact-token prompt cases}"

readonly HOST=127.0.0.1 PORT=8002 GPU_MEM_UTIL=0.92 MAX_MODEL_LEN=246000
readonly NUM_SPEC_TOKENS=3 PROMPT_CASE_REL=ctx16000/turn_04_ask_d_e.json
readonly REPEATS=2 MAX_TOKENS=8
readonly VENV="${R0_ROOT:-/home/base-node/.codex_tasks/qwen38-flashnext-r0}/venv"
readonly LAUNCH_TIMEOUT="${LAUNCH_TIMEOUT:-2400}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
EVENTS="$OUT/events.log"
ACTIVE_PID="" ACTIVE_PGID=""

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
event() { printf '%s run=%s %s\n' "$(timestamp)" "$RUN_ID" "$*" | tee -a "$EVENTS"; }
die() { event "ERROR $*"; exit 2; }
shopt -s nullglob dotglob
existing_outputs=("$OUT"/*)
(("${#existing_outputs[@]}" == 0)) || { echo "ERROR: OUT must be empty: $OUT" >&2; exit 2; }

[[ "$EXPECTED_SHA" =~ ^[0-9a-fA-F]{40}$ ]] || die "EXPECTED_SHA must be a full git SHA"
[[ "$LAUNCH_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die "LAUNCH_TIMEOUT must be positive"
[[ -x "$VENV/bin/python" ]] || die "missing R0 venv: $VENV"
[[ -f "$R0_REPO/tools/serve_cmp170hx_qwen_firstboot.sh" ]] || die "missing serving launcher"
[[ -f "$R0_REPO/tools/r0_prefix_cache_probe.py" ]] || die "missing prefix probe"
[[ -f "$R0_REPO/tools/cmp170hx_qwen_preflight.py" ]] || die "missing Qwen preflight"
[[ -f "$R0_REPO/src/vllm_exl3/exl3.py" ]] || die "missing repo EXL3 source"
[[ -f "$MODEL_DIR/config.json" ]] || die "missing model config"
PROMPT_SOURCE="$PROMPT_CASE_DIR/$PROMPT_CASE_REL"
[[ -f "$PROMPT_SOURCE" ]] || die "missing prompt case: $PROMPT_SOURCE"
ACTUAL_SHA="$(git -C "$R0_REPO" rev-parse HEAD 2>/dev/null)" || die "R0_REPO is not a git checkout"
[[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]] || die "R0_REPO HEAD differs from EXPECTED_SHA"
[[ -z "$(git -C "$R0_REPO" status --porcelain=v1)" ]] || die "R0_REPO is dirty"
INSTALLED_PLUGIN="$VENV/lib/python3.12/site-packages/vllm_exl3/exl3.py"
[[ -f "$INSTALLED_PLUGIN" ]] || die "missing installed EXL3 source"
REPO_PLUGIN_SHA="$(sha256sum "$R0_REPO/src/vllm_exl3/exl3.py" | awk '{print $1}')"
INSTALLED_PLUGIN_SHA="$(sha256sum "$INSTALLED_PLUGIN" | awk '{print $1}')"
[[ "$REPO_PLUGIN_SHA" == "$INSTALLED_PLUGIN_SHA" ]] || die "installed EXL3 source differs from R0_REPO"
export CUDA_HOME="$VENV/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH" VIRTUAL_ENV="$VENV"

port_listener_count() { ss -H -ltn "sport = :$PORT" 2>/dev/null | wc -l | tr -d ' '; }
port_is_open() {
  python - "$HOST" "$PORT" <<'PY'
import socket,sys
s=socket.socket(); s.settimeout(1)
try: s.connect((sys.argv[1],int(sys.argv[2])))
except OSError: raise SystemExit(1)
else: raise SystemExit(0)
finally: s.close()
PY
}
record_port() {
  local path="$1" listeners state=CLOSED
  listeners="$(port_listener_count)"
  if (( listeners > 0 )) || port_is_open; then state=OPEN; fi
  printf 'timestamp_utc=%s\nhost=%s\nport=%s\nlisteners=%s\nstate=%s\n' \
    "$(timestamp)" "$HOST" "$PORT" "$listeners" "$state" > "$path"
}
gpu_snapshot() {
  local path="$1"
  { echo "timestamp_utc=$(timestamp)"
    nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
    echo "compute_processes:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  } > "$path" 2>&1
}
gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/,"");print}'
}
xid_count() { journalctl -k -b -n 50000 --no-pager -o cat 2>/dev/null | tail -c 4000000 | grep -ciE 'NVRM: Xid' || true; }
record_processes() {
  local path="$1" pgid="${2:-}"
  { echo "timestamp_utc=$(timestamp)"
    if [[ -n "$pgid" ]]; then ps -eo pid,ppid,pgid,lstart,args | awk -v p="$pgid" 'NR==1 || $3==p'
    else ps -eo pid,ppid,pgid,lstart,args | head -n 80; fi
  } > "$path" 2>&1 || true
}
installed_source_fingerprint() {
  "$VENV/bin/python" - "$1" <<'PY'
import hashlib,importlib.util,json,pathlib,sys
roots={}; versions={}
for name in ("vllm","vllm_exl3"):
    spec=importlib.util.find_spec(name); loc=list(spec.submodule_search_locations or []) if spec else []
    if not loc and spec and spec.origin: loc=[str(pathlib.Path(spec.origin).parent)]
    roots[name]=[str(pathlib.Path(p).resolve()) for p in loc]
    try: m=__import__(name); versions[name]=getattr(m,"__version__","unknown")
    except Exception as e: versions[name]=f"import-error: {type(e).__name__}: {e}"
h=hashlib.sha256(); count=0
for name in sorted(roots):
  for text in roots[name]:
    root=pathlib.Path(text)
    if not root.is_dir(): continue
    for p in sorted(root.rglob("*.py")):
      rel=f"{name}/{p.relative_to(root).as_posix()}".encode(); h.update(len(rel).to_bytes(8,"big")); h.update(rel)
      with p.open("rb") as f:
        while chunk:=f.read(1024*1024): h.update(chunk)
      count+=1
with open(sys.argv[1],"w") as f:
  json.dump({"sha256_python_sources":h.hexdigest(),"python_file_count":count,"roots":roots,"versions":versions},f,indent=2); f.write("\n")
PY
}
stop_owned_process() {
  [[ -n "$ACTIVE_PID" ]] || return 0
  event "STOP_REQUEST pid=$ACTIVE_PID pgid=${ACTIVE_PGID:-unknown}"
  if [[ -n "$ACTIVE_PGID" ]]; then
    if kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; then
    kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
    local end=$((SECONDS+20))
    while (( SECONDS<end )) && kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; do sleep 1; done
    if kill -0 -- "-$ACTIVE_PGID" 2>/dev/null; then kill -KILL -- "-$ACTIVE_PGID" 2>/dev/null || true; fi
    fi
  elif kill -0 "$ACTIVE_PID" 2>/dev/null; then
    # If setsid ownership could not be proven, signal only our exact child PID.
    kill -TERM "$ACTIVE_PID" 2>/dev/null || true
    local end=$((SECONDS+20))
    while (( SECONDS<end )) && kill -0 "$ACTIVE_PID" 2>/dev/null; do sleep 1; done
    if kill -0 "$ACTIVE_PID" 2>/dev/null; then kill -KILL "$ACTIVE_PID" 2>/dev/null || true; fi
  fi
  local rc=0
  if wait "$ACTIVE_PID" 2>/dev/null; then rc=0; else rc=$?; fi
  printf 'server_exit_status=%s\nserver_exit_utc=%s\n' "$rc" "$(timestamp)" >> "$OUT/startup.txt"
  record_processes "$OUT/processes_after_stop.txt" "$ACTIVE_PGID"
  event "STOPPED pid=$ACTIVE_PID pgid=$ACTIVE_PGID rc=$rc"
  ACTIVE_PID="" ACTIVE_PGID=""
}
finish() {
  local rc=$?
  trap - EXIT INT TERM HUP
  stop_owned_process || true
  command -v nvidia-smi >/dev/null 2>&1 && gpu_snapshot "$OUT/gpu_final.csv" || true
  command -v ss >/dev/null 2>&1 && record_port "$OUT/port_final.txt" || true
  if [[ -n "${XID_INITIAL:-}" ]]; then
    local xf; xf="$(xid_count)"
    printf 'xid_count_after=%s\nxid_delta=%s\n' "$xf" "$((xf-XID_INITIAL))" > "$OUT/xid_final.txt"
    if (( xf!=XID_INITIAL )); then rc=2; event "ERROR Xid count changed"; fi
  fi
  if [[ -n "$ACTIVE_PGID" ]]; then rc=2; fi
  if command -v ss >/dev/null 2>&1 && (( $(port_listener_count)>0 )); then rc=2; event "ERROR port listener remains"; fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gp; gp="$(gpu_compute_pids || true)"
    if [[ -n "$gp" ]]; then rc=2; printf 'remaining_compute_pids=%s\n' "$gp" > "$OUT/gpu_hygiene_failure.txt"; fi
  fi
  printf 'run_id=%s\nfinished_utc=%s\nexit_status=%s\n' "$RUN_ID" "$(timestamp)" "$rc" > "$OUT/run_exit.txt"
  event "RUN_EXIT status=$rc"; exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

for c in nvidia-smi curl setsid ss journalctl sha256sum; do command -v "$c" >/dev/null 2>&1 || die "$c required"; done
(( $(port_listener_count)==0 )) || die "port $PORT already has a listener"
port_is_open && die "port $PORT already accepts connections"
record_port "$OUT/port_before.txt"
gpu_snapshot "$OUT/gpu_before.csv" || die "cannot capture GPU state"
GPU_PIDS="$(gpu_compute_pids)"
[[ -z "$GPU_PIDS" ]] || die "GPU already has compute processes: $GPU_PIDS"
XID_INITIAL="$(xid_count)"; printf 'xid_count_before=%s\n' "$XID_INITIAL" > "$OUT/xid_before.txt"
record_processes "$OUT/processes_before.txt"

# Preserve original ctx16000 case and normalize its token IDs for the probe.
cp "$PROMPT_SOURCE" "$OUT/original_prompt_case.json"
PROMPT_IDS="$OUT/prompt_token_ids.json"
"$VENV/bin/python" - "$PROMPT_SOURCE" "$PROMPT_IDS" <<'PY'
import json,pathlib,sys
s,d=map(pathlib.Path,sys.argv[1:])
try: x=json.loads(s.read_text(encoding="utf-8"))
except (OSError,json.JSONDecodeError) as e: raise SystemExit(f"ERROR: invalid prompt JSON: {e}")
if isinstance(x,dict): x=x.get("prompt_token_ids")
if not isinstance(x,list) or not x: raise SystemExit("ERROR: expected token array or prompt_token_ids field")
if any(isinstance(t,bool) or not isinstance(t,int) or t<0 for t in x): raise SystemExit("ERROR: invalid token IDs")
if not 15000<=len(x)<=16384: raise SystemExit(f"ERROR: unexpected ctx16000 token count {len(x)}")
d.write_text(json.dumps(x,separators=(",",":"))+"\n",encoding="utf-8")
print(f"prompt_token_count={len(x)}")
PY
sha256sum "$PROMPT_SOURCE" "$OUT/original_prompt_case.json" "$PROMPT_IDS" > "$OUT/prompt_sha256.txt"
installed_source_fingerprint "$OUT/installed_source_before.json" || die "installed source fingerprint failed"
cat > "$OUT/run_identity.txt" <<EOF
run_id=$RUN_ID
expected_sha=$EXPECTED_SHA
r0_repo=$R0_REPO
repo_head=$ACTUAL_SHA
repo_status=clean
repo_exl3_sha256=$REPO_PLUGIN_SHA
installed_exl3_sha256=$INSTALLED_PLUGIN_SHA
model_dir=$MODEL_DIR
prompt_case=$PROMPT_SOURCE
prompt_case_sha256=$(awk 'NR==1 {print $1}' "$OUT/prompt_sha256.txt")
venv=$VENV
host=$HOST
port=$PORT
gpu_mem_util=$GPU_MEM_UTIL
max_model_len=$MAX_MODEL_LEN
max_num_batched_tokens=auto
max_num_seqs=1
prefix_caching=1
mtp_num_speculative_tokens=$NUM_SPEC_TOKENS
repeats=$REPEATS
max_tokens=$MAX_TOKENS
EOF

event "RUN_START sha=$ACTUAL_SHA port=$PORT prefix=on mtp_k=$NUM_SPEC_TOKENS"
: > "$OUT/server.log"; : > "$OUT/startup.txt"
env MODEL_DIR="$MODEL_DIR" GPU_MEM_UTIL="$GPU_MEM_UTIL" MAX_MODEL_LEN="$MAX_MODEL_LEN" \
  MAX_NUM_SEQS=1 PORT="$PORT" HOST="$HOST" PREFIX_CACHING=1 NUM_SPEC_TOKENS=3 \
  MAX_NUM_BATCHED_TOKENS= TORCH_PROFILER_DIR= \
  setsid bash "$R0_REPO/tools/serve_cmp170hx_qwen_firstboot.sh" > "$OUT/server.log" 2>&1 < /dev/null &
ACTIVE_PID=$!
observed_pgid="$(ps -o pgid= -p "$ACTIVE_PID" 2>/dev/null | tr -d ' ' || true)"
if [[ "$observed_pgid" != "$ACTIVE_PID" ]]; then
  ACTIVE_PGID=""
  die "setsid group does not match PID; refusing group cleanup"
fi
ACTIVE_PGID="$observed_pgid"
printf 'launch_pid=%s\nlaunch_pgid=%s\nlaunch_utc=%s\n' "$ACTIVE_PID" "$ACTIVE_PGID" "$(timestamp)" > "$OUT/startup.txt"
record_processes "$OUT/processes_start.txt" "$ACTIVE_PGID"
healthy=0; end=$((SECONDS+LAUNCH_TIMEOUT))
while (( SECONDS<end )); do
  if curl -fsS --max-time 5 "http://$HOST:$PORT/health" >/dev/null 2>&1 \
    && curl -fsS --max-time 10 "http://$HOST:$PORT/v1/models" | grep -q '"id"'; then healthy=1; break; fi
  kill -0 "$ACTIVE_PID" 2>/dev/null || break
  sleep 5
done
if (( healthy==0 )); then
  printf 'startup_failed_utc=%s\n' "$(timestamp)" >> "$OUT/startup.txt"
  stop_owned_process; record_port "$OUT/port_after_failure.txt"
  die "engine failed health gate; see $OUT/server.log"
fi
printf 'healthy_utc=%s\n' "$(timestamp)" >> "$OUT/startup.txt"
event "ENGINE_HEALTHY pid=$ACTIVE_PID pgid=$ACTIVE_PGID"
record_port "$OUT/port_healthy.txt"; gpu_snapshot "$OUT/gpu_healthy.csv"
record_processes "$OUT/processes_healthy.txt" "$ACTIVE_PGID"
installed_source_fingerprint "$OUT/installed_source_healthy.json"
cmp -s "$OUT/installed_source_before.json" "$OUT/installed_source_healthy.json" || die "installed source changed before probe"
curl -fsS --max-time 10 "http://$HOST:$PORT/metrics" > "$OUT/metrics_before.prom"

event "PROBE_START prompt_case=$PROMPT_CASE_REL repeats=$REPEATS"
if "$VENV/bin/python" "$R0_REPO/tools/r0_prefix_cache_probe.py" \
  --port "$PORT" --prompt-tokens-file "$PROMPT_IDS" --repeats "$REPEATS" \
  --max-tokens "$MAX_TOKENS" --min-prompt-tokens 15000 --out "$OUT/probe.json" \
  > "$OUT/probe.stdout.log" 2>&1; then
  event "PROBE_PASS"
else
  p_rc=$?
  printf 'probe_exit_status=%s\n' "$p_rc" > "$OUT/probe_status.txt"
  curl -fsS --max-time 10 "http://$HOST:$PORT/metrics" > "$OUT/metrics_after_probe.prom" || true
  stop_owned_process; die "probe failed rc=$p_rc; artifacts retained"
fi
printf 'probe_exit_status=0\n' > "$OUT/probe_status.txt"
curl -fsS --max-time 10 "http://$HOST:$PORT/metrics" > "$OUT/metrics_after_probe.prom"
installed_source_fingerprint "$OUT/installed_source_after.json"
cmp -s "$OUT/installed_source_before.json" "$OUT/installed_source_after.json" || die "installed source changed during probe"
XID_CURRENT="$(xid_count)"
printf 'xid_count_after_probe=%s\nxid_delta=%s\n' "$XID_CURRENT" "$((XID_CURRENT-XID_INITIAL))" > "$OUT/xid_after_probe.txt"
(( XID_CURRENT==XID_INITIAL )) || die "Xid count changed during probe"
gpu_snapshot "$OUT/gpu_before_shutdown.csv"
stop_owned_process; record_port "$OUT/port_after.txt"
(( $(port_listener_count)==0 )) || die "port listener remains after owned engine shutdown"
port_is_open && die "port $PORT still accepts connections after shutdown"
[[ -z "$(gpu_compute_pids)" ]] || die "GPU compute process remains after engine shutdown"
XID_CURRENT="$(xid_count)"
printf 'xid_count_after_shutdown=%s\nxid_delta=%s\n' "$XID_CURRENT" "$((XID_CURRENT-XID_INITIAL))" > "$OUT/xid_after_shutdown.txt"
(( XID_CURRENT==XID_INITIAL )) || die "Xid count changed by shutdown"
event "RUN_COMPLETE status=PASS"

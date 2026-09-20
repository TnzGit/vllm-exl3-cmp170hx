#!/usr/bin/env bash
# Detach a long-running R0 command so it survives the ssh session that started it.
#
# A plain `nohup ... &` from an inline ssh command dies when the session is
# killed. setsid + full redirection + disown keeps it alive.
#
# usage: r0_launch_detached.sh <logfile> <command...>
set -euo pipefail

LOG="$1"
shift

mkdir -p "$(dirname "$LOG")"
setsid bash -c "$*" > "$LOG" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "launched pid=$PID log=$LOG"

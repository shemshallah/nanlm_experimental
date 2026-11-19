#!/usr/bin/env bash
# Launch a long command fully detached from the notebook/kernel.
#
#   ./z3/molab_launch.sh runs/foo.log runs/foo.pid -- \
#       python z3/train_v3.py ... --out runs/foo
#
# nohup ignores SIGHUP; setsid creates a new session/process group, so the
# command does not receive terminal-close signals and its stdout goes to a
# file.  This does NOT keep the command alive if the Molab sandbox/container
# itself is stopped or restarted.  Always pair it with periodic checkpoint
# uploads and a resumable --resume command.
set -u
if [ "$#" -lt 4 ] || [ "$3" != "--" ]; then
    echo "usage: $0 LOG PIDFILE -- COMMAND [ARGS...]" >&2
    exit 2
fi
LOG="$1"
PIDFILE="$2"
shift 3
mkdir -p "$(dirname "$LOG")" "$(dirname "$PIDFILE")"
if command -v setsid >/dev/null 2>&1; then
    nohup setsid "$@" >"$LOG" 2>&1 &
else
    nohup "$@" >"$LOG" 2>&1 &
fi
PID=$!
printf '%s\n' "$PID" >"$PIDFILE"
echo "launched pid=$PID log=$LOG pidfile=$PIDFILE"

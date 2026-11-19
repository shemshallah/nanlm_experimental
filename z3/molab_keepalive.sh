#!/usr/bin/env bash
# Keep a Molab/marimo session warm and detect token/session loss early.
#
# Usage:
#   MARIMO_URL=https://sb-....sb.molab.run/ \
#   MARIMO_TOKEN=... \
#   ./z3/molab_keepalive.sh
#
# Optional:
#   KEEPALIVE_INTERVAL=900     seconds between pings (default 15 min)
#   KEEPALIVE_MAX_HOURS=8      stop after this many hours
#   KEEPALIVE_CODE="..."       code to execute instead of a trivial print
#   MARIMO_EXEC=/path/to/execute-code.sh
#
# Exit codes: 0 = max hours reached, 2 = session/token lost.
set -u

: "${MARIMO_URL:?set MARIMO_URL}"
: "${MARIMO_TOKEN:?set MARIMO_TOKEN}"

INTERVAL="${KEEPALIVE_INTERVAL:-900}"
MAX_HOURS="${KEEPALIVE_MAX_HOURS:-8}"
EXEC="${MARIMO_EXEC:-$HOME/.config/opencode/skills/marimo-pair/scripts/execute-code.sh}"
CODE="${KEEPALIVE_CODE:-print('keepalive')}"
END=$(( $(date +%s) + MAX_HOURS * 3600 ))

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keepalive start url=$MARIMO_URL interval=${INTERVAL}s max=${MAX_HOURS}h"
while [ "$(date +%s)" -lt "$END" ]; do
    TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    HTTP=$(curl -sS -m 20 -o /dev/null -w '%{http_code}' \
        -H "Authorization: Bearer $MARIMO_TOKEN" \
        "$MARIMO_URL/api/sessions" 2>/dev/null || echo 000)
    if [ "$HTTP" != "200" ]; then
        echo "$TS SESSION_LOST http=$HTTP"
        exit 2
    fi
    OUT=$(MARIMO_TOKEN="$MARIMO_TOKEN" bash "$EXEC" --url "$MARIMO_URL" -c "$CODE" 2>&1)
    RC=$?
    OUT_ONE=$(printf '%s' "$OUT" | tr '\n' ' ' | cut -c1-500)
    echo "$TS http=$HTTP rc=$RC $OUT_ONE"
    if [ "$RC" -ne 0 ]; then
        echo "$TS SESSION_LOST execute_rc=$RC"
        exit 2
    fi
    sleep "$INTERVAL"
done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) keepalive done"
exit 0

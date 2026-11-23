#!/bin/bash
# Start the integrated project service (default port 5001)
# Usage: ./run_server.sh [PORT]
# Interpreter priority: $PYTHON > active python3 > python
cd "$(dirname "$0")" || exit 1

PORT="${1:-5001}"
export JEV_DECIDE_CONCURRENCY="${JEV_DECIDE_CONCURRENCY:-8}"
export JEV_DECIDE_TIMEOUT="${JEV_DECIDE_TIMEOUT:-600}"
export JEV_DECIDE_MAX_RETRIES="${JEV_DECIDE_MAX_RETRIES:-0}"

if [ -n "$PYTHON" ]; then
  PY="$PYTHON"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "No Python interpreter found. Install Python 3.10+ or set the PYTHON environment variable." >&2
  exit 1
fi

echo "Using interpreter: $PY"
exec "$PY" -u -c "
from app import app
app.run(host='0.0.0.0', port=$PORT, debug=False, threaded=True)
"

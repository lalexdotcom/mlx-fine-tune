#!/bin/bash
set -e

FINE_TUNE_DIR="$HOME/.mlx-fine-tune"
VENV="$FINE_TUNE_DIR/venv"
LOG_DIR="$FINE_TUNE_DIR/logs"
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
LOG_FILE="$LOG_DIR/eval_$RUN_ID.log"
PID_FILE="$LOG_DIR/eval_$RUN_ID.pid"

mkdir -p "$VENV" "$LOG_DIR"

# Prevent running multiple instances simultaneously
for pid_file in "$LOG_DIR"/*.pid; do
  [ -f "$pid_file" ] || continue
  existing_pid=$(cat "$pid_file")
  if ps -p "$existing_pid" > /dev/null 2>&1; then
    echo "❌ A pipeline is already running (PID: $existing_pid)"
    echo "   Log: $(ls -t $LOG_DIR/*.log | head -1)"
    echo "   Stop: kill $existing_pid"
    exit 1
  fi
done

if [ ! -d "$VENV/bin" ]; then
  echo "Creating venv at $VENV..."
  python3 -m venv "$VENV"
fi

for pkg in pyarrow jinja2 mlx_lm pyyaml; do
  if ! "$VENV/bin/python3" -c "import $pkg" 2>/dev/null; then
    echo "Installing $pkg..."
    "$VENV/bin/pip" install "$pkg" --quiet
  else
    echo "✓ $pkg"
  fi
done

echo ""
echo "Run ID : $RUN_ID"
echo "Logs   : $LOG_FILE"
echo ""

nohup "$VENV/bin/python3" -u evaluate.py \
  --work-dir "$FINE_TUNE_DIR" \
  --run-id "$RUN_ID" \
  "$@" > "$LOG_FILE" 2>&1 &
PYTHON_PID=$!
echo $PYTHON_PID > "$PID_FILE"

echo "PID    : $PYTHON_PID"
echo "Stop   : kill \$(cat $PID_FILE)"

# Follow log output in background
tail -f "$LOG_FILE" &
TAIL_PID=$!

# Trap Ctrl+C to kill tail cleanly without stopping the pipeline
trap "kill $TAIL_PID 2>/dev/null; echo ''; exit 0" INT

# Wait for Python process to finish then clean up tail
wait $PYTHON_PID 2>/dev/null || true
sleep 0.5
kill $TAIL_PID 2>/dev/null || true
wait $TAIL_PID 2>/dev/null || true

echo ""
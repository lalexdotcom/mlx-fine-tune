#!/bin/bash

LOG_DIR="$HOME/.mlx-fine-tune/logs"
LATEST_LOG=$(ls -t "$LOG_DIR"/run_*.log 2>/dev/null | head -1)
LATEST_PID=$(ls -t "$LOG_DIR"/run_*.pid 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
  echo "No log found in $LOG_DIR"
  exit 1
fi

PID=$(cat "$LATEST_PID" 2>/dev/null)

if [ -z "$PID" ] || ! ps -p "$PID" > /dev/null 2>&1; then
  echo "No running pipeline found."
  echo "Last log: $LATEST_LOG"
  echo ""
  tail -20 "$LATEST_LOG"
  exit 0
fi

echo "Pipeline running (PID: $PID)"
echo "Log: $LATEST_LOG"
echo ""

tail -20 -f "$LATEST_LOG" &
TAIL_PID=$!
trap "kill $TAIL_PID 2>/dev/null; echo ''; exit 0" INT
wait $TAIL_PID 2>/dev/null || true
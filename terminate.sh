#!/bin/bash

LOG_DIR="$HOME/.mlx-fine-tune/logs"
LATEST_PID=$(ls -t "$LOG_DIR"/run_*.pid 2>/dev/null | head -1)

if [ -z "$LATEST_PID" ]; then
  echo "No pipeline found."
  exit 0
fi

PID=$(cat "$LATEST_PID" 2>/dev/null)

if [ -z "$PID" ] || ! ps -p "$PID" > /dev/null 2>&1; then
  echo "No running pipeline found."
  exit 0
fi

LATEST_LOG=$(ls -t "$LOG_DIR"/run_*.log | head -1)
echo "Running pipeline:"
echo "  PID : $PID"
echo "  Log : $LATEST_LOG"
echo ""

# Check for --force / -f
FORCE=false
for arg in "$@"; do
  if [ "$arg" = "--force" ] || [ "$arg" = "-f" ]; then
    FORCE=true
  fi
done

if [ "$FORCE" = false ]; then
  read -r -p "Stop this pipeline? [y/N] " confirm
  if [[ ! "$confirm" =~ ^[yY]$ ]]; then
    echo "Cancelled."
    exit 0
  fi
fi

kill "$PID"
echo "✓ Pipeline stopped (PID: $PID)"
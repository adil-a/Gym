#!/usr/bin/env bash
# Usage: ./tmux_keepalive.sh [target] [interval_seconds]
TARGET=${1:-test}
INTERVAL=${2:-5}

echo "Sending Enter to tmux target '$TARGET' every ${INTERVAL}s. Ctrl+C to stop."

while true; do
    tmux send-keys -t "$TARGET" Enter
    sleep "$INTERVAL"
done

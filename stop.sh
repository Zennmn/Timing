#!/usr/bin/env bash
# Stop the Timing service started by start.sh.
set -euo pipefail
cd "$(dirname "$0")"

pid_file=.runtime/server.pid
if [ ! -f "$pid_file" ]; then
    echo "没有记录中的 Timing 服务。"
    exit 0
fi
pid=$(cat "$pid_file")
# Only stop the process if it is still this Timing server, not a reused PID.
if ps -p "$pid" -o command= 2>/dev/null | grep -q -- "-m timing --serve"; then
    kill "$pid"
    for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
    done
    echo "Timing 已关闭。"
else
    echo "没有记录中的 Timing 服务。"
fi
rm -f "$pid_file"

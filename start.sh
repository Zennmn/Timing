#!/usr/bin/env bash
# Start the local Timing page on macOS. --no-browser skips opening it.
set -euo pipefail
cd "$(dirname "$0")"

url="http://127.0.0.1:7861"
open_browser=1
[ "${1:-}" = "--no-browser" ] && open_browser=0

running() { curl -fsS --noproxy 127.0.0.1 --max-time 2 "$url/api/status" 2>/dev/null | grep -q '"Timing"'; }

if running; then
    [ "$open_browser" = 1 ] && open "$url"
    echo "Timing: $url"
    exit 0
fi
[ -x .venv/bin/python ] || { echo "请先运行 ./setup.sh。" >&2; exit 1; }

mkdir -p .runtime
nohup .venv/bin/python -X utf8 -u -m timing --serve > .runtime/server.log 2> .runtime/server-error.log &
echo $! > .runtime/server.pid
for _ in $(seq 1 30); do
    sleep 0.5
    if running; then
        [ "$open_browser" = 1 ] && open "$url"
        echo "Timing: $url"
        exit 0
    fi
    kill -0 "$(cat .runtime/server.pid)" 2>/dev/null || break
done
echo "启动失败，请查看 .runtime/server-error.log。" >&2
exit 1

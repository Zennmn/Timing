#!/usr/bin/env bash
# Double-click in Finder: runs stop.sh in Terminal.
cd "$(dirname "$0")" && ./stop.sh
read -r -p "按回车关闭窗口…" _

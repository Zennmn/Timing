#!/usr/bin/env bash
# Double-click in Finder: runs start.sh in Terminal.
cd "$(dirname "$0")" && ./start.sh
read -r -p "按回车关闭窗口…" _

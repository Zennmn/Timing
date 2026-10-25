#!/usr/bin/env bash
# Double-click in Finder: runs setup.sh in Terminal.
cd "$(dirname "$0")" && ./setup.sh
read -r -p "按回车关闭窗口…" _

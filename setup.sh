#!/usr/bin/env bash
# macOS (Apple silicon) setup: tools check -> locked Python env -> real MPS check -> models.
#   ./setup.sh                 install everything
#   ./setup.sh --skip-models   skip the model step
#   ./setup.sh --check-only    only check, install nothing
set -euo pipefail
cd "$(dirname "$0")"

skip_models=0
check_only=0
for arg in "$@"; do
    case "$arg" in
        --skip-models) skip_models=1 ;;
        --check-only) check_only=1 ;;
        *) echo "未知参数：$arg" >&2; exit 2 ;;
    esac
done

fail() { echo "安装未完成：$1" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "setup.sh 只用于 macOS；Windows 请运行 setup.ps1。"
[ "$(uname -m)" = "arm64" ] || fail "需要 Apple 芯片（M 系列）的 Mac；Intel Mac 没有 MPS。"

missing=()
for tool in uv ffmpeg git; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "缺少工具：${missing[*]}" >&2
    echo "可用 Homebrew 安装：brew install ${missing[*]}" >&2
    echo "（uv 也可按 https://docs.astral.sh/uv/ 的官方方式安装）" >&2
    exit 1
fi
echo "工具检查通过：uv、FFmpeg、Git。"

if [ "$check_only" = 1 ]; then
    if [ -x .venv/bin/python ]; then
        .venv/bin/python -X utf8 -c "import torch; print('PyTorch', torch.__version__, 'MPS', torch.backends.mps.is_available())"
    else
        echo "尚未创建 Python 环境（.venv）。"
    fi
    exit 0
fi

echo "创建 Python 3.12 环境并安装锁定依赖…"
uv sync --frozen

echo "检查 Apple GPU（MPS）能否真实运算…"
.venv/bin/python -X utf8 - <<'EOF' || fail "MPS 运算检查失败，模型步骤未执行。"
import torch
if not torch.backends.mps.is_available():
    raise SystemExit("PyTorch 报告 MPS 不可用。")
a = torch.randn(128, 128, device="mps")
b = (a @ a).cpu()
if not torch.isfinite(b).all():
    raise SystemExit("MPS 矩阵乘法结果异常。")
print(f"MPS 可用：PyTorch {torch.__version__}")
EOF

if [ "$skip_models" = 1 ]; then
    echo "已跳过模型下载；之后可运行 .venv/bin/python -X utf8 scripts/download_models.py。"
else
    echo "下载并校验五组必需模型…"
    .venv/bin/python -X utf8 scripts/download_models.py
fi
echo "安装完成。运行 ./start.sh 启动。"

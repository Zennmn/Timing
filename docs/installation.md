# 首次安装与排错

## 开始安装

1. 在 64 位 Windows 上准备可用的 NVIDIA 显卡和驱动，将源码解压到有写入权限的目录。
2. 双击根目录的 **安装.cmd**。缺少 PowerShell 7 时，按提示用 WinGet 安装；脚本不调用 Windows PowerShell 5.1。
3. 安装器列出缺少的 uv、FFmpeg、Git，并通过 WinGet 引导安装。工具安装器可能要求确认协议或管理员权限；完成后若仍提示找不到工具，关闭窗口重新运行 安装.cmd。
4. Python 3.12 与项目依赖会自动准备。真实 CUDA 运算检查通过后才下载五组模型；完成后双击 **启动.cmd**。

当前硬件验收为 Windows 11、RTX 5070 Ti 16 GB。没有建立其他显卡的统一最低显存要求。环境检查通过表示基础运行条件可用，不等于所有歌曲都不会耗尽显存。没有 NVIDIA GPU 的电脑不在当前安装流程支持范围内。

首次安装需要访问 Python / PyTorch 下载源、GitHub 和 Hugging Face。模型目录当前约 4.27 GiB，Python、CUDA 运行依赖、下载缓存和输出另计。请留足磁盘空间并保持网络连接。无需手动给系统安装 CUDA Toolkit；安装器使用锁定的 PyTorch 依赖，最终以实际 GPU 运算检查为准。

## macOS（Apple 芯片）

1. 需要 M 系列芯片的 Mac；Intel Mac 没有 MPS。用 Homebrew 安装工具：`brew install uv ffmpeg git`。
2. 在源码目录运行 `./setup.sh`，或在访达中双击 **安装.command**。PyTorch 使用 PyPI 的 macOS 版本（与 CUDA 版同为 2.11.0），不需要 CUDA 下载源。
3. 真实 MPS 运算检查通过后才下载五组模型；完成后运行 `./start.sh` 或双击 **启动.command**。

PyTorch 在 MPS 上缺少的两种运算由 Timing 绕开：TIFA 打分中的 int64 `scatter_reduce` 改在 CPU 上计算（只涉及很小的索引张量），Kim 人声模型对复数掩码的 `scatter_add_` 改为实部、虚部分别累加。服务记录写在 `.runtime/server.log` 与 `.runtime/server-error.log`。

## 没有 WinGet

在 Microsoft Store 安装或更新“应用安装程序”，或自行准备下面的工具。手动安装结束后重新打开安装窗口，让新 PATH 生效。

| 工具 | WinGet 命令 | 官方说明 |
| --- | --- | --- |
| PowerShell 7 | `winget install --id Microsoft.PowerShell --exact --source winget` | [Microsoft 安装说明](https://learn.microsoft.com/powershell/scripting/install/install-powershell-on-windows) |
| uv | `winget install --id astral-sh.uv --exact --source winget` | [Astral 安装说明](https://docs.astral.sh/uv/getting-started/installation/) |
| FFmpeg | `winget install --id Gyan.FFmpeg --exact --source winget` | [Windows 构建](https://www.gyan.dev/ffmpeg/builds/) |
| Git | `winget install --id Git.Git --exact --source winget` | [Git for Windows](https://git-scm.com/downloads/win) |

WinGet 的协议和权限提示仍由用户确认；安装器不自动接受协议，也不修改下载安全校验。WinGet 使用方式见 [Microsoft 文档](https://learn.microsoft.com/windows/package-manager/winget/install)。

## 命令行选项

```powershell
# 仅检查；不会安装依赖或下载模型（会保存检查日志）
pwsh.exe -NoProfile -File setup.ps1 -CheckOnly

# 常规安装；缺工具时列出安装方法
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File setup.ps1

# 与双击入口相同：缺少工具时通过 WinGet 引导安装
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File setup.ps1 -InstallPrerequisites

# 只准备 Python / CUDA 环境；应用仍需要五组模型
pwsh.exe -NoProfile -File setup.ps1 -SkipModels
```

`-CheckOnly` 优先于其他选项：不会调用 WinGet、uv 安装或模型下载。已有 Python 环境时会运行一次本地 CUDA 运算检查；环境未创建时只报告基础条件，正式安装后再验证 CUDA。

## 出错后怎么处理

| 提示 / 阶段 | 处理方式 |
| --- | --- |
| 找不到 NVIDIA 工具、驱动无法识别显卡 | 安装适合显卡的 NVIDIA 驱动，重启后重试；模型尚未下载。 |
| 工具已安装但仍提示缺少 | 关闭安装窗口后重新双击 安装.cmd；也可检查工具是否已进入 PATH。 |
| Python 依赖安装失败 | 会自动尝试最多 3 次。检查下载源访问、磁盘空间及日志；重试会复用 uv 缓存。 |
| CUDA 检查失败 | 查看中文原因与 `.runtime/install/gpu.json`；先处理驱动、PyTorch 或运行库问题，再安装模型。 |
| 模型下载中断 | 自动重试后仍失败时重新运行安装。已完成的文件会复用；HTTP 续传是否可用取决于服务器支持。 |
| 文件哈希不符 | 不跳过校验。按错误中的文件名处理或查看模型日志，避免使用损坏文件。 |
| Timing 正在运行 / 另一个安装或启动窗口运行中 | 等任务结束并双击 停止.cmd；等待当前安装或启动结束后再操作。 |

`.runtime/install/` 保存本次安装阶段、GPU 检查结果、安装日志及模型安装记录。日志只保存在本机。重试不会重置歌曲、结果或删除已有模型。安装成功的最终提示只在所有要求的步骤都通过后显示；`-SkipModels` 会明确说明跳过了模型。

已用隔离的工具、GPU 和网络故障样例检查安装分支；没有将本机验证等同于另一台干净电脑的完整首次安装验收。

# Timing

本地中文（普通话）、日语、英语与混唱歌词对齐工具：音频 + 歌词 → 歌声对齐 → AMLL TTML 初稿。

提供本机网页和命令行两种用法；歌词为必填输入；对齐结果直接导出 TTML。没有播放器、波形或时间轴编辑器，初稿需要试听校对。

## 安装与启动

支持并验证的环境：Windows 11、PowerShell 7、NVIDIA CUDA GPU、Python 3.12。当前 CUDA 12.8 依赖配置在 RTX 5070 Ti 16 GB 上验证；其他硬件的显存需求未建立统一下限。macOS 的 Apple 芯片（MPS）另有安装脚本，见下文“macOS（Apple 芯片）”。

下载并解压源码到可写文件夹，然后双击 **安装.cmd**。它会检查 PowerShell 7；缺失时可通过 WinGet 安装，再引导安装 uv、FFmpeg 和 Git。安装器保留各工具的协议和权限提示，按提示完成后继续。系统没有 WinGet 时会给出官方安装地址。NVIDIA 驱动需要提前安装；当前版本不支持 CPU 或 AMD 的安装流程。

安装顺序是：检查系统与 NVIDIA 驱动 → 检查工具 → 创建 Python 3.12 环境并安装锁定依赖 → 验证真实 CUDA 运算 → 下载和校验五组模型。CUDA 未通过时不会下载模型。Python 和 PyTorch 首次下载较大；五组模型当前本机约 4.27 GiB，Python、缓存、临时文件和歌曲输出还需额外空间。

已自行安装 [uv](https://docs.astral.sh/uv/)、[FFmpeg](https://ffmpeg.org/download.html)、Git 和 PowerShell 7 的用户也可以运行：

```powershell
pwsh.exe -NoProfile -ExecutionPolicy Bypass -File setup.ps1
```

setup.ps1 默认安装全部五组必需模型（MMS、TIFA、SOFA、BS 主唱提取、Kim 人声分离）；重复运行可以补装或校验，安装器不删除现存文件。安装时需要网络；下载完成后的本地对齐不自动下载权重。只想跳过模型步骤时加 -SkipModels。

只检查环境、不安装依赖或下载模型：`pwsh.exe -NoProfile -File setup.ps1 -CheckOnly`。安装失败时会显示当前阶段和日志位置；安装记录保存在 `.runtime/install/`。重试时复用已完成的文件和下载缓存；服务器支持分段请求时可续传未完成的 HTTP 文件，不支持时会重新下载该文件。请先关闭正在运行的 Timing 再安装，防止修改运行中的 Python 环境。排错和手动依赖安装见 [安装说明](docs/installation.md)。

双击 启动.cmd，或运行 pwsh.exe -File start.ps1，打开 [本机页面](http://127.0.0.1:7861)。双击 停止.cmd 关闭服务。

### macOS（Apple 芯片）

在 M4 Max（64 GB 统一内存）、macOS 27 上验证。先用 Homebrew 安装工具：`brew install uv ffmpeg git`，然后在源码目录运行 `./setup.sh`，或在访达中双击 **安装.command**。顺序与 Windows 相同：检查工具 → 创建 Python 3.12 环境 → 验证真实 MPS 运算 → 下载和校验五组模型；MPS 未通过时不会下载模型。`./setup.sh --skip-models` 跳过模型步骤，`./setup.sh --check-only` 只检查。`./start.sh`（或双击 **启动.command**）打开本机页面，`./stop.sh`（或 **停止.command**）关闭服务。

模型默认在 Apple GPU 上运行；`TIMING_DEVICE=cpu` 可强制使用 CPU（很慢）。在 Apple GPU 上，“只保留主唱”的 BS-Roformer 较慢（一首约 4 分钟的歌需 6–8 分钟），“保留全部人声”的 Kim 模型连同打轴约 3 分钟，“直接使用所选音轨”约 1.5–2 分钟。

## 使用

1. 选择歌曲，填写正确歌词，或导入 UTF-8 TXT / LRC / JSON / 逐行 TTML。TTML 读取每句的起止时间及歌词，网页转成可编辑的带时间 JSON；也可直接粘贴 TTML。歌词为必填，提交以文本框内容为准。
2. 使用本地方案时选择歌词语言：默认日英混唱；普通话选“中文（普通话）”，中英混唱选“中英混唱”。JSON 行上的 language 优先。使用“LLM 整曲识别”时，由 LLM 听音判断每段语言，不需要手选。
3. 选择音频预处理：
   - 只保留主唱（默认）：用 BS-Roformer 从原曲或导入轨去掉伴奏，并尽量去掉和声；
   - 保留全部人声：用 Kim 去掉伴奏，保留主唱和和声；已导入人声时直接使用导入轨；
   - 直接使用所选音轨：不改动原曲或导入轨。
4. 选择对齐方式：auto 按歌词格式自动选择；完整逐句时间的 LRC / TTML / JSON 用 TIFA，纯文本用 SOFA；也可选 MMS（ctc）、LLM 整曲识别（llm_direct）或网页 LLM 导入（llm_import）。这些方案都支持普通话歌词。
5. 生成并下载 lyrics.ttml 和 alignment.json。背景歌词（role=bg）在“只保留主唱”模式下使用 Kim 完整人声或导入的完整人声，不会退回原曲；报告会注明背景轨的来源。

页面会按歌词格式和所选方式显示适用选项，并检查五组必需模型是否齐全。LLM 功能默认关闭；整曲识别会一次发送完整音轨和全部歌词，由 LLM 判断中日英语言、读音和拆分，再由本地 MMS + TIFA 计时。仍需逐句 LRC / TTML / JSON；不自动听写歌词。缺漏或无效的行会标为待补轴，不自动追加请求。其他 LLM 辅助选项按各自说明发送文字或音频片段。API Key 不写入浏览器存储或任务报告。

只有网页入口的模型可以用“网页 LLM（导入结果）”：生成并复制提示词，你把完整歌曲和提示词发给网页模型，再把 JSON 结果粘贴或导入回来。此模式无需 API Key，应用不会调用 LLM；返回的语言、读音和拆分取代本地词典与分词选择，字时间继续由本地声学模型计算。

两种 LLM 入口都可选择日语“细拆（默认）”或“适度合并（旧版）”提示词。两版都要求英语单词之间分开，词内按演唱需要拆音节，禁止跨词合并；日语分别保持细拆和适度合并规则。网页模式更新或切换提示词后需重新生成并让模型返回新结果，已导入结果按实际分段处理。

命令行示例（路径替换成自己的文件）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m timing "song.flac" --lyrics "song.lrc" --output "song.ttml"
```

中英混唱歌曲在命令末尾加 `--language zh-en`，普通话歌曲用 `--language zh`（默认 `mixed` 覆盖日语、英语与日英混唱）。歌词格式、重叠 / 伴唱、读音覆盖、LLM 与完整 CLI 参数见 [使用说明](docs/usage.md)；模型来源与体积见 [模型清单](docs/models.md)。

## 输出与限制

- 网页任务保存在 data/<任务编号>/，命令行中间记录保存在输出旁的 <文件名>.timing/。
- 时间来自模型预测，10 ms / 20 ms 的网格不代表对应精度。持续音、混唱、分离瑕疵和特殊读音仍需人工检查。
 - 中文按普通话处理，音素来自 TIFA 原生中文词典、MMS 罗马化音素与 SOFA 中文词典；LLM 听音读音返回拼音。不覆盖粤语，混唱的歌可能仍有语言判定错误。
- 个别字对齐失败时保留其他有效单位，失败字在 TTML 中使用零时长编辑占位，JSON 时间保持 null；整句失败则保留输入 LRC 范围并标记待补轴。这类初稿需要补轴后再用于严格校验 / 提交。
- 导入的音轨必须与原曲同起点、同时长。分离失败会报错，不会自动换成原曲；背景歌词没有原曲回退。

## 许可

Timing 自有源码采用 MIT 许可。模型权重、下载的运行库和图标保留各自许可，MIT 不覆盖它们：MMS 模型卡标注非商业许可；BS 主唱权重与 Kim 人声权重许可未明确确认。详见 [第三方来源与许可](THIRD_PARTY.md)。仓库不包含权重、歌曲音频、任务输出或 API Key。

源码目录、测试命令和维护边界见 [架构说明](docs/architecture.md) 与 [发布整理说明](docs/publishing.md)。

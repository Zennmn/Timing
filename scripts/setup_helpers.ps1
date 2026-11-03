# Setup functions are separate so the installation flow can be tested offline.
# This file does nothing when dot-sourced; setup.ps1 is the entry point.

function Get-TimingPrerequisites {
    @(
        [pscustomobject]@{ Command = 'uv'; Name = 'uv'; Package = 'astral-sh.uv'; Url = 'https://docs.astral.sh/uv/getting-started/installation/' }
        [pscustomobject]@{ Command = 'ffmpeg'; Name = 'FFmpeg'; Package = 'Gyan.FFmpeg'; Url = 'https://www.gyan.dev/ffmpeg/builds/' }
        [pscustomobject]@{ Command = 'git'; Name = 'Git'; Package = 'Git.Git'; Url = 'https://git-scm.com/downloads/win' }
    )
}

function Update-TimingProcessPath {
    # Refresh this process only after an installer changes the user's PATH.
    $paths = @($env:PATH, [Environment]::GetEnvironmentVariable('PATH', 'Machine'), [Environment]::GetEnvironmentVariable('PATH', 'User'))
    if ($env:LOCALAPPDATA) { $paths += Join-Path $env:LOCALAPPDATA 'Microsoft/WinGet/Links' }
    $env:PATH = (($paths -join ';').Split(';') | Where-Object { $_ } | Select-Object -Unique) -join ';'
}

function Invoke-TimingCommand {
    param([string]$Executable, [string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    & $Executable @Arguments 2>&1 | ForEach-Object { Write-Host $_.ToString() }
    return $LASTEXITCODE
}

function Test-TimingPlatform {
    if (-not $IsWindows -or -not [Environment]::Is64BitProcess -or [Runtime.InteropServices.RuntimeInformation]::OSArchitecture -ne 'X64') {
        throw '当前安装器支持 64 位 Windows + NVIDIA 显卡。其他系统尚未完成安装验证。'
    }
}

function Test-TimingDriver {
    $command = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    $executable = if ($command) { $command.Source } else { $null }
    if (-not $executable) {
        foreach ($base in @($env:WINDIR, $env:ProgramFiles)) {
            if (-not $base) { continue }
            $relative = if ($base -eq $env:WINDIR) { 'System32/nvidia-smi.exe' } else { 'NVIDIA Corporation/NVSMI/nvidia-smi.exe' }
            $candidate = Join-Path $base $relative
            if (Test-Path -LiteralPath $candidate) { $executable = $candidate; break }
        }
    }
    if (-not $executable) { throw '没有找到 NVIDIA 驱动检查工具。请先安装适合显卡的 NVIDIA 驱动并重启：https://www.nvidia.com/Download/index.aspx' }
    $PSNativeCommandUseErrorActionPreference = $false
    $description = & $executable --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -ne 0 -or -not ($description | Out-String).Trim()) {
        throw 'NVIDIA 驱动无法正常识别显卡。请修复驱动或重启后重试；尚未下载 Python 依赖和模型。'
    }
    Write-Host ('显卡：' + ($description -join '；'))
}

function Test-TimingServiceRunning {
    param([string]$Root)
    try { if ((Invoke-RestMethod 'http://127.0.0.1:7861/api/status' -TimeoutSec 2).app -eq 'Timing') { return $true } }
    catch { }
    # A busy/unresponsive service can still be using this Python environment.
    $candidates = @()
    if ($Root) {
        $pidFile = Join-Path $Root '.runtime/server.pid'
        if (Test-Path -LiteralPath $pidFile) {
            $recordedPid = 0
            try { if ([int]::TryParse((Get-Content -LiteralPath $pidFile -Raw).Trim(), [ref]$recordedPid)) { $candidates += $recordedPid } }
            catch { }
        }
    }
    try { $candidates += @((Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue).OwningProcess) }
    catch { }
    foreach ($candidate in ($candidates | Where-Object { $_ -gt 0 } | Select-Object -Unique)) {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$candidate" -ErrorAction SilentlyContinue
        if ($process -and $process.Name -like 'python*' -and $process.CommandLine -match '-m\s+timing\s+--serve(?:\s|$)') { return $true }
    }
    return $false
}

function Set-TimingInstallState {
    param([string]$Folder, [string]$Stage, [string]$Status, [string]$Message)
    @{ stage = $Stage; status = $Status; message = $Message; updated = (Get-Date -Format 'o') } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Folder 'status.json') -Encoding utf8
}

function Show-TimingPrerequisiteHelp {
    param([object[]]$Missing)
    foreach ($item in $Missing) {
        Write-Host ('缺少 ' + $item.Name + '：winget install --id ' + $item.Package + ' --exact --source winget') -ForegroundColor Yellow
        Write-Host ('手动安装：' + $item.Url)
    }
}

function Invoke-TimingSetup {
    param([string]$Root, [switch]$SkipModels, [switch]$CheckOnly, [switch]$InstallPrerequisites)
    $ErrorActionPreference = 'Stop'
    $folder = Join-Path $Root '.runtime/install'
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    $log = Join-Path $folder ('setup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N').Substring(0, 6) + '.log')
    $transcribing = $false
    $installLock = $null
    $stage = 'preflight'
    try {
        try { $installLock = [IO.File]::Open((Join-Path $folder 'setup.lock'), [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None) }
        catch { Write-Host '另一个安装窗口正在运行，请等它结束后再重试。' -ForegroundColor Yellow; return 1 }
        Start-Transcript -LiteralPath $log -Force | Out-Null
        $transcribing = $true
        Set-Location -LiteralPath $Root
        Write-Host 'Timing 首次安装 / 修复' -ForegroundColor Cyan
        Write-Host ('日志：' + $log)
        Set-TimingInstallState $folder $stage 'running' '正在检查系统和显卡。'
        Test-TimingPlatform
        if (-not $CheckOnly -and (Test-TimingServiceRunning -Root $Root)) { throw 'Timing 服务正在运行。请先双击 停止.cmd，再运行安装；现有任务和模型不会删除。' }
        Write-Host '[1/5] 检查 NVIDIA 驱动（此步不下载文件）'
        Test-TimingDriver

        $stage = 'prerequisites'
        Set-TimingInstallState $folder $stage 'running' '正在检查 uv、FFmpeg 和 Git。'
        $required = @(Get-TimingPrerequisites)
        $missing = @($required | Where-Object { -not (Get-Command $_.Command -ErrorAction SilentlyContinue) })
        if ($missing.Count) {
            Show-TimingPrerequisiteHelp $missing
            if (-not $InstallPrerequisites -or $CheckOnly) { throw '请安装上面列出的工具，或双击 安装.cmd 进入依赖安装引导。检查模式不会安装或下载文件。' }
            if (-not (Get-Command winget -ErrorAction SilentlyContinue)) { throw '找不到 WinGet。请在 Microsoft Store 安装或更新“应用安装程序”，或按上面的地址手动安装工具后重试。' }
            foreach ($item in $missing) {
                Write-Host ('正在通过 WinGet 安装 ' + $item.Name + '；协议和权限提示请在安装器中自行确认。')
                $code = Invoke-TimingCommand 'winget' @('install', '--id', $item.Package, '--exact', '--source', 'winget')
                Update-TimingProcessPath
                if ($code -ne 0 -or -not (Get-Command $item.Command -ErrorAction SilentlyContinue)) {
                    throw ($item.Name + ' 尚未就绪。如刚安装成功，请关闭当前窗口后重新双击 安装.cmd；不会重复下载已完成的模型。')
                }
            }
        }
        Write-Host '[2/5] uv、FFmpeg、Git 已就绪'
        $python = Join-Path $Root '.venv/Scripts/python.exe'
        if ($CheckOnly) {
            if (Test-Path -LiteralPath $python) {
                $stage = 'gpu'
                $code = Invoke-TimingCommand $python @('-X', 'utf8', 'scripts/verify_gpu.py', '--output', (Join-Path $folder 'gpu.json'))
                if ($code -ne 0) { throw 'CUDA 运算检查未通过，请查看上面的说明与 gpu.json。检查模式没有修改 Python 环境或模型。' }
                $stage = 'dictionary'
                $code = Invoke-TimingCommand $python @('-X', 'utf8', 'scripts/download_unidic.py', '--check-only')
                if ($code -ne 0) { throw '日语 UniDic 2025.12 词典未准备完整，请运行正式安装。检查模式未下载文件。' }
            } else { Write-Host '基础条件通过；尚未创建 Python 环境，正式安装时会继续验证 CUDA。' }
            Set-TimingInstallState $folder 'check' 'complete' '环境检查完成；未安装依赖或下载模型。'
            Write-Host '检查完成。正式安装请双击 安装.cmd。' -ForegroundColor Green
            return 0
        }

        $stage = 'environment'
        Set-TimingInstallState $folder $stage 'running' '正在安装锁定的 Python 3.12 与项目依赖。'
        Write-Host '[3/5] 安装 Python 3.12 和项目依赖（首次较大，请保持网络连接）'
        $code = 1
        for ($attempt = 1; $attempt -le 3; $attempt++) {
            $code = Invoke-TimingCommand 'uv' @('sync', '--frozen', '--no-dev', '--python', '3.12')
            if ($code -eq 0) { break }
            if ($attempt -lt 3) { Write-Host ('依赖安装失败，准备第 ' + ($attempt + 1) + '/3 次尝试；复用下载缓存。') -ForegroundColor Yellow; Start-Sleep -Seconds $attempt }
        }
        if ($code -ne 0 -or -not (Test-Path -LiteralPath $python)) { throw 'Python 依赖安装未完成。请检查网络、磁盘空间及日志后重试；尚未下载模型。' }

        $stage = 'gpu'
        Set-TimingInstallState $folder $stage 'running' '正在验证真实 CUDA 运算，尚未下载模型。'
        Write-Host '[4/5] 检查实际 CUDA 运算'
        $code = Invoke-TimingCommand $python @('-X', 'utf8', 'scripts/verify_gpu.py', '--output', (Join-Path $folder 'gpu.json'))
        if ($code -ne 0) { throw 'CUDA 运算检查失败，已在下载模型前停止。请按上面的原因修复驱动 / Python 环境后重试。' }

        $stage = 'dictionary'
        Set-TimingInstallState $folder $stage 'running' '正在下载 / 校验日语 UniDic 2025.12 词典。'
        Write-Host '下载 / 校验 UniDic 2025.12（日语词库约 664 MB 下载，解压后约 1.1 GiB）'
        $code = Invoke-TimingCommand $python @('-X', 'utf8', 'scripts/download_unidic.py')
        if ($code -ne 0) { throw '日语词典安装未完成。已有模型和歌曲保留，请检查网络与日志后重试。' }

        $stage = 'models'
        if ($SkipModels) { Write-Host '[5/5] 按参数跳过模型安装；应用仍需要四组模型才能完整使用。' }
        else {
            Set-TimingInstallState $folder $stage 'running' '正在下载和校验四组模型。'
            Write-Host '[5/5] 下载 / 校验四组模型（已有缓存会直接复用，磁盘占用以模型清单为准）'
            $code = Invoke-TimingCommand $python @('-X', 'utf8', 'scripts/download_models.py')
            if ($code -ne 0) { throw '模型安装未完成。已完成的文件保留，请查看模型日志后重新双击 安装.cmd 继续。' }
        }
        $message = if ($SkipModels) { 'Python 和 CUDA 检查完成；已跳过模型安装。' } else { '配置完成，双击 启动.cmd。' }
        Set-TimingInstallState $folder 'complete' 'complete' $message
        Write-Host $message -ForegroundColor Green
        return 0
    } catch {
        Set-TimingInstallState $folder $stage 'failed' $_.Exception.Message
        Write-Host ('未完成：' + $_.Exception.Message) -ForegroundColor Red
        Write-Host ('日志：' + $log)
        return 1
    } finally {
        if ($transcribing) { Stop-Transcript | Out-Null }
        if ($installLock) { $installLock.Dispose() }
    }
}

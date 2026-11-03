"""Offline behavior tests for the Windows setup orchestration flow.

Each case generates a small PowerShell runner below pytest's temporary
directory. The runner dot-sources the real setup helpers, replaces the OS,
driver, service, and command boundaries with fixtures, and calls the public
Invoke-TimingSetup entry point against a disposable root.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PWSH = shutil.which("pwsh") or shutil.which("pwsh.exe")

pytestmark = pytest.mark.skipif(os.name != "nt", reason="setup.ps1 is Windows-only")


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@dataclass(frozen=True)
class SetupResult:
    returncode: int
    setup_exit: int
    status: dict[str, Any] | None
    calls: tuple[str, ...]
    output: str
    log_text: str
    driver_called: bool
    service_called: bool
    root: Path


def _driver_function(mode: str, marker: Path) -> str:
    if mode not in {"ok", "missing"}:
        raise ValueError(mode)
    body = "    Set-Content -LiteralPath %s -Value 'called'" % _quote(marker)
    if mode == "missing":
        body += "\n    throw 'DRIVER_MISSING_FIXTURE'"
    return "function Test-TimingDriver {\n%s\n}" % body


def _service_function(running: bool, marker: Path) -> str:
    return (
        "function Test-TimingServiceRunning {\n"
        "    Set-Content -LiteralPath %s -Value 'called'\n"
        "    return $%s\n"
        "}" % (_quote(marker), str(running).lower())
    )


def _prerequisite_function(prerequisites: tuple[dict[str, str], ...]) -> str:
    if not prerequisites:
        return "function Get-TimingPrerequisites { return @() }"
    rows = "\n".join(
        "        [pscustomobject]@{ Command = %s; Name = %s; Package = %s; Url = %s }"
        % (
            _quote(item["Command"]),
            _quote(item["Name"]),
            _quote(item["Package"]),
            _quote(item["Url"]),
        )
        for item in prerequisites
    )
    return "function Get-TimingPrerequisites {\n    return @(\n%s\n    )\n}" % rows


def _command_function(record: Path, codes: dict[str, int]) -> str:
    # Behavior labels stay stable when setup adds arguments such as --python 3.12.
    entries = "\n".join("    %s = %d" % (_quote(key), value) for key, value in codes.items())
    return f"""\
$codes = @{{
{entries}
}}
function Invoke-TimingCommand {{
    param([string]$Executable, [string[]]$Arguments)
    $leaf = [IO.Path]::GetFileName($Executable)
    $scriptArgument = @($Arguments | Where-Object {{ $_ -like '*.py' }} | Select-Object -First 1)
    if ($leaf -in @('uv', 'uv.exe')) {{
        $label = 'uv ' + [string]$Arguments[0]
    }} elseif ($scriptArgument) {{
        $label = [IO.Path]::GetFileName([string]$scriptArgument)
    }} else {{
        $label = $leaf + ' ' + ($Arguments -join ' ')
    }}
    Add-Content -LiteralPath {_quote(record)} -Value $label -Encoding utf8
    Write-Host ('MOCK ' + $label)
    if ($codes.ContainsKey($label)) {{ return [int]$codes[$label] }}
    return 0
}}
"""


def _run_setup(
    tmp_path: Path,
    *,
    skip_models: bool = False,
    check_only: bool = False,
    install_prerequisites: bool = False,
    driver: str = "ok",
    service_running: bool = False,
    prerequisites: tuple[dict[str, str], ...] = (),
    command_codes: dict[str, int] | None = None,
    create_python: bool = True,
    hold_lock: bool = False,
) -> SetupResult:
    if PWSH is None:
        pytest.fail("PowerShell 7 (pwsh.exe) is required to run setup tests")

    root = tmp_path / "root"
    root.mkdir()
    record = tmp_path / "calls.txt"
    driver_marker = tmp_path / "driver-called.txt"
    service_marker = tmp_path / "service-called.txt"
    result_file = tmp_path / "setup-exit.txt"
    script = tmp_path / "run-setup.ps1"

    if create_python:
        python = root / ".venv" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"")

    setup_call = (
        "Invoke-TimingSetup -Root $root "
        f"-SkipModels:${str(skip_models).lower()} "
        f"-CheckOnly:${str(check_only).lower()} "
        f"-InstallPrerequisites:${str(install_prerequisites).lower()}"
    )
    if hold_lock:
        setup_block = f"""\
$lockPath = Join-Path $root '.runtime/install/setup.lock'
New-Item -ItemType Directory -Path (Split-Path -Parent $lockPath) -Force | Out-Null
$heldLock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {{
    $code = {setup_call}
}} finally {{
    $heldLock.Dispose()
}}
"""
    else:
        setup_block = f"$code = {setup_call}\n"

    script_text = f"""\
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -lt 7) {{ throw 'PowerShell 7 is required' }}
$repo = {_quote(REPO_ROOT)}
$root = {_quote(root)}
. (Join-Path $repo 'scripts/setup_helpers.ps1')

function Start-Sleep {{ param([int]$Seconds) }}
function Test-TimingPlatform {{ }}
{_driver_function(driver, driver_marker)}
{_service_function(service_running, service_marker)}
{_prerequisite_function(prerequisites)}
{_command_function(record, command_codes or {})}
{setup_block}
[IO.File]::WriteAllText({_quote(result_file)}, [string]$code)
exit $code
"""
    script.write_text(script_text, encoding="utf-8")

    try:
        completed = subprocess.run(
            [PWSH, "-NoProfile", "-File", str(script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("setup flow did not terminate")

    if not result_file.exists():
        pytest.fail(
            "setup runner did not record an exit code\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    setup_exit = int(result_file.read_text(encoding="utf-8").strip())
    status_path = root / ".runtime" / "install" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8-sig")) if status_path.exists() else None
    calls = (
        tuple(line.strip() for line in record.read_text(encoding="utf-8-sig").splitlines() if line.strip())
        if record.exists()
        else ()
    )
    log_text = "\n".join(
        path.read_text(encoding="utf-8-sig", errors="replace")
        for path in sorted((root / ".runtime" / "install").glob("setup-*.log"))
    )
    return SetupResult(
        returncode=completed.returncode,
        setup_exit=setup_exit,
        status=status,
        calls=calls,
        output=completed.stdout + completed.stderr,
        log_text=log_text,
        driver_called=driver_marker.exists(),
        service_called=service_marker.exists(),
        root=root,
    )


def test_missing_driver_stops_before_dependency_or_model_work(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, driver="missing", create_python=False)

    assert result.returncode == 1
    assert result.setup_exit == 1
    assert result.driver_called
    assert result.calls == ()
    assert not (result.root / ".venv").exists()
    assert result.status is not None
    assert result.status["stage"] == "preflight"
    assert result.status["status"] == "failed"
    assert "DRIVER_MISSING_FIXTURE" in result.status["message"]


def test_check_only_missing_tools_prints_help_without_installing(tmp_path: Path) -> None:
    missing = {
        "Command": "timing-missing-tool-c0ffee",
        "Name": "FixtureMissingTool",
        "Package": "Fixture.Missing.Tool",
        "Url": "https://example.invalid/fixture-missing-tool",
    }
    result = _run_setup(
        tmp_path,
        check_only=True,
        prerequisites=(missing,),
        create_python=False,
    )

    assert result.returncode == 1
    assert result.setup_exit == 1
    assert result.driver_called
    assert result.calls == ()
    assert result.status is not None
    assert result.status["stage"] == "prerequisites"
    assert result.status["status"] == "failed"
    for text in (missing["Name"], missing["Package"], missing["Url"]):
        assert text in result.output
    assert "MOCK winget" not in result.output
    assert "MOCK uv sync" not in result.output


def test_failed_gpu_probe_after_sync_prevents_models(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, command_codes={"verify_gpu.py": 7})

    assert result.returncode == 1
    assert result.setup_exit == 1
    assert result.calls == ("uv sync", "verify_gpu.py")
    assert "download_models.py" not in result.calls
    assert result.status is not None
    assert result.status["stage"] == "gpu"
    assert result.status["status"] == "failed"


def test_success_runs_gpu_probe_before_models(tmp_path: Path) -> None:
    result = _run_setup(tmp_path)

    assert result.returncode == 0
    assert result.setup_exit == 0
    assert result.calls == ("uv sync", "verify_gpu.py", "download_unidic.py", "download_models.py")
    assert "四组模型" in result.output
    assert "五组模型" not in result.output
    assert result.status is not None
    assert result.status["stage"] == "complete"
    assert result.status["status"] == "complete"


def test_skip_models_still_syncs_and_verifies_gpu(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, skip_models=True)

    assert result.returncode == 0
    assert result.setup_exit == 0
    assert result.calls == ("uv sync", "verify_gpu.py", "download_unidic.py")
    assert "四组模型" in result.output
    assert "五组模型" not in result.output
    assert result.status is not None
    assert result.status["stage"] == "complete"
    assert result.status["status"] == "complete"


def test_failed_dictionary_install_preserves_models_and_stops_setup(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, command_codes={"download_unidic.py": 5})
    assert result.returncode == result.setup_exit == 1
    assert result.calls == ("uv sync", "verify_gpu.py", "download_unidic.py")
    assert result.status["stage"] == "dictionary" and result.status["status"] == "failed"


def test_dependency_failure_retries_three_times_then_stops_and_logs(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, command_codes={"uv sync": 9})

    assert result.returncode == 1
    assert result.setup_exit == 1
    assert result.calls == ("uv sync",) * 3
    assert "verify_gpu.py" not in result.calls
    assert "download_models.py" not in result.calls
    assert result.status is not None
    assert result.status["stage"] == "environment"
    assert result.status["status"] == "failed"
    log_files = list((result.root / ".runtime" / "install").glob("setup-*.log"))
    assert log_files
    assert result.log_text.count("MOCK uv sync") == 3


def test_running_service_blocks_actual_install(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, service_running=True)

    assert result.returncode == 1
    assert result.setup_exit == 1
    assert result.calls == ()
    assert not result.driver_called
    assert result.service_called
    assert result.status is not None
    assert result.status["stage"] == "preflight"
    assert result.status["status"] == "failed"


def test_concurrent_install_lock_is_rejected_before_state_writes(tmp_path: Path) -> None:
    result = _run_setup(tmp_path, hold_lock=True)

    assert result.returncode == 1
    assert result.setup_exit == 1
    assert result.calls == ()
    assert not result.driver_called
    assert result.status is None


@pytest.mark.parametrize("command_line, expected", [
    ("python.exe -X utf8 -m timing --serve", True),
    ("python.exe unrelated.py", False),
])
def test_unresponsive_service_is_identified_from_pid_without_trusting_other_python(
    tmp_path: Path, command_line: str, expected: bool,
) -> None:
    if PWSH is None:
        pytest.fail("PowerShell 7 (pwsh.exe) is required to run setup tests")
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "server.pid").write_text("31337", encoding="utf-8")
    runner = tmp_path / "service-probe.ps1"
    runner.write_text(f"""\
$ErrorActionPreference = 'Stop'
. {_quote(REPO_ROOT / 'scripts/setup_helpers.ps1')}
function Invoke-RestMethod {{ throw 'fixture HTTP timeout' }}
function Get-NetTCPConnection {{ return @() }}
function Get-CimInstance {{
    param($ClassName, $Filter)
    if ($Filter -ne 'ProcessId=31337') {{ throw 'unexpected process query' }}
    return [pscustomobject]@{{ Name = 'python.exe'; CommandLine = {_quote(command_line)} }}
}}
$actual = Test-TimingServiceRunning -Root {_quote(tmp_path)}
if ($actual -ne ${str(expected).lower()}) {{ throw 'incorrect service result' }}
""", encoding="utf-8")
    completed = subprocess.run([PWSH, "-NoProfile", "-File", str(runner)], cwd=tmp_path,
                               capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_startup_honors_installation_lock_before_using_python(tmp_path: Path) -> None:
    if PWSH is None:
        pytest.fail("PowerShell 7 (pwsh.exe) is required to run setup tests")
    shutil.copyfile(REPO_ROOT / "start.ps1", tmp_path / "start.ps1")
    folder = tmp_path / ".runtime/install"
    folder.mkdir(parents=True)
    runner = tmp_path / "locked-start.ps1"
    runner.write_text(f"""\
$ErrorActionPreference = 'Stop'
$held = [IO.File]::Open({_quote(folder / 'setup.lock')}, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
function Start-Process {{ throw 'unexpected process launch' }}
function Invoke-RestMethod {{ throw 'unexpected service query' }}
try {{
    $blocked = $false
    try {{ & {_quote(tmp_path / 'start.ps1')} -NoBrowser }}
    catch {{
        if ($_.Exception.Message -like '安装或启动正在进行中*') {{ $blocked = $true }}
        else {{ throw }}
    }}
    if (-not $blocked) {{ throw 'startup ignored the install lock' }}
}} finally {{ $held.Dispose() }}
""", encoding="utf-8")
    completed = subprocess.run([PWSH, "-NoProfile", "-File", str(runner)], cwd=tmp_path,
                               capture_output=True, text=True, encoding="utf-8", timeout=20)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (tmp_path / ".venv").exists()

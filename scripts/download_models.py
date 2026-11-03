"""Explicit installation of all four required model groups; inference never invokes this script."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from timing.config import MODELS as MODELS_DIR, initialize
from timing.model_catalog import MODELS, inventory
from download_support import atomic_write_json, error_tail, friendly_failure


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rerun(spec):
    return f".\\.venv\\Scripts\\python.exe -X utf8 .\\scripts\\{spec.installer}"


def install(attempts=3, *, log_dir=None, status_path=None, sleeper=time.sleep):
    """Install all groups with a bounded number of attempts per group.

    A failed group is never reported as complete.  Other groups still get a
    chance to finish so a first install does not have to start over for one
    transient network failure.
    """
    initialize()
    attempts = max(1, int(attempts))
    log_dir = Path(log_dir) if log_dir is not None else None
    status_path = Path(status_path) if status_path is not None else None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "source": "scripts/download_models.py",
        "updated_at": _now(),
        "complete": False,
        "resume": "HTTP 安装器在服务器支持 Range 时保留 .part；Hugging Face 使用官方续传缓存。",
        "groups": {name: {"label": spec.label, "status": "pending", "attempts": 0}
                   for name, spec in MODELS.items()},
    }
    if status_path is not None:
        atomic_write_json(status_path, status)

    failures = []
    for index, (name, spec) in enumerate(MODELS.items(), 1):
        print(f"[{index}/{len(MODELS)}] 开始安装 {name}：{spec.label}", flush=True)
        state = status["groups"][name]
        state["status"] = "running"
        log_path = log_dir / f"{name}.log" if log_dir is not None else None
        if log_path is not None:
            state["log"] = str(log_path.relative_to(ROOT) if log_path.is_relative_to(ROOT) else log_path)
        if status_path is not None:
            status["updated_at"] = _now()
            atomic_write_json(status_path, status)

        last_error = None
        succeeded = False
        for attempt in range(1, attempts + 1):
            state["attempts"] = attempt
            if status_path is not None:
                status["updated_at"] = _now()
                atomic_write_json(status_path, status)
            stream = log_path.open("a", encoding="utf-8") if log_path is not None else None
            try:
                if stream is not None:
                    stream.write(f"\n[{_now()}] {name} 第 {attempt}/{attempts} 次\n")
                    stream.flush()
                env = os.environ.copy()
                env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1")
                if log_path is not None:
                    env["TIMING_DOWNLOAD_LOG"] = str(log_path.resolve())
                subprocess.run(
                    [sys.executable, "-X", "utf8", str(ROOT / "scripts" / spec.installer)],
                    cwd=ROOT,
                    check=True,
                    stderr=stream if stream is not None else None,
                    env=env,
                )
                succeeded = True
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if attempt < attempts:
                    print(f"{name}：第 {attempt}/{attempts} 次未完成，将重试。", flush=True)
                    sleeper(min(2 ** (attempt - 1), 4))
            except OSError as exc:
                last_error = exc
                break
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    print(f"{name}：安装器返回异常，第 {attempt}/{attempts} 次失败，将重试。", flush=True)
                    sleeper(min(2 ** (attempt - 1), 4))
            finally:
                if stream is not None:
                    stream.close()

        if succeeded:
            state["status"] = "complete"
            print(f"[{index}/{len(MODELS)}] {name} 完成。", flush=True)
        else:
            state["status"] = "failed"
            state["error"] = error_tail(str(last_error or ""), "安装器未完成。")[:500]
            failures.append(name)
            details = f"\n详细日志：{state.get('log')}" if state.get("log") else ""
            print(friendly_failure(spec.label, _rerun(spec),
                                   last_error or RuntimeError("安装器未完成")) + details, flush=True)
        status["complete"] = False
        status["updated_at"] = _now()
        if status_path is not None:
            atomic_write_json(status_path, status)

    status["complete"] = not failures
    status["updated_at"] = _now()
    if status_path is not None:
        atomic_write_json(status_path, status)
    if failures:
        print("以下模型组尚未完成：" + "、".join(failures), flush=True)
        for name in failures:
            print(f"  {name}: {_rerun(MODELS[name])}", flush=True)
        print("已成功的模型组会直接复用；失败组不会标记为完成。", flush=True)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description="Timing model installation; defaults to installing all four required groups")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--list", action="store_true", help="List the required groups without downloads")
    action.add_argument("--inventory", action="store_true", help="Report existing folder sizes as JSON; never deletes")
    action.add_argument("--dry-run", action="store_true", help="Print the groups that would be installed")
    args = parser.parse_args(argv)
    if args.list:
        for name, spec in MODELS.items():
            print(f"{name}: {spec.label} — {spec.purpose}")
    elif args.inventory:
        print(json.dumps(inventory(MODELS_DIR), ensure_ascii=False, indent=2))
    elif args.dry_run:
        print(json.dumps({"models": list(MODELS)}, indent=2))
    else:
        failures = install(
            attempts=3,
            log_dir=ROOT / ".runtime" / "install" / "models",
            status_path=ROOT / ".runtime" / "install" / "models-status.json",
        )
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

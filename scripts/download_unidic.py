"""Install the pinned NINJAL UniDic archive for explicit local setup.

Runtime code never calls this module.  The installer validates the complete
dictionary before publishing it, and ``--check-only`` performs the same full
hash audit without touching the network or the installed directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from download_support import atomic_write_json, fetch, sha256_file
from timing import japanese_dictionary


_SHA256 = re.compile(r"[0-9a-f]{64}")
_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


@dataclass(frozen=True)
class DictionarySpec:
    folder: Path
    version: str
    release: str
    archive_url: str
    archive_sha256: str
    required_files: tuple[str, ...]


def _spec() -> DictionarySpec:
    digest = str(japanese_dictionary.ARCHIVE_SHA256 or "").lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError("UniDic 固定 SHA-256 尚未配置，拒绝下载未固定的词典。")
    required = tuple(japanese_dictionary.REQUIRED_FILES)
    if "manifest.json" not in required:
        raise ValueError("UniDic 必需文件清单缺少 manifest.json。")
    return DictionarySpec(
        folder=Path(japanese_dictionary.FOLDER),
        version=str(japanese_dictionary.VERSION),
        release=str(japanese_dictionary.RELEASE),
        archive_url=str(japanese_dictionary.ARCHIVE_URL),
        archive_sha256=digest,
        required_files=required,
    )


def _safe_parts(name: str, *, label: str = "压缩包路径") -> tuple[str, ...]:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"不安全的{label}：{name!r}")
    if any(ord(character) < 32 for character in name):
        raise ValueError(f"不安全的{label}：包含控制字符。")
    raw_name = name.rstrip("/")
    if not raw_name:
        raise ValueError(f"不安全的{label}：空路径。")
    path = PurePosixPath(raw_name)
    raw_parts = raw_name.split("/")
    if path.is_absolute() or any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError(f"不安全的{label}：{name}")
    for part in raw_parts:
        if (":" in part or part.endswith(" ") or part.endswith(".") or
                part.split(".", 1)[0].upper() in _WINDOWS_RESERVED):
            raise ValueError(f"不安全的 Windows {label}：{name}")
    return tuple(raw_parts)


def _archive_files(archive: Path) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
    records: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    try:
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                mode = (item.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise ValueError(f"UniDic 压缩包包含符号链接：{item.filename}")
                kind = stat.S_IFMT(mode)
                if kind and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ValueError(f"UniDic 压缩包包含非普通文件：{item.filename}")
                if item.flag_bits & 0x1:
                    raise ValueError("UniDic 压缩包不能使用加密成员。")
                parts = _safe_parts(item.filename)
                if item.is_dir():
                    continue
                records.append((item, parts))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("UniDic 压缩包无法读取或已损坏。") from exc

    if not records:
        raise ValueError("UniDic 压缩包为空。")
    roots = {parts[0] for _, parts in records if len(parts) > 1}
    if roots and len(roots) == 1 and all(len(parts) > 1 for _, parts in records):
        records = [(item, parts[1:]) for item, parts in records]

    names = ["/".join(parts) for _, parts in records]
    folded = [name.casefold() for name in names]
    if len(folded) != len(set(folded)):
        raise ValueError("UniDic 压缩包包含重复或大小写冲突的文件路径。")
    folded_set = set(folded)
    for name in names:
        parts = name.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]).casefold() in folded_set:
                raise ValueError(f"UniDic 压缩包的文件与目录路径冲突：{name}")
    return records


def _validate_archive(path: Path) -> None:
    _archive_files(Path(path))


def _extract_archive(archive: Path, target: Path) -> None:
    records = _archive_files(archive)
    target.mkdir(parents=True, exist_ok=False)
    root = target.resolve()
    try:
        with zipfile.ZipFile(archive) as source:
            for item, parts in records:
                destination = target.joinpath(*parts)
                if not destination.resolve().is_relative_to(root):
                    raise ValueError(f"UniDic 压缩包路径越界：{item.filename}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as input_stream, destination.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError("UniDic 压缩包损坏或成员校验失败。") from exc


def _make_tagger(folder: Path):
    import fugashi

    dictionary = Path(folder).resolve()
    return fugashi.Tagger(
        f'-r "{(dictionary / "mecabrc").as_posix()}" -d "{dictionary.as_posix()}"'
    )


def _probe_tagger(folder: Path) -> None:
    tagger = _make_tagger(folder)
    readings = 0
    for token in tagger("日本語を歌う"):
        feature = token.feature
        kana = str(getattr(feature, "kana", "") or "")
        pronunciation = str(getattr(feature, "pron", "") or "")
        if kana and pronunciation:
            readings += 1
    if readings == 0:
        raise RuntimeError("UniDic 已解析，但 Fugashi 没有返回 kana/pron 读音字段。")


def _probe_tagger_process(folder: Path) -> None:
    # MeCab can retain memory-mapped dictionary files after a Tagger is freed.
    # A child process must exit before Windows can rename the staged directory.
    code = ("import sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); "
            "from download_unidic import _probe_tagger; _probe_tagger(Path(sys.argv[2]))")
    try:
        result = subprocess.run(
            [sys.executable, '-X', 'utf8', '-c', code,
             str(Path(__file__).resolve().parent), str(Path(folder).resolve())],
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('UniDic Fugashi 读音探测超时。') from exc
    if result.returncode:
        raise RuntimeError('UniDic Fugashi 读音探测失败：' + result.stderr[-1200:])


def _prepare_staged_dictionary(stage: Path, spec: DictionarySpec) -> None:
    mecabrc = stage / "mecabrc"
    if not mecabrc.exists():
        mecabrc.write_bytes(b"")
    version = stage / "version"
    if not version.exists():
        version.write_text(spec.version + "\n", encoding="utf-8", newline="\n")
    elif version.read_text(encoding="utf-8").strip() != spec.version:
        raise ValueError(f"UniDic 压缩包的 version 不是 {spec.version}。")
    missing = [name for name in spec.required_files
               if name != "manifest.json" and not (stage / name).is_file()]
    if missing:
        raise ValueError("UniDic 压缩包缺少必需文件：" + "、".join(missing))


def _asset_rows(folder: Path) -> dict[str, dict[str, int | str]]:
    assets: dict[str, dict[str, int | str]] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"UniDic 目录不能包含符号链接：{path}")
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(folder).as_posix()
        _safe_parts(relative, label="词典文件路径")
        assets[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return assets


def _full_ready(spec: DictionarySpec) -> bool:
    folder = spec.folder
    try:
        if not japanese_dictionary.ready(folder):
            return False
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("version") != spec.version or
                manifest.get("release") != spec.release or
                manifest.get("archive_sha256") != spec.archive_sha256 or
                manifest.get("source") != spec.archive_url or
                (folder / "version").read_text(encoding="utf-8").strip() != spec.version):
            return False
        assets = manifest.get("assets")
        if not isinstance(assets, dict) or not assets:
            return False
        actual = _asset_rows(folder)
        if set(assets) != set(actual):
            return False
        for name, expected in actual.items():
            asset = assets.get(name)
            if (not isinstance(asset, dict) or asset.get("bytes") != expected["bytes"] or
                    asset.get("sha256") != expected["sha256"]):
                return False
        return True
    except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
        return False


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _publish(stage: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.parent / f".{target.name}.previous-{uuid.uuid4().hex}"
    moved_existing = False
    if target.exists() or target.is_symlink():
        os.replace(target, backup)
        moved_existing = True
    try:
        os.replace(stage, target)
    except Exception:
        if moved_existing and not target.exists() and not target.is_symlink():
            os.replace(backup, target)
        raise
    if moved_existing:
        _remove_path(backup)


def install(
    *,
    check_only: bool = False,
    archive: Path | None = None,
    fetch_file: Callable[..., Path] = fetch,
    probe_tagger: Callable[[Path], None] = _probe_tagger_process,
) -> Path | bool:
    """Install or verify the pinned dictionary.

    ``check_only`` is read-only and returns a boolean.  Normal installation
    returns the published dictionary directory.  An existing verified asset is
    reused after a full per-file hash audit and never triggers a download.
    """
    spec = _spec()
    if _full_ready(spec):
        print("UniDic CWJ 2025.12 本地文件完整校验通过，继续使用现有词典。", flush=True)
        return True if check_only else spec.folder
    if check_only:
        return False

    target = spec.folder
    archive = Path(archive) if archive is not None else (
        target.parent / ".downloads" / f"unidic-cwj-{spec.release}.zip"
    )
    fetch_file(
        spec.archive_url,
        archive,
        expected_sha256=spec.archive_sha256,
        label="UniDic CWJ 2025.12 词典包",
        validate=_validate_archive,
    )

    stage = target.parent / f".{target.name}.stage-{uuid.uuid4().hex}"
    try:
        _extract_archive(archive, stage)
        _prepare_staged_dictionary(stage, spec)
        probe_tagger(stage)
        assets = _asset_rows(stage)
        atomic_write_json(stage / "manifest.json", {
            "version": spec.version,
            "release": spec.release,
            "archive_sha256": spec.archive_sha256,
            "source": spec.archive_url,
            "assets": assets,
        })
        if not _full_ready(DictionarySpec(
                folder=stage,
                version=spec.version,
                release=spec.release,
                archive_url=spec.archive_url,
                archive_sha256=spec.archive_sha256,
                required_files=spec.required_files)):
            raise RuntimeError("UniDic 暂存目录的最终完整性校验失败。")
        _publish(stage, target)
        print("UniDic CWJ 2025.12 已校验并安装。", flush=True)
        return target
    finally:
        if stage.exists() or stage.is_symlink():
            _remove_path(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="安装 NINJAL UniDic CWJ 2025.12")
    parser.add_argument("--check-only", action="store_true",
                        help="只校验现有词典和所有文件哈希，不下载或修改文件")
    arguments = parser.parse_args(argv)
    try:
        result = install(check_only=arguments.check_only)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"UniDic CWJ 2025.12 词典未准备完整：{exc}", file=sys.stderr, flush=True)
        return 1
    if arguments.check_only:
        if result is True:
            print("UniDic CWJ 2025.12 词典完整校验通过。", flush=True)
            return 0
        print("UniDic CWJ 2025.12 词典缺失或哈希不一致；请运行 "
              ".\\.venv\\Scripts\\python.exe -X utf8 scripts/download_unidic.py。",
              file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

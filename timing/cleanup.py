"""Preview and remove generated files from terminal alignment jobs.

The job store is the source of truth for whether a job has finished.  This
module only removes exact, application-generated names rooted in a validated
job folder.  In particular, uploaded audio (``source.*`` and ``stem.*``),
exports, reports, and ``job.json`` are never cleanup targets.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterable

from .job_store import load_job


CLEANUP_MODES = ("work", "keep_vocals", "all_audio")
_TERMINAL_STATES = frozenset({"complete", "draft", "failed", "cancelled"})
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")

_WORK_DIRECTORIES = ("separation-input", "stems", "tifa", "tifa_direct")
_WORK_FILES = (
    "bs_input.wav",
    "singing.wav",
    "audio.wav",
    "full_vocals.wav",
    "background_alignment.wav",
    "imported.wav",
)
_KEEP_VOCALS_FILES = ("residual.wav", "backing_vocals.wav")
_ALL_AUDIO_FILES = ("lead_vocals.wav", "vocals.wav")

_FILES_BY_MODE = {
    "work": frozenset(_WORK_FILES),
    "keep_vocals": frozenset((*_WORK_FILES, *_KEEP_VOCALS_FILES)),
    "all_audio": frozenset((*_WORK_FILES, *_KEEP_VOCALS_FILES, *_ALL_AUDIO_FILES)),
}


class CleanupChanged(RuntimeError):
    """The approved cleanup preview no longer describes the files on disk."""


@dataclass(frozen=True)
class _Entry:
    relative: str
    kind: str
    device: int
    inode: int
    size: int
    modified_ns: int


@dataclass(frozen=True)
class _Candidate:
    job_id: str
    relative: str
    directories: tuple[_Entry, ...]
    files: tuple[_Entry, ...]


@dataclass(frozen=True)
class _Snapshot:
    mode: str
    root_directory: _Entry | None
    job_directories: tuple[_Entry, ...]
    candidates: tuple[_Candidate, ...]
    files: int
    bytes: int

    @property
    def token(self) -> str:
        payload = {
            "mode": self.mode,
            "root_directory": self.root_directory.__dict__ if self.root_directory else None,
            "job_directories": [entry.__dict__ for entry in self.job_directories],
            "candidates": [
                {
                    "job": candidate.job_id,
                    "path": candidate.relative,
                    "directories": [entry.__dict__ for entry in candidate.directories],
                    "files": [entry.__dict__ for entry in candidate.files],
                }
                for candidate in self.candidates
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _mode_files(mode: str) -> frozenset[str]:
    try:
        return _FILES_BY_MODE[mode]
    except KeyError:
        raise ValueError("未知的清理范围。") from None


def _is_reparse(info: os.stat_result) -> bool:
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(getattr(info, "st_file_attributes", 0) & reparse_flag)


def _entry(relative: str, info: os.stat_result, kind: str) -> _Entry:
    is_file = kind == "file"
    return _Entry(
        relative=relative,
        kind=kind,
        device=int(info.st_dev),
        inode=int(info.st_ino),
        # Directory timestamps and sizes are implementation details and can
        # change asynchronously after a prior cleanup.  Their device/inode
        # identity plus their exact child inventory is the useful contract.
        size=int(info.st_size) if is_file else 0,
        modified_ns=int(info.st_mtime_ns) if is_file else 0,
    )


def _walk_directory(path: Path, relative: str) -> tuple[tuple[_Entry, ...], tuple[_Entry, ...]] | None:
    """Return a no-follow tree description, or ``None`` when links are present."""
    directories: list[_Entry] = []
    files: list[_Entry] = []
    pending = [(path, relative)]
    while pending:
        current, current_relative = pending.pop()
        try:
            current_info = current.lstat()
        except OSError:
            return None
        if _is_reparse(current_info) or not stat.S_ISDIR(current_info.st_mode):
            return None
        directories.append(_entry(current_relative, current_info, "directory"))
        try:
            with os.scandir(current) as scan:
                children = list(scan)
        except OSError:
            return None
        for child in children:
            child_path = Path(child.path)
            child_relative = f"{current_relative}/{child.name}"
            try:
                # DirEntry.stat is cached and Windows can return zero file IDs
                # from that cache.  A fresh lstat supplies the identity used by
                # the preview token.
                info = child_path.lstat()
            except OSError:
                return None
            if _is_reparse(info):
                return None
            if stat.S_ISDIR(info.st_mode):
                pending.append((child_path, child_relative))
            elif stat.S_ISREG(info.st_mode):
                files.append(_entry(child_relative, info, "file"))
            else:
                return None
    return (
        tuple(sorted(directories, key=lambda entry: entry.relative)),
        tuple(sorted(files, key=lambda entry: entry.relative)),
    )


def _candidate(job_id: str, path: Path, relative: str, info: os.stat_result) -> _Candidate | None:
    if stat.S_ISREG(info.st_mode):
        entry = _entry(relative, info, "file")
        return _Candidate(job_id, relative, (), (entry,))
    if stat.S_ISDIR(info.st_mode):
        tree = _walk_directory(path, relative)
        if tree is None:
            return None
        directories, files = tree
        if files:
            return _Candidate(job_id, relative, directories, files)
    return None


def _snapshot(root: Path, mode: str, active_ids: Iterable[str]) -> _Snapshot:
    files_by_mode = _mode_files(mode)
    try:
        root_info = root.lstat()
    except FileNotFoundError:
        return _Snapshot(mode, None, (), (), 0, 0)
    if _is_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
        raise NotADirectoryError("清理目录无效。")
    active = set(active_ids)
    job_directories: list[_Entry] = []
    candidates: list[_Candidate] = []
    with os.scandir(root) as scan:
        job_entries = sorted(scan, key=lambda item: item.name)
    for job_entry in job_entries:
        job_id = job_entry.name
        if not _JOB_ID.fullmatch(job_id) or job_id in active:
            continue
        job_root = root / job_id
        try:
            job_info = job_root.lstat()
        except OSError:
            continue
        if _is_reparse(job_info) or not stat.S_ISDIR(job_info.st_mode):
            continue
        try:
            job = load_job(job_root)
        except (OSError, TypeError, ValueError):
            continue
        if not isinstance(job, dict) or job.get("state") not in _TERMINAL_STATES:
            continue
        job_candidates: list[_Candidate] = []
        for name in sorted((*files_by_mode, *_WORK_DIRECTORIES)):
            path = job_root / name
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            except OSError:
                job_candidates.clear()
                break
            if _is_reparse(info):
                if name in files_by_mode:
                    continue
                job_candidates.clear()
                break
            expected_directory = name in _WORK_DIRECTORIES
            if expected_directory != stat.S_ISDIR(info.st_mode):
                continue
            candidate = _candidate(job_id, path, f"{job_id}/{name}", info)
            if candidate is not None:
                job_candidates.append(candidate)
        if job_candidates:
            job_directories.append(_entry(job_id, job_info, "directory"))
            candidates.extend(job_candidates)
    candidates.sort(key=lambda candidate: candidate.relative)
    all_files = [entry for candidate in candidates for entry in candidate.files]
    return _Snapshot(
        mode=mode,
        root_directory=_entry(".", root_info, "directory"),
        job_directories=tuple(job_directories),
        candidates=tuple(candidates),
        files=len(all_files),
        bytes=sum(entry.size for entry in all_files),
    )


def preview(root: Path, mode: str, *, active_ids: Iterable[str] = ()) -> dict:
    """Describe terminal job files that the requested cleanup would remove."""
    snapshot = _snapshot(Path(root).absolute(), mode, active_ids)
    return {
        "mode": mode,
        "jobs": len(snapshot.job_directories),
        "files": snapshot.files,
        "bytes": snapshot.bytes,
        "token": snapshot.token,
    }


def _same_identity(left: _Entry, info: os.stat_result) -> bool:
    if _is_reparse(info) or (left.device, left.inode) != (info.st_dev, info.st_ino):
        return False
    if left.kind == "directory":
        return stat.S_ISDIR(info.st_mode)
    return stat.S_ISREG(info.st_mode) and (left.size, left.modified_ns) == (info.st_size, info.st_mtime_ns)


def _matches_path(root: Path, entry: _Entry, directories: dict[Path, _Entry]) -> bool:
    relative = Path(entry.relative)
    # Validate from the root down so a replaced ancestor cannot redirect a leaf
    # lookup, even if the leaf itself still has the identity from the preview.
    for parent in reversed(relative.parents):
        if not _same_identity(directories[parent], (root / parent).lstat()):
            return False
    return _same_identity(entry, (root / relative).lstat())


def _remove_snapshot(root: Path, snapshot: _Snapshot) -> tuple[int, int, int, int]:
    if snapshot.root_directory is None:
        return 0, 0, 0, 0
    deleted_files = 0
    deleted_bytes = 0
    failures = 0
    touched_jobs: set[str] = set()
    job_dirs = {Path(entry.relative): entry for entry in snapshot.job_directories}
    job_dirs[Path('.')] = snapshot.root_directory
    for candidate in snapshot.candidates:
        directories = job_dirs | {Path(entry.relative): entry for entry in candidate.directories}
        for entry in candidate.files:
            path = root / Path(entry.relative)
            try:
                if not _matches_path(root, entry, directories):
                    failures += 1
                    continue
                path.unlink()
            except OSError:
                failures += 1
                continue
            deleted_files += 1
            deleted_bytes += entry.size
            touched_jobs.add(candidate.job_id)
        for entry in sorted(candidate.directories, key=lambda row: row.relative.count("/"), reverse=True):
            path = root / Path(entry.relative)
            try:
                if not _matches_path(root, entry, directories):
                    failures += 1
                    continue
                path.rmdir()
            except OSError:
                failures += 1
                continue
            touched_jobs.add(candidate.job_id)
    return len(touched_jobs), deleted_files, deleted_bytes, failures


def clean(
    root: Path,
    mode: str,
    *,
    active_ids: Iterable[str] = (),
    expected_token: str,
) -> dict:
    """Remove the exact files described by a matching preview token."""
    root = Path(root).absolute()
    snapshot = _snapshot(root, mode, active_ids)
    if snapshot.token != expected_token:
        raise CleanupChanged("清理范围内的文件已变化，请重新预览。")
    jobs, files, removed_bytes, failures = _remove_snapshot(root, snapshot)
    return {
        "mode": mode,
        "jobs": jobs,
        "files": files,
        "bytes": removed_bytes,
        "failed": failures,
    }

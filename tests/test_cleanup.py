import json
import os
from pathlib import Path

import pytest

from timing import cleanup


JOB_A = "0123456789abcdef0123456789abcdef"
JOB_B = "fedcba9876543210fedcba9876543210"


def make_job(root: Path, identifier: str, state: str = "complete") -> Path:
    folder = root / identifier
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(
        json.dumps({"id": identifier, "state": state, "_state_revision": 1}),
        encoding="utf-8",
    )
    return folder


def write_files(folder: Path, files: dict[str, bytes]) -> None:
    for relative, contents in files.items():
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)


def test_work_preview_and_clean_preserve_uploads_and_results(tmp_path):
    folder = make_job(tmp_path, JOB_A)
    write_files(folder, {
        "bs_input.wav": b"12",
        "singing.wav": b"345",
        "separation-input/mix.wav": b"678",
        "stems/vocals.wav": b"1234",
        "tifa/line.wav": b"12",
        "tifa_direct/line.wav": b"1",
        "lead_vocals.wav": b"lead",
        "source.m4a": b"original",
        "stem.wav": b"provided",
        "lyrics.ttml": b"<tt/>",
        "alignment.json": b"{}",
        "lyrics.txt": b"lyrics",
    })

    result = cleanup.preview(tmp_path, "work")
    assert result["jobs"] == 1
    assert result["files"] == 6
    assert result["bytes"] == 15
    assert len(result["token"]) == 64

    cleaned = cleanup.clean(tmp_path, "work", expected_token=result["token"])
    assert cleaned == {"mode": "work", "jobs": 1, "files": 6, "bytes": 15, "failed": 0}
    assert not any((folder / name).exists() for name in cleanup._WORK_DIRECTORIES)
    assert not (folder / "bs_input.wav").exists()
    assert (folder / "lead_vocals.wav").read_bytes() == b"lead"
    for name in ("source.m4a", "stem.wav", "lyrics.ttml", "alignment.json", "lyrics.txt", "job.json"):
        assert (folder / name).exists()


def test_audio_cleanup_modes_are_cumulative(tmp_path):
    folder = make_job(tmp_path, JOB_A)
    write_files(folder, {
        "audio.wav": b"a",
        "full_vocals.wav": b"b",
        "background_alignment.wav": b"c",
        "imported.wav": b"d",
        "residual.wav": b"e",
        "backing_vocals.wav": b"f",
        "lead_vocals.wav": b"g",
        "vocals.wav": b"h",
        "source.wav": b"source",
        "stem.wav": b"stem",
        "lyrics.ttml": b"ttml",
    })

    keep = cleanup.preview(tmp_path, "keep_vocals")
    assert keep["files"] == 6
    assert cleanup.clean(tmp_path, "keep_vocals", expected_token=keep["token"])["files"] == 6
    assert (folder / "lead_vocals.wav").exists()
    assert (folder / "vocals.wav").exists()

    all_audio = cleanup.preview(tmp_path, "all_audio")
    assert all_audio["files"] == 2
    assert cleanup.clean(tmp_path, "all_audio", expected_token=all_audio["token"])["files"] == 2
    assert set(path.name for path in folder.iterdir()) == {
        "job.json", "source.wav", "stem.wav", "lyrics.ttml"
    }


def test_changed_preview_refuses_deletion(tmp_path):
    folder = make_job(tmp_path, JOB_A)
    write_files(folder, {"singing.wav": b"before"})
    result = cleanup.preview(tmp_path, "work")
    (folder / "singing.wav").write_bytes(b"changed size")

    with pytest.raises(cleanup.CleanupChanged):
        cleanup.clean(tmp_path, "work", expected_token=result["token"])
    assert (folder / "singing.wav").read_bytes() == b"changed size"


def test_replaced_parent_is_skipped_even_if_the_leaf_did_not_change(tmp_path, monkeypatch):
    folder = make_job(tmp_path, JOB_A)
    write_files(folder, {"tifa/line.wav": b"same leaf"})
    approved = cleanup.preview(tmp_path, "work")
    remove_snapshot = cleanup._remove_snapshot

    def replace_directory(root, snapshot):
        (folder / 'tifa').rename(folder / 'moved')
        (folder / 'tifa').mkdir()
        (folder / 'moved' / 'line.wav').rename(folder / 'tifa' / 'line.wav')
        return remove_snapshot(root, snapshot)

    monkeypatch.setattr(cleanup, '_remove_snapshot', replace_directory)
    result = cleanup.clean(tmp_path, 'work', expected_token=approved['token'])
    assert result['files'] == 0
    assert result['failed'] > 0
    assert (folder / 'tifa' / 'line.wav').read_bytes() == b'same leaf'


def test_only_matching_file_or_directory_types_are_candidates(tmp_path):
    folder = make_job(tmp_path, JOB_A)
    (folder / "singing.wav").mkdir()
    (folder / "singing.wav" / "keep.txt").write_text("keep", encoding="utf-8")
    (folder / "tifa").write_text("unexpected file", encoding="utf-8")
    (folder / "bs_input.wav").write_bytes(b"candidate")

    result = cleanup.preview(tmp_path, "work")
    assert result["files"] == 1
    cleanup.clean(tmp_path, "work", expected_token=result["token"])
    assert (folder / "singing.wav" / "keep.txt").exists()
    assert (folder / "tifa").read_text(encoding="utf-8") == "unexpected file"
    assert not (folder / "bs_input.wav").exists()


def test_active_nonterminal_and_unknown_jobs_are_skipped(tmp_path):
    complete = make_job(tmp_path, JOB_A)
    active = make_job(tmp_path, JOB_B)
    write_files(complete, {"singing.wav": b"complete"})
    write_files(active, {"singing.wav": b"active"})
    running = make_job(tmp_path, "11111111111111111111111111111111", "running")
    write_files(running, {"singing.wav": b"running"})
    (tmp_path / "abcdefabcdefabcdefabcdefabcdefab").mkdir()
    (tmp_path / "not-a-job").mkdir()
    malformed_id = 'c' * 32
    malformed = make_job(tmp_path, malformed_id)
    (malformed / 'job.json').write_text(json.dumps({'id': malformed_id, 'state': []}), encoding='utf-8')
    write_files(malformed, {'singing.wav': b'unknown state'})

    result = cleanup.preview(tmp_path, "work", active_ids={JOB_B})
    assert result["jobs"] == 1
    assert result["bytes"] == len(b"complete")
    cleanup.clean(tmp_path, "work", active_ids={JOB_B}, expected_token=result["token"])
    assert not (complete / "singing.wav").exists()
    assert (active / "singing.wav").exists()
    assert (running / "singing.wav").exists()
    assert (malformed / 'singing.wav').exists()


def test_directory_link_makes_candidate_ineligible(tmp_path):
    folder = make_job(tmp_path, JOB_A)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")
    try:
        os.symlink(outside, folder / "tifa", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory links are unavailable")
    write_files(folder, {"singing.wav": b"candidate"})

    result = cleanup.preview(tmp_path, "work")
    assert result["files"] == 0
    cleanup.clean(tmp_path, "work", expected_token=result["token"])
    assert (folder / "singing.wav").exists()
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"

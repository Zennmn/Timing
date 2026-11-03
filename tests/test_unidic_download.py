import hashlib
import json
import stat
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

import download_unidic
import download_tifa
from timing import japanese_dictionary


FAKE_FILES = {
    "sys.dic": b"UNIDIC_FAKE_SYS_DICTIONARY_0123456789",
    "unk.dic": b"unknown",
    "matrix.bin": b"matrix",
    "char.bin": b"chars",
    "dicrc": b"dicrc",
    "model.bin": b"model",
    "rewrite.def": b"rewrite",
    "README_unidic-cwj.txt": b"readme",
    "license/COPYING": b"license",
    "license/AUTHORS": b"authors",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def make_archive(path, *, root="", files=None, extra=None, symlink=None):
    files = FAKE_FILES if files is None else files
    prefix = root.rstrip("/") + "/" if root else ""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in files.items():
            archive.writestr(prefix + name, data)
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
        if symlink:
            item = zipfile.ZipInfo(symlink)
            item.create_system = 3
            item.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(item, "../outside")
    return path


def configure(monkeypatch, tmp_path, archive):
    folder = tmp_path / "Models with spaces" / "unidic-cwj-202512"
    monkeypatch.setattr(japanese_dictionary, "FOLDER", folder)
    monkeypatch.setattr(japanese_dictionary, "VERSION", "2025.12")
    monkeypatch.setattr(japanese_dictionary, "RELEASE", "202512")
    monkeypatch.setattr(japanese_dictionary, "ARCHIVE_URL",
                        "https://example.invalid/unidic-cwj-202512.zip")
    monkeypatch.setattr(japanese_dictionary, "ARCHIVE_SHA256", digest(archive.read_bytes()))
    return folder


@pytest.mark.parametrize("root", ["", "unidic-cwj-202512"])
def test_install_extracts_flat_or_rooted_archive_and_publishes_manifest(
        tmp_path, monkeypatch, root):
    archive = make_archive(tmp_path / "fixture.zip", root=root)
    folder = configure(monkeypatch, tmp_path, archive)
    fetched = []
    probed = []

    def fake_fetch(url, destination, **kwargs):
        fetched.append((url, Path(destination), kwargs))
        kwargs["validate"](Path(destination))
        return Path(destination)

    def fake_probe(stage):
        assert (stage / "sys.dic").read_bytes() == FAKE_FILES["sys.dic"]
        assert not folder.exists()
        probed.append(stage)

    result = download_unidic.install(
        archive=archive, fetch_file=fake_fetch, probe_tagger=fake_probe)

    assert result == folder
    assert fetched[0][0] == japanese_dictionary.ARCHIVE_URL
    assert fetched[0][2]["expected_sha256"] == digest(archive.read_bytes())
    assert fetched[0][2]["label"] == "UniDic CWJ 2025.12 词典包"
    assert len(probed) == 1
    assert (folder / "mecabrc").is_file()
    assert (folder / "mecabrc").read_bytes() == b""
    assert (folder / "version").read_text(encoding="utf-8").strip() == "2025.12"
    assert (folder / "model.bin").read_bytes() == b"model"
    assert (folder / "license" / "COPYING").read_bytes() == b"license"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "2025.12"
    assert manifest["release"] == "202512"
    assert manifest["archive_sha256"] == digest(archive.read_bytes())
    assert manifest["source"] == japanese_dictionary.ARCHIVE_URL
    assert "license/COPYING" in manifest["assets"]
    assert "manifest.json" not in manifest["assets"]
    assert japanese_dictionary.ready(folder)
    assert not any(path.name.startswith(".unidic-cwj-202512.")
                   for path in folder.parent.iterdir())


def test_existing_dictionary_is_fully_revalidated_before_reuse(
        tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "fixture.zip")
    folder = configure(monkeypatch, tmp_path, archive)
    download_unidic.install(archive=archive, probe_tagger=lambda _folder: None)

    def unexpected(*_args, **_kwargs):
        pytest.fail("fully verified dictionary must be reused offline")

    assert download_unidic.install(
        archive=archive, fetch_file=unexpected, probe_tagger=unexpected) == folder

    target = folder / "sys.dic"
    size = target.stat().st_size
    target.write_bytes(b"x" * size)
    with pytest.raises(RuntimeError, match="archive unavailable"):
        download_unidic.install(
            archive=archive,
            fetch_file=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("archive unavailable")),
            probe_tagger=lambda _folder: None,
        )
    assert target.read_bytes() == b"x" * size


def test_check_only_is_read_only_and_rejects_a_same_size_hash_change(
        tmp_path, monkeypatch, capsys):
    archive = make_archive(tmp_path / "fixture.zip")
    folder = configure(monkeypatch, tmp_path, archive)
    download_unidic.install(archive=archive, probe_tagger=lambda _folder: None)
    capsys.readouterr()

    assert download_unidic.main(["--check-only"]) == 0
    assert "完整校验通过" in capsys.readouterr().out

    target = folder / "sys.dic"
    target.write_bytes(b"z" * target.stat().st_size)
    assert download_unidic.main(["--check-only"]) == 1
    captured = capsys.readouterr()
    assert "词典缺失或哈希不一致" in captured.err
    assert "scripts/download_unidic.py" in captured.err
    assert not any(path.name.startswith(".unidic-cwj-202512.")
                   for path in folder.parent.iterdir())


def test_check_only_missing_dictionary_returns_one_without_download(
        tmp_path, monkeypatch, capsys):
    archive = make_archive(tmp_path / "fixture.zip")
    folder = configure(monkeypatch, tmp_path, archive)

    def unexpected(*_args, **_kwargs):
        pytest.fail("check-only must never download")

    monkeypatch.setattr(download_unidic, "fetch", unexpected)
    assert download_unidic.main(["--check-only"]) == 1
    captured = capsys.readouterr()
    assert "词典缺失或哈希不一致" in captured.err
    assert not folder.exists()


@pytest.mark.parametrize("unsafe", [
    "../escape",
    "root/../../escape",
    "C:/escape",
    "root/aux.txt",
])
def test_installer_rejects_unsafe_archive_paths(tmp_path, monkeypatch, unsafe):
    archive = make_archive(tmp_path / "fixture.zip", extra={unsafe: b"bad"})
    folder = configure(monkeypatch, tmp_path, archive)
    with pytest.raises(ValueError, match="不安全"):
        download_unidic.install(
            archive=archive,
            fetch_file=lambda _url, destination, **_kwargs: Path(destination),
            probe_tagger=lambda _folder: pytest.fail("unsafe archive must not be probed"),
        )
    assert not folder.exists()


def test_installer_rejects_symlink_members(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "fixture.zip", symlink="dictionary/link")
    folder = configure(monkeypatch, tmp_path, archive)
    with pytest.raises(ValueError, match="符号链接"):
        download_unidic.install(
            archive=archive,
            fetch_file=lambda _url, destination, **kwargs: (
                kwargs["validate"](Path(destination)), Path(destination))[1],
            probe_tagger=lambda _folder: pytest.fail("symlink archive must not be probed"),
        )
    assert not folder.exists()


def test_missing_required_member_and_partial_archive_keep_existing_target(
        tmp_path, monkeypatch):
    placeholder = tmp_path / "placeholder.zip"
    placeholder.write_bytes(b"")
    folder = configure(monkeypatch, tmp_path, placeholder)
    folder.mkdir(parents=True)
    marker = folder / "keep.txt"
    marker.write_text("existing", encoding="utf-8")

    missing = make_archive(
        tmp_path / "missing.zip", files={k: v for k, v in FAKE_FILES.items() if k != "sys.dic"})
    monkeypatch.setattr(japanese_dictionary, "ARCHIVE_SHA256", digest(missing.read_bytes()))
    with pytest.raises(ValueError, match="缺少必需文件"):
        download_unidic.install(
            archive=missing,
            fetch_file=lambda _url, destination, **kwargs: (
                kwargs["validate"](Path(destination)), Path(destination))[1],
            probe_tagger=lambda _folder: None,
        )
    assert marker.read_text(encoding="utf-8") == "existing"

    partial = tmp_path / "partial.zip"
    partial.write_bytes(b"PK\x03\x04partial")
    monkeypatch.setattr(japanese_dictionary, "ARCHIVE_SHA256", digest(partial.read_bytes()))
    with pytest.raises(ValueError, match="无法读取或已损坏"):
        download_unidic.install(
            archive=partial,
            fetch_file=lambda _url, destination, **kwargs: (
                kwargs["validate"](Path(destination)), Path(destination))[1],
            probe_tagger=lambda _folder: None,
        )
    assert marker.read_text(encoding="utf-8") == "existing"
    assert not any(path.name.startswith(".unidic-cwj-202512.")
                   for path in folder.parent.iterdir())


def test_corrupt_member_keeps_existing_target_and_cleans_stage(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "fixture.zip")
    payload = FAKE_FILES["sys.dic"]
    corrupted = archive.read_bytes().replace(payload, b"X" * len(payload), 1)
    assert corrupted != archive.read_bytes()
    archive.write_bytes(corrupted)
    folder = configure(monkeypatch, tmp_path, archive)
    folder.mkdir(parents=True)
    marker = folder / "keep.txt"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="损坏或成员校验失败"):
        download_unidic.install(
            archive=archive,
            fetch_file=lambda _url, destination, **kwargs: (
                kwargs["validate"](Path(destination)), Path(destination))[1],
            probe_tagger=lambda _folder: None,
        )
    assert marker.read_text(encoding="utf-8") == "existing"
    assert not any(path.name.startswith(".unidic-cwj-202512.")
                   for path in folder.parent.iterdir())


def test_tagger_command_quotes_dictionary_paths_with_spaces(tmp_path, monkeypatch):
    calls = []

    def tagger(arguments):
        calls.append(arguments)
        return object()

    monkeypatch.setitem(sys.modules, "fugashi", SimpleNamespace(Tagger=tagger))
    folder = tmp_path / "Dictionary With Spaces"
    folder.mkdir()
    download_unidic._make_tagger(folder)
    resolved = folder.resolve().as_posix()
    assert calls == [f'-r "{resolved}/mecabrc" -d "{resolved}"']


def test_tagger_probe_requires_both_kana_and_pronunciation(tmp_path, monkeypatch):
    class Feature:
        def __init__(self, kana, pron):
            self.kana = kana
            self.pron = pron

    class Token:
        def __init__(self, kana, pron):
            self.feature = Feature(kana, pron)

    monkeypatch.setattr(
        download_unidic, "_make_tagger",
        lambda _folder: lambda _text: [Token("ニホン", "ニホン"), Token("ウタウ", "ウタウ")])
    download_unidic._probe_tagger(tmp_path)

    monkeypatch.setattr(
        download_unidic, "_make_tagger",
        lambda _folder: lambda _text: [Token("ニホン", "")])
    with pytest.raises(RuntimeError, match="kana/pron"):
        download_unidic._probe_tagger(tmp_path)


def test_native_probe_closes_dictionary_mappings_before_publication(tmp_path, monkeypatch):
    if not japanese_dictionary.ready():
        pytest.skip('full UniDic not installed')
    stage = tmp_path / 'Dictionary With Spaces'
    stage.mkdir()
    for name in japanese_dictionary.REQUIRED_FILES:
        if name != 'manifest.json':
            # Same bytes without copying a gigabyte; deleting these links never
            # removes the installed dictionary's separate directory entries.
            (stage / name).hardlink_to(japanese_dictionary.dictionary_dir() / name)
    download_unidic._probe_tagger_process(stage)
    published = stage.rename(tmp_path / 'published')
    for path in published.iterdir():
        path.unlink()
    monkeypatch.setattr(download_unidic.subprocess, 'run',
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr='kana/pron failed'))
    with pytest.raises(RuntimeError, match='kana/pron failed'):
        download_unidic._probe_tagger_process(tmp_path)


def test_tifa_final_readiness_installs_dictionary_without_touching_tifa(monkeypatch):
    calls = []
    monkeypatch.setattr(download_tifa, "_manifest_ok", lambda: True)
    monkeypatch.setattr(download_tifa, "_package_ok", lambda: (True, ""))
    monkeypatch.setattr(download_tifa, "install_dictionary", lambda: calls.append("dictionary"))

    download_tifa.main()

    assert calls == ["dictionary"]

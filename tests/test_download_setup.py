import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_support as support


def digest(data):
    return hashlib.sha256(data).hexdigest()


def script(name):
    spec = importlib.util.spec_from_file_location(f"test_download_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_fetch_reuses_a_completed_verified_file(tmp_path):
    data = b"already complete"
    target = tmp_path / "model.bin"
    target.write_bytes(data)

    def fail(_request):
        pytest.fail("verified completed file must not use the network")

    client = mock_client(fail)
    try:
        assert support.fetch("https://example.invalid/model.bin", target,
                              expected_sha256=digest(data), client=client,
                              log=lambda _message: None) == target
    finally:
        client.close()


def test_fetch_validates_cached_file_without_a_checksum(tmp_path):
    target = tmp_path / "runtime.zip"
    target.write_bytes(b"corrupt")
    seen = []

    def validator(path):
        if path.read_bytes() == b"corrupt":
            raise RuntimeError("bad zip")
        seen.append(path.read_bytes())

    def handler(request):
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, content=b"replacement")

    client = mock_client(handler)
    try:
        support.fetch("https://example.invalid/runtime.zip", target, client=client,
                      log=lambda _message: None, validate=validator)
    finally:
        client.close()
    assert target.read_bytes() == b"replacement"
    assert seen == [b"replacement"]


def test_fetch_retries_a_transient_status(tmp_path):
    data = b"ok"
    target = tmp_path / "file.bin"
    calls = []
    sleeps = []

    def handler(_request):
        calls.append(1)
        return httpx.Response(503) if len(calls) == 1 else httpx.Response(200, content=data)

    client = mock_client(handler)
    try:
        support.fetch("https://example.invalid/file.bin?token=hidden", target,
                      expected_sha256=digest(data), attempts=2, client=client,
                      sleeper=sleeps.append, log=lambda _message: None)
    finally:
        client.close()
    assert target.read_bytes() == data
    assert len(calls) == 2 and sleeps == [1]


def test_fetch_resumes_with_standard_206_content_range(tmp_path):
    target = tmp_path / "file.bin"
    target.with_name(target.name + ".part").write_bytes(b"abc")
    data = b"abcdef"
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["range"] == "bytes=3-"
        return httpx.Response(206, content=b"def", headers={
            "Content-Range": "bytes 3-5/6", "Content-Length": "3"})

    client = mock_client(handler)
    try:
        support.fetch("https://example.invalid/file.bin", target, expected_sha256=digest(data),
                      client=client, log=lambda _message: None)
    finally:
        client.close()
    assert target.read_bytes() == data
    assert not target.with_name(target.name + ".part").exists()
    assert len(requests) == 1


def test_fetch_restarts_when_server_ignores_range_with_200(tmp_path):
    target = tmp_path / "file.bin"
    target.with_name(target.name + ".part").write_bytes(b"stale partial")

    def handler(request):
        assert request.headers["range"] == "bytes=13-"
        return httpx.Response(200, content=b"fresh")

    client = mock_client(handler)
    try:
        support.fetch("https://example.invalid/file.bin", target, expected_sha256=digest(b"fresh"),
                      client=client, log=lambda _message: None)
    finally:
        client.close()
    assert target.read_bytes() == b"fresh"


def test_fetch_rejects_206_range_that_does_not_cover_the_end(tmp_path):
    target = tmp_path / "file.bin"
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"a")

    def handler(_request):
        return httpx.Response(206, content=b"b", headers={
            "Content-Range": "bytes 1-1/4", "Content-Length": "1"})

    client = mock_client(handler)
    try:
        with pytest.raises(support.DownloadError, match="续传范围"):
            support.fetch("https://example.invalid/file.bin", target, attempts=1,
                          expected_sha256=digest(b"abc?"), client=client,
                          log=lambda _message: None)
    finally:
        client.close()
    assert not target.exists() and part.read_bytes() == b"a"


def test_fetch_416_publishes_a_complete_part(tmp_path):
    target = tmp_path / "file.bin"
    part = target.with_name(target.name + ".part")
    part.write_bytes(b"abc")

    def handler(request):
        assert request.headers["range"] == "bytes=3-"
        return httpx.Response(416, headers={"Content-Range": "bytes */3"})

    client = mock_client(handler)
    try:
        support.fetch("https://example.invalid/file.bin", target, expected_sha256=digest(b"abc"),
                      client=client, log=lambda _message: None)
    finally:
        client.close()
    assert target.read_bytes() == b"abc" and not part.exists()


def test_fetch_416_restarts_a_bad_part(tmp_path):
    target = tmp_path / "file.bin"
    target.with_name(target.name + ".part").write_bytes(b"x")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(416, headers={"Content-Range": "bytes */1"})
        assert "range" not in request.headers
        return httpx.Response(200, content=b"good")

    client = mock_client(handler)
    try:
        support.fetch("https://example.invalid/file.bin", target, expected_sha256=digest(b"good"),
                      attempts=2, client=client, sleeper=lambda _delay: None,
                      log=lambda _message: None)
    finally:
        client.close()
    assert target.read_bytes() == b"good" and len(calls) == 2


def test_fetch_without_content_length_reports_unknown_size(tmp_path):
    target = tmp_path / "file.bin"
    messages = []

    def handler(_request):
        response = httpx.Response(200, content=b"small")
        response.headers.pop("content-length", None)
        return response

    client = mock_client(handler)
    try:
        support.fetch("https://example.invalid/file.bin", target, expected_sha256=digest(b"small"),
                      client=client, log=messages.append)
    finally:
        client.close()
    assert target.read_bytes() == b"small"
    assert any("未知大小" in message for message in messages)


def test_fetch_hash_failure_never_publishes_final_file(tmp_path):
    target = tmp_path / "file.bin"

    def handler(_request):
        return httpx.Response(200, content=b"wrong")

    client = mock_client(handler)
    try:
        with pytest.raises(support.DownloadError, match="SHA-256"):
            support.fetch("https://example.invalid/file.bin", target, attempts=1,
                          expected_sha256=digest(b"expected"), client=client,
                          log=lambda _message: None)
    finally:
        client.close()
    assert not target.exists()
    assert target.with_name(target.name + ".part").read_bytes() == b"wrong"


def test_friendly_failure_handles_empty_exception_and_redacts_url_secrets():
    message = support.friendly_failure("TIFA", "rerun", RuntimeError())
    assert "TIFA" in message and "rerun" in message and "未提供详细错误" in message
    safe = support.safe_url("https://user:secret@example.invalid/model.bin?token=secret#fragment")
    assert safe == "https://example.invalid/model.bin"
    assert "secret" not in safe


def test_setup_log_tees_progress_to_console_and_group_log(tmp_path, monkeypatch, capsys):
    log = tmp_path / "ctc.log"
    monkeypatch.setenv("TIMING_DOWNLOAD_LOG", str(log))
    support._setup_log("MMS：已下载 50%")
    assert "MMS：已下载 50%" in capsys.readouterr().out
    assert "MMS：已下载 50%" in log.read_text(encoding="utf-8")


def test_models_retry_each_group_and_never_mark_a_failure_complete(tmp_path, monkeypatch):
    module = script("download_models")
    monkeypatch.setattr(module, "initialize", lambda: None)
    calls = []

    def run(args, **kwargs):
        calls.append(Path(args[-1]).name)
        assert kwargs["check"] is True
        assert kwargs["stderr"] is not None
        group = next(name for name, spec in module.MODELS.items()
                     if spec.installer == Path(args[-1]).name)
        assert Path(kwargs["env"]["TIMING_DOWNLOAD_LOG"]).name == f"{group}.log"
        if args[-1].endswith("download_tifa.py"):
            raise subprocess.CalledProcessError(9, args)

    monkeypatch.setattr(module.subprocess, "run", run)
    status_path = tmp_path / "models-status.json"
    failures = module.install(attempts=3, log_dir=tmp_path / "logs", status_path=status_path,
                              sleeper=lambda _delay: None)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert failures == ["tifa"]
    assert calls.count("download_tifa.py") == 3
    assert status["complete"] is False
    assert status["groups"]["tifa"]["status"] == "failed"
    assert status["groups"]["tifa"]["attempts"] == 3
    assert list(status["groups"]) == ["ctc", "tifa", "bs_lead", "separator"]
    assert all(status["groups"][name]["status"] == "complete"
               for name in ("ctc", "bs_lead", "separator"))
    assert "第 3/3 次" in (tmp_path / "logs" / "tifa.log").read_text(encoding="utf-8")


def test_ctc_forces_only_a_known_corrupt_cached_file(tmp_path, monkeypatch):
    module = script("download_ctc")
    monkeypatch.setattr(module, "FOLDER", tmp_path)
    good = b"good checkpoint"
    monkeypatch.setattr(module, "CHECKPOINT_SHA", digest(good))
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    calls = []

    def snapshot(repo, *, revision, local_dir, allow_patterns, force_download=False):
        calls.append((tuple(allow_patterns), force_download))
        path = Path(local_dir) / "model.safetensors"
        path.write_bytes(good)
        for name in ("config.json", "preprocessor_config.json", "vocab.json"):
            (Path(local_dir) / name).write_text("{}", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr(module, "_package_ok", lambda: (True, ""))
    module.main()
    assert calls[0] == (("model.safetensors",), True)
    assert calls[1][1] is False
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["revision"] == module.REVISION


def test_bs_hf_force_is_limited_to_the_corrupt_pinned_file(tmp_path, monkeypatch):
    module = script("download_bs_karaoke")
    monkeypatch.setattr(module, "FOLDER", tmp_path)
    checkpoint, config = tuple(module.FILES)
    good_ckpt, good_config = b"checkpoint", b"config"
    monkeypatch.setattr(module, "FILES", {checkpoint: digest(good_ckpt), config: digest(good_config)})
    (tmp_path / checkpoint).write_bytes(b"corrupt")
    (tmp_path / config).write_bytes(good_config)
    hf_calls = []

    def download(repo, name, *, revision, local_dir, force_download=False):
        hf_calls.append((name, force_download))
        path = Path(local_dir) / name
        path.write_bytes(good_ckpt if name == checkpoint else good_config)
        return str(path)

    def git_run(args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=b"runtime")

    states = iter([False, True])
    monkeypatch.setattr(module, "hf_hub_download", download)
    monkeypatch.setattr(module.subprocess, "run", git_run)
    monkeypatch.setattr(module, "_manifest_ok", lambda: next(states))
    monkeypatch.setattr(module, "_package_ok", lambda: (True, ""))
    module.main()
    assert hf_calls[0] == (checkpoint, True)
    assert hf_calls[1] == (config, False)


def test_bs_config_loader_accepts_the_pinned_python_tuple_fixture(tmp_path):
    module = script("download_bs_karaoke")
    config = tmp_path / "config.yaml"
    config.write_text("freqs_per_bands: !!python/tuple [2, 4]\n", encoding="utf-8")
    assert module._load_config(config)["freqs_per_bands"] == (2, 4)


def test_interrupted_body_keeps_partial_bytes_and_resumes_next_request(tmp_path):
    prefix = b"a" * support.CHUNK_SIZE
    complete = prefix + b"tail"
    target = tmp_path / "model.bin"
    requests = []

    class InterruptedBody(httpx.SyncByteStream):
        def __iter__(self):
            yield prefix
            raise httpx.ReadError("fixture connection lost")

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=InterruptedBody(),
                                  headers={"Content-Length": str(len(complete))})
        assert request.headers["range"] == f"bytes={len(prefix)}-"
        return httpx.Response(206, content=b"tail", headers={
            "Content-Range": f"bytes {len(prefix)}-{len(complete)-1}/{len(complete)}"})

    with mock_client(handler) as client:
        support.fetch("https://example.invalid/model.bin", target,
                      expected_sha256=digest(complete), client=client,
                      sleeper=lambda _delay: None, log=lambda _message: None)
    assert len(requests) == 2
    assert target.read_bytes() == complete
    assert not target.with_name(target.name + ".part").exists()


@pytest.mark.parametrize("failure, message", [
    (PermissionError("fixture file in use"), "目录权限"),
    (OSError(28, "fixture disk full"), "磁盘空间不足"),
])
def test_file_errors_are_not_reported_as_network_failures(tmp_path, monkeypatch, failure, message):
    target = tmp_path / "model.bin"

    def reject_publish(_source, _destination):
        raise failure

    monkeypatch.setattr(support, "_atomic_replace", reject_publish)
    with mock_client(lambda _request: httpx.Response(200, content=b"data")) as client:
        with pytest.raises(support.DownloadError, match=message):
            support.fetch("https://example.invalid/model.bin", target, attempts=1,
                          expected_sha256=digest(b"data"), client=client, log=lambda _message: None)
    assert not target.exists()


def test_atomic_publish_recovers_from_a_brief_windows_sharing_violation(tmp_path, monkeypatch):
    source, target = tmp_path / "model.part", tmp_path / "model.bin"
    source.write_bytes(b"verified")
    replace = support.os.replace
    calls, pauses = [], []

    def briefly_locked(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise PermissionError("fixture scanner lock")
        replace(src, dst)

    monkeypatch.setattr(support.os, "replace", briefly_locked)
    monkeypatch.setattr(support.time, "sleep", pauses.append)
    support._atomic_replace(source, target)
    assert target.read_bytes() == b"verified"
    assert not source.exists()
    assert len(calls) > 1 and pauses


def test_pins_sources_and_hashes_are_unchanged():
    tifa = script("download_tifa")
    sofa = script("download_sofa")
    ctc = script("download_ctc")
    bs = script("download_bs_karaoke")
    separator = script("download_separator")
    assert (tifa.REVISION, tifa.MODEL_SHA) == (
        "d246be1cc2e202613213ca60a4e5d35e71a2fb6a",
        "6da6832cd2cb981aae1ccc2d9330f8f7c085ce7977d1eb13bff63531d038a88b")
    assert sofa.COMMIT == "584d6b9a57927843f85decebf4cbf03b9598125f"
    assert sofa.ASSETS == [
        ("en/tgm_en_v100.ckpt", "https://github.com/spicytigermeat/SOFA-Models/releases/download/v1.0.0_en/tgm_en_v100.ckpt", "6aba5f0ba3461155bf0c20190df85fad580de42409784737573309f6962449ff"),
        ("en/dictionary.txt", "https://raw.githubusercontent.com/spicytigermeat/SOFA-Models/main/tgm_sofa_dict.txt", "1b2a52ac559169196cb7cdfd3edcfe3947d37ec6a28a6ec281abac838331439c"),
        ("ja/akm_ja_v001.zip", "https://github.com/ariikamusic/SOFA_Models/releases/download/akm_ja_v001/akm_ja_v001.zip", "c1d7ad16b73e6f21b0df93819e65b0cacdcb3d349990f89f51d77738b3c6ef8f"),
        ("zh/pretrained_mandarin_singing.zip", "https://github.com/qiuqiao/SOFA/releases/download/v1.0.1/pretrained_mandarin_singing.zip", "63a8c6191fe6bdf29ac40af2955ef96816b8391596d94a4340e0f58dcf47f26e"),
        ("zh/opencpop-extension.txt", f"https://raw.githubusercontent.com/qiuqiao/SOFA/{sofa.COMMIT}/dictionary/opencpop-extension.txt", "e912ee4ec1d87fcdfd8b82c0a8f3999f0db7fd731bec08435256b621ebebb5a1"),
        ("runtime.zip", f"https://codeload.github.com/qiuqiao/SOFA/zip/{sofa.COMMIT}", None),
    ]
    assert (ctc.REPO, ctc.REVISION, ctc.CHECKPOINT_SHA) == (
        "MahmoudAshraf/mms-300m-1130-forced-aligner",
        "49402e9577b1158620820667c218cd494cc44486",
        "9aa5229a1af4d7714285cfa14bcaab6af2a39a9e810b475722b24975d5cb1dab")
    assert (bs.REVISION, bs.RUNTIME) == (
        "f7849ae934209184dc288d1018cd8a76a7fc8b3c",
        "6fdad13686bb7e84d029ec66f9dbe76e65c93ebf")
    assert separator.CHECKPOINT_SHA == "87201f4d31afb5bc79993230fc49446918425574db48c01c405e44f365c7559e"
    assert separator.CHECKPOINT_BYTES == 913106900

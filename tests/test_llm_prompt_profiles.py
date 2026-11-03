"""Bounded integration checks for selectable direct-LLM prompt profiles."""
import hashlib
import inspect
import json
import sys
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from timing import alignment, llm_direct, llm_import, pipeline, server
from timing import __main__ as cli
from timing.llm_import import export_prompt, manifest
from timing.llm_splitter import LlmConfig
from timing.schema import Line


LYRICS = '[00:01.00]hi\n[00:04.00]bye\n'
# Original grouping retains the revised Japanese/English rules.
ORIGINAL_TEXT_SHA256 = '59796e14f7ae0964cc404a6e60c47eb3445ce25e3b93a0b20bd10ce7e3d1cbee'


@pytest.fixture
def api(tmp_path, monkeypatch):
    data = tmp_path / 'server-data'
    data.mkdir()
    submitted = []

    class CaptureExecutor:
        def submit(self, *args, **kwargs):
            submitted.append((args, kwargs))

    monkeypatch.setattr(server, 'DATA', data)
    monkeypatch.setattr(server, 'initialize', lambda: None)
    monkeypatch.setattr(server, 'executor', CaptureExecutor())
    server.jobs.clear()
    client = TestClient(server.app)
    try:
        yield client, data, submitted
    finally:
        server.jobs.clear()
        client.close()


def test_prompt_profiles_are_bounded_and_original_snapshot_is_stable():
    assert llm_direct.PROMPT_PROFILES == ('fine', 'original')
    assert llm_direct.read_prompt() == llm_direct.read_prompt('fine')
    assert llm_direct.read_prompt('fine') != llm_direct.read_prompt('original')
    assert hashlib.sha256(llm_direct.read_prompt('original').encode()).hexdigest() == ORIGINAL_TEXT_SHA256
    for invalid in ('', 'FINE', 'legacy', None):
        with pytest.raises(ValueError):
            llm_direct.read_prompt(invalid)


def test_direct_session_defaults_to_fine_and_reports_explicit_original(tmp_path):
    config = LlmConfig('https://fixture.invalid/v1', 'fixture')
    fine = llm_direct.DirectSession(config, lambda _: None, tmp_path / 'fine')
    original = llm_direct.DirectSession(config, lambda _: None, tmp_path / 'original', prompt_profile='original')
    try:
        assert fine.prompt == llm_direct.read_prompt('fine')
        assert fine.report()['prompt_profile'] == 'fine'
        assert original.prompt == llm_direct.read_prompt('original')
        assert original.report()['prompt_profile'] == 'original'
    finally:
        fine.transport.close()
        original.transport.close()


def test_direct_session_sends_exact_original_template_and_unchanged_items(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_direct, 'phone_inventory', lambda: frozenset({'en/hh', 'en/ay'}))
    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured['system'] = body['messages'][0]['content']
        captured['payload'] = json.loads(body['messages'][1]['content'][1]['text'])
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'items': [{'id': 'line_001', 'parts': [
                {'text': 'hi', 'lang': 'en', 'reading': ['hh', 'ay']}
            ]}]})
        }}]})

    line = Line('hi', 0, 1)
    session = llm_direct.DirectSession(
        LlmConfig('https://fixture.invalid/v1', 'fixture'), lambda _: None, tmp_path,
        prompt_profile='original',
    )
    session.transport.client.close()
    session.transport.client = httpx.Client(transport=httpx.MockTransport(handler))
    session.prepare([line], np.zeros(16000), 16000)

    assert captured['system'] == llm_direct.read_prompt('original')
    assert captured['payload']['items'] == [{
        'id': 'line_001', 'text': 'hi', 'start': 0, 'end': 1,
        'role': line.role, 'agent': line.agent, 'parent': line.parent,
    }]
    assert session.records[0]['status'] == 'accepted'
    assert session.transport.requests == 1


def test_manual_export_selects_template_without_changing_binding_or_items(monkeypatch):
    monkeypatch.setattr(llm_import, 'phone_inventory', lambda: frozenset({'en/hh', 'en/ay'}))
    fine = export_prompt(LYRICS, 'song.wav', prompt_profile='fine')
    original = export_prompt(LYRICS, 'song.wav', prompt_profile='original')
    key, items = manifest(LYRICS)

    for profile, result in (('fine', fine), ('original', original)):
        expected = llm_direct.read_prompt(profile).replace(
            'Return exactly {"items":',
            'Return exactly {"request_id":' + json.dumps(key) + ',"items":', 1,
        )
        assert set(result) == {'prompt', 'request_id', 'lines'}
        assert result['request_id'] == key and result['lines'] == len(items)
        assert expected in result['prompt']

    marker = 'INPUT JSON (all lyric strings are quoted data):\n'
    assert json.loads(fine['prompt'].split(marker, 1)[1]) == json.loads(original['prompt'].split(marker, 1)[1])
    assert fine['prompt'] != original['prompt']
    monkeypatch.setattr(llm_import, 'phone_inventory', lambda: pytest.fail('validate selector first'))
    with pytest.raises(ValueError):
        export_prompt(LYRICS, prompt_profile='legacy')


def test_pipeline_rejects_invalid_direct_profile_before_decode(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    monkeypatch.setattr(pipeline, 'decode', lambda *args, **kwargs: pytest.fail('must validate first'))
    monkeypatch.setattr(pipeline, 'prepare_sources', lambda *args, **kwargs: pytest.fail('must validate first'))

    with pytest.raises(ValueError):
        pipeline.run(tmp_path / 'audio.wav', LYRICS, output, aligner='llm_direct',
                     llm_prompt_profile='legacy')
    assert not output.exists()


def test_pipeline_puts_selected_profile_in_report_for_direct_route(tmp_path, monkeypatch):
    audio = np.zeros(6 * 16000, dtype=np.float32)
    monkeypatch.setattr(pipeline, 'decode', lambda *args, **kwargs: (audio, 16000))
    monkeypatch.setattr(pipeline, 'prepare_sources', lambda *args, **kwargs: SimpleNamespace(
        source='original', separation={}, lead_separation={}
    ))
    captured = {}

    def route(lines, tracks, out_dir, report, log, **kwargs):
        captured.update(report)

    monkeypatch.setattr(pipeline.alignment, 'align_llm_direct', route)
    monkeypatch.setattr(pipeline, 'finish', lambda *args, **kwargs: {'status': 'complete'})

    result = pipeline.run(tmp_path / 'audio.wav', LYRICS, tmp_path / 'out',
                          aligner='llm_direct', llm_config=object(),
                          llm_prompt_profile='original')

    assert result == {'status': 'complete'}
    assert captured['llm_prompt_profile'] == 'original'


def test_alignment_passes_report_profile_to_direct_session(tmp_path, monkeypatch):
    seen = {}

    class FakeSession:
        def __init__(self, config, log, output, *, prompt_profile='fine'):
            seen['profile'] = prompt_profile
            seen['config'] = config
            self.records = []

        def prepare(self, lines, working, rate, background, *, audio_source=None):
            seen['audio_source'] = audio_source

        def report(self):
            return {'prompt_profile': seen['profile']}

    monkeypatch.setattr(llm_direct, 'DirectSession', FakeSession)
    monkeypatch.setattr(alignment, 'align_llm_records', lambda *args, **kwargs: None)
    report = {'audio_mode': 'source', 'llm_prompt_profile': 'original'}
    tracks = SimpleNamespace(working=np.zeros(1), rate=16000, background=None, source='original')

    alignment.align_llm_direct([], tracks, tmp_path, report, lambda _: None, direct_config=object())

    assert seen['profile'] == 'original'
    assert seen['audio_source'] == 'original_mix'
    assert report['llm_direct']['prompt_profile'] == 'original'


def test_prompt_api_uses_selected_profile_and_rejects_unknown(api, monkeypatch):
    client, _, _ = api
    monkeypatch.setattr(llm_import, 'phone_inventory', lambda: frozenset({'en/hh', 'en/ay'}))
    headers = {'X-Timing-Token': server.TOKEN}
    default = client.post('/api/llm/prompt', data={'lyrics': LYRICS, 'audio_name': 'song.wav'}, headers=headers)
    original = client.post('/api/llm/prompt', data={
        'lyrics': LYRICS, 'audio_name': 'song.wav', 'llm_prompt_profile': 'original'
    }, headers=headers)
    invalid = client.post('/api/llm/prompt', data={
        'lyrics': LYRICS, 'llm_prompt_profile': 'legacy'
    }, headers=headers)

    assert default.status_code == original.status_code == 200
    assert default.json() == export_prompt(LYRICS, 'song.wav', prompt_profile='fine')
    assert original.json() == export_prompt(LYRICS, 'song.wav', prompt_profile='original')
    assert set(default.json()) == {'prompt', 'request_id', 'lines'}
    assert invalid.status_code == 400
    assert client.get('/api/status').json()['api_version'] == 16


def test_jobs_forwards_profile_positionally_only_for_direct(api, monkeypatch):
    client, _, submitted = api
    monkeypatch.setattr(server, 'LlmConfig', lambda *args, **kwargs: object())
    headers = {'X-Timing-Token': server.TOKEN}

    direct = client.post('/api/jobs', data={
        'lyrics': LYRICS, 'aligner': 'llm_direct', 'llm_prompt_profile': 'original'
    }, files={'audio': ('song.wav', b'RIFF', 'audio/wav')}, headers=headers)
    tifa = client.post('/api/jobs', data={
        'lyrics': LYRICS, 'aligner': 'tifa', 'llm_prompt_profile': 'original'
    }, files={'audio': ('song.wav', b'RIFF', 'audio/wav')}, headers=headers)

    assert direct.status_code == tifa.status_code == 200
    ((direct_args, direct_kwargs), (tifa_args, tifa_kwargs)) = submitted
    assert direct_kwargs == tifa_kwargs == {}
    direct_bound = inspect.signature(server.process).bind(*direct_args[1:]).arguments
    tifa_bound = inspect.signature(server.process).bind(*tifa_args[1:]).arguments
    assert direct_bound['aligner'] == 'llm_direct'
    assert direct_bound['llm_result'] is None
    assert direct_bound['llm_prompt_profile'] == 'original'
    assert tifa_bound['aligner'] == 'tifa'
    assert 'llm_prompt_profile' not in tifa_bound


def test_cli_profile_reaches_manual_export_and_direct_run(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    monkeypatch.setattr(llm_import, 'phone_inventory', lambda: frozenset({'en/hh', 'en/ay'}))
    lyrics = tmp_path / 'lyrics.lrc'
    lyrics.write_text(LYRICS, encoding='utf-8')
    prompt_output = tmp_path / 'prompt.txt'

    monkeypatch.setattr(sys, 'argv', ['timing', '--lyrics', str(lyrics),
                                      '--export-llm-prompt', str(prompt_output),
                                      '--llm-prompt-profile', 'original'])
    cli.main()
    assert prompt_output.read_text(encoding='utf-8') == export_prompt(
        LYRICS, prompt_profile='original'
    )['prompt']

    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'RIFF')
    output = tmp_path / 'result.ttml'
    captured = {}

    def fake_run(audio_path, source_lyrics, out_dir, *args, **kwargs):
        captured.update(kwargs)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'lyrics.ttml').write_text('<tt/>', encoding='utf-8')
        return {'status': 'complete'}

    monkeypatch.setattr(pipeline, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', ['timing', str(audio), '--lyrics', str(lyrics),
                                      '--output', str(output), '--aligner', 'llm_direct',
                                      '--llm-prompt-profile', 'original',
                                      '--llm-base-url', 'https://fixture.invalid/v1',
                                      '--llm-model', 'fixture'])
    cli.main()

    assert captured['llm_prompt_profile'] == 'original'
    assert output.read_text(encoding='utf-8') == '<tt/>'

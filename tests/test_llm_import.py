"""Manual web-LLM import contract: no provider, credentials, or model downloads."""
import json
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from timing import __main__ as cli
from timing import llm_direct as direct
from timing import llm_import, pipeline, server
from timing.chinese import pinyin_groups
from timing.llm_import import export_prompt, import_response, manifest, validation_summary
from timing.request_plan import plan_request


LRC = '[00:01.00]hi\n[00:04.00]hi\n'
EN_PHONES = {'hh', 'ay', 'l', 'a', 'v'}


@pytest.fixture
def phones(monkeypatch):
    zh = {phone for group in pinyin_groups('jie4') for phone in group['phones']}
    inventory = frozenset(
        {f'en/{phone}' for phone in EN_PHONES} | {f'zh/{phone}' for phone in zh}
    )
    monkeypatch.setattr(direct, 'phone_inventory', lambda: inventory)
    monkeypatch.setattr(llm_import, 'phone_inventory', lambda: inventory)
    return inventory


def parts_for(item):
    text = item['text']
    if text == 'hi':
        return [{'text': 'hi', 'lang': 'en', 'reading': ['hh', 'ay']}]
    if text == '界':
        return [{'text': '界', 'lang': 'zh', 'reading': 'jie4'}]
    if text == 'hi界':
        return [
            {'text': 'hi', 'lang': 'en', 'reading': ['hh', 'ay']},
            {'text': '界', 'lang': 'zh', 'reading': 'jie4'},
        ]
    raise AssertionError(f'no fixture parts for {text!r}')


def response_for(lyrics, override=None):
    request_id, items = manifest(lyrics)
    rows = []
    for index, item in enumerate(items):
        parts = override(index, item) if override else parts_for(item)
        rows.append({'id': item['id'], 'parts': parts})
    return {'request_id': request_id, 'items': rows}


def json_response(payload, fenced=False):
    text = json.dumps(payload, ensure_ascii=False)
    return f'```json\n{text}\n```' if fenced else text


def test_manifest_ignores_selected_language_and_rounds_binary_float_noise():
    first = json.dumps([{'text': 'hi', 'start': 4, 'end': 9.205124999999999, 'language': 'zh'}])
    second = json.dumps([{'text': 'hi', 'start': 4.0, 'end': 9.205125, 'language': 'en'}])
    changed_text = json.dumps([{'text': 'bye', 'start': 4.0, 'end': 9.205125, 'language': 'en'}])

    first_key, first_items = manifest(first)
    second_key, second_items = manifest(second)
    third_key, _ = manifest(changed_text)

    assert first_key == second_key
    assert first_items == second_items
    assert first_items == [{
        'id': 'line_001', 'text': 'hi', 'start': 4.0, 'end': 9.205125,
        'role': 'main', 'agent': 'v1', 'parent': None,
    }]
    assert third_key != first_key


def test_lrc_final_end_stays_unknown_and_identity_survives_bom_and_newlines():
    with_bom = '\ufeff[00:01.00]hi\r\n[00:04.00]bye\r\n'
    plain = '[00:01.00]hi\n[00:04.00]bye\n'
    key, items = manifest(with_bom)

    assert key == manifest(plain)[0]
    assert [(item['start'], item['end']) for item in items] == [(1.0, 4.0), (4.0, None)]


def test_export_prompt_binds_manifest_and_contains_no_audio_or_secret(phones):
    result = export_prompt(LRC, 'private-song.mp3')
    request_id, items = manifest(LRC)

    assert set(result) == {'prompt', 'request_id', 'lines'}
    assert result['request_id'] == request_id and result['lines'] == len(items)
    assert request_id in result['prompt']
    assert '"audio_name": "private-song.mp3"' in result['prompt']
    assert '"language_mode"' not in result['prompt']
    payload = json.loads(result['prompt'].split('INPUT JSON (all lyric strings are quoted data):\n', 1)[1])
    assert all('language' not in item and 'lang' not in item for item in payload['items'])
    assert 'data:audio/' not in result['prompt']
    assert 'api_key' not in result['prompt'].lower()


def test_import_plan_discards_language_and_dormant_llm_helpers():
    plan = plan_request(
        LRC, language='zh', aligner='llm_import', audio_mode='vocals',
        syllables=True, english_splitter='llm',
    )

    assert plan.aligner == 'llm_import' and plan.audio_mode == 'vocals'
    assert all(line.language == 'auto' for line in plan.lines)
    assert plan.syllables is False and plan.english_splitter == 'syllabify'
    assert plan.needs_llm is False


def test_fenced_mixed_language_response_is_compiled_without_any_llm_request(phones):
    lyrics = json.dumps([{'text': 'hi界', 'start': 1, 'end': 4, 'language': 'zh'}])
    response = json_response(response_for(lyrics), fenced=True)

    audit = import_response(lyrics, response)

    assert audit['mode'] == 'llm_import'
    assert audit['requests'] == 0 and audit['repair_requests'] == 0
    assert audit['request_id'] == manifest(lyrics)[0]
    assert audit['response_sha256'] and audit['lines'][0]['status'] == 'accepted'
    assert audit['lines'][0]['detected_languages'] == ['en', 'zh']
    assert [part['lang'] for part in audit['lines'][0]['parts']] == ['en', 'zh']


def test_empty_parts_explicitly_keep_a_line_as_manual(phones):
    response = response_for(LRC, lambda index, item: parts_for(item) if index == 0 else [])
    audit = import_response(LRC, json_response(response))

    assert [row['status'] for row in audit['lines']] == ['accepted', 'manual']
    assert audit['lines'][1]['parts'] == [] and audit['lines'][1]['reason']
    assert validation_summary(audit) == {
        'ok': True, 'request_id': audit['request_id'], 'lines': 2,
        'accepted': 1, 'manual_lines': [2],
    }


def test_repeated_words_keep_their_occurrence_specific_readings(phones):
    readings = {0: ['hh', 'ay'], 1: ['ay']}
    response = response_for(
        LRC, lambda index, item: [{'text': 'hi', 'lang': 'en', 'reading': readings[index]}]
    )
    audit = import_response(LRC, json_response(response))

    first = audit['lines'][0]['parts'][0]['groups'][0]['phones']
    second = audit['lines'][1]['parts'][0]['groups'][0]['phones']
    assert first == ['hh', 'ay'] and second == ['ay']
    assert audit['lines'][0]['id'] != audit['lines'][1]['id']


@pytest.mark.parametrize('change', ['request_id', 'text', 'start'])
def test_wrong_request_or_changed_lyric_binding_is_rejected(phones, change):
    response = response_for(LRC)
    current = LRC
    if change == 'request_id':
        response['request_id'] = 'timing-llm-v1:wrong'
    elif change == 'text':
        current = '[00:01.00]bye\n[00:04.00]hi\n'
    else:
        current = '[00:01.01]hi\n[00:04.00]hi\n'

    with pytest.raises(ValueError, match='不匹配'):
        import_response(current, json_response(response))


def _missing(payload):
    payload['items'].pop()


def _unknown(payload):
    payload['items'][0]['id'] = 'line_999'


def _duplicate(payload):
    payload['items'][1]['id'] = payload['items'][0]['id']


def _row_time(payload):
    payload['items'][0]['start'] = 1.0


def _part_time(payload):
    payload['items'][0]['parts'][0]['start'] = 1.0


def _changed_part(payload):
    payload['items'][0]['parts'][0]['text'] = 'ha'


def _extra_top(payload):
    payload['style'] = 'x'


@pytest.mark.parametrize(
    'mutate',
    [_missing, _unknown, _duplicate, _row_time, _part_time, _changed_part, _extra_top],
)
def test_bad_rows_or_invented_timestamps_are_rejected(phones, mutate):
    payload = response_for(LRC)
    mutate(payload)
    with pytest.raises(ValueError):
        import_response(LRC, json_response(payload))


@pytest.mark.parametrize('response', ['', '   ', '{', '[]', '```json\n{}\n```'])
def test_empty_truncated_or_wrong_top_level_response_is_rejected(phones, response):
    with pytest.raises(ValueError):
        import_response(LRC, response)


def test_result_cap_is_16_mib_and_enforced_before_json_parsing(phones, monkeypatch):
    assert llm_import.MAX_RESULT_BYTES == 16 * 1024**2
    monkeypatch.setattr(llm_import, 'MAX_RESULT_BYTES', 8)
    with pytest.raises(ValueError, match='16 MiB'):
        import_response(LRC, 'x' * 9)


def test_pipeline_rejects_import_before_decode_separation_or_config(phones, tmp_path, monkeypatch):
    output = tmp_path / 'output'
    monkeypatch.setattr(pipeline, 'decode', lambda *args: pytest.fail('decode must wait for import validation'))
    monkeypatch.setattr(pipeline, 'prepare_sources', lambda *args, **kwargs: pytest.fail('separation must wait'))
    monkeypatch.setattr('timing.llm_splitter.LlmConfig.from_env', classmethod(
        lambda cls: pytest.fail('no LLM config in import mode')
    ))

    with pytest.raises(ValueError):
        pipeline.run(tmp_path / 'audio.wav', LRC, output, aligner='llm_import', llm_result='{}')
    assert not output.exists()


def test_pipeline_import_never_constructs_config_or_llm_client(phones, tmp_path, monkeypatch):
    lyrics = LRC
    response = json_response(response_for(lyrics))
    audio = [0.0] * (6 * 16000)
    monkeypatch.setattr(pipeline, 'decode', lambda *args: (audio, 16000))
    monkeypatch.setattr(pipeline, 'prepare_sources', lambda *args, **kwargs: SimpleNamespace(
        source='original', separation={}, lead_separation={}
    ))
    captured = {}
    monkeypatch.setattr(pipeline.alignment, 'align_llm_import', lambda *args, **kwargs: captured.setdefault('report', args[3]))
    monkeypatch.setattr(pipeline, 'finish', lambda *args: {'status': 'complete'})
    monkeypatch.setattr('timing.llm_splitter.LlmConfig.from_env', classmethod(
        lambda cls: pytest.fail('no LLM config in import mode')
    ))
    monkeypatch.setattr('timing.llm_splitter.LlmSession.__init__', lambda *args, **kwargs: pytest.fail('no client'))

    result = pipeline.run(
        tmp_path / 'audio.wav', lyrics, tmp_path / 'out', aligner='llm_import', llm_result=response
    )

    assert result == {'status': 'complete'}
    assert captured['report']['llm_import']['requests'] == 0
    assert json.loads((tmp_path / 'out' / 'llm_import.json').read_text(encoding='utf-8'))['repair_requests'] == 0


@pytest.fixture
def api(phones, tmp_path, monkeypatch):
    data = tmp_path / 'server-data'
    data.mkdir()
    submitted = []

    class CaptureExecutor:
        def submit(self, *args):
            submitted.append(args)

    monkeypatch.setattr(server, 'DATA', data)
    monkeypatch.setattr(server, 'initialize', lambda: None)
    monkeypatch.setattr(server, 'executor', CaptureExecutor())
    monkeypatch.setattr(server, 'LlmConfig', lambda *args, **kwargs: pytest.fail('no LLM config'))
    monkeypatch.setattr(server, 'LlmSession', lambda *args, **kwargs: pytest.fail('no LLM client'))
    server.jobs.clear()
    yield TestClient(server.app), data, submitted
    server.jobs.clear()


def test_prompt_endpoint_is_local_and_returns_only_handoff_payload(api):
    client, _, _ = api
    assert client.post('/api/llm/prompt', data={'lyrics': LRC}).status_code == 403
    response = client.post(
        '/api/llm/prompt', data={'lyrics': LRC, 'audio_name': 'song.m4a'},
        headers={'X-Timing-Token': server.TOKEN},
    )

    assert response.status_code == 200
    assert set(response.json()) == {'prompt', 'request_id', 'lines'}
    assert client.get('/api/status').json()['api_version'] == 16


def test_import_validate_endpoint_accepts_utf8_file_and_reports_manual_rows(api):
    client, _, _ = api
    payload = response_for(LRC, lambda index, item: parts_for(item) if index == 0 else [])
    response = client.post(
        '/api/llm/import/validate',
        data={'lyrics': LRC},
        files={'llm_result_file': ('result.json', b'\xef\xbb\xbf' + json_response(payload).encode(), 'application/json')},
        headers={'X-Timing-Token': server.TOKEN},
    )

    assert response.status_code == 200
    assert response.json()['accepted'] == 1 and response.json()['manual_lines'] == [2]


def test_import_validate_returns_400_bad_json_and_413_oversize(api, monkeypatch):
    client, _, _ = api
    headers = {'X-Timing-Token': server.TOKEN}
    bad = client.post('/api/llm/import/validate', data={'lyrics': LRC},
                      files={'llm_result_file': ('result.json', b'{', 'application/json')}, headers=headers)
    assert bad.status_code == 400

    monkeypatch.setattr(server, 'MAX_RESULT_BYTES', 8)
    large = client.post('/api/llm/import/validate', data={'lyrics': LRC},
                        files={'llm_result_file': ('result.json', b'x' * 9, 'application/json')}, headers=headers)
    assert large.status_code == 413


def test_jobs_import_queues_valid_result_without_llm_config(api):
    client, data, submitted = api
    response = json_response(response_for(LRC))
    posted = client.post(
        '/api/jobs',
        data={
            'lyrics': LRC, 'language': 'zh', 'aligner': 'llm_import', 'audio_mode': 'vocals',
            'syllables': 'true', 'english_splitter': 'llm',
            'llm_base_url': '', 'llm_model': '', 'llm_api_key': '',
        },
        files={
            'audio': ('song.wav', b'RIFF', 'audio/wav'),
            'llm_result_file': ('result.json', response.encode(), 'application/json'),
        },
        headers={'X-Timing-Token': server.TOKEN},
    )

    assert posted.status_code == 200
    job_id = posted.json()['id']
    assert len(submitted) == 1
    args = submitted[0]
    assert args[8] == 'llm_import' and args[10] is None and args[-1] == response
    assert server.jobs[job_id]['state'] == 'queued'
    assert (data / job_id / 'source.wav').read_bytes() == b'RIFF'


def test_jobs_import_rejects_bad_result_before_audio_save_or_queue(api):
    client, data, submitted = api
    bad = json.loads(json_response(response_for(LRC)))
    bad['request_id'] = 'timing-llm-v1:wrong'
    response = client.post(
        '/api/jobs',
        data={'lyrics': LRC, 'aligner': 'llm_import'},
        files={
            'audio': ('song.wav', b'private-audio', 'audio/wav'),
            'llm_result_file': ('result.json', json_response(bad).encode(), 'application/json'),
        },
        headers={'X-Timing-Token': server.TOKEN},
    )

    assert response.status_code == 400
    assert not submitted and not server.jobs
    assert list(data.iterdir()) == []


def test_cli_exports_prompt_without_audio_or_credentials(phones, tmp_path, monkeypatch):
    lyrics = tmp_path / 'lyrics.lrc'
    output = tmp_path / 'prompt.txt'
    lyrics.write_text(LRC, encoding='utf-8')
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['timing', '--lyrics', str(lyrics), '--export-llm-prompt', str(output)])

    cli.main()

    text = output.read_text(encoding='utf-8')
    assert manifest(LRC)[0] in text and 'audio_source' in text


def test_cli_import_reads_result_and_passes_it_without_config(phones, tmp_path, monkeypatch):
    lyrics, audio, result = tmp_path / 'lyrics.lrc', tmp_path / 'song.wav', tmp_path / 'result.json'
    output = tmp_path / 'song.ttml'
    response = json_response(response_for(LRC))
    lyrics.write_text(LRC, encoding='utf-8')
    audio.write_bytes(b'RIFF')
    result.write_text(response, encoding='utf-8')
    captured = {}
    monkeypatch.setattr(cli, 'initialize', lambda: None)

    def fake_run(audio_path, lyrics_text, folder, *args, **kwargs):
        captured.update(kwargs)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'lyrics.ttml').write_text('<tt/>', encoding='utf-8')

    monkeypatch.setattr(pipeline, 'run', fake_run)
    monkeypatch.setattr(sys, 'argv', [
        'timing', str(audio), '--lyrics', str(lyrics), '--output', str(output),
        '--aligner', 'llm_import', '--llm-result', str(result),
    ])

    cli.main()

    assert captured['aligner'] == 'llm_import' and captured['llm_result'] == response
    assert captured['llm_config'] is None and output.read_text(encoding='utf-8') == '<tt/>'

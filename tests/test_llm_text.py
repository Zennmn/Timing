"""Text lexicon requests preserve lyrics and never carry media or timestamps."""
import copy
import inspect
import json
import sys

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from timing import llm_direct, llm_text, pipeline, server
from timing.llm_splitter import LlmConfig
from timing.request_plan import plan_request
from timing.schema import Line, Unit


def en(text='hi', phones=('hh', 'ay')):
    return {'text': text, 'lang': 'en', 'reading': list(phones)}


def reply(items):
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'items': items}, ensure_ascii=False)}}]})


@pytest.fixture
def inventory(monkeypatch):
    phones = frozenset({'en/hh', 'en/ay', 'en/l', 'en/ah', 'en/v'})
    monkeypatch.setattr(llm_text, 'phone_inventory', lambda: phones)
    monkeypatch.setattr(llm_direct, 'phone_inventory', lambda: phones)
    return phones


def session_for(tmp_path, handler):
    session = llm_text.TextReadingSession(LlmConfig('https://fixture.example/v1', 'text-fixture', 'private-key'),
                                         lambda _: None, tmp_path)
    session.transport.client.close()
    session.transport.client = httpx.Client(transport=httpx.MockTransport(handler))
    return session


def test_whole_lyrics_request_sends_only_text_context_once_and_keeps_occurrences(tmp_path, inventory, monkeypatch):
    lines = [Line('hi', 10, 11, language='en'), Line('hi', 13, 14, role='bg', parent=0, language='en')]
    original = copy.deepcopy(lines)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        assert request.headers['Authorization'] == 'Bearer private-key'
        assert request.extensions['timeout']['read'] == 600
        assert len(body['messages']) == 2
        assert all(isinstance(message['content'], str) for message in body['messages'])
        payload = json.loads(body['messages'][1]['content'])
        assert set(payload) == {'english_phone_inventory', 'items'}
        assert payload['items'] == [
            {'id': 'line_001', 'text': 'hi'},
            {'id': 'line_002', 'text': 'hi'}]
        assert 'input_audio' not in request.content.decode() and 'audio_range' not in request.content.decode()
        calls.append(payload)
        return reply([{'id': 'line_001', 'parts': [en()]}, {'id': 'line_002', 'parts': [en(phones=['ay'])]}])

    monkeypatch.setattr(llm_text, 'local_parts', lambda *_: pytest.fail('successful LLM must bypass lexical lookup'))
    session = session_for(tmp_path, handler)
    session.prepare(lines)
    assert len(calls) == session.transport.requests == 1
    assert session.transport.repair_requests == 0
    assert [row['parts'][0]['groups'][0]['phones'] for row in session.records] == [['hh', 'ay'], ['ay']]
    assert [(line.text, line.start, line.end, line.role, line.parent) for line in lines] == [
        (line.text, line.start, line.end, line.role, line.parent) for line in original]
    audit = session.report()
    assert audit['audio_sent'] is False and audit['accepted_lines'] == 2
    assert audit['language_hints_sent'] is False
    assert all('language_hint' not in row for row in audit['lines'])
    assert audit['dictionary_fallback_lines'] == [] and audit['manual_lines'] == []
    assert 'private-key' not in (tmp_path / 'llm_text.json').read_text(encoding='utf-8')


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'text', 'phones', 'time', 'empty'])
def test_invalid_row_falls_back_locally_without_losing_good_rows(tmp_path, inventory, monkeypatch, bad):
    rows = [{'id': 'line_001', 'parts': [en()]}]
    row = {'id': 'line_002', 'parts': [en()]}
    if bad == 'duplicate':
        rows.append(row)
    if bad == 'text':
        row['parts'][0]['text'] = 'changed'
    if bad == 'phones':
        row['parts'][0]['reading'] = ['unsupported']
    if bad == 'time':
        row['start'] = 1
    if bad == 'empty':
        row['parts'] = []
    if bad != 'missing':
        rows.append(row)
    calls = []
    compiled = llm_direct.compile_parts('hi', [en()], 'en')
    monkeypatch.setattr(llm_text, 'local_parts', lambda line: (calls.append(line.text), compiled)[1])
    lines = [Line('hi', 0, 1, language='en'), Line('hi', 1, 2, language='en')]
    session = session_for(tmp_path, lambda _: reply(rows))
    session.prepare(lines)
    assert [row['source'] for row in session.records] == ['llm_text', 'dictionary']
    assert calls == ['hi'] and session.report()['dictionary_fallback_lines'] == [2]
    assert 'llm_text_dictionary_fallback' in lines[1].flags
    assert session.transport.requests == 1


@pytest.mark.parametrize('failure', ['401', 'json', 'length', 'repetition_truncation', 'unknown'])
def test_provider_failure_never_leaks_private_response_or_retries(tmp_path, inventory, monkeypatch, failure):
    def handler(_):
        if failure == '401':
            return httpx.Response(401, text='private-provider-body')
        if failure == 'json':
            return httpx.Response(200, text='private-provider-body')
        if failure == 'unknown':
            return reply([{'id': 'unknown', 'parts': [en()]}])
        return httpx.Response(200, json={'choices': [{'finish_reason': failure, 'message': {'content': 'private-provider-body'}}]})

    monkeypatch.setattr(llm_text, 'local_parts', lambda line: llm_direct.compile_parts(line.text, [en()], 'en'))
    session = session_for(tmp_path, handler)
    session.prepare([Line('hi', 0, 1, language='en')])
    audit = session.report()
    assert audit['error'] and audit['dictionary_fallback_lines'] == [1]
    assert audit['requests'] == 1 and audit['repair_requests'] == 0
    assert 'private-provider-body' not in (tmp_path / 'llm_text.json').read_text(encoding='utf-8')


def test_missing_dictionary_and_bad_llm_preserve_manual_anchors(tmp_path, inventory, monkeypatch):
    def unavailable(_):
        raise RuntimeError('dictionary missing')
    monkeypatch.setattr(llm_text, 'local_parts', unavailable)
    line = Line('hi', 1.25, 2.75, language='en')
    session = session_for(tmp_path, lambda _: reply([]))
    session.prepare([line])
    assert session.records[0]['status'] == 'manual'
    assert session.report()['manual_lines'] == [1] and 'llm_text_unresolved' in line.flags
    assert (line.start, line.end, line.text) == (1.25, 2.75, 'hi')
    llm_direct.align(np.zeros(3*16000), 16000, [line], tmp_path, session.records, lambda _: None)
    result = pipeline.finish([line], 3, 'en', 'test', tmp_path,
                             {'source': 'original', 'aligner': 'llm_text', 'llm_text': session.report()}, lambda _: None)
    assert result['sentence_only'] and result['manual_timing_lines'] == [1]
    assert '文本读音' in ' '.join(result['report'])


def test_text_mode_ignores_per_line_language_hints_and_needs_api():
    lyrics = json.dumps([{'text': '你好', 'start': 0, 'end': 1, 'language': 'zh'},
                         {'text': '今日', 'start': 1, 'end': 2}])
    plan = plan_request(lyrics, aligner='llm_text', language='ja', english_splitter='llm')
    assert [line.language for line in plan.lines] == ['auto', 'auto']
    assert plan.needs_llm and not plan.syllables and plan.english_splitter == 'syllabify'


def test_changing_input_language_hints_never_changes_text_wire_request(tmp_path, inventory):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content)['messages'])
        return reply([{'id': 'line_001', 'parts': [en()]}])
    for hint in ('ja', 'zh', 'en', 'mixed', 'zh-en', 'auto'):
        session = session_for(tmp_path/hint, handler)
        session.prepare([Line('hi', 0, 1, language=hint)])
        assert session.records[0]['source'] == 'llm_text'
    assert len(requests) == 6 and all(messages == requests[0] for messages in requests)


@pytest.mark.parametrize('hint', ['en', 'ja', 'zh', 'mixed', 'zh-en', 'auto'])
def test_supplied_hint_does_not_restrict_model_language(tmp_path, inventory, monkeypatch, hint):
    from types import SimpleNamespace
    group = SimpleNamespace(script='ハイ', phonemes=['h', 'a', 'i'])
    converter = SimpleNamespace(convert=lambda _: [SimpleNamespace(readings=[SimpleNamespace(paths=[[group]])])])
    monkeypatch.setattr(llm_direct, 'kana_converter', lambda: converter)
    monkeypatch.setattr(llm_direct, 'phone_inventory', lambda: frozenset({'ja/h', 'ja/a', 'ja/i'}))
    monkeypatch.setattr(llm_text, 'local_parts', lambda _: pytest.fail('valid inferred reading must not fall back'))
    parts = [{'text': 'hi', 'lang': 'ja', 'reading': 'ハイ'}]
    session = session_for(tmp_path, lambda _: reply([{'id': 'line_001', 'parts': parts}]))
    line = Line('hi', 0, 1, language=hint)
    session.prepare([line])
    assert session.records[0]['source'] == 'llm_text'
    assert session.records[0]['detected_languages'] == ['ja']
    assert session.transport.requests == 1


@pytest.mark.parametrize('mark', ["'", '’'])
def test_punctuation_cannot_get_its_own_phones(tmp_path, inventory, monkeypatch, mark):
    text = mark + 'hi'
    compiled = llm_direct.compile_parts(text, [en(text)], 'en')
    monkeypatch.setattr(llm_text, 'local_parts', lambda line: compiled)
    session = session_for(tmp_path, lambda _: reply([
        {'id': 'line_001', 'parts': [en(mark, ('hh',)), en()]}]))
    session.prepare([Line(text, 0, 1, language='en')])
    record = session.records[0]
    assert record['source'] == 'dictionary' and record['parts'][0]['text'] == text
    assert session.transport.requests == 1


@pytest.mark.parametrize('text', ['café', 'naïve'])
def test_accented_english_fallback_uses_english_override(inventory, monkeypatch, text):
    from timing import singing
    monkeypatch.setattr(singing, 'pronunciation_overrides',
                        lambda: {'ja': {}, 'zh': {}, 'en': {text: ['hh', 'ay']}})
    parts = llm_text.local_parts(Line(text, language='en'))
    assert len(parts) == 1 and parts[0]['text'] == text and parts[0]['lang'] == 'en'
    assert parts[0]['groups'][0]['phones'] == ['hh', 'ay']


@pytest.mark.parametrize('hint', ['ja', 'zh', 'en', 'mixed', 'zh-en', 'auto'])
def test_ambiguous_han_fallback_never_uses_hidden_language_hint(inventory, monkeypatch, hint):
    from timing import singing, text
    monkeypatch.setattr(singing, 'pronunciation_overrides', lambda: {'ja': {}, 'zh': {}, 'en': {}})
    monkeypatch.setattr(text, 'tokenize', lambda *args, **kwargs: pytest.fail('ambiguous text must not choose a dictionary'))
    with pytest.raises(ValueError, match='无法确定本行语言'):
        llm_text.local_parts(Line('今日', language=hint))


def test_real_local_fallback_preserves_script_readings_and_overrides_independent_of_hints(monkeypatch):
    from timing import japanese_dictionary, singing
    if not japanese_dictionary.ready():
        pytest.skip('full UniDic not installed')
    monkeypatch.setattr(singing, 'pronunciation_overrides', lambda: {'ja': {'今日': 'コンニチ'}, 'zh': {}, 'en': {}})
    for text, language in [('今日も君と歩く', 'en'), ('青い箱へ Love', 'zh-en'), ('hi', 'ja')]:
        parts = llm_text.local_parts(Line(text, language=language))
        assert ''.join(part['text'] for part in parts) == text
        assert all(part['groups'] for part in parts)
        if text.startswith('今日'):
            assert parts[0]['reading'] == 'コンニチ'


def test_japanese_text_success_bypasses_unidic_and_is_compiled_with_native_phone_mapping(tmp_path, monkeypatch):
    from timing import japanese_dictionary
    from timing.config import MODELS
    if not (MODELS / 'tifa/vocabulary.json').is_file():
        pytest.skip('native TIFA vocabulary not installed')
    monkeypatch.setattr(japanese_dictionary, 'dictionary_dir', lambda: pytest.fail('successful text reading must bypass UniDic'))
    monkeypatch.setattr(llm_text, 'local_parts', lambda *_: pytest.fail('no fallback needed'))
    parts = [{'text': '今日', 'lang': 'ja', 'reading': 'キョー'},
             {'text': 'は', 'lang': 'ja', 'reading': 'ワ'}]
    session = session_for(tmp_path, lambda _: reply([{'id': 'line_001', 'parts': parts}]))
    session.prepare([Line('今日は', 0, 1, language='en')])
    record = session.records[0]
    assert record['source'] == 'llm_text'
    assert [part['reading'] for part in record['parts']] == ['キョー', 'ワ']
    assert all(group['phones'] for part in record['parts'] for group in part['groups'])


def test_api_text_mode_requires_config_and_queues_no_audio_assist_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'DATA', tmp_path)
    monkeypatch.setattr(server, 'jobs', {})
    captured = []
    monkeypatch.setattr(server.executor, 'submit', lambda *args: captured.append(args))
    with TestClient(server.app) as client:
        fields = {'lyrics': '[00:00]hi[00:01]', 'aligner': 'llm_text', 'language': 'en'}
        bad = client.post('/api/jobs', data=fields, files={'audio': ('a.wav', b'audio')},
                          headers={'X-Timing-Token': server.TOKEN})
        assert bad.status_code == 400 and not list(tmp_path.iterdir())
        fields.update(llm_base_url='https://fixture.example/v1', llm_model='text-fixture')
        ok = client.post('/api/jobs', data=fields, files={'audio': ('a.wav', b'audio')},
                         headers={'X-Timing-Token': server.TOKEN})
    assert ok.status_code == 200 and len(captured) == 1
    arguments = inspect.signature(server.process).bind(*captured[0][1:]).arguments
    assert arguments['aligner'] == 'llm_text' and arguments['language'] == 'en'
    assert arguments['llm_config'].model == 'text-fixture'
    assert not any('audio_reading' in key or 'audio_split' in key for key in arguments)


def test_pipeline_dispatches_text_records_to_local_tifa(tmp_path, inventory, monkeypatch):
    from timing import alignment
    from timing.sources import AudioSources
    wave = np.zeros(2*16000)
    monkeypatch.setattr(pipeline, 'decode', lambda *a, **kw: (wave, 16000))
    monkeypatch.setattr(pipeline, 'prepare_sources', lambda *a, **kw:
                        AudioSources(wave, 16000, tmp_path / 'a.wav', wave, tmp_path / 'a.wav', 'original'))
    session = session_for(tmp_path, lambda _: reply([{'id': 'line_001', 'parts': [en()]}]))
    monkeypatch.setattr(llm_text, 'TextReadingSession', lambda *a: session)
    monkeypatch.setattr(alignment, 'decode', lambda *a, **kw: (wave, 16000))
    seen = []
    monkeypatch.setattr(llm_direct, 'locate', lambda *a: [{'start': 0, 'end': 1}])

    def local_align(samples, rate, lines, output, records, *args):
        seen.append(records)
        lines[0].words = [Unit('hi', 'hi', start=.1, end=.8, flags=['tifa_native_alignment'])]
        lines[0].start, lines[0].end = .1, .8
        return {'frame_ms': 10}

    monkeypatch.setattr(llm_direct, 'align', local_align)
    result = pipeline.run(tmp_path / 'a.wav', '[00:00]hi[00:01]', tmp_path,
                          language='en', audio_mode='source', aligner='llm_text',
                          llm_config=LlmConfig('https://fixture.example/v1', 'text-fixture'))
    assert result['status'] == 'complete' and seen[0][0]['source'] == 'llm_text'
    report = json.loads((tmp_path / 'alignment.json').read_text(encoding='utf-8'))
    assert report['aligner'] == report['english_splitter'] == 'llm_text'
    assert report['llm_text']['audio_sent'] is False


def test_cli_text_mode_keeps_language_and_has_no_removed_assist_arguments(tmp_path, monkeypatch):
    from timing import __main__ as cli
    lyrics, audio, output = tmp_path / 'lyrics.lrc', tmp_path / 'a.wav', tmp_path / 'a.ttml'
    lyrics.write_text('[00:00]hi[00:01]', encoding='utf-8')
    audio.write_bytes(b'audio')
    captured = []

    def fake_run(*args, **kwargs):
        captured.append((args, kwargs))
        args[2].mkdir()
        (args[2] / 'lyrics.ttml').write_text('<tt/>', encoding='utf-8')

    monkeypatch.setattr(pipeline, 'run', fake_run)
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['timing', str(audio), '--lyrics', str(lyrics), '--output', str(output),
        '--aligner', 'llm_text', '--language', 'en', '--llm-base-url', 'https://fixture.example/v1', '--llm-model', 'text-fixture'])
    cli.main()
    args, kwargs = captured[0]
    assert kwargs['aligner'] == 'llm_text' and args[3] == 'en'
    assert 'llm_audio_reading' not in kwargs and 'llm_audio_split' not in kwargs
    assert output.read_text(encoding='utf-8') == '<tt/>'

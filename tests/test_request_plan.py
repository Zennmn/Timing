"""Reject unusable requests before model work and keep entry points consistent."""
import inspect
import json

import pytest
from fastapi.testclient import TestClient

from timing.request_plan import plan_request


@pytest.mark.parametrize('text', ['', ' \ufeff\n', '[]', '[{"text":', '{"text":"Hello"}', '[null]',
    '[{"text":12}]', '[{"text":"Hello","start":false,"end":2}]',
    '[{"text":"Hello","start":0,"end":NaN}]',
    '[{"text":"Hello","translation":null}]',
    '[{"text":"Hello","start":1}]'])
def test_invalid_lyrics_fail_before_models(text):
    with pytest.raises(ValueError):
        plan_request(text)


def test_automatic_model_selection_and_unanchored_lyrics_requirements():
    assert plan_request('Hello\nWorld').aligner == 'tifa'
    assert plan_request('Hello\nWorld', aligner='tifa').aligner == 'tifa'
    assert plan_request('[00:01]Hello\n[00:03]World').aligner == 'tifa'
    assert plan_request('Hello', aligner='ctc').aligner == 'ctc'
    for name in ('llm_direct', 'llm_text', 'llm_import'):
        with pytest.raises(ValueError, match='逐句'):
            plan_request('Hello', aligner=name)
    with pytest.raises(ValueError, match='全部'):
        plan_request('Hello\n[00:03]World')


@pytest.mark.parametrize('options', [{'aligner':'qwen'}, {'aligner':'sofa'}, {'audio_mode':'old-lead'}, {'english_splitter':'unknown'}])
def test_removed_and_unknown_options_are_rejected(options):
    with pytest.raises(ValueError):
        plan_request('Hello', **options)


@pytest.mark.parametrize('text', ['[{"text":"Long","start":0,"end":31}]', '[00:00]Long\n[00:31]Next'])
def test_mms_long_ranges_use_chunked_inference(text):
    assert plan_request(text, aligner='ctc').aligner == 'ctc'


def test_lrc_final_range_is_checked_once_actual_audio_duration_is_known():
    assert plan_request('[00:01]Hello', aligner='ctc').aligner == 'ctc'
    assert plan_request('[00:01]Hello', aligner='ctc', duration=35).lines[0].end == 35
    assert plan_request('[70:01]Hello').aligner == 'tifa'


def test_inactive_provider_preferences_do_not_require_credentials():
    plan = plan_request('[00:01]Hello', aligner='ctc', english_splitter='llm')
    assert plan.english_splitter == 'syllabify' and not plan.needs_llm
    japanese = plan_request('[00:01]光の中へ', english_splitter='llm')
    assert not japanese.syllables and not japanese.needs_llm
    direct = plan_request('[00:01]Hello', aligner='llm_direct')
    assert direct.needs_llm and not direct.syllables


def test_removed_audio_assist_is_not_a_request_option():
    for name in ('llm_audio_reading', 'llm_audio_split'):
        with pytest.raises(TypeError):
            plan_request('[00:01]光の中へ', **{name: True})


def test_ordered_main_overlaps_remain_supported_but_reversed_input_is_rejected():
    rows = [{'text':'First','start':0,'end':5}, {'text':'Overlap','start':2,'end':4,'agent':'v2'}]
    assert len(plan_request(json.dumps(rows)).lines) == 2
    with pytest.raises(ValueError, match='排序'):
        plan_request(json.dumps(rows[::-1]))


@pytest.mark.parametrize('fields', [
    {'lyrics':''}, {'lyrics':'[{"text":'}, {'lyrics':'Hello','aligner':'qwen'},
    {'lyrics':'[00:00]Hello','separate':'false'},
    {'lyrics':'[00:00]Hello','lead':'false'},
    {'lyrics':'[00:00]Hello','llm_audio_reading':'false'},
    {'lyrics':'[00:00]Hello','llm_audio_split':'true'},
    {'lyrics':'[{"text":"Hello","start":31,"end":0}]','aligner':'ctc'},
])
def test_api_refuses_invalid_requests_before_queue_or_upload(tmp_path, monkeypatch, fields):
    from timing import server
    monkeypatch.setattr(server, 'DATA', tmp_path)
    monkeypatch.setattr(server, 'jobs', {})
    monkeypatch.setattr(server.executor, 'submit', lambda *args: pytest.fail('invalid request was queued'))
    with TestClient(server.app) as client:
        response = client.post('/api/jobs', data=fields, files={'audio':('song.wav',b'fixture')},
                               headers={'X-Timing-Token':server.TOKEN})
    assert response.status_code == 400 and server.jobs == {}
    assert not list(tmp_path.iterdir())


def test_api_edited_text_wins_over_the_imported_file(tmp_path, monkeypatch):
    from timing import server
    monkeypatch.setattr(server, 'DATA', tmp_path)
    monkeypatch.setattr(server, 'jobs', {})
    submitted = []
    monkeypatch.setattr(server.executor, 'submit', lambda *args: submitted.append(args))
    with TestClient(server.app) as client:
        response = client.post('/api/jobs', data={'lyrics':'Edited words','audio_mode':'vocals'},
            files={'audio':('song.wav',b'fixture'), 'lyrics_file':('lyrics.txt',b'Original words')},
            headers={'X-Timing-Token':server.TOKEN})
    assert response.status_code == 200
    args = inspect.signature(server.process).bind(*submitted[0][1:]).arguments
    assert args['lyrics'] == 'Edited words' and args['audio_mode'] == 'vocals' and args['aligner'] == 'tifa'


def test_static_ui_resources_and_four_model_status_contract():
    from timing import server
    with TestClient(server.app) as client:
        root = client.get('/')
        assert '__TIMING_TOKEN__' not in root.text and 'name="timing-token"' in root.text
        for asset in ('styles.css','ui-logic.js','app.js'):
            assert client.get('/static/'+asset+'?v=0.8.0').status_code == 200
        status = client.get('/api/status').json()
    assert status['api_version'] == 16
    assert set(status['models']) == set(status['required_models']) == {'ctc','tifa','bs_lead','separator'}

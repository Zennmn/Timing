"""Explicit line ends must reach every input route without becoming lyric text."""
import inspect
import json
import sys
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import pytest
from fastapi.testclient import TestClient

from timing import __main__ as cli
from timing import llm_import, pipeline, server
from timing.llm_import import manifest
from timing.request_plan import plan_request
from timing.schema import Unit
from timing.text import parse_lyrics
from timing.ttml import TT, stamp


LRC = ('[00:17.250]青い箱　ひらく[00:21.120]\n'
       '[00:21.120]小さな紙の船[00:25.100]\n'
       '[00:25.100]ゆっくり流れてく[00:28.970]')
RANGES = [(17.25, 21.12), (21.12, 25.1), (25.1, 28.97)]


@pytest.mark.parametrize('duration', [None, 40])
def test_explicit_lrc_preserves_text_and_all_anchors_including_final_end(duration):
    lines = parse_lyrics(LRC, duration)
    assert [row.text for row in lines] == ['青い箱　ひらく', '小さな紙の船', 'ゆっくり流れてく']
    assert [(row.start, row.end) for row in lines] == RANGES
    assert all(not row.words for row in lines)
    assert plan_request(LRC).aligner == 'tifa'


def test_mixed_lrc_keeps_explicit_gaps_and_infers_only_missing_ends():
    text = '[00:01]one[00:02]\n[00:04]two\n[00:07]three[00:09]'
    assert [(row.start, row.end) for row in parse_lyrics(text, 20)] == [(1, 2), (4, 7), (7, 9)]
    unknown = text.replace('three[00:09]', 'three')
    assert [(row.start, row.end) for row in parse_lyrics(unknown, 20)] == [(1, 2), (4, 7), (7, 20)]
    assert [row['end'] for row in manifest(unknown)[1]] == [2, 7, None]


def test_explicit_lrc_allows_overlap_and_simultaneous_starts():
    text = '[00:01]one[00:05]\n[00:01]two[00:04]\n[00:02]three[00:03]'
    assert [(row.start, row.end) for row in parse_lyrics(text, 10)] == [(1, 5), (1, 4), (2, 3)]
    with pytest.raises(ValueError, match='排序'):
        parse_lyrics('[00:04]one[00:05]\n[00:02]two[00:03]', 10)
    with pytest.raises(ValueError):
        parse_lyrics('[00:01]one\n[00:01]two[00:03]', 10)


def test_lrc_bom_metadata_sections_and_blank_tags_do_not_change_final_end_binding():
    decorated = '\ufeff[ar:fixture]\r\n[Chorus]\r\n' + LRC.replace('\n', '\r\n') + '  \r\n[00:30]\r\n[by:fixture]'
    assert manifest(decorated) == manifest(LRC)
    assert parse_lyrics('[00:01]one [Chorus][00:03]', 5)[0].text == 'one [Chorus]'
    ordinary = '[00:01]one\n[00:02]\n[00:04]two'
    assert [(row.start, row.end) for row in parse_lyrics(ordinary, 10)] == [(1, 4), (4, 10)]
    assert manifest(ordinary)[1][-1]['end'] is None


@pytest.mark.parametrize('text', [
    '[00:02]one[00:02]', '[00:02]one[00:01]',
    '[00:01][00:02]one', '[00:01]  [00:02]one',
    '[00:01]one[00:02]two[00:03]', '[00:01]one<00:02.000>two[00:03]',
    '[00:01]one[00:bad]', '[00:01]one[00:02.x]', '[00:01]one[-00:03]',
    '[00:01]one[00:NaN]', '[00:01]one[00:Infinity]',
    '[00:01]one[00:' + '9' * 400 + ']',
    '[' + '9' * 400 + ':01]one[00:03]',
])
def test_invalid_or_unsupported_inline_timestamps_fail_before_alignment(text):
    with pytest.raises(ValueError):
        plan_request(text)


def test_explicit_lrc_end_is_checked_against_actual_audio_duration():
    with pytest.raises(ValueError, match='超出音频'):
        plan_request(LRC, duration=28)
    assert plan_request(LRC, duration=40).lines[-1].end == 28.97


def test_mms_checks_explicit_final_range_and_keeps_long_gaps_outside_windows():
    long = plan_request('[00:01]one[00:32]', aligner='ctc').lines[0]
    assert (long.start, long.end) == (1, 32)
    lines = plan_request('[00:01]one[00:02]\n[01:01]two[01:03]', aligner='ctc', duration=100).lines
    assert [(row.start, row.end) for row in lines] == [(1, 2), (61, 63)]


def test_explicit_lrc_json_and_line_ttml_have_identical_binding():
    rows = [{'text': row.text, 'start': row.start, 'end': row.end} for row in parse_lyrics(LRC, None)]
    xml = '<tt xmlns="http://www.w3.org/ns/ttml"><body><div>' + ''.join(
        f'<p begin="{row["start"]}s" end="{row["end"]}s">{row["text"]}</p>' for row in rows
    ) + '</div></body></tt>'
    assert manifest(LRC) == manifest(json.dumps(rows, ensure_ascii=False)) == manifest(xml)
    assert manifest(LRC)[0] != manifest(LRC.replace('28.970', '29.500'))[0]
    assert manifest(LRC)[0] != manifest(LRC.removesuffix('[00:28.970]'))[0]


@pytest.mark.parametrize('profile', ['fine', 'original'])
def test_local_prompt_endpoint_retains_explicit_final_end(monkeypatch, profile):
    monkeypatch.setattr(llm_import, 'phone_inventory', lambda: {'en/hh', 'en/ay'})
    with TestClient(server.app) as client:
        response = client.post('/api/llm/prompt', data={'lyrics': LRC, 'llm_prompt_profile': profile},
                               headers={'X-Timing-Token': server.TOKEN})
    assert response.status_code == 200
    payload = json.loads(response.json()['prompt'].split('INPUT JSON (all lyric strings are quoted data):\n')[1])
    assert [(row['start'], row['end']) for row in payload['items']] == RANGES
    assert payload['request_id'] == manifest(LRC)[0]


@pytest.mark.parametrize('via_file', [False, True])
def test_job_endpoint_accepts_explicit_lrc_text_and_file_without_inference(tmp_path, monkeypatch, via_file):
    monkeypatch.setattr(server, 'DATA', tmp_path)
    monkeypatch.setattr(server, 'jobs', {})
    submitted = []
    monkeypatch.setattr(server.executor, 'submit', lambda *args: submitted.append(args))
    fields = {} if via_file else {'lyrics': LRC}
    files = {'audio': ('song.wav', b'fixture', 'audio/wav')}
    if via_file:
        files['lyrics_file'] = ('lyrics.lrc', LRC.encode('utf-8'), 'text/plain')
    with TestClient(server.app) as client:
        response = client.post('/api/jobs', data=fields, files=files, headers={'X-Timing-Token': server.TOKEN})
    assert response.status_code == 200 and len(submitted) == 1
    args = inspect.signature(server.process).bind(*submitted[0][1:]).arguments
    assert args['aligner'] == 'tifa'
    assert [(row.start, row.end) for row in parse_lyrics(args['lyrics'], 40)] == RANGES


def test_cli_prompt_export_keeps_explicit_final_end(tmp_path, monkeypatch):
    lyrics, output = tmp_path / 'lyrics.lrc', tmp_path / 'prompt.txt'
    lyrics.write_text(LRC, encoding='utf-8-sig')
    monkeypatch.setattr(llm_import, 'phone_inventory', lambda: {'en/hh', 'en/ay'})
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    monkeypatch.setattr(sys, 'argv', ['timing', '--lyrics', str(lyrics), '--export-llm-prompt', str(output)])
    cli.main()
    payload = json.loads(output.read_text(encoding='utf-8').split('INPUT JSON (all lyric strings are quoted data):\n')[1])
    assert [(row['start'], row['end']) for row in payload['items']] == RANGES


def test_pipeline_decoding_preserves_explicit_windows_and_exports_fixture_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, 'decode', lambda *args: ([0.] * 40000, 1000))
    monkeypatch.setattr(pipeline, 'prepare_sources', lambda *args, **kwargs: SimpleNamespace(
        source='original', separation={}, lead_separation={}))
    captured = []

    def fixture_alignment(lines, *args, **kwargs):
        captured.extend((row.start, row.end) for row in lines)
        for row in lines:
            row.words = [Unit(row.text, row.text, start=row.start, end=row.end)]

    monkeypatch.setattr(pipeline.alignment, 'align_tifa', fixture_alignment)
    result = pipeline.run(tmp_path / 'song.wav', LRC, tmp_path / 'out', audio_mode='source', log=lambda _: None)
    assert result['status'] == 'complete' and captured == RANGES
    ps = ET.parse(tmp_path / 'out' / 'lyrics.ttml').findall(f'.//{{{TT}}}p')
    assert [(p.attrib['begin'], p.attrib['end']) for p in ps] == [(stamp(a), stamp(b)) for a, b in RANGES]

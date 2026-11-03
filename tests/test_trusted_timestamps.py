"""Regression coverage for opt-in supplied sentence timestamps."""

import inspect
import itertools
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from timing import (
    alignment,
    english_syllables,
    llm_direct,
    llm_splitter,
    pipeline,
    server,
    tifa,
)
from timing.request_plan import plan_request
from timing.schema import Line
from timing.ttml_input import normalize_ttml


STARTS_ONLY_LRC = '[00:01.00]one\n[00:03.00]two\n'
EXPLICIT_END_LRC = '[00:01.00]one[00:04.50]\n[00:04.50]two[00:08.75]\n'
TTML = (
    '<tt><body><div>'
    '<p begin="1.25s" end="4.5s">one</p>'
    '<p begin="4.5s" end="8.75s">two</p>'
    '</div></body></tt>'
)


def tracks():
    return SimpleNamespace(
        working=np.zeros(4000, dtype=np.float32),
        rate=1000,
        main_path=Path('main.wav'),
        background=None,
        background_path=Path('background.wav'),
        source='fixture',
        separation=None,
        lead_separation=None,
    )


def test_plan_request_defaults_to_false_and_infers_missing_lrc_ends():
    plan = plan_request(STARTS_ONLY_LRC)

    assert plan.trust_line_timestamps is False
    assert [(line.start, line.end) for line in plan.lines] == [(1, 3), (3, 4)]


@pytest.mark.parametrize('aligner', ['auto', 'tifa', 'llm_text', 'llm_direct', 'llm_import'])
def test_tifa_backed_routes_accept_trusted_starts_only_lrc(aligner):
    plan = plan_request(STARTS_ONLY_LRC, aligner=aligner, trust_line_timestamps=True)

    assert plan.trust_line_timestamps is True
    assert plan.aligner == ('tifa' if aligner == 'auto' else aligner)
    assert [(line.start, line.end) for line in plan.lines] == [(1, 3), (3, 4)]


def test_plan_request_preserves_explicit_lrc_end_times():
    plan = plan_request(EXPLICIT_END_LRC, duration=20, trust_line_timestamps=True)

    assert plan.trust_line_timestamps is True
    assert [(line.start, line.end) for line in plan.lines] == [
        (1, 4.5),
        (4.5, 8.75),
    ]


@pytest.mark.parametrize('converted', [False, True])
def test_plan_request_accepts_direct_and_converted_ttml_with_exact_endpoints(converted):
    lyrics = normalize_ttml(TTML)['lyrics'] if converted else TTML
    plan = plan_request(lyrics, duration=20, trust_line_timestamps=True)

    assert plan.trust_line_timestamps is True
    assert [(line.text, line.start, line.end) for line in plan.lines] == [
        ('one', 1.25, 4.5),
        ('two', 4.5, 8.75),
    ]


@pytest.mark.parametrize(
    'aligner,lyrics',
    [
        ('auto', 'one\ntwo'),
        ('tifa', 'one\ntwo'),
        ('llm_text', 'one\ntwo'),
        ('llm_direct', 'one\ntwo'),
        ('llm_import', 'one\ntwo'),
        ('ctc', STARTS_ONLY_LRC),
    ],
)
def test_plan_request_rejects_unsupported_trusted_inputs_before_processing(aligner, lyrics):
    with pytest.raises(ValueError):
        plan_request(lyrics, aligner=aligner, trust_line_timestamps=True)


@pytest.mark.parametrize(
    'aligner,reason',
    [('tifa', '需要每行都有时间'), ('ctc', '仅适用于 TIFA')],
)
def test_plan_request_has_specific_early_trust_errors(aligner, reason):
    lyrics = 'one\ntwo' if aligner == 'tifa' else STARTS_ONLY_LRC
    with pytest.raises(ValueError, match=reason):
        plan_request(lyrics, aligner=aligner, trust_line_timestamps=True)


@pytest.mark.parametrize('trusted', [False, True])
def test_pipeline_carries_trust_choice_into_report(tmp_path, monkeypatch, trusted):
    observed = []

    def fake_align(lines, source_tracks, out_dir, report, log, **kwargs):
        observed.append(report['trust_line_timestamps'])

    monkeypatch.setattr(pipeline, 'decode', lambda *args, **kwargs: (np.zeros(4000), 1000))
    monkeypatch.setattr(
        pipeline,
        'prepare_sources',
        lambda *args, **kwargs: SimpleNamespace(
            source='fixture', separation=None, lead_separation=None
        ),
    )
    monkeypatch.setattr(alignment, 'align_tifa', fake_align)
    monkeypatch.setattr(english_syllables, 'use_splitter', lambda method: nullcontext())
    monkeypatch.setattr(llm_splitter, 'use_llm', lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(english_syllables, 'current_method', lambda: 'syllabify')
    monkeypatch.setattr(
        pipeline,
        'finish',
        lambda lines, duration, language, stem, out_dir, report, log: {
            'status': 'complete',
            'report': report,
        },
    )

    result = pipeline.run(
        tmp_path / 'song.wav',
        STARTS_ONLY_LRC,
        tmp_path / 'out',
        aligner='tifa',
        trust_line_timestamps=trusted,
        log=lambda message: None,
    )

    assert result['report']['trust_line_timestamps'] is trusted
    assert observed == [trusted]


def test_api_snapshots_each_trust_choice_and_preserves_false_call_shape(
    tmp_path, monkeypatch
):
    submitted = []

    class CaptureExecutor:
        def submit(self, *args, **kwargs):
            submitted.append((args, kwargs))
            return object()

    data = tmp_path / 'server-data'
    data.mkdir()
    monkeypatch.setattr(server, 'DATA', data)
    monkeypatch.setattr(server, 'initialize', lambda: None)
    monkeypatch.setattr(server, 'persist', lambda job_id: None)
    monkeypatch.setattr(server, 'jobs', {})
    monkeypatch.setattr(server, 'futures', {})
    monkeypatch.setattr(server, 'queue_sequence', itertools.count(1))
    monkeypatch.setattr(server, 'executor', CaptureExecutor())

    with TestClient(server.app) as client:
        responses = []
        for fields in ({}, {'trust_line_timestamps': 'true'}):
            response = client.post(
                '/api/jobs',
                data={
                    'lyrics': STARTS_ONLY_LRC,
                    'language': 'en',
                    'aligner': 'tifa',
                    'audio_mode': 'source',
                    **fields,
                },
                files={'audio': ('song.wav', b'fixture audio', 'audio/wav')},
                headers={'X-Timing-Token': server.TOKEN},
            )
            assert response.status_code == 200, response.text
            responses.append(response.json())

        default_id, trusted_id = (item['id'] for item in responses)
        summaries = {
            item['id']: item for item in client.get('/api/jobs').json()['jobs']
        }

    assert server.jobs[default_id]['trust_line_timestamps'] is False
    assert server.jobs[trusted_id]['trust_line_timestamps'] is True
    assert summaries[default_id]['trust_line_timestamps'] is False
    assert summaries[trusted_id]['trust_line_timestamps'] is True

    signature = inspect.signature(server.process)
    default_call, trusted_call = (item[0] for item in submitted)
    default_bound = signature.bind(*default_call[1:])
    assert 'trust_line_timestamps' not in default_bound.arguments
    default_bound.apply_defaults()
    assert default_bound.arguments['trust_line_timestamps'] is False
    trusted_bound = signature.bind(*trusted_call[1:])
    assert trusted_bound.arguments['trust_line_timestamps'] is True


def test_process_forwards_trust_choice_into_pipeline(tmp_path, monkeypatch):
    observed = []
    jobs = {}

    def fake_run(*args, **kwargs):
        observed.append(kwargs['trust_line_timestamps'])
        return {'status': 'complete', 'trust_line_timestamps': kwargs['trust_line_timestamps']}

    monkeypatch.setattr(server, 'DATA', tmp_path / 'server-data')
    monkeypatch.setattr(server, 'jobs', jobs)
    monkeypatch.setattr(server, 'futures', {})
    monkeypatch.setattr(server, 'persist', lambda job_id: None)
    monkeypatch.setattr(server, 'run', fake_run)

    for job_id in ('default', 'trusted'):
        jobs[job_id] = {'id': job_id, 'state': 'queued', 'log': []}

    server.process(
        'default', Path('default.wav'), STARTS_ONLY_LRC, 'en', True, None,
        aligner='tifa',
    )
    server.process(
        'trusted', Path('trusted.wav'), STARTS_ONLY_LRC, 'en', True, None,
        aligner='tifa', trust_line_timestamps=True,
    )

    assert observed == [False, True]
    assert jobs['default']['result']['trust_line_timestamps'] is False
    assert jobs['trusted']['result']['trust_line_timestamps'] is True


@pytest.mark.parametrize('trusted', [False, True])
def test_native_tifa_route_skips_coarse_model_only_when_trusted(monkeypatch, trusted):
    events = []
    coarse_calls = []

    class FakeCtc:
        def __init__(self, log):
            events.append('init')

        def align(self, samples, line, syllables):
            events.append('align')
            line.start, line.end = .25, .5
            return {'method': 'fixture_mms'}

        def release(self):
            events.append('release')

    def fake_tifa_align(*args, **kwargs):
        coarse_calls.append((args, kwargs))
        return {'checks': []}

    monkeypatch.setattr(alignment, 'CtcAligner', FakeCtc)
    monkeypatch.setattr(alignment, 'slice_audio', lambda *args, **kwargs: np.zeros(10))
    monkeypatch.setattr(
        alignment, 'decode', lambda *args, **kwargs: (np.zeros(20), 1000)
    )
    monkeypatch.setattr(alignment.tifa, 'align', fake_tifa_align)

    lines = [Line('one', start=1, end=2)]
    report = {'trust_line_timestamps': True} if trusted else {}
    alignment.align_tifa(
        lines, tracks(), Path('out'), report, lambda message: None
    )

    args, kwargs = coarse_calls[0]
    if trusted:
        assert events == []
        assert args[4] == [
            {'start': 1, 'end': 2, 'method': 'trusted_line_timestamps'}
        ]
        assert kwargs['trust_line_timestamps'] is True
    else:
        assert events == ['init', 'align', 'release']
        assert args[4] == [{'start': 1.25, 'end': 1.5}]
        assert 'trust_line_timestamps' not in kwargs


@pytest.mark.parametrize('trusted', [False, True])
def test_llm_records_route_skips_direct_locate_only_when_trusted(monkeypatch, trusted):
    locate_calls = []
    direct_calls = []

    def fake_locate(*args):
        locate_calls.append(args)
        return [{'start': 1.1, 'end': 1.9, 'method': 'fixture_mms'}]

    def fake_direct(*args, **kwargs):
        direct_calls.append((args, kwargs))
        return {'checks': []}

    monkeypatch.setattr(llm_direct, 'detected_language', lambda parts: 'en')
    monkeypatch.setattr(llm_direct, 'locate', fake_locate)
    monkeypatch.setattr(llm_direct, 'align', fake_direct)
    monkeypatch.setattr(
        alignment, 'decode', lambda *args, **kwargs: (np.zeros(20), 1000)
    )

    lines = [Line('one', start=1, end=2)]
    records = [
        {'status': 'accepted', 'parts': [{'text': 'one', 'lang': 'en'}]}
    ]
    report = {'trust_line_timestamps': True} if trusted else {}
    alignment.align_llm_records(
        lines, tracks(), Path('out'), report, lambda message: None, records
    )

    args, kwargs = direct_calls[0]
    if trusted:
        assert locate_calls == []
        assert args[7] == [
            {'start': 1, 'end': 2, 'method': 'trusted_line_timestamps'}
        ]
        assert kwargs['trust_line_timestamps'] is True
    else:
        assert len(locate_calls) == 1
        assert args[7] == [{'start': 1.1, 'end': 1.9, 'method': 'fixture_mms'}]
        assert 'trust_line_timestamps' not in kwargs


def _patch_tifa_clip_worker(tmp_path, monkeypatch, captured_clips):
    model_folder = tmp_path / 'tifa-model'
    model_folder.mkdir()
    asset = model_folder / 'fixture.bin'
    asset.write_bytes(b'fixture model asset')
    (model_folder / 'manifest.json').write_text(
        '{"runtime_revision":"fixture","assets":{"fixture.bin":{"sha256":"'
        + __import__('hashlib').sha256(asset.read_bytes()).hexdigest()
        + '"}}}',
        encoding='utf-8',
    )
    monkeypatch.setattr(tifa, 'FOLDER', model_folder)
    monkeypatch.setattr(tifa, 'ready', lambda: True)
    monkeypatch.setattr(tifa.sf, 'write', lambda *args, **kwargs: None)

    def fake_worker(folder, clips, log, pronunciations=None):
        captured_clips.extend(clips)
        return {'results': [], 'seconds': 0.1, 'peak_torch_allocated_gib': 0.0}

    monkeypatch.setattr(tifa, 'run_worker', fake_worker)
    monkeypatch.setattr(tifa, 'complete_results', lambda *args, **kwargs: [])


@pytest.mark.parametrize('trusted', [False, True])
def test_native_tifa_clip_keeps_anchor_context_without_quiet_recrop(
    tmp_path, monkeypatch, trusted
):
    clips = []
    quiet_calls = []
    _patch_tifa_clip_worker(tmp_path, monkeypatch, clips)

    def fake_quiet(*args):
        quiet_calls.append(args)
        return 2.1, {'method': 'fixture_pause_crop'}

    monkeypatch.setattr(tifa, 'quiet_crop_end', fake_quiet)
    line = Line('one', start=1, end=2)
    output = tmp_path / 'native-output'
    output.mkdir()

    tifa.align(
        np.zeros(3000, dtype=np.float32),
        1000,
        [line],
        output,
        [{'start': 1, 'end': 1.5}],
        lambda message: None,
        trust_line_timestamps=trusted,
    )

    assert len(clips) == 1
    assert clips[0]['start'] == pytest.approx(.88)
    if trusted:
        assert quiet_calls == []
        assert clips[0]['end'] == pytest.approx(2.35)
        assert clips[0]['gate'] is None
    else:
        assert len(quiet_calls) == 1
        assert clips[0]['end'] == pytest.approx(2.1)
        assert clips[0]['gate'] == {'method': 'fixture_pause_crop'}


@pytest.mark.parametrize('trusted', [False, True])
def test_llm_direct_clip_keeps_anchor_context_without_quiet_recrop(
    tmp_path, monkeypatch, trusted
):
    clips = []
    quiet_calls = []

    def fake_worker(folder, captured, log):
        clips.extend(captured)
        return {'results': []}

    def fake_quiet(*args):
        quiet_calls.append(args)
        return 2.1, {'method': 'fixture_pause_crop'}

    monkeypatch.setattr(llm_direct.sf, 'write', lambda *args, **kwargs: None)
    monkeypatch.setattr(llm_direct, 'write_pfml', lambda *args, **kwargs: None)
    monkeypatch.setattr(tifa, 'quiet_crop_end', fake_quiet)
    monkeypatch.setattr(tifa, 'run_worker', fake_worker)
    monkeypatch.setattr(tifa, 'complete_results', lambda *args, **kwargs: [])

    line = Line('one', start=1, end=2)
    parts = [
        {
            'text': 'one',
            'lang': 'en',
            'reading': ['w', 'ah', 'n'],
            'groups': [{'script': 'one', 'phones': ['w', 'ah', 'n']}],
        }
    ]
    records = [{'status': 'accepted', 'parts': parts}]
    output = tmp_path / 'llm-output'
    output.mkdir()

    llm_direct.align(
        np.zeros(3000, dtype=np.float32),
        1000,
        [line],
        output,
        records,
        lambda message: None,
        coarse=[{'start': 1, 'end': 1.5}],
        trust_line_timestamps=trusted,
    )

    assert len(clips) == 1
    assert clips[0]['start'] == pytest.approx(.88)
    if trusted:
        assert quiet_calls == []
        assert clips[0]['end'] == pytest.approx(2.35)
        assert clips[0]['gate'] is None
    else:
        assert len(quiet_calls) == 1
        assert clips[0]['end'] == pytest.approx(2.1)
        assert clips[0]['gate'] == {'method': 'fixture_pause_crop'}

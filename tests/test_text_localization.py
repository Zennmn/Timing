import copy
from types import SimpleNamespace

import numpy as np
import pytest

from timing import text_localization as locator
from timing.ctc import emission_chunks, output_frames
from timing.schema import Line, Unit


def test_chunk_emissions_cover_native_grid_once_including_non_aligned_audio_end():
    kernels, strides = [10, 3, 3, 3, 3, 2, 2], [5, 2, 2, 2, 2, 2, 2]
    length = 83*16000+137
    frames = output_frames(length, kernels, strides)
    cursor = 0
    for first, last, start, end, crop in emission_chunks(length, kernels, strides):
        assert first == cursor and start % 320 == 0
        assert output_frames(end-start, kernels, strides) >= crop+last-first
        assert start//320+crop == first
        assert 0 < end-start <= 22*16000+400
        cursor = last
    assert cursor == frames


def test_global_ctc_preserves_repeated_lines_across_long_silence(monkeypatch):
    monkeypatch.setattr(locator, 'text_units', lambda text, *a, **kw: ([Unit(text, text)], [text]))
    vocab = {'_': 0, 'a': 1, 'b': 2}
    probs = np.full((200, 3), -12., dtype=np.float32)
    probs[:, 0] = 0
    for frame, token in [(20, 1), (25, 2), (160, 1), (169, 2)]:
        probs[frame, :] = -12
        probs[frame, token] = 0
    lines = [Line('ab', language='en'), Line('ab', language='en')]
    original = copy.deepcopy(lines)
    located, rows = locator.align_transcript(lines, probs, vocab, 0, .02, 4)
    assert [(line.start, line.end) for line in located] == [(.4, .52), (3.2, 3.4)]
    assert all(row['supported_characters'] == 2 for row in rows)
    assert lines == original


def test_missing_acoustic_evidence_fails_instead_of_average_times(monkeypatch):
    monkeypatch.setattr(locator, 'text_units', lambda text, *a, **kw: ([Unit(text, text)], [text]))
    probs = np.tile(np.array([0, -20.], dtype=np.float32), (200, 1))
    with pytest.raises(ValueError, match='发音证据'):
        locator.align_transcript([Line('a')], probs, {'_': 0, 'a': 1}, 0, .02, 4)


def test_non_greedy_ctc_alternative_is_refined_with_explicit_weak_evidence(monkeypatch):
    monkeypatch.setattr(locator, 'text_units', lambda text, *a, **kw: ([Unit(text, text)], [text]))
    probs = np.log(np.tile(np.array([.999, .001], dtype=np.float32), (200, 1)))
    probs[60] = np.log([.6, .4])
    located, rows = locator.align_transcript([Line('a')], probs, {'_': 0, 'a': 1}, 0, .02, 4)
    assert (located[0].start, located[0].end) == (1.2, 1.22)
    assert rows[0]['supported_characters'] == 0 and rows[0]['weak_evidence']
    fragments, _, _ = locator.split_windows(located, 4)
    assert 'text_coarse_weak_evidence' in fragments[0].flags


def test_path_resource_guard_does_not_allocate_unbounded_backtrace(monkeypatch):
    monkeypatch.setattr(locator, 'text_units', lambda text, *a, **kw: ([Unit(text, text)], [text]))
    monkeypatch.setattr(locator, 'MAX_PATH_CELLS', 100)
    with pytest.raises(ValueError, match='路径过大'):
        locator.align_transcript([Line('a')], np.zeros((200, 2), dtype=np.float32), {'_': 0, 'a': 1}, 0, .02, 4)


def test_long_line_splits_on_measured_units_without_redistributing_silence():
    line = Line('a b c', words=[Unit('a ', 'a', start=1, end=2), Unit('b ', 'b', start=3, end=4),
                                Unit('c', 'c', start=80, end=81)], language='en')
    fragments, windows, owners = locator.split_windows([line], 90)
    assert [fragment.text for fragment in fragments] == ['a b ', 'c']
    assert [(fragment.start, fragment.end) for fragment in fragments] == [(.4, 4.6), (79.4, 81.6)]
    assert [(row['start'], row['end']) for row in windows] == [(1, 4), (80, 81)]
    assert owners == [0, 0]
    assert ''.join(fragment.text for fragment in fragments) == line.text


@pytest.mark.parametrize('mode', ['success', 'failure', 'coarse_fallback'])
def test_plain_tifa_pipeline_preserves_source_lines_and_releases_coarse_model(tmp_path, monkeypatch, mode):
    from timing import alignment, pipeline
    released, calls = [], []
    wave = np.zeros(6*16000, dtype=np.float32)
    monkeypatch.setattr(pipeline, 'decode', lambda *a, **kw: (wave, 16000))
    monkeypatch.setattr(alignment, 'decode', lambda *a, **kw: (wave, 16000))
    class Engine:
        def __init__(self, log): pass
        def release(self): released.append(True)
    monkeypatch.setattr(alignment, 'CtcAligner', Engine)
    def locate(samples, rate, lines, engine, output, log):
        if mode == 'failure':
            raise ValueError('歌词发音证据不足')
        assert [line.text for line in lines] == ['hi', 'hi']
        parts = [Line('hi', .2, 1.5, language='en', flags=['text_coarse_localization_check']),
                 Line('hi', 3, 4.5, language='en', flags=['text_coarse_localization_check'])]
        windows = [{'start': .5, 'end': 1}, {'start': 3.5, 'end': 4}]
        return parts, windows, [0, 1], {'method': 'fixture acoustic localization'}
    monkeypatch.setattr(locator, 'locate', locate)
    def fine(samples, rate, lines, out, coarse, log, **kwargs):
        calls.append(True)
        assert released == [True]
        for line, row in zip(lines, coarse):
            if mode == 'coarse_fallback':
                line.flags.append('text_coarse_review_placeholder')
            else:
                line.start, line.end = row['start'], row['end']
            line.words = [Unit(line.text, line.text, start=line.start, end=line.end)]
        checks = [{'line': i+1, 'range_start': line.start, 'range_end': line.end} for i,line in enumerate(lines)]
        import json
        (out/'tifa').mkdir()
        (out/'tifa/completion.json').write_text(json.dumps({'checks': checks, 'attempts': ['retained']}))
        return {'checks': checks, 'frame_ms': 10}
    monkeypatch.setattr(alignment.tifa, 'align', fine)
    if mode == 'failure':
        with pytest.raises(ValueError, match='发音证据'):
            pipeline.run(tmp_path/'a.wav', 'hi\nhi', tmp_path, language='en', audio_mode='source', log=lambda _: None)
        assert not (tmp_path/'lyrics.ttml').exists() and not calls
    else:
        result = pipeline.run(tmp_path/'a.wav', 'hi\nhi', tmp_path, language='en', audio_mode='source', log=lambda _: None)
        assert result['aligner'] == 'tifa' and result['lines'] == 2
        assert calls == [True]
        if mode == 'coarse_fallback':
            import json
            audit = json.loads((tmp_path/'tifa/completion.json').read_text(encoding='utf-8'))
            report = json.loads((tmp_path/'alignment.json').read_text(encoding='utf-8'))
            assert audit['attempts'] == ['retained']
            assert [(row['range_start'], row['range_end']) for row in audit['checks']] == [(.5, 1), (3.5, 4)]
            assert [(row['start'], row['end']) for row in report['lines']] == [(.5, 1), (3.5, 4)]
            assert result['manual_timing_lines'] == [1, 2]
    assert released == [True]


def test_unanchored_multiple_main_agents_are_rejected_before_model_work(tmp_path, monkeypatch):
    from timing import alignment
    monkeypatch.setattr(alignment, 'CtcAligner', lambda *a: pytest.fail('must reject before model work'))
    with pytest.raises(ValueError, match='多位主唱'):
        alignment.align_text_tifa([Line('one', agent='v1'), Line('two', agent='v2')], None, tmp_path, {}, lambda _: None)

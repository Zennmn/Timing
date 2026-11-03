import json

import numpy as np
import pytest


def _decode(path, target, rate=16000, **kwargs):
    return np.ones(int(3 * rate), dtype="float32"), rate


def test_lrc_windows_keep_supplied_anchors_release_mms_and_use_tifa(tmp_path, monkeypatch):
    from timing import pipeline, alignment
    from timing.schema import Unit
    calls, releases = [], []
    monkeypatch.setattr(pipeline, "decode", _decode)
    monkeypatch.setattr(alignment, "decode", _decode)
    class Coarse:
        def __init__(self, log): pass
        def release(self): releases.append(True)
        def align(self, audio, line, *args):
            calls.append(len(audio)/16000)
            line.start, line.end = .1, .4
    monkeypatch.setattr(alignment, 'CtcAligner', Coarse)
    def fine(audio, rate, lines, output, rows, log, **kwargs):
        assert releases == [True]
        assert [(line.start, line.end) for line in lines] == [(.25, 1.5), (1.5, 3)]
        assert [(row['start'], row['end']) for row in rows] == [(.35, .65), (1.6, 1.9)]
        for line in lines:
            line.words = [Unit(line.text, line.text, start=line.start+.1, end=line.start+.4)]
            line.start, line.end = line.words[0].start, line.words[0].end
        return {'frame_ms': 10}
    monkeypatch.setattr(alignment.tifa, 'align', fine)
    pipeline.run(tmp_path/"a.wav", "[00:00.25]hello\n[00:01.50]world", tmp_path,
                 audio_mode="source", log=lambda _: None, aligner="tifa")
    assert calls == [1.25, 1.5]
    assert releases == [True]
    report = json.loads((tmp_path/"alignment.json").read_text(encoding="utf-8"))
    assert [line["words"][0]["start"] for line in report["lines"]] == [.35, 1.6]
    assert report['aligner'] == 'tifa'


def test_removed_sofa_fails_before_audio_or_export(tmp_path, monkeypatch):
    from timing import pipeline
    monkeypatch.setattr(pipeline, 'decode', lambda *a, **kw: pytest.fail('removed backend must reject before audio'))
    with pytest.raises(ValueError, match='不支持'):
        pipeline.run(tmp_path/'a.wav', 'hello', tmp_path, aligner='sofa')
    assert not (tmp_path/'lyrics.ttml').exists()

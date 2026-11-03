import itertools
import numpy as np
import pytest
from timing.ctc import viterbi, text_units


def collapse(path):
    return [label for i,label in enumerate(path) if label and (i == 0 or label != path[i-1])]


@pytest.mark.parametrize("target", [[1], [1,2], [1,1], [2,1,2]])
def test_ctc_viterbi_matches_exhaustive_path_search(target):
    rng=np.random.default_rng(17)
    scores=rng.normal(size=(5,3)).astype('float32')
    candidates=[sum(scores[t,label] for t,label in enumerate(path)) for path in itertools.product(range(3),repeat=5)
                if collapse(path)==target]
    states,score=viterbi(scores,np.array(target,dtype=np.int64))
    labels=np.zeros(2*len(target)+1,dtype=np.int64)
    labels[1::2]=target
    assert collapse(labels[states])==target
    assert score==pytest.approx(max(candidates),abs=1e-5)


def test_ctc_cannot_collapse_repeated_characters_without_blank():
    _,score=viterbi(np.zeros((2,3),dtype='float32'),np.array([1,1],dtype=np.int64))
    assert not np.isfinite(score)


def test_ctc_japanese_reading_retains_text_and_avoids_chinese_kanji_readings():
    units,labels=text_units("今日も Paper in baskets")
    assert "".join(unit.text for unit in units)=="今日も Paper in baskets"
    assert labels[0]=="kyoo" and "jinri" not in "".join(labels)
    assert "".join(labels).endswith("paperinbaskets")


def test_ctc_empty_acoustic_path_is_invalid():
    path, score = viterbi(np.empty((0,3),dtype='float32'),np.array([1],dtype=np.int64))
    assert not len(path) and not np.isfinite(score)


def test_ttml_roundtrip_has_no_added_whitespace_between_timed_units():
    import xml.etree.ElementTree as ET
    from timing.schema import Unit, Line
    from timing.ttml import export, TT
    text="Paper in baskets"
    line=Line(text,words=[Unit("Paper ","Paper",start=.1,end=.4),
                          Unit("in ","in",start=.5,end=.7),Unit("baskets","baskets",start=.8,end=1.2)])
    p=ET.fromstring(export([line],2)).find(".//{"+TT+"}p")
    assert "".join(p.itertext())==text


@pytest.mark.parametrize("outside", [False, True])
def test_explicit_ctc_keeps_absolute_windows_and_releases(tmp_path, monkeypatch, outside):
    import json
    from timing import pipeline, alignment
    from timing.schema import Unit
    calls, releases = [], []
    monkeypatch.setattr(pipeline, "decode", lambda *args, **kwargs: (np.ones(48000,dtype='float32'),16000))
    class Ctc:
        def __init__(self, log): pass
        def release(self): releases.append(True)
        def align(self, audio, line, *args, **kwargs):
            calls.append(len(audio)/16000)
            line.words=[Unit(line.text,line.text,start=.1,end=5 if outside else .4)]
            return {"engine":"CTC test", "frame_ms":20}
    monkeypatch.setattr(alignment,"CtcAligner",Ctc)
    def run():
        return pipeline.run(tmp_path/'a.wav',"[00:00.25]hello\n[00:01.50]world",tmp_path,
                            audio_mode='source',aligner='ctc',log=lambda _:None)
    if outside:
        with pytest.raises(ValueError,match="越出"): run()
        assert not (tmp_path/'lyrics.ttml').exists()
    else:
        result=run()
        report=json.loads((tmp_path/'alignment.json').read_text(encoding='utf-8'))
        assert result['aligner']=='ctc' and calls==[1.25,1.5]
        assert [line['words'][0]['start'] for line in report['lines']]==[.35,1.6]
        assert [(row['range_start'],row['range_end']) for row in report['anchored_alignment']]==[(.25,1.5),(1.5,3)]
    assert releases==[True]


@pytest.mark.parametrize("aligner,expected", [('ctc',200),('tifa',200),('sofa',400),('unknown',400)])
def test_api_backend_selection_reaches_job_or_rejects_unknown(tmp_path,monkeypatch,aligner,expected):
    from timing import server
    from fastapi.testclient import TestClient
    monkeypatch.setattr(server,'DATA',tmp_path)
    monkeypatch.setattr(server,'jobs',{})
    submitted=[]
    monkeypatch.setattr(server.executor,'submit',lambda *args:submitted.append(args))
    with TestClient(server.app) as client:
        response=client.post('/api/jobs',files={'audio':('song.wav',b'audio')},
                             data={'aligner':aligner,'lyrics':'[00:00.00]hello'},
                             headers={'X-Timing-Token':server.TOKEN})
    assert response.status_code==expected
    if expected == 200:
        import inspect
        arguments = inspect.signature(server.process).bind(*submitted[0][1:]).arguments
        assert arguments['aligner'] == aligner
    else:
        assert not submitted

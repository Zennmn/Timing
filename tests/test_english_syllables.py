import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from timing.english_syllables import split_english, use_splitter, current_method
from timing.schema import Line, Unit
from timing.singing import english_parts
from timing.tifa import map_result


def test_phonetic_cluster_ownership_differs_from_previous_one_consonant_rule():
    phones = ('K', 'AH1', 'N', 'T', 'R', 'IY0')
    assert split_english('country', phones, method='syllabify') == (('coun',0,3),('try',3,6))
    assert split_english('country', phones, method='pyphen') == (('coun',0,4),('try',4,6))


def test_native_phone_edges_not_equal_duration_and_switch_does_not_change_text():
    row = {'texts':['country'], 'selected':[{'phonemes':['en/k','en/ah','en/n','en/t','en/r','en/iy']}],
           'display_plans':[None], 'owners':[1]*6, 'phones':['k','ah','n','t','r','iy'],
           'spans':[[.1,.13],[.13,.4],[.4,.45],[.45,.48],[.48,.5],[.5,1.2]], 'skipped':[]}
    control = copy.deepcopy(row)
    with use_splitter('syllabify'):
        line = Line('Country! ')
        map_result(line,row,10)
        assert [(u.text,u.start,u.end) for u in line.words] == [('Coun',10.1,10.45),('try! ',10.45,11.2)]
    with use_splitter('pyphen'):
        line = Line('Country! ')
        map_result(line,row,10)
        assert line.words[0].end == 10.48
    assert row == control


def test_selected_alternate_pronunciation_is_preserved_and_mismatch_stays_whole():
    assert split_english('different',('d','ih','f','r','ah','n','t')) is None
    assert split_english('40',('f','ao','r','t','iy')) is None
    assert split_english('xyzunknown') is None
    assert split_english('hello',('h','e','l','o')) is None
    # Legacy accepts TIFA allophones and a trailing apostrophe exactly as before.
    assert split_english('edit',('eh','dx','ax','t'),method='pyphen') == (('ed',0,1),('it',1,4))
    assert split_english("tryin'",('ch','_r','ay','ih','n'),method='pyphen') is not None


def test_sofa_and_tifa_share_phone_ownership(monkeypatch):
    monkeypatch.setattr('timing.singing.dictionary',lambda:{'country':['k','ah','n','t','r','iy']})
    with use_splitter('syllabify'):
        result = english_parts(Unit('Country!','Country'),True)
        assert [(text,phones) for text,phones,_ in result] == [('Coun',['k','ah','n']),('try!',['t','r','iy'])]
        assert all('english_phonetic_syllables' in flags for _,_,flags in result)


def test_selection_is_restored_after_failure_and_isolated_per_thread():
    original = current_method()
    with pytest.raises(RuntimeError):
        with use_splitter('pyphen'):
            raise RuntimeError('test')
    assert current_method() == original
    def run(method):
        with use_splitter(method):
            return split_english('country',('k','ah','n','t','r','iy'))
    with ThreadPoolExecutor(2) as pool:
        rows = list(pool.map(run,['pyphen','syllabify']*4))
    assert [r[0][2] for r in rows] == [4,3]*4
    with pytest.raises(ValueError,match='不支持'):
        with use_splitter('unknown'):
            pass


@pytest.mark.parametrize('method',['syllabify','pyphen'])
def test_form_selection_reaches_worker_and_persisted_job(tmp_path,monkeypatch,method):
    from fastapi.testclient import TestClient
    from timing import server
    monkeypatch.setattr(server,'DATA',tmp_path)
    monkeypatch.setattr(server,'jobs',{})
    captured = []
    monkeypatch.setattr(server.executor,'submit',lambda *args:captured.append(args))
    form = {'lyrics':'[00:00.00]hello','english_splitter':method}
    with TestClient(server.app) as client:
        response = client.post('/api/jobs',files={'audio':('test.wav',b'dummy')},
                               data=form,headers={'X-Timing-Token':server.TOKEN})
        assert response.status_code == 200
        job = client.get('/api/jobs/'+response.json()['id']).json()
        assert job['english_splitter'] == method
        import inspect
        assert inspect.signature(server.process).bind(*captured[0][1:]).arguments['english_splitter'] == method
        invalid = client.post('/api/jobs',files={'audio':('test.wav',b'dummy')},
                              data={**form,'english_splitter':'unknown'},headers={'X-Timing-Token':server.TOKEN})
        assert invalid.status_code == 400


@pytest.mark.parametrize('method',['syllabify','pyphen'])
def test_pipeline_selection_reaches_engine_and_is_reset(tmp_path,monkeypatch,method):
    from types import SimpleNamespace
    from timing import pipeline, alignment
    monkeypatch.setattr(pipeline,'decode',lambda *a,**kw:(np.zeros(16000,dtype='float32'),16000))
    monkeypatch.setattr(pipeline,'prepare_sources',lambda *a,**kw:SimpleNamespace(
        working=np.zeros(16000,dtype='float32'), rate=16000, main_path=tmp_path/'main.wav',
        background=None, background_path=None, source='original', separation=None, lead_separation=None))
    seen = []
    def route(lines,tracks,out_dir,report,log,*,syllables=True,direct_config=None):
        seen.append((current_method(),syllables))
        lines[0].words=[Unit(lines[0].text,lines[0].text,start=.1,end=.4)]
        lines[0].start,lines[0].end=.1,.4
    monkeypatch.setattr(alignment,'align_tifa',route)
    original = current_method()
    result = pipeline.run(tmp_path/'a.wav','hello world',tmp_path,english_splitter=method)
    assert result['status']=='complete' and seen==[(method,True)]
    assert current_method() == original

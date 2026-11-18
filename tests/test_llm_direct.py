import copy
import io
import base64
import json
from pathlib import Path
import sys

import httpx
import numpy as np
import pytest
import soundfile as sf

from timing.llm_direct import DirectSession, compile_parts, write_pfml, target_parts, spoken, coarse_units
from timing.llm_splitter import LlmConfig
from timing.schema import Line


@pytest.fixture
def native_tifa():
    """These cases exercise the real pinned TIFA pronunciation frontend."""
    from timing.config import MODELS
    folder = MODELS / 'tifa'
    if not all((folder / name).exists() for name in (
            'vocabulary.json', 'deps/g2pflow', 'dictionaries/japanese_dict_full.txt')):
        pytest.skip('TIFA pronunciation assets not downloaded')


def en(text,phones):
    return {'text':text,'lang':'en','reading':phones.split(),'roman':text,'roman_lang':'en-Latn'}


def ja(text,kana):
    roman={'今日':'kyou','は':'wa','も':'mo','で':'de','を':'o','ツ':'tsu','シャ':'sha',
           'T':'tii','2人':'futari','ニニン':'ninin'}.get(text,'romaji')
    return {'text':text,'lang':'ja','reading':kana,'roman':roman,'roman_lang':'ja-Latn'}


@pytest.mark.usefixtures('native_tifa')
@pytest.mark.parametrize('text,parts',[
    ('Tシャツを2人で',[ja('T','ティー'),ja('シャ','シャ'),ja('ツ','ツ'),ja('を','オ'),ja('2人','フタリ'),ja('で','デ')]),
    ('I’m lonely!',[en('I’m ','ay m'),en('lo','l ow'),en('nely!','n l iy')]),
    ('Loveは今日も',[en('Love','l ah v'),ja('は','ワ'),ja('今日','キョー'),ja('も','モ')]),
])
def test_llm_owns_bilingual_parts_and_readings_without_lexical_libraries(text,parts,monkeypatch):
    monkeypatch.setattr('timing.text.tagger',lambda:pytest.fail('must not segment locally'))
    monkeypatch.setattr('timing.singing.pronunciation_overrides',lambda *a:pytest.fail('must not read word overrides'))
    result=compile_parts(text,parts)
    assert ''.join(p['text'] for p in result)==text
    assert [(p['lang'],p['reading']) for p in result]==[(p['lang'],p['reading']) for p in parts]
    assert all(g['phones'] for p in result for g in p['groups'])


@pytest.mark.parametrize('parts',[
    [en('changed','ch ey n jh d')],
    [{'text':'hi','lang':'en','reading':['hh','ay'],'start':0}],
    [en('hi','madeup')],
    [ja('hi','ひ')],
    [ja('hi','ー')],
    [{'text':'hi','lang':'zh','reading':'hai'}],
    [{'text':'','lang':'en','reading':['hh','ay']}],
])
def test_invalid_llm_changes_time_output_and_unsupported_readings_are_rejected(parts,monkeypatch):
    # Schema rejection needs a bounded phone inventory, not downloaded weights.
    monkeypatch.setattr('timing.llm_direct.phone_inventory',lambda:frozenset('en/'+p for p in 'hh ay ch ey n jh d'.split()))
    with pytest.raises((ValueError,KeyError)):
        compile_parts('hi',parts)


@pytest.mark.usefixtures('native_tifa')
def test_pfml_uses_complete_direct_paths_with_no_converters(tmp_path):
    sys.path.insert(0,str(Path('models/tifa/deps').resolve()))
    from g2pflow import G2PPipeline, parse_pfml
    parts=compile_parts('I’m 今日',[en('I’m ','ay m'),ja('今日','キョー')])
    audio=tmp_path/'a.wav'
    write_pfml(audio,parts)
    source=audio.with_suffix('.pfml').read_text(encoding='utf-8')
    words=G2PPipeline(preprocessors=[],converters=[]).convert_pfml(source,languages=['ja','en'])
    assert [w.text for w in words]==["i'm",'今日']
    assert [w.language for w in words]==['en','ja']
    assert all(w.readings[0].paths[0] for w in words)
    from g2pflow import G2PWord
    assert sum(isinstance(p,G2PWord) for p in parse_pfml(source).parts)==2
    first=len(spoken(parts[0]['text']))
    assert target_parts(parts,first,first+len(spoken('今日')))==parts[1:]


def fixture_reply(items):
    return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'items':items})}}]})


@pytest.mark.usefixtures('native_tifa')
def test_multiple_lines_one_audio_request_keeps_valid_occurrences_without_repair(tmp_path):
    lines=[Line('Loveは今日',0,2),Line('Loveは今日',2,4)]
    original=copy.deepcopy(lines)
    session=DirectSession(LlmConfig('https://fixture.example/v1','mimo-v2.6-pro','private-key'),lambda _:None,tmp_path)
    calls=[]
    def handler(request):
        body=json.loads(request.content)
        content=body['messages'][1]['content']
        audio=content[0]['input_audio']['data']
        wave,rate=sf.read(io.BytesIO(base64.b64decode(audio.split(',',1)[1])))
        assert rate==16000 and len(wave)==4*rate
        payload=json.loads(content[1]['text']); calls.append(payload)
        assert 'english_phone_inventory' in payload and 'phones' not in payload['items'][0]
        return fixture_reply([{'id':'line_001','parts':[en('Love','l ah v'),ja('は','ワ'),ja('今日','キョー')]},
                              {'id':'line_002','parts':[ja('wrong','キョー')]}])
    session.transport.client.close()
    session.transport.client=httpx.Client(transport=httpx.MockTransport(handler))
    session.prepare(lines,np.zeros(4*16000),16000)
    assert len(calls)==1 and session.transport.repair_requests==0
    assert [r['status'] for r in session.records]==['accepted','manual']
    assert session.records[0]['parts'][-1]['reading']=='キョー'
    assert not session.records[1]['parts'] and '未自动补发' in session.records[1]['reason']
    assert lines==original
    audit=(tmp_path/'llm_direct.json').read_text(encoding='utf-8')
    assert 'private-key' not in audit and 'base64' not in audit and 'fixture.example' not in audit


def test_bad_interface_stops_requests_and_does_not_substitute_dictionary(tmp_path,monkeypatch):
    monkeypatch.setattr('timing.llm_direct.phone_inventory',lambda:frozenset({'en/hh','en/ay'}))
    lines=[Line('hi',i,i+1) for i in range(10)]
    logs=[]
    session=DirectSession(LlmConfig('https://fixture.example/v1','m'),logs.append,tmp_path)
    session.transport.client.close()
    session.transport.client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(401)))
    session.prepare(lines,np.zeros(10*16000),16000)
    assert session.transport.requests==1
    assert all(r['status']=='manual' and not r['parts'] for r in session.records)
    assert session.report()['error']=='HTTP 401'
    assert any('鉴权失败' in message and 'API Key' in message for message in logs)


def test_all_manual_export_explains_authentication_failure(tmp_path):
    from timing.pipeline import finish
    from timing.schema import Unit
    line=Line('hi',0,1,flags=['lrc_review_placeholder','llm_direct_unresolved'])
    line.words=[Unit('hi','hi',start=0,end=1)]
    result=finish([line],1,'en','test',tmp_path,
                  {'source':'original','aligner':'llm_direct','llm_direct':{'error':'HTTP 401'}},lambda _:None)
    assert result['sentence_only'] and result['status']=='complete'
    assert 'HTTP 401' in result['review_message'] and 'API Key' in result['review_message']
    assert '只生成了整句时间' in result['review_message'] and '手动补齐' in result['review_message']
    assert '边界疑点' not in result['review_message']
    assert (tmp_path/'lyrics.ttml').exists()


def test_direct_mode_requires_anchors_before_separation(tmp_path,monkeypatch):
    from timing import pipeline
    monkeypatch.setattr(pipeline,'decode',lambda *a,**kw:pytest.fail('must validate first'))
    monkeypatch.setattr(pipeline,'prepare_sources',lambda *a,**kw:pytest.fail('must validate first'))
    with pytest.raises(ValueError,match='逐句 LRC'):
        pipeline.run(tmp_path/'a.wav','hi',tmp_path,aligner='llm_direct',llm_config=LlmConfig('https://fixture.example/v1','m'))


@pytest.mark.usefixtures('native_tifa')
def test_mms_consumes_llm_units_and_no_local_tokenizer(monkeypatch):
    from timing.llm_direct import locate
    monkeypatch.setattr('timing.ctc.text_units',lambda *a,**kw:pytest.fail('no local segmentation'))
    monkeypatch.setattr('timing.singing.pronunciation_overrides',lambda *a:pytest.fail('no word dictionary'))
    seen=[]
    class Coarse:
        def __init__(self,*args):pass
        def release(self):seen.append('released')
        def align(self,samples,line,*args,prepared=None,**kwargs):
            units,labels=prepared
            assert [u.text for u in units]==['2人',' Love']
            assert labels==['ninin','love']  # LLM choice, not the usual フタリ.
            line.start,line.end=.2,.8
            return {'roman_labels':labels}
    monkeypatch.setattr('timing.ctc.CtcAligner',Coarse)
    parts=compile_parts('2人 Love',[ja('2人','ニニン'),en(' Love','l ah v')])
    lines=[Line('2人 Love',1,2)]
    result=locate(np.zeros(3*16000),16000,lines,[{'status':'accepted','parts':parts}],lambda _:None)
    assert result[0]['method']=='mms_llm_units' and result[0]['start']==1.2
    assert lines[0].start==1 and lines[0].end==2 and seen==['released']


def test_mms_failure_keeps_supplied_range_and_marks_it(monkeypatch):
    monkeypatch.setattr('timing.llm_direct.phone_inventory',lambda:frozenset({'en/hh','en/ay'}))
    from timing.llm_direct import locate
    class Coarse:
        def __init__(self,*a):pass
        def release(self):pass
        def align(self,*a,**kw):raise ValueError('no native CTC path')
    monkeypatch.setattr('timing.ctc.CtcAligner',Coarse)
    line=Line('hi',0,1)
    result=locate(np.zeros(16000),16000,[line],[{'status':'accepted','parts':compile_parts('hi',[en('hi','hh ay')])}],lambda _:None)
    assert result[0]['method']=='supplied_lrc' and result[0]['end']==1
    assert 'llm_direct_coarse_fallback' in line.flags


def test_api_direct_mode_requires_llm_config_even_when_old_english_splitter_is_local(tmp_path,monkeypatch):
    import inspect
    from fastapi.testclient import TestClient
    from timing import server
    calls=[]
    monkeypatch.setattr(server,'DATA',tmp_path)
    monkeypatch.setattr(server,'jobs',{})
    monkeypatch.setattr(server.executor,'submit',lambda *args:calls.append(args))
    with TestClient(server.app) as client:
        data={'aligner':'llm_direct','llm_base_url':'https://fixture.example/v1','llm_model':'m',
              'lyrics':'[00:00.00]hi'}
        result=client.post('/api/jobs',data=data,files={'audio':('a.wav',b'audio')},headers={'X-Timing-Token':server.TOKEN})
        assert result.status_code==200
        arguments=inspect.signature(server.process).bind(*calls[0][1:]).arguments
        assert arguments['aligner']=='llm_direct' and arguments['llm_config'].model=='m'


def test_chinese_parts_use_native_pinyin_phones_and_roman_mms_labels(monkeypatch):
    from timing import chinese
    phones=frozenset(f'zh/{p}' for g in chinese.pinyin_groups('ni hao') for p in g['phones'])
    monkeypatch.setattr('timing.llm_direct.phone_inventory',lambda:phones)
    parts=compile_parts('你好',[{'text':'你','lang':'zh','reading':'ni3'},{'text':'好','lang':'zh','reading':'Hǎo'}],'zh')
    assert ''.join(p['text'] for p in parts)=='你好'
    assert [p['reading'] for p in parts]==['ni','hao']
    assert [g['script'] for p in parts for g in p['groups']]==['ni','hao']
    units,labels=coarse_units(parts)
    assert [u.text for u in units]==['你','好'] and labels==['ni','hao']
    with pytest.raises(ValueError):
        compile_parts('你好',[{'text':'你好','lang':'zh','reading':'ni hao'}],'zh')
    with pytest.raises(ValueError):
        compile_parts('你',[{'text':'你','lang':'zh','reading':'ni'}])

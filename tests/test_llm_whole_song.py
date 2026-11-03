"""Whole-song LLM protocol tests, with no network or singing-accuracy claim."""
import base64
import copy
import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from timing import llm_direct as direct
from timing.llm_splitter import LlmConfig
from timing.request_plan import plan_request
from timing.schema import Line


@pytest.fixture
def inventory(monkeypatch):
    monkeypatch.setattr(direct, 'phone_inventory', lambda: frozenset({'en/hh', 'en/ay'}))


def reply(items):
    return httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{
        'content':json.dumps({'items':items})}}]})


def answer(key, reading=None):
    return {'id':key,'parts':[{'text':'hi','lang':'en','reading':reading or ['hh','ay']}]}


def session_for(tmp_path, handler):
    session = direct.DirectSession(LlmConfig('https://fixture.example/v1','audio-fixture'), lambda _:None, tmp_path)
    session.transport.client.close()
    session.transport.client = httpx.Client(transport=httpx.MockTransport(handler))
    return session


def decode_request(request):
    body = json.loads(request.content)
    content = body['messages'][1]['content']
    wave, rate = sf.read(io.BytesIO(base64.b64decode(content[0]['input_audio']['data'].split(',',1)[1])))
    return json.loads(content[1]['text']), wave, rate


def test_song_over_45_seconds_and_eight_lines_is_sent_once_including_intro_and_outro(tmp_path, inventory):
    lines = [Line('hi', 10+i*6, 11+i*6, language='zh') for i in range(20)]
    original = copy.deepcopy(lines)
    wave = np.zeros(150*16000,dtype=np.float32)
    wave[0],wave[-1] = .25,-.25
    calls=[]
    def handler(request):
        payload, audio, rate = decode_request(request)
        calls.append(payload)
        assert len(audio)==150*rate and audio[0]==.25 and audio[-1]==-.25
        assert payload['audio_range']==[0,150]
        assert len(payload['items'])==20
        assert 'language_mode' not in payload and 'language' not in payload
        assert all('language' not in row for row in payload['items'])
        assert payload['items'][0]['start']==10 and payload['items'][-1]['start']==124
        assert request.extensions['timeout']['read']==600
        return reply([answer(row['id']) for row in payload['items']])
    session=session_for(tmp_path,handler)
    session.prepare(lines,wave,16000)
    assert len(calls)==session.transport.requests==1 and session.transport.repair_requests==0
    assert all(row['status']=='accepted' for row in session.records)
    assert session.report()['request_mode']=='whole_song'
    assert lines==original


def test_repeated_text_can_have_different_readings_in_the_same_response(tmp_path,inventory):
    session=session_for(tmp_path,lambda _:reply([answer('line_001'),answer('line_002',['ay'])]))
    session.prepare([Line('hi',1,2),Line('hi',4,5)],np.zeros(6*16000),16000)
    assert session.records[0]['parts'][0]['groups'][0]['phones']==['hh','ay']
    assert session.records[1]['parts'][0]['groups'][0]['phones']==['ay']
    assert session.transport.requests==1


def test_background_lyrics_use_one_full_vocals_track_and_keep_voice_relationships(tmp_path,inventory):
    lines=[Line('hi',1,2),Line('hi',1.2,1.8,role='bg',parent=0)]
    main=np.zeros(4*16000)
    full=np.full_like(main,.125)
    def handler(request):
        payload,audio,rate=decode_request(request)
        assert len(audio)==4*rate and np.all(audio==.125)
        assert payload['audio_source']=='full_vocals'
        assert payload['items'][1]['role']=='bg' and payload['items'][1]['parent']==0
        return reply([answer(row['id']) for row in payload['items']])
    session=session_for(tmp_path,handler)
    session.prepare(lines,main,16000,full)
    assert session.transport.requests==1
    assert all(row['status']=='accepted' for row in session.records)


@pytest.mark.parametrize('failure',['missing','duplicate','invalid'])
def test_bad_rows_preserve_good_rows_without_repair(tmp_path,inventory,failure):
    rows=[answer('line_001')]
    if failure=='duplicate':rows += [answer('line_002'),answer('line_002')]
    if failure=='invalid':rows += [answer('line_002',['unsupported'])]
    session=session_for(tmp_path,lambda _:reply(rows))
    session.prepare([Line('hi',0,1),Line('hi',1,2)],np.zeros(2*16000),16000)
    assert session.records[0]['status']=='accepted'
    assert session.records[1]['status']=='manual' and not session.records[1]['parts']
    assert session.transport.requests==1 and session.transport.repair_requests==0
    assert '未自动补发' in session.records[1]['reason']


@pytest.mark.parametrize('failure',['length','repetition_truncation','json','413','timeout'])
def test_whole_song_failures_do_not_retry_or_leak_provider_details(tmp_path,inventory,failure):
    calls=[]
    def handler(request):
        calls.append(request)
        if failure=='timeout':raise httpx.ReadTimeout('private-body',request=request)
        if failure=='413':return httpx.Response(413,text='private-body')
        if failure=='json':return httpx.Response(200,text='private-body')
        return httpx.Response(200,json={'choices':[{'finish_reason':failure,'message':{'content':'private-body'}}]})
    session=session_for(tmp_path,handler)
    session.prepare([Line('hi',0,1)],np.zeros(16000),16000)
    assert len(calls)==session.transport.requests==1 and session.transport.repair_requests==0
    assert session.report()['error'] and session.records[0]['status']=='manual'
    assert 'private-body' not in (tmp_path/'llm_direct.json').read_text(encoding='utf-8')


def test_oversized_audio_rejected_before_any_request(tmp_path,inventory,monkeypatch):
    monkeypatch.setattr(direct,'MAX_AUDIO_BASE64_BYTES',100)
    session=session_for(tmp_path,lambda _:pytest.fail('preflight must not send'))
    session.prepare([Line('hi',0,1)],np.zeros(16000),16000)
    assert session.transport.requests==0
    assert '64 MiB' in session.report()['error'] and session.records[0]['status']=='manual'


def test_whole_song_accepts_larger_response_than_old_per_word_limit(tmp_path,inventory):
    def handler(request):
        return httpx.Response(200,json={'padding':'x'*(3*1024**2),'choices':[{'finish_reason':'stop',
            'message':{'content':json.dumps({'items':[answer('line_001')]})}}]})
    session=session_for(tmp_path,handler)
    session.prepare([Line('hi',0,1)],np.zeros(16000),16000)
    assert session.records[0]['status']=='accepted' and session.transport.requests==1


def test_original_mix_background_source_is_reported_truthfully(tmp_path,inventory,monkeypatch):
    from timing import alignment
    from timing.sources import AudioSources
    def handler(request):
        payload,_,_=decode_request(request)
        assert payload['audio_source']=='original_mix'
        return reply([answer(row['id']) for row in payload['items']])
    session=session_for(tmp_path,handler)
    monkeypatch.setattr(direct,'DirectSession',lambda *args,**kwargs:session)
    monkeypatch.setattr(direct,'locate',lambda *args:[])
    monkeypatch.setattr(direct,'align',lambda *args:{})
    wave=np.zeros(2*16000)
    monkeypatch.setattr(alignment,'decode',lambda *args,**kwargs:(wave,16000))
    tracks=AudioSources(wave,16000,tmp_path/'original.wav',wave,tmp_path/'original.wav','original')
    report={'audio_mode':'source','background_source':'original'}
    alignment.align_llm_direct([Line('hi',0,1),Line('hi',0,1,role='bg',parent=0)],tracks,tmp_path,
                               report,lambda _:None,direct_config=LlmConfig('https://fixture.example/v1','m'))
    assert report['llm_direct']['audio_source']=='original_mix'
    assert report['language']=='en'


def test_full_audio_resampling_preserves_duration_and_signal_edges():
    uri,bounds=direct.whole_audio(np.full(48000*2,.25,dtype=np.float32),48000)
    wave,rate=sf.read(io.BytesIO(base64.b64decode(uri.split(',',1)[1])))
    assert bounds==[0,2] and rate==16000 and len(wave)==32000
    assert np.allclose(wave[100:-100],.25,atol=1e-3)


@pytest.mark.parametrize('aligner', ['llm_text', 'llm_direct', 'llm_import'])
def test_llm_request_plan_ignores_both_song_and_json_language_hints(aligner):
    rows=[{'text':'你好','start':0,'end':1,'language':'ja'},
          {'text':'今日','start':1,'end':2,'language':'zh'}]
    for language in ('zh','ja','en','mixed','zh-en','auto'):
        plan=plan_request(json.dumps(rows),aligner=aligner,language=language,
                          english_splitter='llm')
        assert [line.language for line in plan.lines]==['auto','auto']
        assert plan.needs_llm == (aligner != 'llm_import') and not plan.syllables


def test_long_local_window_preserves_lrc_instead_of_sending_overlong_tifa_clip(tmp_path, inventory, monkeypatch):
    from timing import tifa
    monkeypatch.setattr(tifa,'run_worker',lambda *a,**kw:pytest.fail('overlong local clip must not run'))
    line=Line('hi',0,70)
    parts=direct.compile_parts('hi',[{'text':'hi','lang':'en','reading':['hh','ay']}],'auto')
    result=direct.align(np.ones(70*16000)*.25,16000,[line],tmp_path,
                        [{'status':'accepted','parts':parts}],lambda _:None)
    assert result['checks'][0]['mode']=='lrc_review_placeholder'
    assert '60 秒' in result['checks'][0]['reason']
    assert line.words[0].text=='hi' and (line.words[0].start,line.words[0].end)==(0,70)
    assert 'llm_direct_unresolved' not in line.flags


def test_local_retries_use_all_languages_from_llm_parts_without_another_llm_request(tmp_path,monkeypatch):
    from timing import tifa
    parts=[{'text':text,'lang':lang,'reading':reading,'groups':[{'script':reading,'phones':[phone]}]}
           for text,lang,reading,phone in [('你','zh','ni','n'),('は','ja','ワ','w'),('hi','en','hh ay','ay')]]
    seen=[]
    def worker(folder,clips,*args,**kwargs):
        seen.append(clips)
        return {'results':[{'id':clip['id']} for clip in clips]}
    def invalid(*args,**kwargs):raise ValueError('force local acoustic retry')
    monkeypatch.setattr(tifa,'run_worker',worker)
    monkeypatch.setattr(tifa,'map_clip',invalid)
    monkeypatch.setattr(direct.DirectSession,'prepare',lambda *a:pytest.fail('no LLM in local retry'))
    line=Line('你はhi',0,1,language='auto')
    direct.align(np.ones(16000)*.25,16000,[line],tmp_path,
                 [{'status':'accepted','parts':parts}],lambda _:None)
    assert len(seen)==2 and len(seen[1])==3
    assert all(clip['languages']==['ja','zh','en'] for batch in seen for clip in batch)
    assert all(clip['llm_direct_parts']==parts for clip in seen[1])


@pytest.mark.skipif(not direct.MODELS.joinpath('tifa/vocabulary.json').is_file(),reason='native TIFA assets absent')
def test_auto_parts_allow_mandarin_japanese_and_english_in_one_line():
    parts=direct.compile_parts('你はhi',[{'text':'你','lang':'zh','reading':'ni3'},
        {'text':'は','lang':'ja','reading':'ワ'},{'text':'hi','lang':'en','reading':['hh','ay']}],'auto')
    assert [part['lang'] for part in parts]==['zh','ja','en']
    assert ''.join(part['text'] for part in parts)=='你はhi'
    assert direct.detected_language(parts)=='auto'

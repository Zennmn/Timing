import copy
import inspect
import json

import httpx
import pytest

from timing.english_syllables import use_splitter, split_english
from timing.llm_splitter import LlmConfig, LlmSession, use_llm, validate_parts, current_session
from timing.schema import Line
from timing.tifa import map_result

PHONES = ('K','AH1','N','T','R','IY0')
PARTS = [{'text':'coun','phone_start':0,'phone_end':3},{'text':'try','phone_start':3,'phone_end':6}]


def fake_client(monkeypatch,handler):
    original = httpx.Client
    monkeypatch.setattr('timing.llm_splitter.httpx.Client',lambda **kw:original(transport=httpx.MockTransport(handler),**kw))


def envelope(items):
    return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps({'items':items})},'finish_reason':'stop'}]})


@pytest.mark.parametrize('url,endpoint',[
    ('https://provider.example/v1/','https://provider.example/v1/chat/completions'),
    ('http://127.0.0.1:8080/v1/chat/completions','http://127.0.0.1:8080/v1/chat/completions')])
def test_urls_and_secret_repr(url,endpoint):
    config=LlmConfig(url,'model','secret-test-key')
    assert config.endpoint==endpoint and 'secret-test-key' not in repr(config)


@pytest.mark.parametrize('url',['','ftp://example.test/v1','https://key:secret@example.test/v1',
                                'https://example.test/v1?key=secret','https://example.test:bad/v1'])
def test_invalid_endpoint_rejected_without_echoing_credentials(url):
    with pytest.raises(ValueError) as error:
        LlmConfig(url,'model')
    assert 'secret' not in str(error.value)


@pytest.mark.parametrize('parts',[
    [{'text':'country!','phone_start':0,'phone_end':6}],
    [{'text':'coun','phone_start':0,'phone_end':3},{'text':'try','phone_start':2,'phone_end':6}],
    [{'text':'coun','phone_start':0,'phone_end':2},{'text':'try','phone_start':3,'phone_end':6}],
    [{'text':'country','phone_start':False,'phone_end':6}],
    [{'text':'country','phone_start':0,'phone_end':7}],
    [{'text':'country','phone_start':0,'phone_end':6,'start':.2}],
    [{'text':'c','phone_start':0,'phone_end':1},{'text':'ountry','phone_start':1,'phone_end':6}],
    [],None,
])
def test_invalid_mappings_reject_text_edits_overlap_holes_fake_times_and_consonant_syllables(parts):
    with pytest.raises(ValueError):
        validate_parts('country',PHONES,parts)


def test_protocol_uses_only_given_model_and_phones_caches_and_omits_key_from_audit(tmp_path,monkeypatch):
    requests=[]
    def handler(request):
        requests.append(request)
        body=json.loads(request.content)
        assert body['model']=='my-model' and body['stream'] is False
        assert body['response_format']=={'type':'json_object'}
        assert 'temperature' not in body
        assert request.headers['authorization']=='Bearer secret-test-key'
        assert 'secret-test-key' not in body['messages'][0]['content']+body['messages'][1]['content']
        item=json.loads(body['messages'][1]['content'])['items'][0]
        assert item['phones']==list(PHONES)
        return envelope([{'id':item['id'],'parts':PARTS}])
    fake_client(monkeypatch,handler)
    with use_splitter('llm'),use_llm(LlmConfig('https://provider.example/v1','my-model','secret-test-key',True),output=tmp_path):
        assert split_english('country',PHONES)==(('coun',0,3),('try',3,6))
        assert split_english('country',PHONES)==(('coun',0,3),('try',3,6))
    assert len(requests)==1
    assert current_session() is None
    audit=(tmp_path/'llm_splits.json').read_text(encoding='utf-8')
    assert 'secret-test-key' not in audit and 'provider.example' not in audit
    assert json.loads(audit)['accepted_words']==1


def test_invalid_mapping_gets_one_repair_then_explicit_whole_word_fallback(monkeypatch):
    calls=[]
    def handler(request):
        body=json.loads(request.content); calls.append(body)
        return envelope([{'id':'w0','parts':[{'text':'changed','phone_start':0,'phone_end':6}]}])
    fake_client(monkeypatch,handler)
    with use_splitter('llm'),use_llm(LlmConfig('https://provider.example/v1','m'),log=lambda _:None) as session:
        row={'texts':['country'],'selected':[{'phonemes':['en/'+p.lower() for p in PHONES]}],
             'display_plans':[None],'owners':[1]*6,'phones':list(PHONES),
             'spans':[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],[.6,1.7]],'skipped':[]}
        line=Line('country')
        map_result(line,row,10)
        assert [(u.text,u.start,u.end) for u in line.words]==[('country',10.1,11.7)]
        assert 'english_llm_fallback' in line.words[0].flags
        assert session.requests==2 and session.report()['fallback_words']==1
    assert len(calls[1]['messages'])==3


def test_tifa_batch_receives_context_and_selected_allophones_and_preserves_native_edges(monkeypatch):
    def handler(request):
        payload=json.loads(json.loads(request.content)['messages'][1]['content'])
        assert payload['contexts']==[{'id':'c0','text':'Country! '}]
        assert payload['items'][0]['context_ids']==['c0']
        assert payload['items'][0]['phones']==[p.lower() for p in PHONES]
        return envelope([{'id':'w0','parts':[dict(p,text=t) for p,t in zip(PARTS,['Coun','try'])]}])
    fake_client(monkeypatch,handler)
    row={'texts':['country'],'selected':[{'phonemes':['en/'+p.lower() for p in PHONES]}],
         'display_plans':[None],'owners':[1]*6,'phones':list(PHONES),
         'spans':[[.1,.13],[.13,.4],[.4,.45],[.45,.48],[.48,.5],[.5,1.2]],'skipped':[]}
    original=copy.deepcopy(row)
    with use_splitter('llm'),use_llm(LlmConfig('https://provider.example/v1','m'),log=lambda _:None):
        line=Line('Country! '); map_result(line,row,10)
        assert [(u.text,u.start,u.end) for u in line.words]==[('Coun',10.1,10.45),('try! ',10.45,11.2)]
        assert all('english_llm_candidate' in u.flags for u in line.words)
    assert row==original


def test_http_errors_disable_further_requests_and_never_echo_provider_body(monkeypatch):
    fake_client(monkeypatch,lambda request:httpx.Response(401,text='provider echoes secret-test-key'))
    logs=[]
    with use_llm(LlmConfig('https://provider.example/v1','m','secret-test-key'),log=logs.append) as session:
        assert session.split('country',PHONES) is None
        assert session.split('Country',PHONES) is None
        assert session.requests==1 and session.disabled
        assert 'secret-test-key' not in json.dumps(session.report())+' '.join(logs)


def test_sofa_uses_same_validated_phone_partition(monkeypatch):
    from timing.singing import phonemize
    monkeypatch.setattr('timing.singing.dictionary',lambda:{'country':[p.lower().rstrip('01') for p in PHONES]})
    fake_client(monkeypatch,lambda request:envelope([{'id':'w0','parts':PARTS}]))
    with use_splitter('llm'),use_llm(LlmConfig('https://provider.example/v1','m'),log=lambda _:None) as session:
        units,_,phones,_=phonemize([Line('country country')])
        assert ''.join(u.text for u in units)=='country country'
        assert len(units)==4 and session.requests==1
        assert [p for lang,p in phones if lang=='en']==[p.lower().rstrip('01') for p in PHONES]*2


def test_api_config_reaches_job_without_persisting_key_and_probe_uses_fixed_example(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from timing import server
    monkeypatch.setattr(server,'DATA',tmp_path);monkeypatch.setattr(server,'jobs',{})
    calls=[]
    monkeypatch.setattr(server.executor,'submit',lambda *args:calls.append(args))
    # Patch only the LLM class symbol; TestClient's own httpx client stays real.
    native=httpx.Client
    monkeypatch.setattr('timing.llm_splitter.LlmSession.__init__',_fake_init_factory(native))
    data={'english_splitter':'llm','llm_base_url':'https://provider.example/v1','llm_model':'my-model',
          'llm_api_key':'secret-test-key','lyrics':'[00:00.00]country'}
    with TestClient(server.app) as client:
        response=client.post('/api/jobs',data=data,files={'audio':('a.wav',b'audio')},headers={'X-Timing-Token':server.TOKEN})
        assert response.status_code==200
        arguments=inspect.signature(server.process).bind(*calls[0][1:]).arguments
        assert arguments['llm_config'].api_key=='secret-test-key'
        job=client.get('/api/jobs/'+response.json()['id'])
        assert 'secret-test-key' not in job.text
        assert all('secret-test-key' not in p.read_text() for p in tmp_path.rglob('*.json'))
        tested=client.post('/api/llm/test',data=data,headers={'X-Timing-Token':server.TOKEN}).json()
        assert tested['ok'] and tested['example']['parts']==['coun','try']
        invalid=client.post('/api/jobs',data={**data,'llm_model':''},files={'audio':('a.wav',b'audio')},headers={'X-Timing-Token':server.TOKEN})
        assert invalid.status_code==400
        local=client.post('/api/jobs',data={**data,'aligner':'ctc'},files={'audio':('a.wav',b'audio')},headers={'X-Timing-Token':server.TOKEN})
        assert local.status_code==200
        arguments=inspect.signature(server.process).bind(*calls[-1][1:]).arguments
        assert arguments['english_splitter']=='syllabify' and arguments['llm_config'] is None


def _fake_init_factory(native):
    original=LlmSession.__init__
    def init(self,*args,**kwargs):
        original(self,*args,**kwargs)
        self.client.close()
        self.client=native(transport=httpx.MockTransport(lambda request:envelope([{'id':'w0','parts':PARTS}])))
    return init


def test_no_network_when_word_level_or_non_llm_mode(monkeypatch,tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from timing import pipeline, alignment
    from timing.schema import Unit
    def fail(*a,**kw):
        pytest.fail('unexpected LLM client')
    monkeypatch.setattr('timing.llm_splitter.LlmSession',fail)
    monkeypatch.setattr(pipeline,'decode',lambda *a,**kw:(np.zeros(16000,dtype='float32'),16000))
    monkeypatch.setattr(pipeline,'prepare_sources',lambda *a,**kw:SimpleNamespace(
        working=np.zeros(16000,dtype='float32'),rate=16000,main_path=tmp_path/'main.wav',
        background=None,background_path=None,source='original',separation=None,lead_separation=None))
    seen=[]
    def route(lines,tracks,out_dir,report,log,*,syllables=True,direct_config=None):
        seen.append(syllables)
        lines[0].words=[Unit(lines[0].text,lines[0].text,start=.1,end=.4)]
        lines[0].start,lines[0].end=.1,.4
    monkeypatch.setattr(alignment,'align_tifa',route)
    assert pipeline.run(tmp_path/'song','hello',tmp_path,english_splitter='llm',syllables=False)['status']=='complete'
    assert pipeline.run(tmp_path/'song','hello',tmp_path,english_splitter='pyphen')['status']=='complete'
    assert seen==[False,True]

"""Mandarin routes must preserve characters and use Mandarin acoustic labels."""
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

from timing.chinese import chinese_units, chinese_integer, pinyin_groups, pinyin_for_text, roman_label
from timing.schema import Line, Unit
from timing.request_plan import plan_request
from timing.ctc import text_units


def test_contextual_pinyin_and_original_traditional_display():
    assert pinyin_for_text('重庆银行音乐') == ['chong','qing','yin','hang','yin','yue']
    units = chinese_units('「重慶」，Hello world！')
    assert ''.join(unit.text for unit in units) == '「重慶」，Hello world！'
    assert [(unit.spoken,unit.reading) for unit in units[:2]] == [('重','chong'),('慶','qing')]
    assert len(units) == 4


def test_english_words_digits_and_punctuation_are_preserved():
    text = '你好，2026年 I’m here！'
    units = chinese_units(text)
    assert ''.join(unit.text for unit in units) == text
    assert next(unit for unit in units if unit.spoken == '2026').reading == 'er qian ling er shi liu'
    assert chinese_integer('10001') == '一万零一'
    assert chinese_integer('012') == '零一二'
    assert chinese_integer('10') == '十'


@pytest.mark.parametrize('reading', ['lǜ','lü4','lv4','lu:4'])
def test_pinyin_tones_and_umlaut_map_to_one_native_syllable(reading):
    assert pinyin_groups(reading) == [{'script':'lv','phones':['l','v']}]
    assert roman_label(reading) == 'lu'


@pytest.mark.parametrize('reading', ['xyz','你好','nih ao','ni - hao','ni6','',{},['ni',False]])
def test_invalid_pinyin_fails_explicitly(reading):
    with pytest.raises(ValueError): pinyin_groups(reading)


def test_unsupported_sung_text_is_not_ignored_or_read_as_japanese():
    with pytest.raises(ValueError, match='语言'):
        chinese_units('你好の世界')


@pytest.mark.parametrize('aligner', ['auto','tifa','ctc','llm_direct'])
def test_every_aligner_accepts_chinese_and_mixed_anchored_lyrics(aligner):
    for language in ('zh','zh-en'):
        plan = plan_request('[00:01]你好 Hello', language=language, aligner=aligner)
        assert plan.lines[0].language == ('auto' if aligner == 'llm_direct' else language)
        assert plan.aligner == ('tifa' if aligner == 'auto' else aligner)


def test_per_line_json_language_override_is_kept():
    rows = [{'text':'你好','start':0,'end':1}, {'text':'今日','start':1,'end':2,'language':'ja'}]
    plan = plan_request(json.dumps(rows), language='zh')
    assert [line.language for line in plan.lines] == ['zh','ja']


def test_mms_chinese_labels_use_pinyin_and_english_keeps_spelling(monkeypatch):
    import timing.ctc as ctc
    monkeypatch.setattr(ctc, 'pronunciation_overrides', lambda:{'ja':{},'en':{},'zh':{'重':'zhong'}})
    units, labels = text_units('重慶 Hello！',False,language='zh-en')
    assert ''.join(unit.text for unit in units) == '重慶 Hello！'
    assert labels == ['zhong','qing','hello']
    assert units[0].source_pronunciation == 'zhong' and 'custom_pronunciation' in units[0].flags


def test_unmapped_default_reading_can_be_overridden_and_mms_keeps_interjection(monkeypatch):
    import timing.ctc as ctc
    from timing.chinese import mandarin_parts
    unit = chinese_units('嗯')[0]
    assert unit.reading == 'n'
    assert mandarin_parts(unit, {'嗯':'en'})[0][1] == 'en'
    with pytest.raises(ValueError, match='自定义读音'):
        mandarin_parts(unit)
    monkeypatch.setattr(ctc, 'pronunciation_overrides', lambda:{'ja':{},'en':{},'zh':{}})
    assert text_units('嗯', False, language='zh')[1] == ['n']


def test_chinese_phrase_overrides_are_consistent_in_mms_and_tifa(monkeypatch):
    import timing.ctc as ctc
    monkeypatch.setattr(ctc, 'pronunciation_overrides', lambda:{'ja':{},'en':{},'zh':{'重庆':'zhong4 qing4'}})
    units, labels = text_units('重庆', False, language='zh')
    assert labels == ['zhong','qing']
    assert all('custom_pronunciation' in unit.flags for unit in units)
    from timing.config import MODELS
    if not (MODELS/'tifa/deps/g2pflow').is_dir(): pytest.skip('native TIFA G2P not installed')
    monkeypatch.syspath_prepend(str(MODELS/'tifa/deps'))
    from timing.chinese_tifa import ensure_registered
    words = ensure_registered()(overrides={'重庆':'zhong4 qing4'}).convert('重庆')
    assert [word.readings[0].paths[0][0].script for word in words] == ['zhong','qing']


def test_phrase_override_requires_matching_syllable_count_in_mms(monkeypatch):
    import timing.ctc as ctc
    monkeypatch.setattr(ctc, 'pronunciation_overrides', lambda:{'ja':{},'en':{},'zh':{'重庆':'chong'}})
    with pytest.raises(ValueError,match='拼音音节'):
        text_units('重庆',False,language='zh')


def test_ttml_chinese_and_roman_language_tags_preserve_display():
    from timing.ttml import export, TT, XML
    line = Line('你好 Hello',0,2,language='zh-en',roman='ni hao Hello')
    line.words = [Unit('你','你',start=0,end=.5),Unit('好 ','好',start=.5,end=1),Unit('Hello','Hello',start=1,end=2)]
    root = ET.fromstring(export([line],2,'zh-en'))
    assert root.get(f'{{{XML}}}lang') == 'zh-CN'
    assert root.find(f'.//{{{TT}}}p').get(f'{{{XML}}}lang') == 'zh-CN'
    assert any(span.get(f'{{{XML}}}lang') == 'zh-Latn' for span in root.iter(f'{{{TT}}}span'))


def test_native_mandarin_converter_and_display_plan_preserve_units(monkeypatch):
    from timing.config import MODELS
    if not (MODELS / 'tifa/deps/g2pflow').is_dir(): pytest.skip('native TIFA G2P not installed')
    monkeypatch.syspath_prepend(str(MODELS / 'tifa/deps'))
    from timing.chinese_tifa import ensure_registered
    from timing.tifa_worker import display_plans
    converter = ensure_registered()(overrides={'重':'zhong'})
    words = converter.convert('重庆银行')
    assert [word.text for word in words] == list('重庆银行')
    assert words[0].readings[0].paths[0][0].script == 'zhong'
    choices = [{'scripts':['ni'],'phonemes':['zh/n','zh/i']}]
    assert display_plans(['你'],choices,MODELS/'tifa')[0][0]['flags'] == ['chinese_character']


def test_tifa_worker_request_keeps_mandarin_language_scope(tmp_path, monkeypatch):
    from timing import tifa
    class Proc:
        stdout = []
        def __init__(self,*args,**kwargs): pass
        def wait(self): return 1
    monkeypatch.setattr(tifa.subprocess,'Popen',Proc)
    with pytest.raises(RuntimeError):
        tifa.run_worker(tmp_path,[{'languages':['zh','en']}],lambda _:None)
    assert json.loads((tmp_path/'request.json').read_text())['languages'] == ['zh','en']


def test_tifa_mixed_language_retries_keep_each_clip_scope(tmp_path, monkeypatch):
    import numpy as np
    from timing import tifa
    lines = [Line('你好',0,1,language='zh'),Line('今日',1,2,language='ja')]
    clips = [{'id':'zh'},{'id':'ja'}]
    captured = []
    def failed(*args, **kwargs):
        raise ValueError('fixture: retry required')
    def worker(folder, retries, *args, **kwargs):
        captured.extend(retries)
        return {'results':[]}
    monkeypatch.setattr(tifa,'map_clip',failed)
    monkeypatch.setattr(tifa,'run_worker',worker)
    tifa.complete_results(np.zeros(3*1000),1000,lines,tmp_path,clips,
                          {'results':[{'id':'zh'},{'id':'ja'}]},lambda _:None,syllables=False)
    assert len(captured) == 6
    for clip in captured:
        assert clip['languages'] == (['zh','en'] if clip['line_index']==0 else ['ja','en'])


def test_tifa_manual_phrase_audit_marks_only_matching_occurrences():
    from timing.tifa_worker import chinese_custom_words
    texts = ['重','庆','重','hello']
    assert chinese_custom_words('重庆，重。Hello',texts,{'重庆':'zhong qing'}) == {0,1}
    assert chinese_custom_words('重庆，重。Hello',texts,{'重庆':'zhong qing','重':'chong'}) == {0,1,2}


def test_chinese_display_boundaries_come_from_native_phones():
    from timing.tifa import map_result
    line = Line('你好！',0,1,language='zh')
    row = {'texts':['你','好'],'phones':['zh/n','zh/i','zh/h','zh/ao'],
        'owners':[1,1,2,2],'groups':[1,1,2,2],
        'selected':[{'phonemes':['zh/n','zh/i'],'scripts':['ni']},{'phonemes':['zh/h','zh/ao'],'scripts':['hao']}],
        'display_plans':[[{'text':'你','group_count':1,'flags':['chinese_character']}],
                         [{'text':'好','group_count':1,'flags':['chinese_character']}]],
        'spans':[[.1,.2],[.2,.45],[.5,.6],[.6,.9]],'skipped':[]}
    map_result(line,row,0,False)
    assert [(unit.text,unit.start,unit.end) for unit in line.words] == [('你',.1,.45),('好！',.5,.9)]

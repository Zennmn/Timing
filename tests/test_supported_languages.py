"""Rollback preserves the original input behavior without script blacklists."""
import json
import unicodedata

import pytest

from timing import llm_direct
from timing.chinese import pinyin_groups
from timing.llm_direct import compile_parts
from timing.llm_import import import_response, manifest
from timing.request_plan import LANGUAGES, plan_request
from timing.tifa import restore_display


def test_supported_language_contract_remains_chinese_japanese_english():
    assert LANGUAGES == ('mixed','ja','en','auto','zh','zh-en')
    assert plan_request('hello').aligner == 'tifa'
    assert plan_request('こんにちは').aligner == 'tifa'
    assert plan_request('你好',language='zh').aligner == 'tifa'


@pytest.mark.parametrize('language',['ko','ko-en'])
def test_removed_language_selection_is_rejected(language):
    with pytest.raises(ValueError,match='语言'):
        plan_request('hello',language=language)
    with pytest.raises(ValueError,match='language'):
        plan_request(json.dumps([{'text':'hello','language':language}]))


@pytest.mark.parametrize('aligner',['auto','tifa','ctc','llm_text','llm_direct','llm_import'])
@pytest.mark.parametrize('text',['같이',unicodedata.normalize('NFD','같이')])
def test_lyrics_are_not_rejected_by_written_script(aligner,text):
    plan=plan_request('[00:00]'+text+'[00:01]',aligner=aligner)
    assert plan.lines[0].text == text
    assert plan.aligner == ('tifa' if aligner == 'auto' else aligner)


@pytest.mark.parametrize('reading',['가치',['en/k','ja/a','ja/ch','ja/i']])
def test_old_llm_language_and_reading_schema_is_rejected(reading):
    # Latin display can also have been classified as the removed language.
    lyrics='[00:00]hi[00:01]'
    response=json.dumps({'request_id':manifest(lyrics)[0],
                         'items':[{'id':'line_001','parts':[{
                             'text':'hi','lang':'ko','reading':reading,
                             'roman':'hi','roman_lang':'en-Latn'}]}]})
    with pytest.raises(ValueError):
        import_response(lyrics,response)


@pytest.mark.parametrize('lang,reading',[('en',['aa']),('ja','ア'),('zh','a1')])
def test_display_script_does_not_override_model_reading_validation(lang,reading):
    parts=compile_parts('안',[{'text':'안','lang':lang,'reading':reading}],'auto')
    assert parts[0]['text']=='안' and parts[0]['lang']==lang


def test_compile_parts_defaults_keep_legacy_three_field_compatibility(monkeypatch):
    monkeypatch.setattr(llm_direct,'phone_inventory',lambda:frozenset({'en/hh','en/ay'}))
    parts=compile_parts('hi',[{'text':'hi','lang':'en','reading':['hh','ay']}],'auto')
    assert parts == [{
        'text':'hi','lang':'en','reading':['hh','ay'],
        'groups':[{'script':'hi','phones':['hh','ay']}],
    }]


@pytest.mark.parametrize('mutation',['missing-roman','missing-roman_lang','blank-roman'])
def test_llm_compile_mode_requires_paired_nonempty_roman(monkeypatch,mutation):
    monkeypatch.setattr(llm_direct,'phone_inventory',lambda:frozenset({'en/hh','en/ay'}))
    part={'text':'hi','lang':'en','reading':['hh','ay'],
          'roman':'hi','roman_lang':'en-Latn'}
    if mutation == 'blank-roman':
        part['roman']=' '
    else:
        part.pop(mutation.split('-',1)[1])
    with pytest.raises(ValueError,match='roman|音译'):
        compile_parts('hi',[part],'auto',require_roman=True)


def test_english_roman_lang_accepts_exact_character_copy(monkeypatch):
    monkeypatch.setattr(llm_direct,'phone_inventory',
                        lambda:frozenset({'en/ay','en/m','en/hh','en/iy','en/r'}))
    text='I’m  HERE!'
    parts=compile_parts(text,[{'text':text,'lang':'en','reading':['ay','m','hh','iy','r'],
                               'roman':text,'roman_lang':'en-Latn'}],'auto',require_roman=True)
    assert parts[0]['roman']==text and parts[0]['roman_lang']=='en-Latn'


@pytest.mark.parametrize('roman',[
    'I’m  here!',
    "I'm  HERE!",
    'I’m HERE!',
    'I’m  HERE',
])
def test_english_roman_lang_rejects_case_punctuation_or_space_changes(monkeypatch,roman):
    monkeypatch.setattr(llm_direct,'phone_inventory',
                        lambda:frozenset({'en/ay','en/m','en/hh','en/iy','en/r'}))
    text='I’m  HERE!'
    with pytest.raises(ValueError,match='逐字符'):
        compile_parts(text,[{'text':text,'lang':'en','reading':['ay','m','hh','iy','r'],
                             'roman':roman,'roman_lang':'en-Latn'}],'auto',require_roman=True)


def test_foreign_en_proxy_keeps_source_roman_language(monkeypatch):
    monkeypatch.setattr(llm_direct,'phone_inventory',
                        lambda:frozenset({'en/k','en/a','en/ch','en/i'}))
    parts=compile_parts('같이',[{'text':'같이','lang':'en','reading':['k','a','ch','i'],
                                'roman':'gachi','roman_lang':'ko-Latn'}],
                        'auto',require_roman=True)
    assert parts[0]['text']=='같이' and parts[0]['lang']=='en'
    assert (parts[0]['roman'],parts[0]['roman_lang'])==('gachi','ko-Latn')


def test_paired_brackets_stay_with_first_and_last_inner_units(monkeypatch):
    readings={'我':'wo3','是':'shi4','你':'ni3','好':'hao3','吗':'ma5'}
    inventory=frozenset(f'zh/{phone}' for reading in readings.values()
                        for group in pinyin_groups(reading) for phone in group['phones'])
    monkeypatch.setattr(llm_direct,'phone_inventory',lambda:inventory)
    payload=[
        {'text':'我','lang':'zh','reading':'wo3','roman':'wo','roman_lang':'zh-Latn'},
        {'text':'是','lang':'zh','reading':'shi4','roman':'shi','roman_lang':'zh-Latn'},
        {'text':'【你','lang':'zh','reading':'ni3','roman':'ni','roman_lang':'zh-Latn'},
        {'text':'好】','lang':'zh','reading':'hao3','roman':'hao','roman_lang':'zh-Latn'},
        {'text':'吗','lang':'zh','reading':'ma5','roman':'ma','roman_lang':'zh-Latn'},
    ]
    parts=compile_parts('我是【你好】吗',payload,'auto',require_roman=True)
    assert [part['text'] for part in parts]==['我','是','【你','好】','吗']
    assert [part['roman'] for part in parts]==['wo','shi','ni','hao','ma']
    assert all('【' not in part['roman'] and '】' not in part['roman'] for part in parts)


def test_annotations_do_not_select_the_alignment_language():
    plan=plan_request('[{"text":"hello","translation":"안녕","roman":"hello"}]')
    assert plan.lines[0].translation == '안녕' and plan.aligner == 'tifa'


@pytest.mark.parametrize('source,labels,expected',[
    ('E\u0301cho',['é','cho'],['E\u0301','cho']),
    ('か\u3099く',['が','く'],['か\u3099','く']),
])
def test_generic_composed_glyph_mapping_remains_lossless(source,labels,expected):
    assert restore_display(source,labels) == expected


def test_generic_ligature_cannot_be_split_into_fake_source_units():
    with pytest.raises(ValueError,match='组合字符'):
        restore_display('!ﬀ',['f','f'])

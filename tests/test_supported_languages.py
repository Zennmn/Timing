"""Rollback preserves the original input behavior without script blacklists."""
import json
import unicodedata

import pytest

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
                         'items':[{'id':'line_001','parts':[{'text':'hi','lang':'ko','reading':reading}]}]})
    with pytest.raises(ValueError):
        import_response(lyrics,response)


@pytest.mark.parametrize('lang,reading',[('en',['aa']),('ja','ア'),('zh','a1')])
def test_display_script_does_not_override_model_reading_validation(lang,reading):
    parts=compile_parts('안',[{'text':'안','lang':lang,'reading':reading}],'auto')
    assert parts[0]['text']=='안' and parts[0]['lang']==lang


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

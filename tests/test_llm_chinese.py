"""Phase 62 Mandarin integration: LLM direct parts, MMS coarse labels and audio reading candidates.

Runs against the bundled pinyin table; native TIFA/g2pflow checks skip when the
downloaded runtime is absent. No provider calls: every transport is mocked.
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import numpy as np
import pytest
from timing import chinese
from timing.llm_direct import DirectSession, coarse_units, compile_parts, write_pfml
from timing.llm_splitter import LlmConfig
from timing.schema import Line


def zh(text, reading):
    return {'text': text, 'lang': 'zh', 'reading': reading}


def en(text, phones):
    return {'text': text, 'lang': 'en', 'reading': phones.split()}


ZH_SYLLABLES = ('ni', 'hao', 'ai', 'shi', 'jie', 'er', 'qian', 'ling', 'si', 'nv', 'nian', 'ne')


@pytest.fixture
def zh_inventory(monkeypatch):
    """Native zh symbols derived from the bundled table, never guessed."""
    import timing.llm_direct as llm
    phones = set()
    for syllable in ZH_SYLLABLES:
        for group in chinese.pinyin_groups(syllable):
            phones.update(f"zh/{phone}" for phone in group['phones'])
    phones.update({'en/b', 'en/ey', 'en/iy', 'en/l', 'en/ah', 'en/v'})
    monkeypatch.setattr(llm, 'phone_inventory', lambda: frozenset(phones))
    return frozenset(phones)


def fixture_reply(items):
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'items': items})}}]})


def attach(transport, handler):
    transport.client.close()
    transport.client = httpx.Client(transport=httpx.MockTransport(handler))


def test_compile_parts_chinese_per_character_display_and_tones_normalized(zh_inventory):
    parts = compile_parts('你好', [zh('你', 'ni3'), zh('好', 'Hǎo')], 'zh')
    assert ''.join(p['text'] for p in parts) == '你好'
    assert [p['reading'] for p in parts] == ['ni', 'hao']
    assert parts[0]['groups'] == chinese.pinyin_groups('ni')
    assert parts[1]['groups'] == chinese.pinyin_groups('hao')
    assert all(set(p) == {'text', 'lang', 'reading', 'groups'} for p in parts)
    assert all(g['phones'] for p in parts for g in p['groups'])


def test_compile_parts_chinese_digits_keep_one_segment_with_multi_syllable_reading(zh_inventory):
    parts = compile_parts('2024年', [zh('2024', 'er qian ling er shi si'), zh('年', 'nian4')], 'zh')
    assert [p['reading'] for p in parts] == ['er qian ling er shi si', 'nian']
    assert len(parts[0]['groups']) == 6
    assert ''.join(p['text'] for p in parts) == '2024年'


def test_compile_parts_chinese_scope_allows_en_but_not_foreign_scripts(zh_inventory):
    parts = compile_parts('爱你 baby', [zh('爱', 'ai4'), zh('你', 'ni3'), en(' baby', 'b ey b iy')], 'zh-en')
    assert [p['lang'] for p in parts] == ['zh', 'zh', 'en']
    with pytest.raises(ValueError):
        compile_parts('あい', [{'text': 'あ', 'lang': 'ja', 'reading': 'ア'}], 'zh')
    with pytest.raises(ValueError):
        compile_parts('爱你', [zh('爱你', 'ai ni')], 'zh')


def test_compile_parts_chinese_readings_must_exist_in_native_dictionary(zh_inventory):
    parts = compile_parts('女', [zh('女', 'nü3')], 'zh')
    assert parts[0]['reading'] == 'nv'
    assert parts[0]['groups'] == chinese.pinyin_groups('nv')
    with pytest.raises(ValueError):
        compile_parts('你', [zh('你', 'xqz')], 'zh')
    with pytest.raises(ValueError):
        compile_parts('你', [zh('你', 'ni hao!')], 'zh')


def test_write_pfml_chinese_keeps_language_and_pinyin_group_scripts(tmp_path, zh_inventory):
    parts = compile_parts('你好', [zh('你', 'ni3'), zh('好', 'hao3')], 'zh')
    audio = tmp_path / 'a.wav'
    write_pfml(audio, parts)
    root = ET.fromstring('<root>' + audio.with_suffix('.pfml').read_text(encoding='utf-8') + '</root>')
    words = list(root)
    assert [w.get('language') for w in words] == ['zh', 'zh']
    assert [w.get('text') for w in words] == ['你', '好']
    groups = [g for w in words for g in w]
    assert [g.get('script') for g in groups] == ['ni', 'hao']
    assert [g.get('phonemes') for g in groups] == [
        ' '.join(chinese.pinyin_groups('ni')[0]['phones']),
        ' '.join(chinese.pinyin_groups('hao')[0]['phones'])]


def test_coarse_units_chinese_use_roman_pinyin_labels_without_timing(zh_inventory):
    parts = compile_parts('你好 Love', [zh('你', 'ni3'), zh('好', 'hao3'), en(' Love', 'l ah v')], 'zh-en')
    units, labels = coarse_units(parts)
    assert [u.text for u in units] == ['你', '好', ' Love']
    assert labels == ['ni', 'hao', 'love']
    assert [u.reading for u in units] == ['ni', 'hao', 'l ah v']
    assert all(u.start is None and u.end is None for u in units)
    units, labels = coarse_units(compile_parts('女', [zh('女', 'nü3')], 'zh'))
    assert labels == ['nu']


def test_direct_session_chinese_parts_detected_language_and_audit(tmp_path, zh_inventory):
    lines = [Line('你好', 0, 2, language='zh'), Line('爱你 baby', 2, 4, language='zh-en')]
    session = DirectSession(LlmConfig('https://fixture.example/v1', 'mimo-v2.6-pro', 'private-key'), lambda _: None, tmp_path)
    calls = []
    def handler(request):
        body = json.loads(request.content)
        content = body['messages'][1]['content']
        payload = json.loads(content[1]['text'])
        calls.append(payload)
        assert all('language' not in item for item in payload['items'])
        assert 'language_mode' not in payload and 'language' not in payload
        return fixture_reply([
            {'id': 'line_001', 'parts': [zh('你', 'ni3'), zh('好', 'hao3')]},
            {'id': 'line_002', 'parts': [zh('爱', 'ai4'), zh('你', 'ni3'), en(' baby', 'b ey b iy')]}])
    session.transport.client.close()
    session.transport.client = httpx.Client(transport=httpx.MockTransport(handler))
    session.prepare(lines, np.zeros(4 * 16000), 16000)
    assert len(calls) == 1
    assert 'Mandarin parts' in session.prompt and 'pinyin' in session.prompt
    assert [r['status'] for r in session.records] == ['accepted', 'accepted']
    assert [p['reading'] for p in session.records[0]['parts']] == ['ni', 'hao']
    assert [p['lang'] for p in session.records[1]['parts']] == ['zh', 'zh', 'en']
    assert session.records[1]['parts'][1]['groups'] == chinese.pinyin_groups('ni')
    audit = json.loads((tmp_path / 'llm_direct.json').read_text(encoding='utf-8'))
    assert 'private-key' not in json.dumps(audit)
    for record in audit['lines']:
        for part in record['parts']:
            assert set(part) == {'text', 'lang', 'reading', 'groups'}
    assert lines[0].start == 0 and lines[0].end == 2


def test_direct_payload_ignores_selected_language_for_han_only_lines(tmp_path, zh_inventory):
    lines = [Line('你好', 0, 2, language='ja'), Line('今日', 2, 4, language='zh')]
    session = DirectSession(LlmConfig('https://fixture.example/v1', 'm'), lambda _: None, tmp_path)
    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body['messages'][1]['content'][1]['text'])
        assert all('language' not in item for item in payload['items'])
        return fixture_reply([{'id': 'line_001', 'parts': [zh('你', 'ni3'), zh('好', 'hao3')]},
                              {'id': 'line_002', 'parts': []}])
    attach(session.transport, handler)
    session.prepare(lines, np.zeros(4 * 16000), 16000)
    assert session.records[0]['status'] == 'accepted'
    assert session.records[1]['status'] == 'manual'
    assert '未确定' in session.records[1]['reason']


def test_chinese_compilation_with_installed_native_inventory():
    from timing.config import MODELS
    if not (MODELS / 'tifa/vocabulary.json').exists():
        pytest.skip('TIFA vocabulary not downloaded')
    from timing.llm_direct import phone_inventory
    assert any(symbol.startswith('zh/') for symbol in phone_inventory())
    parts = compile_parts('你好', [zh('你', 'ni'), zh('好', 'hao')], 'zh')
    assert [p['reading'] for p in parts] == ['ni', 'hao']


def test_pfml_chinese_parses_with_native_g2pflow(tmp_path):
    from timing.config import MODELS
    model = MODELS / 'tifa'
    if not (model / 'deps/g2pflow').exists() or not (model / 'vocabulary.json').exists():
        pytest.skip('TIFA runtime not downloaded')
    sys.path.insert(0, str(model / 'deps'))
    from g2pflow import G2PPipeline, G2PWord, parse_pfml
    parts = compile_parts('你好', [zh('你', 'ni3'), zh('好', 'hao3')], 'zh')
    audio = tmp_path / 'a.wav'
    write_pfml(audio, parts)
    source = audio.with_suffix('.pfml').read_text(encoding='utf-8')
    words = G2PPipeline(preprocessors=[], converters=[]).convert_pfml(source, languages=['zh', 'en'])
    assert [w.language for w in words] == ['zh', 'zh']
    assert all(w.readings[0].paths[0] for w in words)
    assert sum(isinstance(p, G2PWord) for p in parse_pfml(source).parts) == 2

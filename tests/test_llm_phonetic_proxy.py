"""Regression coverage for manually supplied English-phone proxy readings."""
import json

import httpx
import pytest

from timing import llm_direct, llm_text
from timing.llm_splitter import LlmConfig
from timing.schema import Line


# This synthetic surface exercises punctuation through the transport contract;
# it does not assert a linguistic or acoustic pronunciation for any language.
PROXY_SURFACE = "x'qa: lo-vri"
PROXY_PHONES = ['k', 's', 'aa', 'l', 'ow', 'v', 'r', 'iy', 'a', 'ch', 'i']


def en(text, phones, *, roman=None, roman_lang='en-Latn'):
    return {'text': text, 'lang': 'en', 'reading': list(phones),
            'roman': text if roman is None else roman, 'roman_lang': roman_lang}


def reply(parts):
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'items': [{'id': 'line_001', 'parts': parts}]}, ensure_ascii=False)}}]})


@pytest.fixture
def proxy_inventory(monkeypatch):
    phones = frozenset(f'en/{phone}' for phone in PROXY_PHONES)
    monkeypatch.setattr(llm_text, 'phone_inventory', lambda: phones)
    monkeypatch.setattr(llm_direct, 'phone_inventory', lambda: phones)
    return phones


def session_for(tmp_path, handler):
    session = llm_text.TextReadingSession(
        LlmConfig('https://fixture.example/v1', 'text-fixture', 'private-key'),
        lambda _: None, tmp_path, word_transliteration=True)
    session.transport.client.close()
    session.transport.client = httpx.Client(transport=httpx.MockTransport(handler))
    return session


def test_english_phone_proxy_accepts_non_english_latin_surface_without_local_lookup(
        tmp_path, proxy_inventory, monkeypatch):
    wire_items = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body['messages'][1]['content'])
        wire_items.extend(payload['items'])
        return reply([en(PROXY_SURFACE, PROXY_PHONES)])

    monkeypatch.setattr(
        llm_text, 'local_parts',
        lambda *_: pytest.fail('accepted proxy phones must bypass local lexicon fallback'))
    line = Line(PROXY_SURFACE, 4.25, 7.75, language='en', roman='legacy pronunciation')
    session = session_for(tmp_path, handler)
    session.prepare([line])

    record = session.records[0]
    assert [item['text'] for item in wire_items] == [PROXY_SURFACE]
    assert wire_items[0]['roman_reference'] == 'legacy pronunciation'
    assert record['status'] == 'accepted'
    assert record['source'] == 'llm_text'
    assert record['detected_languages'] == ['en']
    assert len(record['parts']) == 1
    part = record['parts'][0]
    assert part['text'] == PROXY_SURFACE
    assert part['lang'] == 'en'
    assert part['reading'] == PROXY_PHONES
    assert (part['roman'], part['roman_lang']) == (PROXY_SURFACE, 'en-Latn')
    assert part['groups'] == [{'script': PROXY_SURFACE, 'phones': PROXY_PHONES}]
    assert (line.text, line.start, line.end) == (PROXY_SURFACE, 4.25, 7.75)
    assert line.roman == 'legacy pronunciation'
    assert 'llm_text_reading_candidate' in line.flags
    audit = session.report()
    assert audit['accepted_lines'] == 1
    assert audit['dictionary_fallback_lines'] == []
    assert audit['manual_lines'] == []
    assert audit['roman_references_sent'] is True
    assert session.transport.requests == 1


def test_foreign_pronunciation_keeps_en_phones_and_source_roman_language(
        tmp_path, proxy_inventory, monkeypatch):
    wire_items = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body['messages'][1]['content'])
        wire_items.extend(payload['items'])
        return reply([en('같이', ['k', 'a', 'ch', 'i'],
                         roman='gachi', roman_lang='ko-Latn')])

    monkeypatch.setattr(
        llm_text, 'local_parts',
        lambda *_: pytest.fail('accepted proxy phones must bypass local lexicon fallback'))
    line = Line('같이', 4.25, 7.75, language='en', roman='gach-i legacy')
    session = session_for(tmp_path, handler)
    session.prepare([line])

    assert wire_items[0]['roman_reference'] == 'gach-i legacy'
    record = session.records[0]
    assert record['status'] == 'accepted'
    assert record['detected_languages'] == ['en']
    assert (line.text, line.start, line.end) == ('같이', 4.25, 7.75)
    part = record['parts'][0]
    assert part['text'] == '같이' and part['lang'] == 'en'
    assert (part['roman'], part['roman_lang']) == ('gachi', 'ko-Latn')
    assert session.report()['roman_references_sent'] is True
    assert session.transport.requests == 1


@pytest.mark.parametrize('missing', ['roman', 'roman_lang'])
def test_text_response_rejects_a_part_without_required_roman(
        tmp_path, proxy_inventory, monkeypatch, missing):
    def unavailable(*_):
        raise RuntimeError('local lexicon unavailable')

    part = en(PROXY_SURFACE, PROXY_PHONES)
    part.pop(missing)
    monkeypatch.setattr(llm_text, 'local_parts', unavailable)
    line = Line(PROXY_SURFACE, 4.25, 7.75, language='en')
    session = session_for(tmp_path, lambda _: reply([part]))
    session.prepare([line])

    record = session.records[0]
    assert record['status'] == 'manual' and record['source'] == 'unresolved'
    assert 'roman' in record['reason']
    assert record['fallback_error'] == 'local lexicon unavailable'
    assert session.transport.requests == 1


@pytest.mark.parametrize(
    'reading', [[], ['not-in-the-model-inventory']],
    ids=['missing-phones', 'unsupported-phone'])
def test_proxy_with_invalid_phones_stays_unresolved(
        tmp_path, proxy_inventory, monkeypatch, reading):
    def unavailable(*_):
        raise RuntimeError('local lexicon unavailable')

    monkeypatch.setattr(llm_text, 'local_parts', unavailable)
    line = Line(PROXY_SURFACE, 4.25, 7.75, language='en')
    session = session_for(
        tmp_path, lambda _: reply([en(PROXY_SURFACE, reading)]))
    session.prepare([line])

    record = session.records[0]
    assert record['status'] == 'manual'
    assert record['source'] == 'unresolved'
    assert record['parts'] == []
    assert record['reason']
    assert record['fallback_error'] == 'local lexicon unavailable'
    assert session.report()['manual_lines'] == [1]
    assert (line.text, line.start, line.end) == (PROXY_SURFACE, 4.25, 7.75)
    assert 'llm_text_unresolved' in line.flags
    assert 'llm_text_reading_candidate' not in line.flags
    assert session.transport.requests == 1

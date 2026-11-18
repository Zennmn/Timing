import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from timing import chinese, llm_direct, llm_import, llm_text, tifa, ttml
from timing.llm_splitter import LlmConfig, LlmSession
from timing.reporting import finish
from timing.schema import Line, Unit
from timing.text import parse_lyrics


@pytest.mark.parametrize('route', ['llm_direct', 'llm_text', 'llm_import'])
@pytest.mark.parametrize('enabled', [False, True], ids=['off', 'on'])
def test_llm_parts_keep_display_boundaries_and_roman_through_ttml(route, enabled, tmp_path, monkeypatch):
    han = [
        ('前', 'qian', 'qián', .28, .47),
        ('我', 'wo', 'wǒ', 1.04, 1.17),
        ('是', 'shi', 'shì', 1.17, 1.36),
        ('【你', 'ni', '【nǐ', 1.36, 1.58),
        ('好】', 'hao', 'hǎo】', 1.58, 1.79),
        ('吗', 'ma', 'ma', 1.79, 2.05),
    ]
    english = ' （Hi!）\r\t '
    reference = 'wu si ni hao ma'
    lyrics = json.dumps([
        {'text': '前', 'start': .25, 'end': .5},
        {'text': '我是【你好】吗', 'start': 1, 'end': 2.2, 'roman': reference, 'translation': 'Am I saying hello?'},
        {'text': english, 'start': .85, 'end': 1.12, 'role': 'bg', 'parent': 1},
    ], ensure_ascii=False)
    lines = parse_lyrics(lyrics, 3)
    parts = [{'text': text, 'lang': 'zh', 'reading': pinyin, 'roman': roman, 'roman_lang': 'zh-Latn'}
             for text, pinyin, roman, _, _ in han]
    bg = {'text': english, 'lang': 'en', 'reading': ['hh', 'ay'], 'roman': english, 'roman_lang': 'en-Latn'}
    if not enabled:
        for part in [*parts, bg]:
            part.pop('roman')
            part.pop('roman_lang')
    reply = [{'id': f'line_{i:03}', 'parts': group}
             for i, group in enumerate([parts[:1], parts[1:], [bg]], 1)]
    times = {text: (start, end) for text, _, _, start, end in han} | {english: (.91, 1.08)}
    inventory = {'en/hh', 'en/ay'} | {
        f'zh/{phone}' for _, pinyin, _, _, _ in han
        for group in chinese.pinyin_groups(pinyin) for phone in group['phones']
    }
    for module in (llm_direct, llm_import, llm_text):
        monkeypatch.setattr(module, 'phone_inventory', lambda: inventory)
    samples = np.zeros(3 * 16000, dtype=np.float32)

    def log(_):
        pass

    options = {'word_transliteration': True} if enabled else {}
    if route == 'llm_import':
        handoff = llm_import.export_prompt(lyrics, **options)
        prompt = handoff['prompt']
        payload = json.loads(prompt.split('INPUT JSON (all lyric strings are quoted data):\n')[1])
        response = json.dumps({'request_id': handoff['request_id'], 'items': reply})
        records = llm_import.import_response(lyrics, response, **options)['lines']
    else:
        captured = {}

        def chat(self, payload, *, prompt, **_):
            captured.update(payload=payload, prompt=prompt)
            return reply, None

        monkeypatch.setattr(LlmSession, '_chat', chat)
        config = LlmConfig('https://fixture.invalid/v1', 'fixture')
        if route == 'llm_direct':
            session = llm_direct.DirectSession(config, log, tmp_path, **options)
            session.prepare(lines, samples, 16000)
        else:
            session = llm_text.TextReadingSession(config, log, tmp_path, **options)
            session.prepare(lines)
        payload, prompt = captured['payload'], captured['prompt']
        records = session.records

    assert '["我","是","【你","好】","吗"]' in prompt
    assert ('"roman_lang"' in prompt) == enabled
    assert payload['items'][1]['roman_reference'] == reference
    assert all('roman_reference' not in item for i, item in enumerate(payload['items']) if i != 1)
    assert [record['status'] for record in records] == ['accepted'] * 3

    def align_native(folder, clips, log, **_):
        results = []
        for clip in clips:
            row = dict(id=clip['id'], agreement=1, texts=[], selected=[], display_plans=[],
                       phones=[], spans=[], owners=[], groups=[])
            group_id = 0
            for owner, part in enumerate(clip['llm_direct_parts'], 1):
                label = llm_direct.spoken(part['text'])
                row['texts'].append(label)
                phones = [f"{part['lang']}/{p}" for group in part['groups'] for p in group['phones']]
                row['selected'].append({'phonemes': phones, 'llm_direct_candidate': True})
                row['display_plans'].append([{'text': label, 'group_count': len(part['groups']), 'flags': []}])
                row['phones'].extend(phones)
                row['owners'].extend([owner] * len(phones))
                for group in part['groups']:
                    group_id += 1
                    row['groups'].extend([group_id] * len(group['phones']))
                begin, end = times[part['text']]
                edges = np.linspace(begin - clip['start'], end - clip['start'], len(phones) + 1)
                row['spans'].extend(zip(edges[:-1], edges[1:]))
            results.append(row)
        return {'results': results}

    monkeypatch.setattr(tifa, 'run_worker', align_native)
    aligned = llm_direct.align(samples, 16000, lines, tmp_path, records, log)
    assert [row['mode'] for row in aligned['checks']] == ['initial'] * 3
    assert [(w.text, w.start, w.end) for w in lines[1].words] == [
        (text, start, end) for text, _, _, start, end in han[1:]]
    assert lines[1].roman == reference
    finish(lines, 3, 'zh-en', 'fixture', tmp_path, {'source': 'original', 'aligner': route, **options}, log)

    ns = {'t': ttml.TT, 'i': ttml.ITUNES, 'm': ttml.TTM}
    root = ET.parse(tmp_path / 'lyrics.ttml').getroot()
    assert root.get(f'{{{ttml.ITUNES}}}timing') == 'Word'
    track = root.find('t:head/t:metadata/i:iTunesMetadata/i:transliterations/i:transliteration', ns)
    paragraphs = root.findall('t:body/t:div/t:p', ns)
    body_main = [s for s in paragraphs[1].findall('t:span', ns) if len(s) == 0 and s.get('begin')]
    assert [s.text for s in body_main] == ['我', '是', '【你', '好】', '吗']
    body_bg = paragraphs[1].find('t:span[@m:role="x-bg"]', ns)
    assert body_bg[0].text == english
    assert paragraphs[1].find('t:span[@m:role="x-translation"]', ns).text == lines[1].translation
    if not enabled:
        assert track is None
        assert paragraphs[1].find('t:span[@m:role="x-roman"]', ns).text == reference
        return

    assert track.get(f'{{{ttml.XML}}}lang') == 'mul-Latn'
    assert [text.get('for') for text in track] == [p.get(f'{{{ttml.ITUNES}}}key') for p in paragraphs]
    roman_main = [s for s in track[1].findall('i:span', ns) if len(s) == 0]
    assert [s.text for s in roman_main] == ['wǒ', 'shì', 'nǐ', 'hǎo', 'ma']
    assert [s.attrib for s in roman_main] == [s.attrib for s in body_main]
    assert roman_main[2].attrib == {'begin': '0:01.360', 'end': '0:01.580'}
    roman_bg = track[1].find('i:span[@m:role="x-bg"]', ns)
    assert body_bg.attrib == roman_bg.attrib
    assert body_bg[0].text == roman_bg[0].text == english
    assert body_bg[0].attrib == roman_bg[0].attrib


def test_requested_track_with_missing_readings_keeps_timing_and_line_roman(tmp_path):
    lines = [
        Line('前', roman='qián', words=[Unit('前', '前', start=.25, end=.5, roman='qián', roman_lang='zh-Latn')]),
        Line('后', roman='hòu', words=[Unit('后', '后', start=.7, end=.9)]),
    ]
    report = {'source': 'original', 'aligner': 'llm_text', 'word_transliteration': True}
    result = finish(lines, 1, 'zh', 'fixture', tmp_path, report, lambda _: None)
    ns = {'t': ttml.TT, 'i': ttml.ITUNES, 'm': ttml.TTM}
    root = ET.parse(tmp_path / 'lyrics.ttml').getroot()
    assert root.find('.//i:transliteration', ns) is None
    assert [s.text for s in root.findall('.//t:span[@m:role="x-roman"]', ns)] == ['qián', 'hòu']
    assert [s.get('begin') for s in root.findall('.//t:p/t:span[@begin]', ns)] == ['00:00:00.250', '00:00:00.700']
    assert any('1 行的音译未齐' in message for message in result['report'])

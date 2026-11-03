import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from timing.llm_import import export_prompt, manifest
from timing.request_plan import plan_request
from timing.schema import Unit
from timing.text import parse_lyrics
from timing.ttml import TT, TTM, XML
from timing.ttml_input import normalize_ttml, parse_time


def document(content, *, language='en', body='', root='', div=''):
    language = '' if language is None else f' xml:lang="{language}"'
    return (f'<tt xmlns="{TT}" xmlns:ttm="{TTM}"{language} {root}>'
            f'<body {body}><div {div}>{content}</div></body></tt>')


def line(text='Hello &amp; you', begin='00:01.250', end='00:04.500', extra=''):
    return f'<p begin="{begin}" end="{end}" {extra}>{text}</p>'


@pytest.mark.parametrize('value,expected', [
    ('01:02:03.456', 3723.456), ('02:03.456', 123.456), ('3.125', 3.125),
    ('3125ms', 3.125), ('3.125s', 3.125), ('1.5m', 90), ('0.5h', 1800),
])
def test_media_timestamps(value, expected):
    assert parse_time(value) == pytest.approx(expected)


@pytest.mark.parametrize('value', ['-1s', '+1s', 'NaN', 'Infinity', '1e3s', '2f', '2t',
                                         '01:61.000', '00:00:60.000', '00:00:01:23'])
def test_unsupported_or_invalid_timestamps(value):
    with pytest.raises(ValueError, match='TTML'):
        parse_time(value)


def test_anchored_ttml_keeps_explicit_final_end_without_audio():
    lyrics = '\ufeff' + document(line() + line('again', '4s', '6.875s'))
    lines = parse_lyrics(lyrics, None)
    assert [(row.text, row.start, row.end) for row in lines] == [
        ('Hello & you', 1.25, 4.5), ('again', 4, 6.875)]
    assert all(row.words == [] for row in lines)
    plan = plan_request(lyrics)
    assert plan.aligner == 'tifa' and not plan.needs_llm
    assert parse_lyrics(lyrics, 20)[-1].end == 6.875
    with pytest.raises(ValueError, match='超出音频'):
        parse_lyrics(lyrics, 6)


def test_namespace_prefixes_no_key_and_no_namespace_are_supported():
    lyrics = f'<t:tt xmlns:t="{TT}"><t:body><t:p begin="1s" end="2s">君</t:p></t:body></t:tt>'
    assert parse_lyrics(lyrics, 3)[0].text == '君'
    plain = '<tt><body><p begin="1s" end="2s">hi</p></body></tt>'
    assert parse_lyrics(plain, 3)[0].start == 1


def test_div_sections_have_absolute_times_not_additive_offsets():
    lyrics = document('<div begin="10s" end="15s">' + line('hello', '11s', '13s') + '</div>', body='dur="20s"')
    row = parse_lyrics(lyrics, 20)[0]
    assert (row.start, row.end) == (11, 13)
    with pytest.raises(ValueError, match='相对偏移'):
        parse_lyrics(lyrics.replace('begin="11s"', 'begin="1s"'), 20)


def test_inline_annotations_backgrounds_and_duet_are_not_merged_into_main_text():
    lyrics = document(line(
        '<span>君</span><span>と me</span>'
        '<span ttm:role="x-translation">你和我</span>'
        '<span ttm:role="x-roman">kimi to me</span>'
        '<span ttm:role="x-bg" begin="2s" end="4s" ttm:agent="v2" xml:lang="en">'
        'ah<span ttm:role="x-translation">啊</span></span>',
        extra='ttm:agent="v1" xml:lang="ja"') + line('again', '3s', '6s', 'ttm:agent="v2"'))
    main, bg, duet = parse_lyrics(lyrics, 10)
    assert main.text == '君と me'
    assert main.translation == '你和我' and main.roman == 'kimi to me'
    assert main.language == 'auto'
    assert (bg.text, bg.start, bg.end, bg.role, bg.parent, bg.agent) == ('ah', 2, 4, 'bg', 0, 'v2')
    assert bg.translation == '啊' and bg.language == 'auto'
    assert duet.agent == 'v2' and duet.role == 'main'
    assert plan_request(lyrics).has_background


def test_background_without_own_clocks_inherits_whole_sentence_range():
    main, bg = parse_lyrics(document(line('hello<span ttm:role="x-bg">ah</span>')), 10)
    assert (bg.start, bg.end) == (main.start, main.end)


def test_background_infers_range_from_nested_singing_spans_and_ignores_annotations():
    lyrics = document(line('hello<span ttm:role="x-bg">('
                           '<span><span begin="4s" end="4.5s">yeah </span>'
                           '<span begin="4.6s" dur="1s">baby</span></span>)'
                           '<span ttm:role="x-translation" begin="1s" end="9s">翻译</span>'
                           '</span>'))
    main, bg = parse_lyrics(lyrics, 10)
    assert (main.start, main.end) == (1.25, 4.5)
    assert (bg.text, bg.start, bg.end, bg.translation) == ('(yeah baby)', 4, 5.6, '翻译')
    assert bg.words == []
    normalized = normalize_ttml(lyrics)
    converted = parse_lyrics(normalized['lyrics'], 10)
    assert (converted[1].start, converted[1].end) == (4, 5.6)


@pytest.mark.parametrize(('attrs', 'text', 'expected'), [
    ('begin="2s" end="8s"', '<span begin="3s" end="4s">ah</span>', (2, 8)),
    ('begin="2s"', '<span begin="3s" end="6s">ah</span>', (2, 6)),
    ('end="8s"', '<span begin="3s" end="4s">ah</span>', (3, 8)),
    ('', 'oh <span begin="3s" end="4s">yeah</span>', (1.25, 4.5)),
    ('', '<span begin="3s">ah</span>', (3, 4.5)),
    ('', '<span end="3s">ah</span>', (1.25, 3)),
    ('begin="2s" dur="5s"', '<span begin="3s" end="4s">ah</span>', (2, 7)),
])
def test_background_explicit_and_inherited_boundaries(attrs, text, expected):
    lyrics = document(line(f'hello<span ttm:role="x-bg" {attrs}>{text}</span>'))
    bg = parse_lyrics(lyrics, 10)[1]
    assert (bg.start, bg.end) == expected


def test_invalid_timed_background_text_is_rejected():
    lyrics = document(line('hello<span ttm:role="x-bg"><span begin="5s" end="4s">ah</span></span>'))
    with pytest.raises(ValueError, match='伴唱'):
        parse_lyrics(lyrics, 10)


def test_arbitrary_agent_ids_map_consistently_and_do_not_collapse_third_voice():
    lyrics = document(line('a', '1s', '3s', 'ttm:agent="singerA"') +
                      line('b', '2s', '4s', 'ttm:agent="singerB"') +
                      line('c', '4s', '5s', 'ttm:agent="singerA"'))
    assert [row.agent for row in parse_lyrics(lyrics, 10)] == ['v1', 'v2', 'v1']
    three = lyrics.replace('begin="4s" end="5s" ttm:agent="singerA"',
                           'begin="4s" end="5s" ttm:agent="singerC"')
    assert [row.agent for row in parse_lyrics(three, 10)] == ['v1', 'v2', 'v3']


def test_apple_v3_and_group_voice_identifiers_survive_json_and_export():
    from timing.ttml import export
    from xml.etree import ElementTree as ET
    lyrics = document(line('one', '1s', '3s', 'ttm:agent="v1"') +
                      line('two', '2s', '4s', 'ttm:agent="v2"') +
                      line('three', '3s', '5s', 'ttm:agent="v3"') +
                      line('together', '4s', '6s', 'ttm:agent="v1000"'))
    rows = parse_lyrics(normalize_ttml(lyrics)['lyrics'], 10)
    assert [row.agent for row in rows] == ['v1', 'v2', 'v3', 'v1000']
    for row in rows:
        row.words = [Unit(row.text, row.text, start=row.start, end=row.end)]
    root = ET.fromstring(export(rows, 10))
    assert [row.get(f'{{{TTM}}}agent') for row in root.findall(f'.//{{{TT}}}p')] == ['v1', 'v2', 'v3', 'v1000']


def test_apple_sidecars_use_first_track_not_language_labels():
    itunes = 'http://music.apple.com/lyric-ttml-internal'
    head = '''<head><metadata><iTunesMetadata xmlns="http://music.apple.com/lyric-ttml-internal">
      <translations>
        <translation xml:lang="en"><text for="L2">第二</text><text for="L1">第一</text></translation>
        <translation type="subtitle" xml:lang="zh-Hans"><text for="L1">错误一</text><text for="L2">错误二</text></translation>
      </translations>
      <transliterations>
        <transliteration xml:lang="fr"><text for="L2">ni</text><text for="L1">ichi</text></transliteration>
        <transliteration xml:lang="ja-Latn"><text for="L1">wrong-romaji</text></transliteration>
      </transliterations>
    </iTunesMetadata></metadata></head>'''
    lyrics = document(line('hello', '1s', '3s', 'xmlns:itunes="' + itunes + '" itunes:key="L1"') +
                      line('bye', '2s', '4s', 'xmlns:itunes="' + itunes + '" itunes:key="L2"'))
    lyrics = lyrics.replace('<body ', head + '<body ', 1)
    rows = parse_lyrics(lyrics, 10)
    assert [row.translation for row in rows] == ['第一', '第二']
    assert [row.roman for row in rows] == ['ichi', 'ni']
    with pytest.raises(ValueError, match='行 ID'):
        parse_lyrics(lyrics.replace('for="L2"', 'for="missing"'), 10)
    with pytest.raises(ValueError, match='重复'):
        parse_lyrics(lyrics.replace('itunes:key="L2"', 'itunes:key="L1"'), 10)


@pytest.mark.parametrize(('location', 'language'), [
    ('root', 'zh-Hans'), ('body', 'en'), ('div', 'ja'),
    ('paragraph', 'fr'), ('span', 'not_a_language'), ('background', ''),
])
def test_xml_lang_never_sets_main_or_background_language(location, language):
    metadata = f'xml:lang="{language}"'
    span = f' {metadata}' if location == 'span' else ''
    background = f' {metadata}' if location == 'background' else ''
    lyrics = document(
        line(f'<span{span}>hello</span>'
             f'<span ttm:role="x-bg" begin="2s" end="3s" ttm:agent="v2"{background}>ah</span>',
             extra=metadata if location == 'paragraph' else ''),
        language=language if location == 'root' else None,
        body=metadata if location == 'body' else '',
        div=metadata if location == 'div' else '')
    main, bg = parse_lyrics(lyrics, 10)
    assert (main.language, bg.language) == ('auto', 'auto')
    assert (main.text, main.start, main.end, main.role, main.agent) == ('hello', 1.25, 4.5, 'main', 'v1')
    assert (bg.text, bg.start, bg.end, bg.role, bg.parent, bg.agent) == ('ah', 2, 3, 'bg', 0, 'v2')


def test_language_metadata_does_not_change_normalized_lyrics_or_manifest():
    plain = ('<span>hello</span><span xml:space="preserve"> there</span>'
             '<span ttm:role="x-translation">你好</span>'
             '<span ttm:role="x-bg" begin="2s" end="3.5s" ttm:agent="v2">'
             'ah<span ttm:role="x-translation">啊</span></span>')
    annotated = (plain
                 .replace('<span>hello</span>', '<span xml:lang="ko">hello</span>')
                 .replace('<span xml:space="preserve">', '<span xml:space="preserve" xml:lang="invalid">')
                 .replace('ttm:role="x-bg"', 'ttm:role="x-bg" xml:lang="ja"'))
    baseline = document(line(plain, extra='ttm:agent="v1"'), language=None)
    lyrics = document(line(annotated, extra='ttm:agent="v1" xml:lang="en"'),
                      language='zh-Hans', body='xml:lang="fr"', div='xml:lang="not_a_language"')
    raw_rows = parse_lyrics(lyrics, 10)
    assert [asdict(row) for row in raw_rows] == [asdict(row) for row in parse_lyrics(baseline, 10)]
    assert [row.language for row in raw_rows] == ['auto', 'auto']
    normalized = normalize_ttml(lyrics)
    assert normalized == normalize_ttml(baseline)
    assert [row['language'] for row in json.loads(normalized['lyrics'])] == ['auto', 'auto']
    assert [asdict(row) for row in parse_lyrics(normalized['lyrics'], 10)] == [asdict(row) for row in raw_rows]
    assert manifest(lyrics) == manifest(baseline) == manifest(normalized['lyrics'])


@pytest.mark.parametrize('agent', ['v0', 'v-1', 'v1 v2', '', 'v1000000000', None, 3])
def test_json_rejects_malformed_voice_identifiers(agent):
    with pytest.raises(ValueError, match='agent'):
        parse_lyrics(json.dumps([{'text': 'hi', 'start': 1, 'end': 3, 'agent': agent}]), 10)


def test_word_ttml_can_supply_line_text_but_never_reuses_existing_word_times():
    text = '<span begin="1.5s" end="2s">Hello </span><span begin="2.5s" end="3s">you</span>'
    row = parse_lyrics(document(line(text)), 10)[0]
    assert row.text == 'Hello you' and row.words == []
    assert (row.start, row.end) == (1.25, 4.5)


def test_xml_default_whitespace_preserve_mode_entities_and_br():
    lyrics = document(line('\n  Hello <span>&amp; </span>you<br/>again\n'))
    assert parse_lyrics(lyrics, 10)[0].text == 'Hello & you\nagain'
    lyrics = document(line('  Hello  <span> you </span> ', extra='xml:space="preserve"'))
    assert parse_lyrics(lyrics, 10)[0].text == '  Hello   you  '


def test_inline_xml_space_overrides_do_not_erase_preserved_spaces():
    lyrics = document(line('hi <span xml:space="preserve">  you  </span> again'))
    assert parse_lyrics(lyrics, 10)[0].text == 'hi   you   again'
    lyrics = document(line('  hi <span xml:space="default">  you  </span> again  ', extra='xml:space="preserve"'))
    assert parse_lyrics(lyrics, 10)[0].text == '  hi  you  again  '
    lyrics = document(line(' hi <span> </span> you '))
    assert parse_lyrics(lyrics, 10)[0].text == 'hi you'


def test_formatting_only_newlines_between_spans_do_not_add_cjk_spaces():
    lyrics = document(line('\n <span>甲</span>\n  <span>乙</span>\n'))
    assert parse_lyrics(lyrics, 10)[0].text == '甲乙'
    lyrics = document(line('<span>Hello</span> <span>you</span>'))
    assert parse_lyrics(lyrics, 10)[0].text == 'Hello you'


def test_duration_end_and_empty_interlude():
    lyrics = document('<p begin="0s" end="1s"/>'
                      '<p begin="2s" dur="3s">hello</p>')
    assert [(row.text, row.start, row.end) for row in parse_lyrics(lyrics, 10)] == [('hello', 2, 5)]


@pytest.mark.parametrize('lyrics,reason', [
    ('<tt>', 'XML'), ('<html><body>hello</body></html>', '根元素'),
    ('<!DOCTYPE tt [<!ENTITY text "hello">]><tt><body><p begin="1s" end="2s">&text;</p></body></tt>', 'DTD'),
    ('<tt/>', 'body'), (document('<p begin="1s">hello</p>'), 'begin'),
    (document(line('hello', '2s', '1s')), '结束'),
    (document(line(), body='timeContainer="seq"'), 'seq'),
    (document(line(), body='begin="1s"'), '相对'),
    (document(line('hello', '1t', '3t')), 'tick'),
    (document(line(), root='xmlns:ttp="http://www.w3.org/ns/ttml#parameter" ttp:timeBase="smpte"'), 'media'),
    (document(line('<span ttm:role="x-unknown">hello</span>')), '角色'),
    (document(line('hi<span ttm:role="x-translation">一</span><span ttm:role="x-translation">two</span>')), '多份'),
    (document(line('hi<span ttm:role="x-bg">a<span ttm:role="x-bg">b</span></span>')), '嵌套'),
    (document(line('hi<span ttm:role="x-translation">你好<span ttm:role="x-bg">啊</span></span>')), '嵌套'),
    (document(line('hi', extra='ttm:agent="a b"')), '多歌手'),
])
def test_invalid_or_unrepresentable_ttml_fails_clearly(lyrics, reason):
    with pytest.raises(ValueError, match=reason):
        parse_lyrics(lyrics, 10)


def test_deep_or_oversized_input_is_bounded():
    deep = document('<div>' * 70 + line() + '</div>' * 70)
    with pytest.raises(ValueError, match='嵌套'):
        normalize_ttml(deep)
    with pytest.raises(ValueError, match='2 MB'):
        normalize_ttml(' ' * (2 * 1024**2 + 1))


def test_normalized_json_and_raw_xml_share_line_data_and_llm_identity(monkeypatch):
    lyrics = document(line() + line('again', '5s', '8s'))
    normalized = normalize_ttml(lyrics)
    assert normalized['lines'] == 2
    assert [asdict(row) for row in parse_lyrics(lyrics, 10)] == [asdict(row) for row in parse_lyrics(normalized['lyrics'], 10)]
    key, items = manifest(lyrics)
    assert (key, items) == manifest(normalized['lyrics'])
    assert items[-1]['end'] == 8
    assert key != manifest(lyrics.replace('end="8s"', 'end="9s"'))[0]
    monkeypatch.setattr('timing.llm_import.phone_inventory', lambda: {'en/ay'})
    assert '"end": 8.0' in export_prompt(lyrics)['prompt']


def test_normalization_endpoint_is_local_and_returns_no_model_artifacts():
    from timing import server
    lyrics = document(line())
    with TestClient(server.app) as client:
        assert client.post('/api/lyrics/ttml', data={'lyrics': lyrics}).status_code == 403
        response = client.post('/api/lyrics/ttml', data={'lyrics': lyrics}, headers={'X-Timing-Token': server.TOKEN})
        assert response.status_code == 200
        assert response.json() == normalize_ttml(lyrics)
        bad = client.post('/api/lyrics/ttml', data={'lyrics': '<tt>'}, headers={'X-Timing-Token': server.TOKEN})
        assert bad.status_code == 400 and 'XML' in bad.json()['detail']


def test_api_ttml_file_and_text_inputs_both_reach_anchored_queue(tmp_path, monkeypatch):
    from timing import server
    calls = []
    monkeypatch.setattr(server, 'DATA', tmp_path)
    monkeypatch.setattr(server, 'initialize', lambda: None)
    monkeypatch.setattr(server, 'executor', SimpleNamespace(submit=lambda *args: calls.append(args)))
    monkeypatch.setattr(server, 'jobs', {})
    lyrics = document(line())
    with TestClient(server.app) as client:
        for fields, files in [({}, {'lyrics_file': ('input.ttml', lyrics.encode(), 'application/ttml+xml')}),
                              ({'lyrics': lyrics}, {})]:
            response = client.post('/api/jobs', data={'audio_mode': 'source', **fields},
                                   files={'audio': ('song.wav', b'audio'), **files},
                                   headers={'X-Timing-Token': server.TOKEN})
            assert response.status_code == 200
    assert len(calls) == 2 and all(call[3] == lyrics and call[8] == 'tifa' for call in calls)


def test_pipeline_preserves_ttml_sentence_windows_and_annotations(tmp_path, monkeypatch):
    from timing import alignment, pipeline
    monkeypatch.setattr(pipeline, 'decode', lambda *args, **kwargs: (np.zeros(160000, dtype='float32'), 16000))
    monkeypatch.setattr(pipeline, 'prepare_sources', lambda *args, **kwargs:
                        SimpleNamespace(source='original', separation=None, lead_separation=None))
    observed = []

    def align(lines, tracks, out_dir, report, log, **kwargs):
        for row in lines:
            observed.append((row.text, row.start, row.end, row.translation))
            # Fixture output only: actual model quality is outside an input-format test.
            row.words = [Unit(row.text, row.text, start=row.start + .1, end=row.end - .1)]

    monkeypatch.setattr(alignment, 'align_tifa', align)
    lyrics = document(line('hi<span ttm:role="x-translation">你好</span>'))
    result = pipeline.run(tmp_path / 'song.wav', lyrics, tmp_path, audio_mode='source', log=lambda _: None)
    assert observed == [('hi', 1.25, 4.5, '你好')]
    assert result['status'] == 'complete'
    output = (tmp_path / 'lyrics.ttml').read_text(encoding='utf-8')
    assert '你好' in output and '00:00:01.350' in output


def test_cli_ttml_input_and_prompt_export(tmp_path, monkeypatch):
    from timing import __main__ as cli, pipeline
    lyrics = document(line('hi'))
    source, output, audio = tmp_path / 'input.ttml', tmp_path / 'output.ttml', tmp_path / 'song.wav'
    source.write_text(lyrics, encoding='utf-8-sig')
    audio.write_bytes(b'audio')
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    seen = []

    def run(audio_path, text, folder, *args, **kwargs):
        seen.append(parse_lyrics(text, 10))
        folder.mkdir()
        (folder / 'lyrics.ttml').write_text('fixture export', encoding='utf-8')

    monkeypatch.setattr(pipeline, 'run', run)
    monkeypatch.setattr(sys, 'argv', ['timing', str(audio), '--lyrics', str(source), '--output', str(output)])
    cli.main()
    assert seen[0][0].end == 4.5 and output.read_text() == 'fixture export'
    prompt = tmp_path / 'prompt.txt'
    monkeypatch.setattr('timing.llm_import.phone_inventory', lambda: {'en/ay'})
    monkeypatch.setattr(sys, 'argv', ['timing', '--lyrics', str(source), '--export-llm-prompt', str(prompt)])
    cli.main()
    assert manifest(lyrics)[0] in prompt.read_text(encoding='utf-8')

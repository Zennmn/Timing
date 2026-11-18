import json
import xml.etree.ElementTree as ET
import pytest

from timing.schema import Line, Unit
from timing.text import tokenize, parse_lyrics
from timing.ttml import export, stamp, TT, TTM, ITUNES, XML


@pytest.mark.parametrize("text", ["君の名前 beautiful world!", "今日も I love you, forever.", "名前を呼んで！", "　愛してる。  Hello & goodbye!", "きゃっと、キャット。", "Don't stop — 2026年", "🎵君とme♪"])
def test_tokenization_preserves_display_text(text):
    for fine in (True, False):
        units = tokenize(text, fine=fine)
        assert "".join(w.text for w in units) == text
        assert all(w.spoken for w in units)


def test_indivisible_japanese_reading():
    units = tokenize("今日の君")
    assert units[0].text == "今日"
    assert units[0].reading


def test_english_syllables_are_candidates_with_real_model_boundaries_required():
    units = tokenize("beautiful ")
    assert len(units) > 1
    assert all(w.start is None and w.end is None for w in units)
    assert all("english_syllable_candidate" in w.flags for w in units)


def test_lrc_and_json_anchor_parsing():
    lines = parse_lyrics("[ar:test]\n[00:01.250]君の名前\n[00:04.000]hello", 8)
    assert [(x.start, x.end) for x in lines] == [(1.25, 4), (4, 8)]
    rows = [{"text": "hello", "start": 1, "end": 3}, {"text": "ah", "start": 2, "end": 4, "role": "bg", "parent": 0}]
    assert parse_lyrics(json.dumps(rows), 8)[1].parent == 0


def test_bad_anchors_and_unanchored_background_rejected():
    with pytest.raises(ValueError):
        parse_lyrics('[{"text":"hello","start":8,"end":3}]', 10)
    with pytest.raises(ValueError):
        parse_lyrics('[{"text":"hello"},{"text":"ah","role":"bg","parent":0}]', 10)


def test_main_starts_must_be_ordered_but_overlapping_ranges_are_supported():
    with pytest.raises(ValueError, match="排序"):
        parse_lyrics(json.dumps([{"text": "a", "start": 5, "end": 6},
                                 {"text": "b", "start": 3, "end": 4}]), 10)
    lines = parse_lyrics(json.dumps([{"text": "a", "start": 0, "end": 5},
                                     {"text": "b", "start": 2, "end": 8}]), 10)
    assert [(line.start, line.end) for line in lines] == [(0, 5), (2, 8)]


def test_ttml_roundtrip_spaces_escaping_overlaps_background():
    a = Line("Hello & you", agent="v1", words=[Unit("Hello & ", "Hello", start=1, end=2), Unit("you", "you", start=2.2, end=3)])
    b = Line("君", agent="v2", words=[Unit("君", "君", start=2.5, end=4)])
    bg = Line("ah", role="bg", parent=0, words=[Unit("ah", "ah", start=.5, end=3.5)])
    root = ET.fromstring(export([a, b, bg], 5, "mixed", "a & b"))
    ps = root.findall(f".//{{{TT}}}p")
    assert len(ps) == 2
    assert ps[0].attrib["begin"] == "00:00:00.500"
    assert ps[0].attrib["end"] == "00:00:03.500"
    assert [p.attrib[f"{{{ITUNES}}}key"] for p in ps] == ["L1", "L2"]
    spans = list(ps[0])
    assert spans[0].attrib[f"{{{TTM}}}role"] == "x-bg"
    assert "".join(x.text for x in spans[1:]) == "Hello & you"


@pytest.mark.parametrize(("language", "expected"), [
    ("ja", "ja"), ("en", "en"), ("zh", "zh-CN"),
    ("zh-en", "mul"), ("mixed", "mul"), ("auto", "und"),
])
def test_song_language_only_export_uses_root_language_and_omits_line_languages(language, expected):
    line = Line("hello", language="en", words=[Unit("hello", "hello", start=1, end=2)])
    root = ET.fromstring(export([line], 5, language, include_line_languages=False))
    assert root.attrib[f"{{{XML}}}lang"] == expected
    assert all(f"{{{XML}}}lang" not in p.attrib for p in root.findall(f".//{{{TT}}}p"))


def test_llm_finish_exports_only_song_level_language(tmp_path):
    from timing.reporting import finish
    line = Line("hello", language="en", words=[Unit("hello", "hello", start=.1, end=.8)])
    finish([line], 1, "ja", "test", tmp_path,
           {"source": "original", "aligner": "llm_direct"}, lambda _: None)
    root = ET.parse(tmp_path / "lyrics.ttml").getroot()
    paragraph = root.find(f".//{{{TT}}}p")
    assert root.attrib[f"{{{XML}}}lang"] == "ja"
    assert paragraph is not None and f"{{{XML}}}lang" not in paragraph.attrib


def test_non_llm_export_keeps_per_line_language_by_default():
    line = Line("hello", language="en", words=[Unit("hello", "hello", start=1, end=2)])
    root = ET.fromstring(export([line], 5, "ja"))
    paragraph = root.find(f".//{{{TT}}}p")
    assert paragraph is not None and paragraph.attrib[f"{{{XML}}}lang"] == "en"


def test_tifa_auto_language_uses_audio_selected_pronunciation(tmp_path):
    from timing.reporting import finish
    from timing.tifa import detect_selected_language
    raw = {"results": [{"selected": [
        {"phonemes": ["en/HH", "en/AH"]}, {"phonemes": ["en/L"]},
    ]}]}
    assert detect_selected_language(raw) == "en"
    assert detect_selected_language({"results": [{"selected": [
        {"phonemes": ["ja/k"]}, {"phonemes": ["en/HH"]},
    ]}]}) == "mixed"
    assert detect_selected_language({"results": [{"selected": [{"phonemes": ["ja/k"]}]}]}) == "ja"
    line = Line("hello", language="mixed", words=[Unit("hello", "hello", start=.1, end=.8)])
    report = {"source": "original", "aligner": "tifa",
              "tifa": {"detected_language": "en"}}
    finish([line], 1, "mixed", "test", tmp_path, report, lambda _: None)
    root = ET.parse(tmp_path / "lyrics.ttml").getroot()
    assert root.attrib[f"{{{XML}}}lang"] == "en"
    assert f"{{{XML}}}lang" not in root.find(f".//{{{TT}}}p").attrib
    assert report["language"] == "en"
    assert report["language_detection"] == "tifa_selected_pronunciation"


@pytest.mark.parametrize(("text", "expected"), [
    ("hello world", "en"), ("こんにちは", "ja"), ("Hello こんにちは", "mul"),
    ("你好 Hello", "und"),
])
def test_ctc_auto_language_uses_script_fallback_at_song_root(tmp_path, text, expected):
    from timing.reporting import finish
    line = Line(text, language="auto", words=[Unit(text, text, start=.1, end=.8)])
    report = {"source": "original", "aligner": "ctc"}
    finish([line], 1, "mixed", "test", tmp_path, report, lambda _: None)
    root = ET.parse(tmp_path / "lyrics.ttml").getroot()
    assert root.attrib[f"{{{XML}}}lang"] == expected
    assert f"{{{XML}}}lang" not in root.find(f".//{{{TT}}}p").attrib


@pytest.mark.parametrize("start,end", [(0, 0), (2, 1), (0, 6), (float("nan"), 2), (1, 1.0001)])
def test_export_rejects_invalid_or_collapsed_boundaries(start, end):
    with pytest.raises(ValueError):
        export([Line("a", words=[Unit("a", "a", start=start, end=end)])], 5)


def test_export_rejects_missing_text():
    with pytest.raises(ValueError):
        export([Line("hello world", words=[Unit("hello", "hello", start=1, end=2)])], 5)


def test_timestamp_rollover():
    assert stamp(59.9996) == "00:01:00.000"


def test_exportable_timing_is_not_presented_as_accuracy_pass(tmp_path):
    import json
    from timing.pipeline import finish
    line = Line("ab", words=[Unit("a", "a", start=.1, end=.11),
                             Unit("b", "b", start=.11, end=.7)])
    result = finish([line], 1, "en", "test", tmp_path, {"source": "original"}, lambda _: None)
    assert result["status"] == "complete" and result["review_lines"] == 1
    report = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    assert report["quality"]["status"] == "needs_review"
    assert report["quality"]["issues"][0]["reasons"] == ["short_unit_check"]


def test_server_blocks_cross_origin_and_unknown_downloads():
    from fastapi.testclient import TestClient
    from timing.server import app
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.post("/api/jobs").status_code == 403
        assert client.get("/api/jobs/../../pyproject.toml").status_code == 404
        assert client.get("/api/jobs/" + "0" * 32 + "/files/ttml").status_code == 404

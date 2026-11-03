import numpy as np
import pytest
from timing.schema import Line, Unit
from timing.singing import phonemize, kana_phones, english_parts


@pytest.fixture
def english_dictionary(monkeypatch):
    import timing.singing as singing
    table = {"paper": ["p", "ey", "p", "er"], "in": ["ih", "n"],
             "baskets": ["b", "ae", "s", "k", "ah", "t", "s"]}
    monkeypatch.setattr(singing, "dictionary", lambda: table)


def test_opening_is_phoneme_aligned_not_split_by_duration(english_dictionary):
    units, owners, sequence, mapping = phonemize([Line("Paper in baskets")])
    assert "".join(u.text for u in units) == "Paper in baskets"
    assert all(u.start is None and u.end is None for u in units)
    assert [phone for language, phone in sequence if language == "en"] == ["p", "ey", "p", "er", "ih", "n", "b", "ae", "s", "k", "ah", "t", "s"]
    assert all(i in mapping for i in range(len(units)))


@pytest.mark.parametrize("text", ["「Paper」って書いた", "机には青い箱がある", "今日は紙を数える", "キャット、ストーリー。"])
def test_phoneme_mapping_retains_display_and_all_languages(text, english_dictionary):
    units, owners, sequence, mapping = phonemize([Line(text)])
    assert "".join(u.text for u in units) == text
    assert all(i in mapping for i in range(len(units)))
    assert all(len(phone) > 0 for language, phone in sequence)


def test_japanese_mora_gemination_and_nasal():
    assert kana_phones("キャット") == ["ky", "a", "cl", "t", "o"]
    assert kana_phones("キョーン") == ["ky", "o", "o", "N"]


@pytest.mark.parametrize("text,expected", [("どうせ", ["d", "o", "o", "s", "e"]),
                                           ("今日", ["ky", "o", "o"]),
                                           ("昨日", ["k", "i", "n", "o", "o"])])
def test_actual_japanese_pronunciation_does_not_invent_written_vowels(text, expected):
    units, _, sequence, _ = phonemize([Line(text)])
    assert "".join(unit.text for unit in units) == text
    assert [phone for language, phone in sequence if language == "ja"] == expected
    assert all(unit.source_pronunciation for unit in units)


def test_unknown_english_word_is_not_silently_dropped(english_dictionary):
    with pytest.raises(ValueError, match="需要补充读音"):
        english_parts(Unit("xyzzzt", "xyzzzt"), True)


@pytest.mark.parametrize("source", ["original", "separated", "provided"])
def test_singing_receives_full_band_source_not_asr_waveform(tmp_path, monkeypatch, source):
    from timing import pipeline, alignment
    song, stem = tmp_path / "song.wav", tmp_path / "stem.wav"
    asr_wave = np.zeros(16000, dtype="float32")
    singing_wave = np.ones(44100, dtype="float32")
    reads, separations = [], []

    def decode(path, target, rate=16000, *, floating=False):
        reads.append((path, rate, floating))
        return (singing_wave, 44100) if rate == 44100 else (asr_wave, 16000)

    def separate(*args):
        separations.append(True)
        return stem, {}

    class Coarse:
        def __init__(self, log): pass
        def release(self): pass
        def align(self, audio, line, *args): line.start, line.end = .1, .7

    def fine(audio, rate, lines, *args, **kwargs):
        assert audio is singing_wave and rate == 44100
        lines[0].words = [Unit("hello", "hello", start=.1, end=.7)]
        lines[0].start, lines[0].end = .1, .7
        return {"frame_ms": 10}

    monkeypatch.setattr(pipeline, "decode", decode)
    monkeypatch.setattr(alignment, "decode", decode)
    monkeypatch.setattr("timing.sources.decode", decode)
    monkeypatch.setattr(alignment, 'CtcAligner', Coarse)
    monkeypatch.setattr(alignment.tifa, 'align', fine)
    monkeypatch.setattr("timing.sources.separate_vocals", separate)
    mode = "source" if source == "original" else "vocals"
    pipeline.run(song, "[00:00]hello", tmp_path, "en",
                 vocals_path=stem if source == "provided" else None,
                 log=lambda _: None, audio_mode=mode)
    assert reads[-1] == (song if source == "original" else stem, 44100, True)
    assert separations == ([True] if source == "separated" else [])


def test_local_pronunciation_overrides_are_used_without_rewriting_lyrics(tmp_path, monkeypatch, english_dictionary):
    import timing.singing as singing
    path = tmp_path / "pronunciations.json"
    path.write_text('{"ja":{"今日":"キョー"},"en":{"xyzzzt":["z","iy"]}}', encoding="utf-8")
    monkeypatch.setattr(singing, "OVERRIDES", path)
    units, _, phones, _ = phonemize([Line("今日 xyzzzt")], syllables=False)
    assert "".join(unit.text for unit in units) == "今日 xyzzzt"
    assert [(lang, phone) for lang, phone in phones if lang != "sp"] == [
        ("ja", "ky"), ("ja", "o"), ("ja", "o"), ("en", "z"), ("en", "iy")]
    assert all("custom_pronunciation" in unit.flags for unit in units)

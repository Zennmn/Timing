from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "tifa"
NINETIES_PHONES = ["n", "ay", "n", "t", "iy", "z"]
FORTY_PHONES = ["f", "ao", "r", "t", "iy"]


@pytest.fixture(scope="module")
def native_g2p():
    if not (MODELS / "deps/g2pflow").is_dir() or not (MODELS / "dictionaries/ds_cmudict-07b.txt").is_file():
        pytest.skip("TIFA pronunciation assets not downloaded")
    sys.path.insert(0, str(MODELS / "deps"))
    import g2pflow
    return g2pflow


def _clip(tmp_path, text):
    audio = tmp_path / "line_001.wav"
    audio.with_suffix(".txt").write_text(text, encoding="utf-8")
    return [{"id": "line_001", "audio": str(audio)}]


def _native_pipeline(dictionary):
    from g2pflow import G2PPipeline
    from g2pflow.converters.dictionary import DictionaryConverter
    from g2pflow.preprocessors.simple import FilterPunctuation, LowercasePreprocessor, StripWhitespacePreprocessor
    from timing.tifa_pronunciation import ensure_apostrophes_registered

    return G2PPipeline(
        preprocessors=[ensure_apostrophes_registered()(), FilterPunctuation(),
                       StripWhitespacePreprocessor(), LowercasePreprocessor()],
        converters=[DictionaryConverter(str(dictionary))],
    )


def _phones(word):
    return [phone for reading in word.readings for path in reading.paths
            for group in path for phone in group.phonemes]


def _row(words):
    selected = [{"phonemes": _phones(word)} for word in words]
    phones = [phone for candidate in selected for phone in candidate["phonemes"]]
    owners = [index + 1 for index, candidate in enumerate(selected)
              for _ in candidate["phonemes"]]
    spans = [[index * .1, (index + 1) * .1] for index in range(len(phones))]
    return {
        "texts": [word.text for word in words],
        "selected": selected,
        "display_plans": [None] * len(words),
        "owners": owners,
        "groups": list(owners),
        "phones": phones,
        "spans": spans,
        "skipped": [],
    }


@pytest.mark.parametrize("decade,key", [
    ("90s", "90s"),
    ("'90s", "'90s"),
    ("’90s", "'90s"),
    ("90's", "90's"),
    ("90’s", "90's"),
    ("90s'", "90s'"),
    ("'90s'", "'90s'"),
    ("’90s’", "'90s'"),
])
def test_number_dictionary_claims_nineties_apostrophe_variants(native_g2p, tmp_path, decade, key):
    from g2pflow.converters.dictionary import DictionaryConverter, load_pronunciation_dict
    from timing.tifa_worker import number_dictionary

    path = number_dictionary(_clip(tmp_path, f"We played this in the {decade}."), MODELS, tmp_path)
    assert path is not None
    assert load_pronunciation_dict(path)[key][0] == NINETIES_PHONES
    assert DictionaryConverter(str(path)).accepts_word(key)

    words = _native_pipeline(path).convert(f"({decade})", languages=["en"])
    assert [(word.text, _phones(word)) for word in words] == [(key, NINETIES_PHONES)]


def test_number_dictionary_keeps_ordinary_integer_readings(native_g2p, tmp_path):
    from g2pflow.converters.dictionary import load_pronunciation_dict
    from timing.tifa_worker import number_dictionary

    path = number_dictionary(_clip(tmp_path, "It was 40 years"), MODELS, tmp_path)
    assert path is not None
    assert load_pronunciation_dict(path)["40"][0] == FORTY_PHONES

    words = _native_pipeline(path).convert("(40)", languages=["en"])
    assert [(word.text, _phones(word)) for word in words] == [("40", FORTY_PHONES)]


def test_decade_punctuation_and_curly_apostrophe_survive_map_and_export(native_g2p, tmp_path):
    from timing.schema import Line
    from timing.tifa import map_result
    from timing.tifa_worker import number_dictionary
    from timing.ttml import export

    original = "’90s, 40!"
    path = number_dictionary(_clip(tmp_path, original), MODELS, tmp_path)
    assert path is not None
    words = _native_pipeline(path).convert(original, languages=["en"])
    assert [(word.text, _phones(word)) for word in words] == [
        ("'90s", NINETIES_PHONES),
        ("40", FORTY_PHONES),
    ]

    line = Line(original, language="en")
    map_result(line, _row(words), 0, syllables=False)
    assert [word.text for word in line.words] == ["’90s, ", "40!"]
    assert "".join(word.text for word in line.words) == original
    assert all("number_reading_candidate" in word.flags for word in line.words)

    root = ET.fromstring(export([line], line.end, language="en"))
    spans = root.findall(".//{http://www.w3.org/ns/ttml}span")
    assert "".join(span.text or "" for span in spans) == original


def test_unrelated_alphanumeric_tokens_are_not_claimed(native_g2p, tmp_path):
    from timing.tifa_worker import number_dictionary

    text = "Tracks version90s, 90sx, 190s, 1990s, mp90s, x90s, and 90sec"
    assert number_dictionary(_clip(tmp_path, text), MODELS, tmp_path) is None

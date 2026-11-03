"""Pronunciation overrides for the native TIFA path.

dictionaries/pronunciations.json already drives the SOFA path. This module
brings the same table to TIFA without touching the pinned runtime:

* Japanese entries force a kana reading for the matching segmented word.
  Matching reuses the project tokenizer (fugashi + UniDic), mirroring the
  SOFA word-level semantics.
* English entries are injected as a pronunciation dictionary in the same
  way English number readings are injected.

Lyric text and display units are never rewritten; conversion failures raise
instead of being skipped.
"""
import json
from pathlib import Path

CONVERTER_ID = "timing-ja-overrides"
NUMBER_CONVERTER_ID = 'timing-ja-numbers'
LOANWORD_CONVERTER_ID = 'timing-ja-loanwords'
APOSTROPHE_PREPROCESSOR_ID = 'timing-apostrophes'


def ensure_apostrophes_registered():
    """Normalize before upstream punctuation filtering can split contractions."""
    from g2pflow.registry import preprocessor, get_preprocessor, list_preprocessors
    if APOSTROPHE_PREPROCESSOR_ID in list_preprocessors():
        return get_preprocessor(APOSTROPHE_PREPROCESSOR_ID)
    from g2pflow.preprocessors.base import Preprocessor
    from .text import canonical_apostrophes

    class ApostrophePreprocessor(Preprocessor):
        def process(self, tokens):
            return [canonical_apostrophes(token) for token in tokens]

    preprocessor(id=APOSTROPHE_PREPROCESSOR_ID)(ApostrophePreprocessor)
    return ApostrophePreprocessor


def prepare_configs(overrides, languages, folder, model_dir):
    """Write the effective override files and return g2p converter configs.

    The returned entries are inserted before the built-in converters so an
    override claim wins over the MeCab reading candidates for that word.
    Nothing is returned when the table has no entry for the active languages.
    """
    folder = Path(folder)
    model_dir = Path(model_dir)
    configs = []
    japanese = dict(overrides.get("ja") or {}) if "ja" in languages else {}
    if japanese:
        path = folder / "japanese_overrides.json"
        path.write_text(json.dumps({"ja": japanese}, ensure_ascii=False, indent=2), encoding="utf-8")
        configs.append({
            "id": CONVERTER_ID,
            "language": "ja",
            "kwargs": {
                "overrides_path": str(path),
                "dict_path": str(model_dir / "dictionaries" / "japanese_dict_full.txt"),
            },
        })
    english = dict(overrides.get("en") or {}) if "en" in languages else {}
    if english:
        rows = []
        for word, phones in english.items():
            for phone in phones:
                if any(ch.isspace() for ch in phone):
                    raise ValueError(f"英语自定义读音「{word}」的音素不能包含空白：{phone!r}")
            rows.append(word + "\t" + " ".join(phones))
        path = folder / "english_overrides.txt"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        configs.append({"id": "dictionary", "language": "en", "kwargs": {"dict_path": str(path)}})
    return configs


def ensure_registered():
    """Register the Japanese override converter with g2pflow (idempotent)."""
    from g2pflow.registry import converter as register
    from g2pflow.registry import get_converter, list_converters
    if CONVERTER_ID in list_converters():
        return get_converter(CONVERTER_ID)

    from itertools import product

    from g2pflow.converters.base import Converter, G2PReading, G2PWord
    from g2pflow.converters.japanese import JapaneseKanaConverter

    class JapaneseOverrideConverter(Converter):
        """Whole-word kana reading overrides for Japanese singing input."""

        def __init__(self, overrides_path, dict_path):
            self._kana = JapaneseKanaConverter(dict_path=dict_path)
            rows = json.loads(Path(overrides_path).read_text(encoding="utf-8-sig"))
            table = rows.get("ja") if isinstance(rows, dict) else None
            if not isinstance(table, dict) or not table:
                raise ValueError("日语自定义读音表为空或格式错误。")
            self._overrides = {str(word): str(reading) for word, reading in table.items()}
            self._longest = max(len(word) for word in self._overrides)
            self._spans_cache = {}

        def _spans(self, text):
            if text in self._spans_cache:
                return self._spans_cache[text]
            from timing.text import tagger
            spans, cursor = [], 0
            for node in tagger()(text):
                surface = node.surface
                if not surface:
                    continue
                index = text.find(surface, cursor)
                if index < 0:
                    continue
                spans.append((index, index + len(surface), surface))
                cursor = index + len(surface)
            if len(self._spans_cache) > 256:
                self._spans_cache.clear()
            self._spans_cache[text] = spans
            return spans

        def find(self, text):
            """Claim the earliest segmented word listed in the override table."""
            if not any(word in text for word in self._overrides):
                return None
            spans = self._spans(text)
            best = None
            for i, (start, _, _) in enumerate(spans):
                if best is not None and start > best[0]:
                    break
                joined = ""
                for j in range(i, len(spans)):
                    joined += spans[j][2]
                    if len(joined) > self._longest:
                        break
                    if joined in self._overrides:
                        best = (start, spans[j][1])
            return best

        def _reading(self, text, kana):
            try:
                words = self._kana._convert(kana)
            except KeyError as exc:
                raise ValueError(f"日语自定义读音「{text}」含无法转换的假名：{exc}") from exc
            alternatives = [[path for reading in word.readings for path in reading.paths] for word in words]
            if not alternatives or any(not paths for paths in alternatives):
                raise ValueError(f"日语自定义读音「{text}」无法转换为音素：{kana}")
            paths, seen = [], set()
            for parts in product(*alternatives):
                path = [group for part in parts for group in part]
                key = tuple((group.script, tuple(group.phonemes)) for group in path)
                if path and key not in seen:
                    seen.add(key)
                    paths.append(path)
            if not paths:
                raise ValueError(f"日语自定义读音「{text}」无法转换为音素：{kana}")
            return G2PReading(paths=paths)

        def _convert_word(self, text):
            kana = self._overrides.get(text)
            if kana is None:
                raise ValueError(f"日语自定义读音表没有「{text}」。")
            return G2PWord(text, readings=[self._reading(text, kana)])

        def _convert(self, text):
            return [self._convert_word(text)]

    register(id=CONVERTER_ID, language="ja")(JapaneseOverrideConverter)
    return JapaneseOverrideConverter


def ensure_numbers_registered():
    """Use the same context-sensitive numeric readings as MMS and SOFA."""
    from g2pflow.registry import converter as register, get_converter, list_converters
    if NUMBER_CONVERTER_ID in list_converters():
        return get_converter(NUMBER_CONVERTER_ID)
    from g2pflow.converters.base import G2PWord
    from g2pflow.converters.japanese import JapaneseKanaConverter
    from timing.numeric_lyrics import japanese_numeric_spans
    base = ensure_registered()

    class JapaneseNumberConverter(base):
        def __init__(self,dict_path):
            self._kana = JapaneseKanaConverter(dict_path=dict_path)

        def find(self,text):
            span = next(japanese_numeric_spans(text),None)
            return span[:2] if span else None

        def _convert_word(self,text):
            spans = list(japanese_numeric_spans(text))
            if len(spans)!=1 or spans[0][:2]!=(0,len(text)):
                raise ValueError(f'无法转换日语数字词：{text}')
            return G2PWord(text,readings=[self._reading(text,spans[0][2])])

    register(id=NUMBER_CONVERTER_ID,language='ja')(JapaneseNumberConverter)
    return JapaneseNumberConverter


def ensure_loanwords_registered():
    from g2pflow.registry import converter as register, get_converter, list_converters
    if LOANWORD_CONVERTER_ID in list_converters():
        return get_converter(LOANWORD_CONVERTER_ID)
    from g2pflow.converters.base import G2PWord
    from g2pflow.converters.japanese import JapaneseKanaConverter
    from .japanese_loanwords import japanese_loanword_spans
    base = ensure_registered()

    class JapaneseLoanwordConverter(base):
        def __init__(self, dict_path):
            self._kana = JapaneseKanaConverter(dict_path=dict_path)

        def find(self, text):
            span = next(japanese_loanword_spans(text), None)
            return span[:2] if span else None

        def _convert_word(self, text):
            spans = list(japanese_loanword_spans(text))
            if len(spans) != 1 or spans[0][:2] != (0, len(text)):
                raise ValueError(f'无法转换日语混写词：{text}')
            return G2PWord(text, readings=[self._reading(text, spans[0][2])])

    register(id=LOANWORD_CONVERTER_ID, language='ja')(JapaneseLoanwordConverter)
    return JapaneseLoanwordConverter

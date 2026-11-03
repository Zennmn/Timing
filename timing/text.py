"""Preserve display text exactly; alignment units are explicit model inputs."""
import json
import re
import unicodedata
import math
from functools import lru_cache

from .schema import Line, Unit, valid_agent
from .english_syllables import split_english, current_method

LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’][A-Za-z]+)*|[0-9]+")
JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff々〆〤ー]+")
SMALL = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
LRC = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)$")
LRC_END = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]$")
LRC_INLINE_TIME = re.compile(r"[\[<]\s*-?\d+:")


def kept(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text) if unicodedata.category(c)[0] in "LN" or c in "'’")


def canonical_apostrophes(text):
    """Model spelling only; display text always retains the original glyph."""
    return text.replace('’', "'")


@lru_cache(maxsize=1)
def tagger():
    from .japanese_dictionary import make_tagger
    return make_tagger()


@lru_cache(maxsize=1)
def hyphenator():
    import pyphen
    return pyphen.Pyphen(lang="en_US")


def japanese_units(text, fine):
    result = []
    for word in tagger()(text):
        surface = word.surface
        reading = getattr(word.feature, "kana", "") or ""
        pronunciation = getattr(word.feature, "pron", "") or reading
        if not fine:
            result.append(Unit(surface, kept(surface), reading, source_pronunciation=pronunciation))
            continue
        # Keep kanji groups indivisible: 今日 may be きょう, 明日 may be あした.
        parts = re.findall(r"[\u3400-\u9fff々〆〤]+|[\u3040-\u30ffー]", surface)
        merged = []
        for part in parts:
            if part in SMALL and merged and not re.search(r"[\u3400-\u9fff]", merged[-1]):
                merged[-1] += part
            else:
                merged.append(part)
        for part in merged:
            result.append(Unit(part, kept(part), reading if len(merged) == 1 else "",
                               flags=["japanese_fine_unit"] if len(merged) > 1 else []))
    return result


def tokenize(text, fine=True, english_syllables=True):
    from .numeric_lyrics import japanese_numeric_spans
    from .japanese_loanwords import japanese_loanword_spans
    numeric = {start:(end,reading) for start,end,reading in japanese_numeric_spans(text)}
    loanwords = {start:(end,reading) for start,end,reading in japanese_loanword_spans(text)}
    units = []
    leading = ""
    pos = 0
    while pos < len(text):
        latin = LATIN.match(text, pos)
        japanese = JAPANESE.match(text, pos)
        if pos in loanwords:
            end,reading = loanwords[pos]
            surface = text[pos:end]
            units.append(Unit(surface,kept(surface),reading,source_pronunciation=reading))
            pos = end
        elif pos in numeric:
            end,reading = numeric[pos]
            surface = text[pos:end]
            units.append(Unit(surface,kept(surface),reading,flags=['number_reading_candidate'],
                              source_pronunciation=reading))
            pos = end
        elif latin:
            word = latin.group()
            parts = [word]
            if fine and english_syllables and word.isalpha() and word.isascii():
                if current_method() == "pyphen":
                    parts = hyphenator().inserted(word, hyphen="|").split("|")
                else:
                    division = split_english(word)
                    if division:
                        parts = [part for part,_,_ in division]
            units.extend(Unit(p, kept(p), flags=["english_syllable_candidate"] if len(parts) > 1 else []) for p in parts)
            pos = latin.end()
        elif japanese:
            units.extend(japanese_units(japanese.group(), fine))
            pos = japanese.end()
        else:
            if units:
                units[-1].text += text[pos]
            else:
                leading += text[pos]
            pos += 1
    if leading and units:
        units[0].text = leading + units[0].text
    if "".join(u.text for u in units) != text and kept(text):
        raise ValueError("分词未能保留原歌词，请检查特殊字符。")
    return [u for u in units if u.spoken]


def parse_lrc(text: str) -> list[Line]:
    """Read plain/LRC rows, leaving omitted ends unset for the caller to infer."""
    lines = []
    for raw in text.lstrip("\ufeff").splitlines():
        raw = raw.strip()
        if not raw or re.match(r"^\[(?:ar|ti|al|by|offset|length):", raw, re.I):
            continue
        match = LRC.match(raw)
        if match:
            body = match[3].strip()
            if LRC.match(body):
                raise ValueError("请把多个 LRC 时间标签展开为独立歌词行，以保留每次副歌。")
            tail = LRC_END.search(body)
            end = None
            if tail:
                end = float(tail[1]) * 60 + float(tail[2])
                body = body[:tail.start()].rstrip()
            if LRC_INLINE_TIME.search(body):
                raise ValueError("LRC 每行仅支持 [开始]歌词 或 [开始]歌词[结束]；请检查时间标签。")
            if body:
                start = float(match[1]) * 60 + float(match[2])
                lines.append(Line(body, start=start, end=end))
        elif not (raw.startswith("[") and raw.endswith("]")):
            lines.append(Line(raw))
    return lines


def parse_lyrics(text: str, duration: float | None) -> list[Line]:
    text = text.lstrip("\ufeff").strip()
    if not text:
        return []
    if text.startswith('<'):
        from .ttml_input import parse_ttml
        lines = parse_ttml(text)
        validate_lines(lines, duration)
        return lines
    first = text.splitlines()[0]
    metadata = re.match(r"^\[(?:ar|ti|al|by|offset|length):", first, re.I)
    section = re.fullmatch(r"\[[A-Za-z\u3040-\u30ff\u3400-\u9fff\u1100-\u11ff\u3130-\u318f\uac00-\ud7af][A-Za-z\u3040-\u30ff\u3400-\u9fff\u1100-\u11ff\u3130-\u318f\uac00-\ud7af 0-9_-]*\]", first)
    if text.startswith('{') or (text.startswith('[') and not LRC.match(first) and not metadata and not section):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError('JSON 歌词格式有误，请检查括号、引号和逗号。') from None
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError('JSON 歌词应为数组，每一项是一行歌词。')
        allowed = {"text", "start", "end", "agent", "role", "parent", "language", "translation", "roman"}
        if any('text' not in row for row in rows):
            raise ValueError('每行 JSON 歌词都需要 text 字段。')
        lines = [Line(**{k: v for k, v in row.items() if k in allowed}) for row in rows]
        validate_lines(lines, duration)
        return lines
    lines = parse_lrc(text)
    next_start = duration
    for line in reversed(lines):
        if line.start is not None:
            if line.end is None:
                line.end = next_start if next_start is not None else line.start + 1
            next_start = line.start
    validate_lines(lines, duration)
    return lines


def validate_lines(lines, duration):
    if not lines:
        raise ValueError("没有可对齐的歌词。")
    previous_main_start = -1
    for i, line in enumerate(lines):
        if not isinstance(line.text, str) or not kept(line.text):
            raise ValueError(f"第 {i + 1} 行没有可演唱文字。")
        if not isinstance(line.translation, str) or not isinstance(line.roman, str):
            raise ValueError(f'第 {i + 1} 行的 translation 和 roman 必须是文字。')
        if line.role not in ("main", "bg") or not valid_agent(line.agent):
            raise ValueError("role 仅支持 main/bg，agent 请使用 v1、v2、v3 等正整数歌手标记（最多9位数字）。")
        if line.language not in ("auto", "ja", "en", "mixed", "zh", "zh-en"):
            raise ValueError("language 仅支持 auto、ja、en、mixed、zh、zh-en。")
        if (line.start is None) != (line.end is None):
            raise ValueError(f"第 {i + 1} 行必须同时指定 start 和 end。")
        if line.start is not None:
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in (line.start, line.end)):
                raise ValueError(f'第 {i + 1} 行的 start 和 end 必须是有限数字，单位为秒。')
            if not 0 <= line.start < line.end or (duration is not None and line.end > duration + .05):
                raise ValueError(f"第 {i + 1} 行时间范围超出音频，或结束时间不晚于开始时间。")
            if line.role == 'main':
                if line.start < previous_main_start:
                    raise ValueError('主唱歌词请按开始时间排序；重叠演唱可以保留。')
                previous_main_start = line.start
        if line.role == "bg":
            if type(line.parent) is not int or not 0 <= line.parent < len(lines) or lines[line.parent].role != "main":
                raise ValueError("背景行必须通过 parent 指向主唱行的零起始下标。")
            if line.start is None:
                raise ValueError("重叠背景歌词请提供 start/end 时间范围。")

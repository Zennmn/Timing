"""AMLL-compatible timed TTML; validate all spans before producing an artifact."""
import math
import xml.etree.ElementTree as ET

TT = "http://www.w3.org/ns/ttml"
TTM = "http://www.w3.org/ns/ttml#metadata"
ITUNES = "http://music.apple.com/lyric-ttml-internal"
XML = "http://www.w3.org/XML/1998/namespace"
ET.register_namespace("", TT)
ET.register_namespace("ttm", TTM)
ET.register_namespace("itunes", ITUNES)


def stamp(seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("时间戳必须是非负有限值。")
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{sec:02}.{ms:03}"


def unresolved(word):
    return 'tifa_unresolved_unit' in word.flags and word.start is None and word.end is None


def bounds(line):
    return (line.words[0].start if line.words[0].start is not None else line.start,
            line.words[-1].end if line.words[-1].end is not None else line.end)


def export_times(line, index):
    """Unresolved text uses a zero-duration editing marker at a native edge.

    This supplies the attributes required by AMLL without inventing a positive
    word duration or moving any aligned unit. JSON times remain unset.
    """
    word = line.words[index]
    if not unresolved(word):
        return word.start,word.end
    after = next((w.start for w in line.words[index+1:] if w.start is not None),None)
    before = next((w.end for w in reversed(line.words[:index]) if w.end is not None),None)
    point = after if after is not None else before
    if point is None:
        raise ValueError("待补轴单位没有可用的原生边界占位。")
    return point,point


def check(lines, duration):
    for i, line in enumerate(lines):
        if not line.words or "".join(w.text for w in line.words) != line.text:
            raise ValueError(f"第 {i + 1} 行文字丢失，未导出。")
        previous_end = -1
        for word in line.words:
            if unresolved(word):
                continue
            if word.start is None or word.end is None or not all(math.isfinite(x) for x in (word.start, word.end)):
                raise ValueError(f"第 {i + 1} 行存在未对齐单位。")
            if not (0 <= word.start < word.end <= duration + .001) or round(word.start * 1000) >= round(word.end * 1000):
                raise ValueError(f"第 {i + 1} 行「{word.text}」时间无效，请缩小范围重新对齐。")
            if word.start + .001 < previous_end:
                raise ValueError(f"第 {i + 1} 行单位顺序冲突，请重新对齐。")
            previous_end = word.end
        if any(unresolved(w) for w in line.words):
            begin,end = bounds(line)
            if (not any(w.start is not None for w in line.words) or
                    begin is None or end is None or not all(math.isfinite(x) for x in (begin,end)) or
                    not 0 <= begin < end <= duration + .001):
                raise ValueError(f"第 {i + 1} 行部分对齐范围无效。")
            if any(w.start is not None and not begin <= w.start < w.end <= end for w in line.words):
                raise ValueError(f"第 {i + 1} 行单位超出部分对齐范围。")


def export(lines, duration, language="ja", title=""):
    check(lines, duration)
    export_language = ('zh-CN' if language in ('zh', 'zh-en')
                       else 'en' if language == 'en' else 'mul' if language == 'auto' else 'ja')
    root = ET.Element(f"{{{TT}}}tt", {f"{{{XML}}}lang": export_language})
    head = ET.SubElement(root, f"{{{TT}}}head")
    metadata = ET.SubElement(head, f"{{{TT}}}metadata")
    if title:
        ET.SubElement(metadata, f"{{{TTM}}}title").text = title
    for agent in sorted({line.agent for line in lines}):
        ET.SubElement(metadata, f"{{{TTM}}}agent", {f"{{{XML}}}id": agent, "type": "person"})
    div = ET.SubElement(ET.SubElement(root, f"{{{TT}}}body", {"dur": stamp(duration)}), f"{{{TT}}}div")

    def timed_words(parent, line):
        for index,w in enumerate(line.words):
            start,end = export_times(line,index)
            attrs = {"begin": stamp(start), "end": stamp(end)}
            ET.SubElement(parent, f"{{{TT}}}span", attrs).text = w.text
        if line.translation:
            ET.SubElement(parent, f"{{{TT}}}span", {f"{{{TTM}}}role": "x-translation", f"{{{XML}}}lang": "zh-CN"}).text = line.translation
        if line.roman:
            effective_language = line.language if line.language != 'auto' else language
            roman_language = ('zh-Latn' if effective_language in ('zh','zh-en')
                              else 'en' if effective_language == 'en' else 'ja-Latn')
            ET.SubElement(parent, f"{{{TT}}}span", {f"{{{TTM}}}role": "x-roman", f"{{{XML}}}lang": roman_language}).text = line.roman

    mains = sorted(((i, line) for i, line in enumerate(lines) if line.role == "main"), key=lambda pair: bounds(pair[1])[0])
    for seq, (idx, line) in enumerate(mains, 1):
        backgrounds = [b for b in lines if b.role == "bg" and b.parent == idx]
        group = [line] + backgrounds
        begin = min(bounds(x)[0] for x in group)
        end = max(bounds(x)[1] for x in group)
        p = ET.SubElement(div, f"{{{TT}}}p", {"begin": stamp(begin), "end": stamp(end), f"{{{TTM}}}agent": line.agent, f"{{{ITUNES}}}key": f"L{seq}"})
        if line.language in ('zh', 'zh-en'):
            p.set(f'{{{XML}}}lang', 'zh-CN')
        elif line.language in ('ja','en'):
            p.set(f'{{{XML}}}lang', line.language)
        for bg in [x for x in backgrounds if bounds(x)[0] < bounds(line)[0]]:
            el = ET.SubElement(p, f"{{{TT}}}span", {f"{{{TTM}}}role": "x-bg", "begin": stamp(bounds(bg)[0]), "end": stamp(bounds(bg)[1])})
            timed_words(el, bg)
        timed_words(p, line)
        for bg in [x for x in backgrounds if bounds(x)[0] >= bounds(line)[0]]:
            el = ET.SubElement(p, f"{{{TT}}}span", {f"{{{TTM}}}role": "x-bg", "begin": stamp(bounds(bg)[0]), "end": stamp(bounds(bg)[1])})
            timed_words(el, bg)
    # Lyric paragraphs are mixed content. Pretty-print indentation would add
    # whitespace between timed spans, changing their rendered display text.
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)

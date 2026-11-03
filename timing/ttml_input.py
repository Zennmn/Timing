"""Read AMLL line-timed TTML as anchored lyrics, without reusing word timing."""
import json
import math
import re
from dataclasses import asdict
from decimal import Decimal

from defusedxml import ElementTree as ET
from defusedxml.common import DefusedXmlException

from .schema import Line, valid_agent

TT = 'http://www.w3.org/ns/ttml'
TTM = 'http://www.w3.org/ns/ttml#metadata'
TTP = 'http://www.w3.org/ns/ttml#parameter'
XML = 'http://www.w3.org/XML/1998/namespace'
ITUNES = 'http://music.apple.com/lyric-ttml-internal'
MAX_BYTES = 2 * 1024**2
ANNOTATIONS = {'x-translation': 'translation', 'x-roman': 'roman'}


def is_ttml(text):
    return isinstance(text, str) and text.lstrip('\ufeff \t\r\n').startswith('<')


def parse_time(value):
    """Media clock (hours optional) or a media offset in h/m/s/ms."""
    value = str(value).strip()
    clock = re.fullmatch(r'(?:(\d+):)?(\d{1,2}):(\d{2}(?:\.\d+)?)', value)
    offset = re.fullmatch(r'(\d+(?:\.\d+)?)(ms|s|m|h)?', value)
    try:
        if clock and int(clock[2]) < 60 and Decimal(clock[3]) < 60:
            seconds = float(Decimal(clock[1] or 0) * 3600 + Decimal(clock[2]) * 60 + Decimal(clock[3]))
        elif offset:
            seconds = float(Decimal(offset[1]) * {'ms': Decimal('.001'), 's': 1, 'm': 60, 'h': 3600, None: 1}[offset[2]])
        else:
            raise ValueError('TTML 时间格式无效；请使用 时:分:秒.毫秒、分:秒.毫秒 或秒数，不支持帧和 tick。')
    except OverflowError:
        raise ValueError('TTML 时间必须是有限数字。') from None
    if not math.isfinite(seconds):
        raise ValueError('TTML 时间必须是有限数字。')
    return seconds


def _role(node):
    return node.get(f'{{{TTM}}}role', '')


def _text(node, namespace, *, skip_roles=False, preserve=False, trim=False):
    """Join XML mixed content; only collapse XML whitespace in default mode."""
    chunks = []

    def append(value, keep):
        if value:
            if not keep and not value.strip(' \t\r\n') and ('\n' in value or '\r' in value):
                return  # AMLL ignores indentation-only nodes between spans.
            value = value if keep else re.sub(r'[\t\r\n ]+', ' ', value)
            if not keep and chunks and not chunks[-1][1] and chunks[-1][0].endswith(' ') and value.startswith(' '):
                value = value[1:]
            if value:
                chunks.append([value, keep])

    def visit(parent, keep):
        keep = {'preserve': True, 'default': False}.get(parent.get(f'{{{XML}}}space'), keep)
        append(parent.text, keep)
        for child in parent:
            if not (skip_roles and _role(child)):
                if child.tag == f'{namespace}br':
                    append('\n', True)
                else:
                    visit(child, keep)
            append(child.tail, keep)

    visit(node, preserve)
    if trim:
        for chunk in chunks:
            if chunk[1]:
                break
            chunk[0] = chunk[0].lstrip(' \t\r\n')
            if chunk[0]:
                break
        for chunk in reversed(chunks):
            if chunk[1]:
                break
            chunk[0] = chunk[0].rstrip(' \t\r\n')
            if chunk[0]:
                break
    return ''.join(value for value, _ in chunks)


def _content(node, namespace, preserve):
    text = _text(node, namespace, skip_roles=True, preserve=preserve, trim=True)
    annotations = {'translation': '', 'roman': ''}
    seen = set()

    def visit(parent, keep):
        keep = {'preserve': True, 'default': False}.get(parent.get(f'{{{XML}}}space'), keep)
        for child in parent:
            role = _role(child)
            if role in ANNOTATIONS:
                field = ANNOTATIONS[role]
                if field in seen:
                    raise ValueError('同一句 TTML 有多份翻译或罗马音；当前每行只支持各一份，请先选择需要保留的版本。')
                seen.add(field)
                annotations[field] = _text(child, namespace, preserve=keep, trim=True)
            elif role != 'x-bg':
                visit(child, keep)

    visit(node, preserve)
    return text, annotations


def _range(node, inherited=None):
    begin = parse_time(node.get('begin')) if 'begin' in node.attrib else inherited[0] if inherited else None
    end = parse_time(node.get('end')) if 'end' in node.attrib else None
    if 'dur' in node.attrib:
        if begin is None:
            raise ValueError('TTML 每句需要明确的 begin 和 end（或 dur）。')
        duration_end = begin + parse_time(node.get('dur'))
        end = min(end, duration_end) if end is not None else duration_end
    if end is None and inherited:
        end = inherited[1]
    if begin is None or end is None or not 0 <= begin < end:
        raise ValueError('TTML 每句需要有效的 begin 和 end（或 dur），且结束晚于开始。')
    return begin, end


def _background_range(node, inherited):
    """Find a background line's range without treating annotations as vocals."""
    ranges = []

    def visit(current, bounds):
        begin = parse_time(current.get('begin')) if 'begin' in current.attrib else bounds[0]
        end = parse_time(current.get('end')) if 'end' in current.attrib else bounds[1]
        if 'dur' in current.attrib:
            duration_end = begin + parse_time(current.get('dur'))
            end = min(end, duration_end) if 'end' in current.attrib else duration_end

        def add_text(value):
            # Untimed punctuation around timed words does not extend the singing.
            if value and any(char.isalnum() for char in value):
                if not 0 <= begin < end:
                    raise ValueError('TTML 伴唱需要有效的开始和结束时间。')
                ranges.append((begin, end))

        add_text(current.text)
        for child in current:
            if not _role(child):
                visit(child, (begin, end))
            add_text(child.tail)

    visit(node, inherited)
    if not ranges:
        return _range(node, inherited)
    begin = parse_time(node.get('begin')) if 'begin' in node.attrib else min(item[0] for item in ranges)
    duration_start = parse_time(node.get('begin')) if 'begin' in node.attrib else inherited[0]
    if 'end' in node.attrib:
        end = parse_time(node.get('end'))
        if 'dur' in node.attrib:
            end = min(end, duration_start + parse_time(node.get('dur')))
    elif 'dur' in node.attrib:
        end = duration_start + parse_time(node.get('dur'))
    else:
        end = max(item[1] for item in ranges)
    if not 0 <= begin < end:
        raise ValueError('TTML 伴唱需要有效的开始和结束时间。')
    return begin, end


def parse_ttml(text):
    if not isinstance(text, str) or len(text.encode('utf-8')) > MAX_BYTES:
        raise ValueError('歌词最大 2 MB。')
    try:
        root = ET.fromstring(text.lstrip('\ufeff \t\r\n'), forbid_dtd=True)
    except (ET.ParseError, DefusedXmlException, ValueError):
        raise ValueError('TTML XML 格式有误，或包含不支持的 DTD / 实体声明。') from None
    if root.tag not in (f'{{{TT}}}tt', 'tt'):
        raise ValueError('请选择 TTML 歌词文件，根元素应为 TTML 的 tt。')
    namespace = f'{{{TT}}}' if root.tag.startswith('{') else ''
    pending = [(root, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if depth > 64 or count > 50000:
            raise ValueError('TTML 嵌套过深或节点过多，请简化歌词文件。')
        pending.extend((child, depth + 1) for child in node)
    if root.get(f'{{{TTP}}}timeBase', 'media') != 'media':
        raise ValueError('仅支持 media 时间基准的 TTML；请先转换为歌曲起点的时间。')
    bodies = root.findall(f'{namespace}body')
    if len(bodies) != 1:
        raise ValueError('TTML 需要一个 body 歌词正文。')
    agents = {}
    lines = []
    keyed_lines = {}

    def agent_id(value):
        value = value.strip() or 'v1'
        if len(value.split()) != 1:
            raise ValueError('TTML 同一句的多歌手标记暂不支持，请拆为独立歌词行。')
        if valid_agent(value):
            agents.setdefault(value, value)
            return value
        if value not in agents:
            number = 1
            while f'v{number}' in agents.values():
                number += 1
            agents[value] = f'v{number}'
        return agents[value]

    # Reserve explicit ids before mapping arbitrary TTML agent identifiers.
    for node in bodies[0].iter():
        value = node.get(f'{{{TTM}}}agent', '')
        if valid_agent(value):
            agents[value] = value

    def walk(node, agent='', preserve=False, section=(None, None)):
        # xml:lang describes the source document, not the pronunciation route.
        agent = node.get(f'{{{TTM}}}agent', agent)
        preserve = {'preserve': True, 'default': False}.get(node.get(f'{{{XML}}}space'), preserve)
        if node.get('timeContainer', 'par') != 'par':
            raise ValueError('暂不支持 TTML seq 相对时间容器，请导出使用歌曲绝对时间的 TTML。')
        if node.tag == f'{namespace}p':
            add_line(node, agent, preserve)
            begin, end = _range(node)
            if ((section[0] is not None and begin < section[0] - .001) or
                    (section[1] is not None and end > section[1] + .001)):
                raise ValueError('TTML 句子时间超出 div 段落范围；需要使用歌曲绝对时间，不能使用相对偏移。')
            return
        if node.tag not in (f'{namespace}body', f'{namespace}div'):
            raise ValueError('TTML 正文仅支持 body / div / p 歌词结构。')
        # AMLL section div clocks are absolute, just like paragraph clocks.
        # They are never added to p.begin. Parent-relative TTML is out of scope.
        if node.tag == f'{namespace}body' and any(key in node.attrib for key in ('begin', 'end')):
            raise ValueError('暂不支持 body 带 begin/end 的相对 TTML，请导出歌曲绝对时间的逐行 TTML。')
        for key in ('begin', 'end', 'dur'):
            if key in node.attrib:
                parse_time(node.get(key))
        if node.tag == f'{namespace}div':
            left = parse_time(node.get('begin')) if 'begin' in node.attrib else section[0]
            right = parse_time(node.get('end')) if 'end' in node.attrib else section[1]
            if left is not None and right is not None and left >= right:
                raise ValueError('TTML div 段落时间无效。')
            section = (max(left, section[0]) if left is not None and section[0] is not None else left,
                       min(right, section[1]) if right is not None and section[1] is not None else right)
        for child in node:
            walk(child, agent, preserve, section)

    def add_line(node, agent, preserve):
        if _role(node):
            raise ValueError('TTML 翻译和伴唱请使用歌词 p 内的 span 标记。')
        for child in node.iter():
            if child.tag not in (f'{namespace}p', f'{namespace}span', f'{namespace}br'):
                raise ValueError('TTML 歌词句内仅支持 span 和 br。')
            if _role(child) not in ('', 'x-bg', *ANNOTATIONS):
                raise ValueError('TTML 包含暂不支持的歌词角色标记，请先整理为主歌词、伴唱、翻译和罗马音。')
            if _role(child) in ANNOTATIONS and any(_role(descendant) for descendant in child.iter() if descendant is not child):
                raise ValueError('TTML 内联翻译或罗马音中不能嵌套伴唱等角色，请把标注放到对应的 x-bg 歌词内。')
            if child.get('timeContainer', 'par') != 'par':
                raise ValueError('暂不支持 TTML seq 相对时间容器，请导出使用歌曲绝对时间的 TTML。')
            for key in ('begin', 'end', 'dur'):
                if key in child.attrib:
                    parse_time(child.get(key))
        start, end = _range(node)
        text_value, notes = _content(node, namespace, preserve)
        if not text_value.strip() and not list(node):
            return  # Empty timed paragraphs are interlude markers, not lyrics.
        parent = len(lines)
        main = Line(text_value, start, end, agent=agent_id(agent), language='auto', **notes)
        lines.append(main)
        key = node.get(f'{{{ITUNES}}}key')
        if key:
            if key in keyed_lines:
                raise ValueError('TTML 包含重复的 itunes:key，无法关联翻译，请为每句保留唯一行 ID。')
            keyed_lines[key] = [parent]
        for background in node.iter(f'{namespace}span'):
            if _role(background) != 'x-bg':
                continue
            if any(_role(child) == 'x-bg' for child in background.iter() if child is not background):
                raise ValueError('TTML 不支持嵌套多层伴唱，请拆成同级 x-bg。')
            bg_preserve = {'preserve': True, 'default': False}.get(background.get(f'{{{XML}}}space'), preserve)
            bg_text, bg_notes = _content(background, namespace, bg_preserve)
            bg_start, bg_end = _background_range(background, (start, end))
            lines.append(Line(bg_text, bg_start, bg_end, role='bg', parent=parent,
                              agent=agent_id(background.get(f'{{{TTM}}}agent', agent)),
                              language='auto', **bg_notes))
            if key:
                keyed_lines[key].append(len(lines) - 1)

    walk(bodies[0], root.get(f'{{{TTM}}}agent', ''),
         root.get(f'{{{XML}}}space') == 'preserve')
    _sidecars(root, namespace, lines, keyed_lines)
    return lines


def _sidecars(root, namespace, lines, keyed_lines):
    """Apple stores subtitle translations/transliterations in head by line key."""
    for group, track_name, field in (('translations', 'translation', 'translation'),
                                    ('transliterations', 'transliteration', 'roman')):
        tracks = root.findall(f'{namespace}head/.//{{{ITUNES}}}{group}/{{{ITUNES}}}{track_name}')
        if not tracks:
            continue
        # The application has one subtitle field; use XML document order.
        # Track language labels are metadata and are not reliable selectors.
        track = tracks[0]
        seen = set()

        def assign(index, text):
            existing = getattr(lines[index], field)
            if existing and existing != text:
                raise ValueError('TTML 内联与 head 中的翻译或罗马音冲突，请保留需要的版本。')
            setattr(lines[index], field, text)

        for entry in track:
            if entry.tag != f'{{{ITUNES}}}text':
                raise ValueError('Apple TTML 翻译结构暂不支持，请使用通过 for 关联行 ID 的 text。')
            key = entry.get('for', '')
            if key not in keyed_lines or key in seen:
                raise ValueError('Apple TTML 翻译的 for 行 ID 不存在或重复，无法安全关联。')
            seen.add(key)
            indices = keyed_lines[key]
            value = _text(entry, namespace, skip_roles=True, trim=True)
            assign(indices[0], value)
            bg_texts = [child for child in entry.iter() if _role(child) == 'x-bg']
            if len(bg_texts) > 1 or (bg_texts and len(indices) != 2):
                raise ValueError('Apple TTML 背景翻译无法与唯一的伴唱行对应，请整理 x-bg。')
            if bg_texts:
                assign(indices[1], _text(bg_texts[0], namespace, trim=True))


def normalize_ttml(text):
    """Browser bridge: use the exact same parser and validators as CLI/jobs."""
    from .text import validate_lines
    lines = parse_ttml(text)
    validate_lines(lines, None)
    rows = [{key: value for key, value in asdict(line).items() if key not in ('words', 'flags')} for line in lines]
    lyrics = json.dumps(rows, ensure_ascii=False, indent=2)
    if len(lyrics.encode('utf-8')) > MAX_BYTES:
        raise ValueError('转换后的歌词超过 2 MB，请缩小 TTML 文件。')
    return {'lyrics': lyrics, 'lines': len(lines)}

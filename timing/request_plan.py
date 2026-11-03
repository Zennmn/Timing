"""Shared request validation for the API, CLI and alignment pipeline."""
from dataclasses import dataclass
import re

from .schema import Line
from .text import parse_lyrics

ALIGNERS = ('auto', 'tifa', 'ctc', 'llm_text', 'llm_direct', 'llm_import')
AUDIO_MODES = ('lead', 'vocals', 'source')
SPLITTERS = ('syllabify', 'pyphen', 'llm')
LANGUAGES = ('mixed', 'ja', 'en', 'auto', 'zh', 'zh-en')
MAX_LYRICS_BYTES = 2 * 1024**2


@dataclass(frozen=True)
class RequestPlan:
    lines: list[Line]
    aligner: str
    audio_mode: str
    syllables: bool
    english_splitter: str
    needs_llm: bool
    has_background: bool
    trust_line_timestamps: bool = False


def plan_request(lyrics, *, duration=None, language='mixed', aligner='auto',
                 audio_mode='lead', syllables=True, english_splitter='syllabify',
                 trust_line_timestamps=False):
    if language not in LANGUAGES:
        raise ValueError('不支持的语言选项。')
    if aligner not in ALIGNERS:
        raise ValueError('不支持的对齐模型。')
    if audio_mode not in AUDIO_MODES:
        raise ValueError('请选择只保留主唱、保留全部人声，或直接使用音轨。')
    if english_splitter not in SPLITTERS:
        raise ValueError('不支持的英语音节划分方案。')
    if not isinstance(trust_line_timestamps, bool):
        raise ValueError('信任逐句时间轴必须为布尔值。')
    if not isinstance(lyrics, str) or not lyrics.strip('\ufeff \t\r\n'):
        raise ValueError('请粘贴歌词，或导入 TXT、LRC、JSON、逐行 TTML 歌词文件。')
    if len(lyrics.encode('utf-8')) > MAX_LYRICS_BYTES:
        raise ValueError('歌词最大 2 MB。')
    lines = parse_lyrics(lyrics, duration)
    for line in lines:
        if line.language == 'auto':
            line.language = language
    anchored = [line.start is not None for line in lines]
    mains = [line for line in lines if line.role == 'main']
    if not mains:
        raise ValueError('至少需要一行主歌词。')
    main_anchors = [line.start is not None for line in mains]
    if any(main_anchors) and not all(main_anchors):
        raise ValueError('主唱歌词需全部有逐句时间，或全部使用纯文本。')
    if not any(main_anchors) and len({line.agent for line in mains}) > 1:
        raise ValueError('多位主唱的纯文本歌词可能重叠，请分别处理，或提供逐句 LRC／TTML 时间。')
    if aligner == 'auto':
        aligner = 'tifa'
    if trust_line_timestamps:
        if not all(anchored):
            raise ValueError('信任逐句时间轴需要每行都有时间，请提供 LRC／TTML／带完整时间的 JSON。')
        if aligner == 'ctc':
            raise ValueError('跳过 MMS 粗定位仅适用于 TIFA 及 LLM + TIFA 模式。')
    if aligner in ('llm_text', 'llm_direct', 'llm_import') and not all(anchored):
        raise ValueError('LLM 模式需要逐句 LRC／TTML／带完整时间的 JSON；纯文本请使用自动、TIFA 或 MMS。')
    if aligner in ('llm_text', 'llm_direct', 'llm_import'):
        # Each LLM determines language; input tags must not constrain its output
        # or silently select a different dictionary on a failed text response.
        for line in lines:
            line.language = 'auto'
    english = any(re.search(r'[A-Za-z]', line.text) for line in lines)
    direct = aligner in ('llm_text', 'llm_direct', 'llm_import')
    effective_syllables = bool(syllables and english and not direct)
    splitter = english_splitter if effective_syllables else 'syllabify'
    # MMS uses spelling labels, so a dormant phoneme splitter never requires API credentials.
    if aligner == 'ctc' and splitter == 'llm':
        splitter = 'syllabify'
    needs_llm = aligner in ('llm_text', 'llm_direct') or (effective_syllables and splitter == 'llm')
    return RequestPlan(lines, aligner, audio_mode, effective_syllables, splitter,
                       bool(needs_llm), any(line.role == 'bg' for line in lines), trust_line_timestamps)

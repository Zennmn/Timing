"""Contextual text-to-reading input; time and audio stay in the local aligner."""
import hashlib
import json
import re
from pathlib import Path

from .llm_direct import compile_parts, phone_inventory
from .llm_splitter import LlmSession

PROMPT = Path(__file__).with_name('prompts') / 'llm_text.txt'
TIMEOUT = 600
MAX_RESPONSE_BYTES = 16 * 1024**2


def local_parts(line):
    """Build validated fallback input without a second provider request."""
    from .chinese import HAN_RE
    from .singing import pronunciation_overrides, english_parts, japanese_parts
    from .schema import Unit
    from .text import LATIN, tokenize
    from .english_syllables import use_splitter
    # Supplied language labels must not become a hidden fallback constraint.
    # Kana gives a local Japanese baseline; Han alone does not distinguish
    # Japanese from Mandarin, so do not manufacture one after an LLM failure.
    kana = bool(re.search(r'[\u3041-\u3096\u30a1-\u30fa]', line.text))
    if HAN_RE.search(line.text) and not kana:
        raise ValueError('仅凭汉字无法确定本行语言；LLM 读音未通过，需人工补轴。')
    if not kana and not re.search(r'[A-Za-zÀ-ÖØ-öø-ÿ]', line.text):
        raise ValueError('本地词典无法在没有语言提示时确定本行读音。')
    words = tokenize(line.text, fine=False)
    overrides = pronunciation_overrides()
    result = []
    for index, word in enumerate(words):
        if LATIN.fullmatch(word.spoken) and (not word.spoken.isdigit() or not kana):
            with use_splitter('syllabify'):
                phones = english_parts(word, False, overrides['en'])[0][1]
            result.append({'text': word.text, 'lang': 'en', 'reading': [p.lower() for p in phones]})
        else:
            reading = overrides['ja'].get(word.spoken) or word.source_pronunciation or word.reading
            if not reading:
                raise ValueError('本地词典没有可用的日语读音。')
            for text, sound in japanese_parts(Unit(word.text, word.spoken, reading), True):
                result.append({'text': text, 'lang': 'ja', 'reading': sound})
    return compile_parts(line.text, result, 'auto')


class TextReadingSession:
    def __init__(self, config, log, output):
        self.prompt = PROMPT.read_text(encoding='utf-8')
        self.transport = LlmSession(config, log)
        self.log, self.output = log, Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.records = []
        self.error = None

    def report(self):
        return {'mode': 'llm_text', 'model': self.transport.config.model,
                'requests': self.transport.requests, 'repair_requests': 0,
                'thinking': self.transport.report()['thinking'],
                'prompt_sha256': hashlib.sha256(self.prompt.encode('utf-8')).hexdigest(),
                'request_mode': 'whole_lyrics_text', 'audio_sent': False, 'language_hints_sent': False,
                'automatic_repairs': False, 'error': self.error,
                'accepted_lines': sum(r.get('source') == 'llm_text' for r in self.records),
                'dictionary_fallback_lines': [r['line'] for r in self.records if r.get('source') == 'dictionary'],
                'manual_lines': [r['line'] for r in self.records if r['status'] == 'manual'],
                'lines': self.records,
                'note': 'LLM 只接收整首歌词文字，不接收预设语言、音频或时间。语言和读音由模型判断；本地音素校验不限制其为输入标注的语言，词典回退会逐行标记。'}

    def _fallback(self, line, record, reason):
        record['reason'] = reason
        try:
            parts = local_parts(line)
            if not parts:
                raise ValueError('本地词典没有返回读音。')
        except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
            record['fallback_error'] = str(exc)
            line.flags.append('llm_text_unresolved')
            self.log(f'第 {record["line"]} 行文本读音及词典回退均未通过，保留整句时间供人工补轴。')
            return
        record.update(parts=parts, status='accepted', source='dictionary',
                      detected_languages=sorted({p['lang'] for p in parts}))
        line.flags.append('llm_text_dictionary_fallback')
        self.log(f'第 {record["line"]} 行文本读音未采用，使用本地词典回退。')

    def prepare(self, lines):
        self.records = [{'line': i+1, 'id': f'line_{i+1:03}', 'text': line.text,
                         'status': 'manual', 'parts': [], 'source': 'unresolved'}
                        for i, line in enumerate(lines)]
        try:
            if not lines:
                return
            # Construct a fresh allowlist: no audio path, time, title, source file
            # name, translated lyrics or previously measured timing can escape.
            payload = {'english_phone_inventory': sorted(p.split('/', 1)[1] for p in phone_inventory() if p.startswith('en/')),
                       'items': [{'id': record['id'], 'text': line.text}
                                 for record, line in zip(self.records, lines)]}
            self.log(f'LLM 文本读音：{len(lines)} 行歌词，一次请求，不发送音频…')
            rows, error = self.transport._chat(payload, prompt=self.prompt, timeout=TIMEOUT,
                                               max_response_bytes=MAX_RESPONSE_BYTES)
            self.error = error
            by_id = {record['id']: [] for record in self.records}
            if not error:
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] not in by_id:
                        self.error = '响应包含无效或未知的歌词行 ID。'
                        break
                    by_id[row['id']].append(row)
            for line, record in zip(lines, self.records):
                if self.error:
                    self._fallback(line, record, self.error)
                    continue
                try:
                    matches = by_id[record['id']]
                    if len(matches) != 1 or set(matches[0]) != {'id', 'parts'}:
                        raise ValueError('每个歌词行 ID 需要且只允许一个 id/parts 对象。')
                    parts = compile_parts(line.text, matches[0]['parts'], 'auto')
                    if not parts:
                        raise ValueError('LLM 未确定本行读音。')
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    self._fallback(line, record, str(exc))
                    continue
                record.update(parts=parts, status='accepted', source='llm_text',
                              detected_languages=sorted({p['lang'] for p in parts}))
                line.flags.append('llm_text_reading_candidate')
            self.log(f'文本读音完成：{sum(r["source"] == "llm_text" for r in self.records)}/{len(lines)} 行采用 LLM；其余结果已标记。')
        finally:
            try:
                (self.output / 'llm_text.json').write_text(json.dumps(self.report(), ensure_ascii=False, indent=2), encoding='utf-8')
            finally:
                self.transport.close()

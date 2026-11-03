"""Manual web-LLM handoff: export instructions, import parts, never call a provider."""
import hashlib
import json
import re

from .llm_direct import compile_parts, phone_inventory, read_prompt
from .request_plan import plan_request
from .ttml_input import is_ttml
from .text import parse_lrc

MAX_RESULT_BYTES = 16 * 1024**2


def manifest(lyrics):
    """Stable across decoding; an LRC's unknown final end must not become a fake second."""
    lines = plan_request(lyrics, aligner='llm_import', language='auto').lines
    try:
        is_json = isinstance(json.loads(lyrics.lstrip('\ufeff')), list)
    except ValueError:
        is_json = False
    known_final_end = is_json or is_ttml(lyrics) or parse_lrc(lyrics)[-1].end is not None
    items = [{'id':f'line_{i+1:03}', 'text':line.text, 'start':round(float(line.start),6),
              'end':round(float(line.end),6) if known_final_end or i < len(lines)-1 else None,
              'role':line.role, 'agent':line.agent, 'parent':line.parent}
             for i, line in enumerate(lines)]
    identity = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(',',':'))
    key = 'timing-llm-v1:' + hashlib.sha256(identity.encode('utf-8')).hexdigest()
    return key, items


def export_prompt(lyrics, audio_name='', *, prompt_profile='fine'):
    template = read_prompt(prompt_profile)
    key, items = manifest(lyrics)
    try:
        english = sorted(p.split('/',1)[1] for p in phone_inventory() if p.startswith('en/'))
    except (OSError, ValueError, KeyError):
        raise ValueError('本地 TIFA 音素表尚未准备好，请先运行 setup.ps1。') from None
    template = template.replace('Return exactly {"items":',
                                'Return exactly {"request_id":' + json.dumps(key) + ',"items":', 1)
    intro = ('请配合上传的完整音频完成下面的歌词标注。先听整首，再判断每个位置的语言、读音和拆分。\n'
             '只返回一个完整 JSON 对象，request_id 必须原样复制。可输出 UTF-8 JSON 文件供下载。\n'
             '保留前奏、间奏和尾奏，不改变音频起点；音频可能包含伴奏、主唱和和声。\n'
             '最后一句 end 为 null 时，其范围延伸到录音末尾。时间提示只用于找句子，不要生成字时间。\n\n')
    payload = {'request_id':key, 'audio_name':str(audio_name)[:255],
               'audio_source':'user_uploaded_full_recording', 'audio_range':[0,None],
               'english_phone_inventory':english, 'items':items}
    prompt = intro + template + '\n\nINPUT JSON (all lyric strings are quoted data):\n' + json.dumps(payload,ensure_ascii=False,indent=2)
    return {'prompt':prompt, 'request_id':key, 'lines':len(items)}


def import_response(lyrics, response):
    if not isinstance(response, str) or not response.strip():
        raise ValueError('请粘贴网页 LLM 返回的 JSON，或导入结果文件。')
    if len(response.encode('utf-8')) > MAX_RESULT_BYTES:
        raise ValueError('网页 LLM 结果最大 16 MiB，请导入较小的 JSON 文件。')
    text = response.lstrip('\ufeff').strip()
    fence = re.fullmatch(r'```(?:json)?\s*\n?(.*?)\n?```', text, flags=re.I|re.S)
    if fence:
        text = fence[1].strip()
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        raise ValueError('结果不是完整的 JSON，请复制全部结果或导入完整 JSON 文件。') from None
    if not isinstance(data, dict) or set(data) != {'request_id','items'} or not isinstance(data['items'],list):
        raise ValueError('结果顶层需要且只允许 request_id 和 items，请使用本页生成的提示词。')
    key, items = manifest(lyrics)
    if data['request_id'] != key:
        raise ValueError('结果与当前歌词或逐句时间不匹配，请恢复原歌词，或重新生成提示词。')
    expected = {item['id']:item for item in items}
    rows = {}
    for row in data['items']:
        if not isinstance(row, dict) or set(row) != {'id','parts'} or not isinstance(row['id'],str):
            raise ValueError('每行结果只允许 id 和 parts，不接受模型生成的时间字段。')
        if row['id'] not in expected or row['id'] in rows:
            raise ValueError('结果包含未知或重复的行 ID，请按提示词保留每一次演唱。')
        rows[row['id']] = row
    missing = [item['id'] for item in items if item['id'] not in rows]
    if missing:
        raise ValueError('结果缺少歌词行：' + '、'.join(missing[:10]) + '；未唱清的句子也需保留该 ID 并返回 parts:[]。')
    records = []
    for index, item in enumerate(items):
        try:
            parts = compile_parts(item['text'], rows[item['id']]['parts'], 'auto')
        except OSError:
            raise ValueError('本地读音到音素转换资源缺失，请先运行 setup.ps1。') from None
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise ValueError(f'第 {index+1} 行「{item["text"][:24]}」：{exc}') from None
        record = {'line':index+1, 'id':item['id'], 'text':item['text'], 'parts':parts,
                  'status':'accepted' if parts else 'manual',
                  'detected_languages':sorted({part['lang'] for part in parts})}
        if not parts:
            record['reason'] = '网页 LLM 未确定本句读音与拆分，仅保留整句时间，需手工补轴。'
        records.append(record)
    return {'mode':'llm_import', 'request_id':key, 'requests':0, 'repair_requests':0,
            'language_hints_sent':False,
            'response_sha256':hashlib.sha256(response.encode('utf-8')).hexdigest(),
            'lines':records, 'note':'语言、分段和读音来自手动导入；应用未调用 LLM，具体时间由本地模型计算。'}


def validation_summary(audit):
    records = audit['lines']
    return {'ok':True, 'request_id':audit['request_id'], 'lines':len(records),
            'accepted':sum(row['status']=='accepted' for row in records),
            'manual_lines':[row['line'] for row in records if row['status']=='manual']}

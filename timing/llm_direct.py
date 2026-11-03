"""One whole-song LLM annotation request; local native acoustic timing only."""
import base64
import copy
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import soundfile as sf

from . import chinese
from .config import MODELS
from .llm_splitter import LlmSession
from .schema import Unit
from .text import kept, canonical_apostrophes

PROMPT = Path(__file__).with_name('prompts')/'llm_direct.txt'
PROMPT_PROFILES = ('fine', 'original')
# A local memory/request guard, not a claim about a provider's upload limit.
MAX_AUDIO_BASE64_BYTES = 64 * 1024**2
WHOLE_SONG_TIMEOUT = 600
MAX_RESPONSE_BYTES = 16 * 1024**2


def read_prompt(profile='fine'):
    if profile not in PROMPT_PROFILES:
        raise ValueError('日语拆分请选择 fine（细拆）或 original（适度合并，旧版）。')
    path = PROMPT if profile == 'fine' else PROMPT.with_name('llm_direct_original.txt')
    return path.read_text(encoding='utf-8')


def whole_audio(samples, rate):
    """Encode every sample, including silence, without shifting the song clock."""
    wave = np.asarray(samples)
    if (wave.ndim != 1 or not len(wave) or not isinstance(rate, (int, np.integer)) or
            rate <= 0 or not np.isfinite(wave).all()):
        raise ValueError('完整音轨无效，请检查音频文件。')
    target_frames = math.ceil(len(wave) * 16000 / rate)
    encoded_bytes = 4 * math.ceil((44 + target_frames * 2) / 3)
    if encoded_bytes > MAX_AUDIO_BASE64_BYTES:
        raise ValueError('完整音轨编码后超过本地 64 MiB 限制，未请求 LLM；请使用较短音轨。')
    if rate != 16000:
        from scipy.signal import resample_poly
        factor = math.gcd(int(rate), 16000)
        wave = resample_poly(wave.astype(np.float32), 16000 // factor, int(rate) // factor)
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format='WAV', subtype='PCM_16')
    audio = 'data:audio/wav;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
    return audio, [0, round(len(samples) / rate, 6)]


@lru_cache(maxsize=1)
def phone_inventory():
    return frozenset(json.loads((MODELS/'tifa/vocabulary.json').read_text())['symbols'])


def acoustic_languages(parts):
    scopes = {part['lang'] for part in parts}
    return [lang for lang in ('ja', 'zh', 'en') if lang in scopes]


@lru_cache(maxsize=1)
def kana_converter():
    sys.path.insert(0, str(MODELS/'tifa/deps'))
    from g2pflow.converters.japanese import JapaneseKanaConverter
    return JapaneseKanaConverter(dict_path=str(MODELS/'tifa/dictionaries/japanese_dict_full.txt'))


def spoken(text):
    return canonical_apostrophes(kept(text)).casefold()


def detected_language(parts):
    languages = frozenset(part['lang'] for part in parts)
    return {frozenset(['zh']):'zh', frozenset(['ja']):'ja', frozenset(['en']):'en',
            frozenset(['zh','en']):'zh-en', frozenset(['ja','en']):'mixed'}.get(languages,'auto')


def compile_parts(text, parts, language='ja'):
    if not isinstance(parts, list) or len(parts)>min(256,len(text)):
        raise ValueError('parts 必须为有界分段数组。')
    if not parts:
        return []
    languages = (('ja','zh','en') if language == 'auto' else ('en',) if language == 'en'
                 else ('zh','en') if chinese.is_chinese(language) else ('ja','en'))
    compiled = []
    for part in parts:
        if not isinstance(part, dict) or set(part)!={'text','lang','reading'}:
            raise ValueError('分段仅允许 text/lang/reading；禁止时间或额外字段。')
        surface, lang, reading = part['text'], part['lang'], part['reading']
        if (not isinstance(surface,str) or not surface or
                not any(c.isalnum() for c in spoken(surface)) or lang not in languages):
            raise ValueError('分段文字、语言无效；标点和空格请附在有发音的分段。')
        if lang=='en':
            if not isinstance(reading,list) or not 1<=len(reading)<=80 or any(not isinstance(p,str) for p in reading):
                raise ValueError('英语 reading 需要1到80个音素组成的数组。')
            phones = [p.lower() for p in reading]
            groups = [{'script':surface,'phones':phones}]
        elif lang=='zh':
            if not isinstance(reading,str) or not 1<=len(reading)<=200:
                raise ValueError('中文 reading 需要空格分隔的拼音音节字符串。')
            if len(chinese.HAN_RE.findall(surface))>1:
                raise ValueError('中文显示分段每段最多一个汉字，请按字拆分。')
            try:
                groups = chinese.pinyin_groups(reading)
            except ValueError as exc:
                raise ValueError(f'中文读音无效：{exc}') from exc
            reading = ' '.join(group['script'] for group in groups)
            phones = [p for group in groups for p in group['phones']]
        else:
            if not isinstance(reading,str) or not re.fullmatch(r'[ァ-ヺー]{1,80}',reading):
                raise ValueError('日语 reading 仅允许1到80个片假名和长音符。')
            try:
                words = kana_converter().convert(reading)
            except Exception as exc:
                raise ValueError('假名不能转换为声学模型音素。') from exc
            groups = [{'script':g.script, 'phones':list(g.phonemes)}
                      for word in words for g in word.readings[0].paths[0]]
            phones = [p for g in groups for p in g['phones']]
        if not phones or any(f'{lang}/{p}' not in phone_inventory() for p in phones):
            raise ValueError('读音包含本地声学模型不支持的音素，或只有静音。')
        compiled.append({'text':surface,'lang':lang,'reading':reading,'groups':groups})
    if ''.join(p['text'] for p in compiled)!=text:
        raise ValueError('分段拼接必须与原歌词完全相同，不可增删或改写字符。')
    if sum(len(g['phones']) for p in compiled for g in p['groups'])>1024:
        raise ValueError('本句音素过多，请拆分逐句范围。')
    return compiled


def write_pfml(audio, parts):
    nodes = []
    for part in parts:
        language = part['lang']
        node = ET.Element('word',text=spoken(part['text']),language=language)
        for group in part['groups']:
            ET.SubElement(node,'group',script=group['script'],phonemes=' '.join(group['phones']))
        nodes.append(ET.tostring(node,encoding='unicode'))
    Path(audio).with_suffix('.pfml').write_text('\n'.join(nodes),encoding='utf-8')


def target_parts(parts, start, end):
    result, cursor = [], 0
    for part in parts:
        right = cursor+len(spoken(part['text']))
        if cursor<end and right>start:
            if cursor<start or right>end:
                raise ValueError('LLM 分段跨越歌词句界。')
            result.append(part)
        cursor=right
    return result


class DirectSession:
    def __init__(self, config, log, output, *, prompt_profile='fine'):
        self.prompt=read_prompt(prompt_profile)
        self.prompt_profile=prompt_profile
        self.transport=LlmSession(config,log)
        self.log, self.output=log,Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.records=[]
        self.disabled=False
        self.error=None
        self.audio_source=None
        self.audio_range=None

    def report(self):
        return {'mode':'llm_direct','model':self.transport.config.model,
                'prompt_profile':self.prompt_profile,
                'requests':self.transport.requests,'repair_requests':self.transport.repair_requests,
                'thinking':self.transport.report()['thinking'],
                'prompt_sha256':hashlib.sha256(self.prompt.encode()).hexdigest(),
                'request_mode':'whole_song', 'language_mode':'audio_detected', 'language_hints_sent':False,
                'automatic_repairs':False, 'audio_source':self.audio_source,
                'audio_range':self.audio_range,'response_limit_bytes':MAX_RESPONSE_BYTES,
                'error':self.error,
                'lines':self.records,'note':'完整音轨与歌词一次发送；语言、分段和读音由 LLM 判断，不自动补发。时间来自本地 TIFA，需听音复校。'}

    def save(self):
        (self.output/'llm_direct.json').write_text(json.dumps(self.report(),ensure_ascii=False,indent=2),encoding='utf-8')

    def prepare(self, lines, samples, rate, background=None, *, audio_source=None):
        self.records=[{'line':i+1,'id':f'line_{i+1:03}','text':line.text,'status':'manual','parts':[]}
                      for i,line in enumerate(lines)]
        try:
            if not lines:
                return
            # Full vocals contains both voices; never ask about bg lyrics using
            # a lead-only stem.  It is still one uncut track and one request.
            use_background = any(line.role == 'bg' for line in lines) and background is not None
            wave = background if use_background else samples
            self.audio_source = audio_source or ('full_vocals' if use_background else 'alignment_track')
            try:
                audio, bounds = whole_audio(wave, rate)
            except ValueError as exc:
                self.error=str(exc)
                self.log(f'LLM 整曲输入未发送：{self.error} 保留整句时间供手工补轴。')
                for record in self.records:
                    record['reason']=self.error+' 需人工处理。'
                return
            self.audio_range = bounds
            items=[{'id':record['id'],'text':line.text,'start':line.start,'end':line.end,
                    'role':line.role,'agent':line.agent,'parent':line.parent}
                   for record,line in zip(self.records,lines)]
            english=sorted(p.split('/',1)[1] for p in phone_inventory() if p.startswith('en/'))
            payload={'audio_range':bounds,'audio_source':self.audio_source,
                     'english_phone_inventory':english,'items':items}
            self.log(f'LLM 整曲识别：完整 {bounds[1]:.2f} 秒音轨、{len(lines)} 行歌词，一次请求…')
            rows,error=self.transport._chat(payload,audio=audio,prompt=self.prompt,timeout=WHOLE_SONG_TIMEOUT,
                                            max_response_bytes=MAX_RESPONSE_BYTES)
            if error:
                self.error=error
                hint=('；鉴权失败，请检查 API Key 和接口地址。' if error == 'HTTP 401' else
                      '；接口拒绝完整音频，请检查其音频大小、时长和格式限制。' if error.startswith(('HTTP 413','HTTP 400')) else '')
                self.log(f'LLM 整曲请求失败：{error}{hint} 未自动重发。')
                for record in self.records:
                    record['reason']=error+'；未自动重发，需人工处理。'
                return
            by_id={record['id']:[] for record in self.records}
            for row in rows:
                if isinstance(row,dict) and isinstance(row.get('id'),str) and row['id'] in by_id:
                    by_id[row['id']].append(row)
            for line,record in zip(lines,self.records):
                matches=by_id[record['id']]
                try:
                    if len(matches)!=1 or set(matches[0])!={'id','parts'}:
                        raise ValueError('每行需要且只允许一个 id/parts 对象。')
                    parts=compile_parts(line.text,matches[0]['parts'],'auto')
                except (ValueError,KeyError,IndexError,TypeError) as exc:
                    record['reason']=str(exc)+' 未自动补发，需人工处理。'
                    continue
                record.update(parts=parts,status='accepted' if parts else 'manual',audio_range=bounds,
                              detected_languages=sorted({part['lang'] for part in parts}))
                if not parts:
                    record['reason']='LLM 未确定本句读音／拆分，未自动补发，需人工处理。'
            accepted=sum(record['status']=='accepted' for record in self.records)
            self.log(f'LLM 整曲返回：{accepted}/{len(lines)} 行可用；其余保留整句时间，不追加请求。')
        finally:
            self.save()
            self.transport.close()


def coarse_units(parts):
    units,labels=[],[]
    flat=[p for part in parts for g in part['groups'] for p in g['phones']]
    position=0
    for part in parts:
        phones=[p for g in part['groups'] for p in g['phones']]
        if part['lang']=='en':
            label=spoken(part['text'])
            if not re.fullmatch("[a-z']+",label):
                raise ValueError('英语显示文字无法直接用作 MMS 字符标签，保留 LRC 范围。')
        elif part['lang']=='zh':
            label=chinese.roman_label(part['reading'])
            if not re.fullmatch('[a-z]+',label):
                raise ValueError('中文读音不能转为 MMS 标签，保留 LRC 范围。')
        else:
            roman=[]
            for p in phones:
                if p=='cl':
                    following=flat[position+1].split('/',1)[-1] if position+1<len(flat) else 't'
                    roman.append('t' if following in ('ch','ts') else following[0].lower())
                else:
                    roman.append(p.lower())
                position+=1
            label=''.join(roman)
            if not re.fullmatch("[a-z']+",label):
                raise ValueError('日语读音不能转为 MMS 标签，保留 LRC 范围。')
        if part['lang']!='ja':
            position+=len(phones)
        reading=part['reading'] if isinstance(part['reading'],str) else ' '.join(part['reading'])
        units.append(Unit(part['text'],kept(part['text']),reading,source_pronunciation=reading))
        labels.append(label)
    return units,labels


def locate(samples, rate, lines, records, log, background=None):
    from .ctc import CtcAligner
    engine=CtcAligner(log)
    result=[]
    try:
        for index,(line,record) in enumerate(zip(lines,records)):
            row={'start':line.start,'end':line.end,'method':'supplied_lrc'}
            if record['status']=='accepted':
                wave=background if line.role=='bg' and background is not None else samples
                probe=copy.deepcopy(line)
                try:
                    prepared=coarse_units(record['parts'])
                    log(f'MMS 使用 LLM 分段粗定位 {index+1}/{len(lines)}：{line.text[:32]}')
                    info=engine.align(wave[round(line.start*rate):round(line.end*rate)],probe,False,
                                      sample_rate=rate,prepared=prepared)
                    row.update(start=line.start+probe.start,end=line.start+probe.end,method='mms_llm_units',evidence=info)
                except ValueError as exc:
                    row['reason']=str(exc)
                    line.flags.append('llm_direct_coarse_fallback')
                    log(f'第 {index+1} 行 MMS 粗定位未采用，保留输入 LRC 范围；TIFA 仍使用 LLM 读音。')
            result.append(row)
    finally:
        engine.release()
    return result


def align(samples, rate, lines, output, records, log, background=None, coarse=None, *, trust_line_timestamps=False):
    from . import tifa
    folder=Path(output)/'tifa_direct'
    folder.mkdir(exist_ok=True)
    eligible={i for i,r in enumerate(records) if r['status']=='accepted'}
    good, clips, local_failures=[],[],{}
    for i in sorted(eligible):
        line=lines[i]
        neighbors=[j for j in (i-1,i,i+1) if j in eligible and
                   (lines[j].role,lines[j].agent)==(line.role,line.agent)]
        if max(lines[j].end for j in neighbors)-min(lines[j].start for j in neighbors)>59:
            neighbors=[i]
        wave=background if line.role=='bg' and background is not None else samples
        a=round(max(0,min(lines[j].start for j in neighbors)-.12)*rate)
        limit=min(len(wave)/rate,max(lines[j].end for j in neighbors)+.35)
        gate=None
        if coarse and not trust_line_timestamps:
            last=neighbors[-1]
            limit,gate=tifa.quiet_crop_end(wave,rate,coarse[last]['end'],limit,coarse[last]['start'])
        b=round(limit*rate)
        if not .1 <= (b-a)/rate <= 60:
            local_failures[i]='本地 TIFA 句段需在 0.1 到 60 秒之间，请拆分或缩小该行 LRC 范围。'
            continue
        audio=folder/f'line_{i+1:03}.wav'
        sf.write(audio,wave[a:b],rate,subtype='FLOAT')
        parts=[part for j in neighbors for part in records[j]['parts']]
        audio.with_suffix('.txt').write_text(''.join(spoken(p['text']) for p in parts),encoding='utf-8')
        write_pfml(audio,parts)
        start=sum(len(spoken(lines[j].text)) for j in neighbors if j<i)
        clips.append({'id':f'line_{i+1:03}','audio':str(audio.resolve()),'start':a/rate,'end':b/rate,
                      'gate':gate,'target_start':start,'target_end':start+len(spoken(line.text)),
                      'llm_direct_parts':parts,'source_line':i+1,
                      'languages':acoustic_languages(parts)})
        good.append(i)
    checks=[]
    if clips:
        raw=tifa.run_worker(folder,clips,log)
        subset=[lines[i] for i in good]
        checks=tifa.complete_results(samples,rate,subset,folder,clips,raw,log,syllables=False,background=background)
        for check,index in zip(checks,good):
            check['line']=index+1
    for i,line in enumerate(lines):
        if i in good:
            continue
        line.words=[Unit(line.text,kept(line.text),start=round(line.start,3),end=round(line.end,3),flags=['lrc_review_placeholder'])]
        line.flags.append('lrc_review_placeholder')
        if i not in eligible:
            line.flags.append('llm_direct_unresolved')
        reason=local_failures.get(i) or records[i].get('reason','LLM 未提供有效拆分')
        checks.append({'line':i+1,'mode':'lrc_review_placeholder','reason':reason})
        log(f'第 {i+1} 行仅保留整句 LRC 时间，需人工补轴。{reason}')
    checks.sort(key=lambda c:c['line'])
    return {'model':'TIFA-1.0-ST','input':'LLM explicit PFML; no lexical converters','checks':checks,'frame_ms':10}

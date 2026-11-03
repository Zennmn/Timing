"""Optional Chat Completions mapping; accepts indices, never model-made times."""
import hashlib
import json
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

PROMPT_PATH = Path(__file__).with_name('prompts') / 'english_syllables.txt'
_SESSION = ContextVar('llm_syllable_session', default=None)
VOWELS = frozenset('AA AE AH AO AW AX AY EH ER EY IH IY OW OY UH UW'.split())
MAX_BATCH_WORDS = 24


def vowel(phone):
    return re.sub(r'[012]$', '', phone.rsplit('/',1)[-1].upper()) in VOWELS


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    api_key: str = field(default='', repr=False)
    json_mode: bool = False
    thinking_mode: str = 'default'
    thinking_format: str = 'reasoning_effort'
    reasoning_effort: str = 'high'
    api_key_header: str = 'Authorization'
    api_key_format: str = 'auto'

    def __post_init__(self):
        try:
            parsed = urlsplit(self.base_url.strip())
            valid = (parsed.scheme in ('http','https') and parsed.hostname and
                     not parsed.username and not parsed.password and not parsed.query and not parsed.fragment)
            parsed.port
        except ValueError:
            valid = False
        if not valid:
            raise ValueError('请填写有效的 LLM Base URL（含 http:// 或 https://，不含密钥或查询参数）。')
        if not self.model.strip() or len(self.model) > 200:
            raise ValueError('请填写 LLM 模型名。')
        if len(self.api_key) > 8192 or not self.api_key.isascii() or any(c in self.api_key for c in '\r\n'):
            raise ValueError('LLM API Key 格式无效。')
        if not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}", self.api_key_header):
            raise ValueError('API Key 请求头名称需为1到128个合法英文 HTTP 头字符，不能包含空格、冒号或换行。')
        if self.api_key_header.lower() in {'host','content-length','transfer-encoding','connection','content-type'}:
            raise ValueError('API Key 请求头名称不能使用 HTTP 传输或内容类型字段。')
        if self.api_key_format not in ('auto','bearer','raw'):
            raise ValueError('API Key 发送格式请选择自动、Bearer 或原样发送。')
        if self.thinking_mode not in ('default','on','off'):
            raise ValueError('思考模式请选择跟随平台、开启或关闭。')
        if self.thinking_format not in ('reasoning_effort','enable_thinking','thinking'):
            raise ValueError('不支持的思考参数格式。')
        if self.reasoning_effort and not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,31}',self.reasoning_effort):
            raise ValueError('思考强度请填写最多32位的英文、数字、下划线或连字符。')
        if self.thinking_mode == 'on':
            if self.reasoning_effort.lower() == 'none':
                raise ValueError('开启思考时不能使用 none；要关闭请选择关闭模式。')
            if self.thinking_format == 'reasoning_effort' and not self.reasoning_effort:
                raise ValueError('reasoning_effort 格式开启时需要填写思考强度。')

    def thinking_parameters(self):
        """Wire parameters requested by the user, not proof of provider execution."""
        if self.thinking_mode == 'default':
            return {}
        enabled = self.thinking_mode == 'on'
        if self.thinking_format == 'reasoning_effort':
            return {'reasoning_effort':self.reasoning_effort if enabled else 'none'}
        result = ({'enable_thinking':enabled} if self.thinking_format == 'enable_thinking'
                  else {'thinking':{'type':'enabled' if enabled else 'disabled'}})
        if enabled and self.reasoning_effort:
            result['reasoning_effort'] = self.reasoning_effort
        return result

    @property
    def endpoint(self):
        value = self.base_url.strip().rstrip('/')
        return value if value.endswith('/chat/completions') else value + '/chat/completions'

    def auth_headers(self):
        if not self.api_key:
            return {}
        bearer = self.api_key_format == 'bearer' or (self.api_key_format == 'auto' and self.api_key_header.lower() == 'authorization')
        return {self.api_key_header: ('Bearer ' if bearer else '') + self.api_key}

    @classmethod
    def from_env(cls):
        return cls(os.environ.get('TIMING_LLM_BASE_URL',''), os.environ.get('TIMING_LLM_MODEL',''),
                   os.environ.get('TIMING_LLM_API_KEY',''), os.environ.get('TIMING_LLM_JSON_MODE','0') == '1',
                   os.environ.get('TIMING_LLM_THINKING_MODE','default'),
                   os.environ.get('TIMING_LLM_THINKING_FORMAT','reasoning_effort'),
                   os.environ.get('TIMING_LLM_REASONING_EFFORT','high'),
                   os.environ.get('TIMING_LLM_API_KEY_HEADER','Authorization'),
                   os.environ.get('TIMING_LLM_API_KEY_FORMAT','auto'))


def validate_parts(word, phones, parts):
    if not isinstance(parts,list) or not 1 <= len(parts) <= min(len(word),len(phones)):
        raise ValueError('parts 数量无效')
    cursor, text, result = 0, '', []
    for part in parts:
        if not isinstance(part,dict) or set(part) != {'text','phone_start','phone_end'}:
            raise ValueError('字段无效，禁止生成时间或新音素')
        piece,a,b = part['text'],part['phone_start'],part['phone_end']
        if (not isinstance(piece,str) or not piece or type(a) is not int or type(b) is not int or
                a != cursor or not 0 <= a < b <= len(phones)):
            raise ValueError('音素范围不连续或越界')
        if len(parts)>1 and (sum(vowel(p) for p in phones[a:b]) != 1 or not any(c.isalpha() for c in piece)):
            raise ValueError('每个细分音节必须对应一个元音核')
        result.append((piece,a,b)); text += piece; cursor = b
    if text != word or cursor != len(phones):
        raise ValueError('原文或音素覆盖不完整')
    return tuple(result)


class LlmSession:
    def __init__(self, config, log=print):
        self.config, self.log = config, log
        self.prompt = PROMPT_PATH.read_text(encoding='utf-8')
        self.cache, self.records = {}, []
        self.requests, self.failures, self.disabled = 0, 0, False
        self.repair_requests = 0
        self.request_batches = []
        self.client = httpx.Client(timeout=httpx.Timeout(300 if config.thinking_mode == 'on' else 90,connect=15),follow_redirects=False)

    def close(self):
        self.client.close()

    def system_prompt(self):
        return self.prompt

    def _chat(self, payload, repair=False, *, audio=None, prompt=None, timeout=None, max_response_bytes=2*1024**2):
        system = prompt or self.system_prompt()
        content = json.dumps(payload,ensure_ascii=False)
        if audio:
            content = [{'type':'input_audio','input_audio':{'data':audio}}, {'type':'text','text':content}]
        messages = [{'role':'system','content':system}, {'role':'user','content':content}]
        if repair:
            messages.append({'role':'user','content':'Your previous mapping failed validation. Correct only the remaining IDs and follow the system fallback for uncertainty. Return the exact JSON schema only.'})
        body = {'model':self.config.model.strip(),'messages':messages,'stream':False}
        body.update(self.config.thinking_parameters())
        if self.config.json_mode:
            body['response_format'] = {'type':'json_object'}
        headers = self.config.auth_headers()
        self.requests += 1
        self.repair_requests += int(repair)
        self.request_batches.append({'request':self.requests,'stage':'repair' if repair else 'initial',
                                     'words':len(payload['items'])})
        # Never include provider response bodies or exception text in logs: a
        # provider can echo Authorization headers or its credential-bearing URL.
        try:
            request_options = {'timeout':httpx.Timeout(timeout,connect=15)} if timeout is not None else {}
            with self.client.stream('POST',self.config.endpoint,json=body,headers=headers,**request_options) as response:
                if not 200 <= response.status_code < 300:
                    if response.status_code == 400 and self.config.thinking_mode != 'default':
                        return None, 'HTTP 400（请检查接口支持的思考参数格式、强度与 JSON 模式）'
                    return None, f'HTTP {response.status_code}'
                raw = bytearray()
                for chunk in response.iter_bytes():
                    raw.extend(chunk)
                    if len(raw) > max_response_bytes:
                        return None, '响应过大'
            envelope = json.loads(raw)
            choice = envelope['choices'][0]
            if choice.get('finish_reason') in ('length','repetition_truncation','content_filter','tool_calls'):
                return None, '响应未完成或未返回文本'
            content = choice['message']['content']
            if not isinstance(content,str):
                return None, '缺少文本响应'
            content = content.strip()
            if content.startswith('```') and content.endswith('```'):
                content = re.sub(r'^```(?:json)?\s*','',content,flags=re.I)[:-3].strip()
            data = json.loads(content)
            if not isinstance(data,dict) or set(data) != {'items'} or not isinstance(data['items'],list):
                return None, 'JSON 结构无效'
            return data['items'], None
        except httpx.TimeoutException:
            return None, '连接或响应超时'
        except httpx.HTTPError:
            return None, '接口连接失败'
        except (ValueError,KeyError,IndexError,TypeError):
            return None, '响应不是可用的 Chat Completions JSON'

    def prepare(self, entries, context=''):
        self.prepare_many([(entries,context)])

    def key(self,word,phones):
        return (word,tuple(phones))

    def prepare_many(self, groups):
        """Collect across the song before sending. Share context text by ID.

        Dedup stays exact on surface word + selected phones, as before. Keep
        up to two distinct occurrences as context without changing the cache
        key or substituting a different pronunciation.
        """
        contexts = {}
        for entries,context in groups:
            for word,phones in entries:
                key = self.key(word,phones)
                if key in self.cache:
                    continue
                if not re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]*)*",word) or sum(vowel(p) for p in phones) < 2:
                    self.cache[key] = {'parts':None,'status':'whole_word'}
                    continue
                examples = contexts.setdefault(key,[])
                if context and context[:2000] not in examples and len(examples)<2:
                    examples.append(context[:2000])
        pending = list(contexts)
        if pending:
            self.log(f'LLM 汇总：{len(pending)} 个不同的新词／读音，最多 {MAX_BATCH_WORDS} 个一批。')
        for start in range(0,len(pending),MAX_BATCH_WORDS):
            batch = pending[start:start+MAX_BATCH_WORDS]
            items = [{'id':f'w{i}','word':key[0],'phones':list(key[1])} for i,key in enumerate(batch)]
            unresolved = {item['id']:key for item,key in zip(items,batch)}
            errors = {}
            if self.disabled:
                reason = '此前接口请求失败，本任务停止继续请求'
            else:
                reason = '映射校验未通过'
                errors = {}
                for attempt in range(2):
                    stage = '修正重试' if attempt else '批量首发'
                    self.log(f'LLM 英语拆分：{len(unresolved)} 个新词（{stage}，第 {self.requests+1} 次请求）…')
                    context_ids = {}
                    active = []
                    for item in items:
                        if item['id'] not in unresolved:
                            continue
                        ids = []
                        for context in contexts[unresolved[item['id']]]:
                            if context not in context_ids:
                                context_ids[context] = f'c{len(context_ids)}'
                            ids.append(context_ids[context])
                        active.append({**item,'context_ids':ids})
                    payload = {'contexts':[{'id':i,'text':text} for text,i in context_ids.items()], 'items':active}
                    if attempt:
                        payload['validation_errors'] = [{'id':i,'reason':errors.get(i,reason)} for i in unresolved]
                    rows,error = self._chat(payload,repair=bool(attempt))
                    if error:
                        reason = error
                        if error.startswith('HTTP ') or error in ('连接或响应超时','接口连接失败'):
                            self.disabled = True
                            break
                        continue
                    errors = {}
                    for item_id,key in list(unresolved.items()):
                        matches = [r for r in rows if isinstance(r,dict) and r.get('id') == item_id]
                        if len(matches) != 1:
                            errors[item_id] = '词 ID 缺失或重复'
                            continue
                        row = matches[0]
                        if set(row) != {'id','parts'}:
                            errors[item_id] = '只允许 id 与 parts 字段'
                            continue
                        try:
                            division = validate_parts(*key[:2],row['parts'])
                        except ValueError as exc:
                            errors[item_id] = str(exc)
                            continue
                        self.cache[key] = {'parts':division,'status':'accepted'}
                        del unresolved[item_id]
                    if not unresolved:
                        break
            if unresolved:
                self.failures += 1
                if self.failures >= 2:
                    self.disabled = True
                self.log(f'LLM 拆分未采用：{reason}；{len(unresolved)} 个词保留整词模型时间。')
            else:
                self.failures = 0
            for item_id,key in unresolved.items():
                self.cache[key] = {'parts':None,'status':'fallback','reason':errors.get(item_id,reason) if not self.disabled else reason}
            for key in batch:
                value = self.cache[key]
                self.records.append({'word':key[0],'phones':list(key[1]),**value})

    def split(self,word,phones):
        self.prepare([(word,phones)])
        return self.cache[self.key(word,phones)]['parts']

    def fallback(self,word,phones):
        return self.cache.get(self.key(word,phones),{}).get('status') == 'fallback'

    def report(self):
        return {'model':self.config.model,'protocol':'chat_completions','prompt_sha256':hashlib.sha256(self.system_prompt().encode()).hexdigest(),
                'thinking':{'mode':self.config.thinking_mode,'format':self.config.thinking_format,
                            'requested_parameters':self.config.thinking_parameters()},
                'requests':self.requests,'fallback_words':sum(r['status']=='fallback' for r in self.records),
                'initial_requests':self.requests-self.repair_requests,'repair_requests':self.repair_requests,
                'request_batches':self.request_batches,'max_batch_words':MAX_BATCH_WORDS,
                'accepted_words':sum(r['status']=='accepted' for r in self.records),'words':self.records,
                'note':'仅为文字与音素对应候选；时间来自原生对齐模型，未经过听音验收。'}


def current_session():
    return _SESSION.get()


@contextmanager
def use_llm(config=None,log=print,output=None):
    if config is None:
        yield None
        return
    session = LlmSession(config,log)
    parameters = config.thinking_parameters()
    log('LLM 思考设置：'+(json.dumps(parameters,ensure_ascii=False)+'（请求参数，是否执行由接口决定）' if parameters else '跟随平台默认，不发送思考参数。'))
    token = _SESSION.set(session)
    try:
        yield session
    finally:
        _SESSION.reset(token)
        session.close()
        if output is not None:
            output.mkdir(parents=True,exist_ok=True)
            (output/'llm_splits.json').write_text(json.dumps(session.report(),ensure_ascii=False,indent=2),encoding='utf-8')

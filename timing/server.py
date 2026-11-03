"""Loopback-only upload form and serialized GPU jobs."""
import json
import os
import secrets
import shutil
import threading
import traceback
import logging
import itertools
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from . import __version__, API_VERSION
from .config import ROOT, DATA, initialize
from .pipeline import run
from .separation import separator_ready
from .ctc import ready as ctc_ready
from .tifa import ready as tifa_ready
from .bs_lead import ready as bs_ready
from .llm_splitter import LlmConfig, LlmSession
from .job_store import save_job, load_job
from .request_plan import plan_request, MAX_LYRICS_BYTES, LANGUAGES
from .llm_import import export_prompt, import_response, validation_summary, MAX_RESULT_BYTES
from .llm_direct import read_prompt

app = FastAPI(title="Timing", docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="timing-gpu")
lock = threading.RLock()
jobs = {}
futures = {}
queue_sequence = itertools.count(1)
MAX_QUEUE_JOBS = 50
ACTIVE_STATES = {'uploading', 'queued', 'running'}
TOKEN = secrets.token_urlsafe(32)
ALLOWED = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".mp4", ".aiff"}


@app.middleware("http")
async def local_only(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        if not secrets.compare_digest(request.headers.get("X-Timing-Token", ""), TOKEN):
            return JSONResponse({"detail": "请从本机 Timing 页面提交。"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "仅接受同源请求。"}, status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "web" / "index.html").read_text(encoding="utf-8").replace("__TIMING_TOKEN__", TOKEN)


@app.get("/api/status")
def status():
    from .japanese_dictionary import status as dictionary_status
    japanese_dictionary = dictionary_status()
    models = {
        "separator": separator_ready(),
        "ctc": ctc_ready(),
        "tifa": tifa_ready(),
        "bs_lead": bs_ready(),
    }
    with lock:
        queued = sum(j['state'] in ACTIVE_STATES for j in jobs.values())
        waiting = sum(j['state'] == 'queued' for j in jobs.values())
        running = sum(j['state'] == 'running' for j in jobs.values())
    return {'app': 'Timing', 'version': __version__, 'api_version': API_VERSION,
            'languages': list(LANGUAGES),
            'english_splitters': ['syllabify', 'pyphen', 'llm'], 'models': models,
            'required_models': list(models), 'japanese_dictionary': japanese_dictionary,
            'ready': all(models.values()) and japanese_dictionary['ready'], 'queued': queued,
            'running': running, 'waiting': waiting, 'queue_limit': MAX_QUEUE_JOBS}


def audio_basename(filename):
    return str(filename or 'audio.wav').replace('\\', '/').rsplit('/', 1)[-1] or 'audio.wav'


def ttml_filename(job):
    original = job.get('original_filename')
    stem = Path(audio_basename(original)).stem if original else str(job.get('name') or 'audio')
    stem = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', stem).strip().rstrip(' .')[:230].rstrip(' .') or 'audio'
    if re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', stem, re.I):
        stem = '_' + stem
    return stem + '.ttml'


def job_view(job):
    """Copy public state and derive the current waiting position under one lock."""
    with lock:
        view = dict(job)
        waiting = sorted((j for j in jobs.values() if j['state'] == 'queued'),
                         key=lambda j: j.get('queue_order', 0))
        view['queue_position'] = next((i for i, row in enumerate(waiting, 1)
                                       if row['id'] == job['id']), None)
    view['original_filename'] = view.get('original_filename') or ''
    view['download_name'] = ttml_filename(job)
    view.setdefault('result', None)
    view.setdefault('error', None)
    return view


def persist(job_id):
    with lock:
        jobs[job_id]['updated_at'] = time.time()
        try:
            saved = save_job(DATA / job_id, jobs[job_id])
            if saved == 'checkpoint':
                jobs[job_id]['persistence_notice'] = '进度主文件暂时被占用，已保存恢复副本，计算继续。'
            else:
                jobs[job_id].pop('persistence_notice', None)
        except OSError:
            # A progress-log update must never throw through GPU/LLM inference.
            # Real result-write errors still propagate from the pipeline.
            jobs[job_id]['persistence_notice'] = '进度暂未写入磁盘，计算仍继续；请保持应用运行。'
            logging.getLogger(__name__).warning('Job-state save unavailable for %s; continuing in memory', job_id)


def process(job_id, audio_path, lyrics, language, syllables, vocals_path, audio_mode='lead', aligner="auto", english_splitter="syllabify", llm_config=None, llm_result=None, llm_prompt_profile='fine', trust_line_timestamps=False):
    def log(message):
        with lock:
            jobs[job_id]["log"] = (jobs[job_id]["log"] + [message])[-150:]
            persist(job_id)
    with lock:
        if jobs[job_id]['state'] == 'cancelled':
            return
        jobs[job_id]["state"] = "running"
        persist(job_id)
    try:
        extra = {'llm_result':llm_result} if aligner == 'llm_import' else {}
        if aligner == 'llm_direct':
            extra['llm_prompt_profile'] = llm_prompt_profile
        extra['trust_line_timestamps'] = trust_line_timestamps
        result = run(audio_path, lyrics, DATA / job_id, language, syllables, vocals_path, log, audio_mode=audio_mode, aligner=aligner, english_splitter=english_splitter, llm_config=llm_config, **extra)
        with lock:
            jobs[job_id].update({"state": result["status"], "result": result})
    except Exception as exc:
        with lock:
            jobs[job_id].update({"state": "failed", "error": str(exc)})
        (DATA / job_id / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        log("处理失败：" + str(exc))
    finally:
        with lock:
            futures.pop(job_id, None)
            jobs[job_id]['finished_at'] = time.time()
            persist(job_id)


async def save_upload(upload, target, max_bytes):
    size = 0
    with target.open("wb") as f:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise HTTPException(413, "上传文件过大。音频最大 1 GB，歌词最大 2 MB。")
            f.write(chunk)
    if not size:
        raise HTTPException(400, "上传文件为空。")


@app.post("/api/jobs")
async def submit(request: Request, audio: UploadFile = File(...), lyrics: str = Form(""), language: str = Form("mixed"),
                 syllables: bool = Form(True), lyrics_file: UploadFile | None = File(None),
                 vocals: UploadFile | None = File(None), audio_mode: str = Form('lead'),
                 aligner: str = Form("auto"), english_splitter: str = Form("syllabify"),
                 llm_base_url: str = Form(""), llm_model: str = Form(""),
                 llm_api_key: str = Form(""), llm_json_mode: bool = Form(False),
                 llm_thinking_mode: str = Form('default'), llm_thinking_format: str = Form('reasoning_effort'),
                 llm_reasoning_effort: str = Form(''),
                 llm_api_key_header: str = Form('Authorization'), llm_api_key_format: str = Form('auto'),
                 llm_result_file: UploadFile | None = File(None), llm_prompt_profile: str = Form('fine'),
                 trust_line_timestamps: bool = Form(False)):
    initialize()
    form = await request.form()
    if any(key in form for key in ('separate', 'lead')):
        raise HTTPException(400, '音频处理选项已更新，请刷新页面后重新选择。API 请使用 audio_mode。')
    if any(key in form for key in ('llm_audio_reading', 'llm_audio_split')):
        raise HTTPException(400, '听音辅助读音和听音拆分已移除，请刷新页面。纯文本读音请使用 aligner=llm_text。')
    # An edited text field wins over an imported file, just as in the UI.
    if not lyrics.strip() and lyrics_file and lyrics_file.filename:
        raw = await lyrics_file.read(MAX_LYRICS_BYTES + 1)
        if len(raw) > MAX_LYRICS_BYTES:
            raise HTTPException(413, '歌词最大 2 MB。')
        try:
            lyrics = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            raise HTTPException(400, '歌词文件请保存为 UTF-8 编码。') from None
    try:
        plan = plan_request(lyrics, language=language, aligner=aligner, audio_mode=audio_mode,
                            syllables=syllables, english_splitter=english_splitter,
                            trust_line_timestamps=trust_line_timestamps)
        if plan.aligner == 'llm_direct':
            read_prompt(llm_prompt_profile)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from None
    llm_result = None
    if plan.aligner == 'llm_import':
        llm_result = await read_llm_result(llm_result_file)
        try:
            await run_in_threadpool(import_response, lyrics, llm_result)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
    llm_config = None
    if plan.needs_llm:
        try:
            llm_config = LlmConfig(llm_base_url,llm_model,llm_api_key,llm_json_mode,
                                   llm_thinking_mode,llm_thinking_format,llm_reasoning_effort,
                                   llm_api_key_header,llm_api_key_format)
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from None
    with lock:
        if sum(j["state"] in ACTIVE_STATES for j in jobs.values()) >= MAX_QUEUE_JOBS:
            raise HTTPException(429, f"队列已满（最多 {MAX_QUEUE_JOBS} 首），请等待或移除尚未开始的歌曲。")
        job_id = uuid4().hex
        original = audio_basename(audio.filename)
        jobs[job_id] = {"id": job_id, "state": "uploading", "log": [], "name": Path(original).stem,
                        'original_filename': original, 'created_at': time.time(),
                        "english_splitter": plan.english_splitter if plan.syllables else "word",
                        'audio_mode':plan.audio_mode, 'aligner':plan.aligner,
                        'language':language, 'syllables':plan.syllables,
                        'llm_prompt_profile':llm_prompt_profile,
                        'trust_line_timestamps':plan.trust_line_timestamps}
        jobs[job_id]['download_name'] = ttml_filename(jobs[job_id])
    folder = DATA / job_id
    try:
        folder.mkdir()
        suffix = Path(original).suffix.lower()
        if suffix not in ALLOWED:
            raise HTTPException(400, "请选择常见音频格式，例如 WAV、FLAC、MP3、M4A。")
        audio_path = folder / ("source" + suffix)
        await save_upload(audio, audio_path, 1024**3)
        vocals_path = None
        if vocals and vocals.filename:
            vocal_suffix = Path(vocals.filename).suffix.lower()
            if vocal_suffix not in ALLOWED:
                raise HTTPException(400, "不支持的人声音频格式。")
            vocals_path = folder / ("stem" + vocal_suffix)
            await save_upload(vocals, vocals_path, 1024**3)
        (folder / 'lyrics.txt').write_text(lyrics, encoding='utf-8')
        extra = [llm_result] if plan.aligner == 'llm_import' else [None,llm_prompt_profile] if plan.aligner == 'llm_direct' else []
        if plan.trust_line_timestamps:
            extra = [llm_result, llm_prompt_profile if plan.aligner == 'llm_direct' else 'fine', True]
        with lock:
            jobs[job_id]["state"] = "queued"
            jobs[job_id]['queue_order'] = next(queue_sequence)
            persist(job_id)
            futures[job_id] = executor.submit(process, job_id, audio_path, lyrics, language, plan.syllables, vocals_path,
                                             plan.audio_mode, plan.aligner, plan.english_splitter, llm_config,
                                             *extra)
            return {'id': job_id, 'name': jobs[job_id]['name'], 'download_name': jobs[job_id]['download_name']}
    except Exception:
        with lock:
            jobs.pop(job_id, None)
            futures.pop(job_id, None)
        # No recursive deletion: all uploaded artifacts have known paths.
        try:
            if folder.is_dir():
                for entry in folder.iterdir():
                    if entry.is_file():
                        entry.unlink()
                folder.rmdir()
        except OSError:
            logging.getLogger(__name__).warning('Could not remove incomplete upload %s', job_id)
        raise


def get_job(job_id):
    if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
        raise HTTPException(404, "任务不存在。")
    with lock:
        if job_id in jobs:
            return job_view(jobs[job_id])
    try:
        job = load_job(DATA / job_id)
    except (OSError, ValueError):
        raise HTTPException(503, '进度文件暂时无法读取，请稍后重试。') from None
    if job is not None:
        if job["state"] in ("uploading", "queued", "running"):
            job["state"] = "failed"
            job["error"] = "应用已重启，上次任务中断。请重新提交。"
        return job_view(job)
    raise HTTPException(404, "任务不存在。")


@app.get('/api/jobs')
def list_jobs():
    """Show only jobs submitted since this server process started."""
    with lock:
        current = {identifier: job_view(job) for identifier, job in jobs.items()}
    active = [job for job in current.values() if job['state'] in ACTIVE_STATES]
    recent = [job for job in current.values() if job['state'] not in ACTIVE_STATES]
    active.sort(key=lambda job: ({'running': 0, 'queued': 1, 'uploading': 2}[job['state']],
                                  job.get('queue_order', 0), job.get('created_at', 0)))
    recent.sort(key=lambda job: job.get('created_at', 0), reverse=True)
    selected = active + recent[:max(0, MAX_QUEUE_JOBS - len(active))]
    summaries = []
    for job in selected:
        summary = {key: value for key, value in job.items() if not key.startswith('_')}
        summary['log'] = list(job.get('log', []))[-1:]
        summaries.append(summary)
    return {'jobs': summaries, 'running': sum(job['state'] == 'running' for job in active),
            'waiting': sum(job['state'] == 'queued' for job in active), 'limit': MAX_QUEUE_JOBS}


@app.post('/api/jobs/{job_id}/cancel')
def cancel_job(job_id: str):
    with lock:
        job = get_job(job_id)
        future = futures.get(job_id)
        if job['state'] != 'queued' or future is None or not future.cancel():
            raise HTTPException(409, '只能移除尚未开始的歌曲；该任务已开始或已结束。')
        jobs[job_id].update(state='cancelled', finished_at=time.time())
        jobs[job_id]['log'].append('已从等待队列移除。')
        futures.pop(job_id, None)
        persist(job_id)
        return job_view(jobs[job_id])


async def read_llm_result(upload):
    if upload is None:
        raise HTTPException(400, '请粘贴或导入网页 LLM 返回的 JSON。')
    raw = await upload.read(MAX_RESULT_BYTES + 1)
    if len(raw) > MAX_RESULT_BYTES:
        raise HTTPException(413, '网页 LLM 结果最大 16 MiB。')
    try:
        return raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise HTTPException(400, '网页 LLM 结果请保存为 UTF-8 JSON。') from None


@app.post('/api/llm/prompt')
async def llm_prompt(lyrics: str = Form(''), audio_name: str = Form(''), llm_prompt_profile: str = Form('fine')):
    try:
        return await run_in_threadpool(export_prompt, lyrics, audio_name, prompt_profile=llm_prompt_profile)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400,str(exc)) from None


@app.post('/api/lyrics/ttml')
async def import_ttml(lyrics: str = Form('')):
    from .ttml_input import normalize_ttml
    try:
        return await run_in_threadpool(normalize_ttml, lyrics)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from None


@app.post('/api/llm/import/validate')
async def validate_llm_import(lyrics: str = Form(''), llm_result_file: UploadFile | None = File(None)):
    result = await read_llm_result(llm_result_file)
    try:
        audit = await run_in_threadpool(import_response, lyrics, result)
        return validation_summary(audit)
    except (ValueError, TypeError) as exc:
        raise HTTPException(400,str(exc)) from None


@app.post('/api/llm/test')
async def llm_test(llm_base_url: str = Form(''), llm_model: str = Form(''),
                   llm_api_key: str = Form(''), llm_json_mode: bool = Form(False),
                   llm_thinking_mode: str = Form('default'), llm_thinking_format: str = Form('reasoning_effort'),
                   llm_reasoning_effort: str = Form(''), llm_api_key_header: str = Form('Authorization'),
                   llm_api_key_format: str = Form('auto')):
    try:
        config = LlmConfig(llm_base_url,llm_model,llm_api_key,llm_json_mode,
                           llm_thinking_mode,llm_thinking_format,llm_reasoning_effort,
                           llm_api_key_header,llm_api_key_format)
    except ValueError as exc:
        raise HTTPException(400,str(exc)) from None
    def probe():
        session = LlmSession(config,log=lambda _:None)
        try:
            phones = ('K','AH1','N','T','R','IY0')
            parts = session.split('country',phones)
            if parts is None or session.fallback('country',phones):
                reason = session.cache[('country',phones)].get('reason','未返回有效映射')
                return {'ok':False,'message':'接口测试未通过：'+reason}
            return {'ok':True,'message':'接口与拆分校验通过；这不代表歌词准确率。',
                    'thinking':session.report()['thinking'],
                    'example':{'word':'country','parts':[p[0] for p in parts]},'requests':session.requests}
        finally:
            session.close()
    return await run_in_threadpool(probe)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return get_job(job_id)


@app.get("/api/jobs/{job_id}/files/{kind}")
def download(job_id: str, kind: str):
    job = get_job(job_id)
    allowed = {"ttml": "lyrics.ttml", "report": "alignment.json", "draft": "draft.lrc", "vocals": "vocals.wav",
               "lead": "lead_vocals.wav", "backing": "backing_vocals.wav", "residual": "residual.wav"}
    if kind not in allowed or (kind in ("ttml", "report") and job["state"] != "complete") or (kind == "draft" and job["state"] != "draft"):
        raise HTTPException(404, "文件尚未生成。")
    if kind == "vocals" and (job["state"] not in ("complete", "draft") or
                              job.get("result", {}).get("source") not in ("provided_vocals", "separated_vocals")):
        raise HTTPException(404, "该任务没有可下载的人声轨。")
    if kind in ("lead", "backing", "residual") and (job["state"] not in ("complete", "draft") or
                                         not job.get("result", {}).get("lead_separated")):
        raise HTTPException(404, "该任务没有主唱／和声分轨。")
    path = DATA / job_id / allowed[kind]
    if not path.exists():
        raise HTTPException(404, "文件不存在。")
    media = {"ttml": "application/ttml+xml", "report": "application/json", "draft": "text/plain", "vocals": "audio/wav",
             "lead": "audio/wav", "backing": "audio/wav", "residual": "audio/wav"}[kind]
    return FileResponse(path, filename=job['download_name'] if kind == 'ttml' else allowed[kind], media_type=media)

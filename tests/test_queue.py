"""API regressions for the single-worker multi-song queue.

The fixture drives the real HTTP route and a real one-worker executor, while
stubbing only the alignment pipeline.  It therefore exercises queue state,
persistence, cancellation, and download headers without models or providers.
"""
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from email.message import Message
import itertools
import json
from pathlib import Path
import threading

import pytest
from fastapi.testclient import TestClient

from timing import job_store, server


SUMMARY_FIELDS = {
    'state', 'name', 'original_filename', 'download_name', 'queue_position',
    'log', 'result', 'error',
}


def lyrics(label):
    return f'[00:01.00]{label}\n[00:04.00]{label}\n'


def headers():
    return {'X-Timing-Token': server.TOKEN}


class FixtureRun:
    """A deterministic, event-gated stand-in for the alignment pipeline."""

    def __init__(self, data):
        self.data = data
        self.calls = []
        self.failures = {}
        self.started = {}
        self.released = {}
        self.finished = {}
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def _new_event(self, mapping, job_id):
        with self._lock:
            return mapping.setdefault(job_id, threading.Event())

    def wait_started(self, job_id, timeout=5):
        if not self._new_event(self.started, job_id).wait(timeout):
            raise AssertionError(f'job never started: {job_id}')

    def release(self, job_id):
        self._new_event(self.released, job_id).set()

    def fail(self, job_id, message):
        with self._lock:
            self.failures[job_id] = message

    def release_all(self):
        with self._lock:
            events = list(self.released.values())
        for event in events:
            event.set()

    def call(self, job_id):
        with self._lock:
            return next(row for row in self.calls if row['job_id'] == job_id)

    def __call__(self, audio_path, source_lyrics, output, language, syllables,
                 vocals_path, log, **options):
        job_id = Path(output).name
        gate = self._new_event(self.released, job_id)
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append({
                'job_id': job_id,
                'audio_path': Path(audio_path),
                'lyrics': source_lyrics,
                'output': Path(output),
                'language': language,
                'syllables': syllables,
                'vocals_path': vocals_path,
                'options': dict(options),
            })
        started = self._new_event(self.started, job_id)
        finished = self._new_event(self.finished, job_id)
        try:
            log('fixture running ' + job_id)
            started.set()
            if not gate.wait(5):
                raise AssertionError(f'fixture release missing for {job_id}')
            with self._lock:
                failure = self.failures.get(job_id)
            if failure:
                raise RuntimeError(failure)

            Path(output).mkdir(exist_ok=True)
            (Path(output) / 'lyrics.ttml').write_text(
                f'<tt xmlns="http://www.w3.org/ns/ttml">{source_lyrics}</tt>',
                encoding='utf-8',
            )
            (Path(output) / 'alignment.json').write_text('{}', encoding='utf-8')
            return {
                'status': 'complete',
                'source': 'fixture',
                'audio_mode': options.get('audio_mode'),
                'aligner': options.get('aligner'),
                'language': language,
                'lyrics': source_lyrics,
            }
        finally:
            with self._lock:
                self.active -= 1
            finished.set()


class ApiHarness:
    def __init__(self, client, data, runs, monkeypatch, executor):
        self.client = client
        self.data = data
        self.runs = runs
        self.monkeypatch = monkeypatch
        self.executor = executor

    def submit(self, filename, label, *, vocals=None, **fields):
        data = {
            'lyrics': lyrics(label),
            'language': 'en',
            'syllables': 'true',
            'audio_mode': 'lead',
            'aligner': 'ctc',
            'english_splitter': 'syllabify',
        }
        data.update({key: str(value) if isinstance(value, bool) else value
                     for key, value in fields.items()})
        files = {'audio': (filename, b'fixture audio ' + label.encode(), 'audio/wav')}
        if vocals is not None:
            name, payload = vocals
            files['vocals'] = (name, payload, 'audio/wav')
        response = self.client.post('/api/jobs', data=data, files=files, headers=headers())
        assert response.status_code == 200, response.text
        return response.json()

    def queue(self):
        response = self.client.get('/api/jobs')
        assert response.status_code == 200, response.text
        document = response.json()
        assert set(document) == {'jobs', 'running', 'waiting', 'limit'}
        assert document['limit'] == server.MAX_QUEUE_JOBS
        for summary in document['jobs']:
            assert SUMMARY_FIELDS <= set(summary), summary
            assert len(summary['log']) <= 1
        return document


@pytest.fixture
def api(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    runs = FixtureRun(data)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='timing-queue-test')

    monkeypatch.setattr(server, 'DATA', data)
    monkeypatch.setattr(server, 'initialize', lambda: None)
    monkeypatch.setattr(server, 'jobs', {})
    monkeypatch.setattr(server, 'futures', {})
    monkeypatch.setattr(server, 'queue_sequence', itertools.count(1))
    monkeypatch.setattr(server, 'MAX_QUEUE_JOBS', 50)
    monkeypatch.setattr(server, 'executor', executor)
    monkeypatch.setattr(server, 'run', runs)
    monkeypatch.setattr(server, 'LlmConfig', lambda *args, **kwargs: pytest.fail('unexpected LLM config'))
    monkeypatch.setattr(server, 'LlmSession', lambda *args, **kwargs: pytest.fail('unexpected LLM session'))

    with TestClient(server.app) as client:
        try:
            yield ApiHarness(client, data, runs, monkeypatch, executor)
        finally:
            runs.release_all()
    executor.shutdown(wait=True)


def _by_id(document):
    return {job['id']: job for job in document['jobs']}


def _future(job_id):
    future = server.futures.get(job_id)
    assert future is not None
    assert isinstance(future, Future)
    return future


def _finish_run(api, job_id):
    future = _future(job_id)
    api.runs.release(job_id)
    return future.result(timeout=5)


def _filename_from_content_disposition(value):
    message = Message()
    message['content-disposition'] = value
    return message.get_filename()


def test_fifo_positions_cancel_only_waiting_and_one_worker(api):
    assert server.executor._max_workers == 1
    first = api.submit('First take.mp3', 'first lyric')
    first_id = first['id']
    api.runs.wait_started(first_id)

    second = api.submit('Second take.wav', 'second lyric')
    third = api.submit('Third take.flac', 'third lyric')
    second_id, third_id = second['id'], third['id']

    document = api.queue()
    assert (document['running'], document['waiting']) == (1, 2)
    jobs = _by_id(document)
    assert jobs[first_id]['state'] == 'running'
    assert jobs[first_id]['queue_position'] is None
    assert jobs[first_id]['log'] == ['fixture running ' + first_id]
    assert jobs[second_id]['state'] == 'queued'
    assert jobs[second_id]['queue_position'] == 1
    assert jobs[third_id]['queue_position'] == 2
    for job_id in (first_id, second_id, third_id):
        assert jobs[job_id]['result'] is None
        assert jobs[job_id]['error'] is None

    assert api.client.post(
        f'/api/jobs/{second_id}/cancel', headers={'X-Timing-Token': 'wrong'},
    ).status_code == 403
    assert api.client.post(
        f'/api/jobs/{first_id}/cancel', headers=headers(),
    ).status_code == 409

    second_future = _future(second_id)
    cancelled = api.client.post(f'/api/jobs/{second_id}/cancel', headers=headers())
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()['state'] == 'cancelled'
    assert cancelled.json()['queue_position'] is None
    assert second_future.cancelled()
    assert api.client.post(
        f'/api/jobs/{second_id}/cancel', headers=headers(),
    ).status_code == 409
    assert api.client.post(
        '/api/jobs/' + 'f' * 32 + '/cancel', headers=headers(),
    ).status_code == 404

    document = api.queue()
    jobs = _by_id(document)
    assert (document['running'], document['waiting']) == (1, 1)
    assert jobs[third_id]['queue_position'] == 1
    assert jobs[second_id]['state'] == 'cancelled'

    _finish_run(api, first_id)
    api.runs.wait_started(third_id)
    _finish_run(api, third_id)

    called = [row['job_id'] for row in api.runs.calls]
    assert called == [first_id, third_id]
    assert api.runs.max_active == 1
    jobs = _by_id(api.queue())
    assert jobs[third_id]['state'] == 'complete'
    assert jobs[third_id]['queue_position'] is None
    assert jobs[third_id]['result']['status'] == 'complete'
    assert jobs[third_id]['error'] is None
    assert api.client.post(
        f'/api/jobs/{first_id}/cancel', headers=headers(),
    ).status_code == 409
    assert json.loads((api.data / second_id / 'job.json').read_text(encoding='utf-8'))['state'] == 'cancelled'


def test_failure_does_not_consume_following_job_or_its_options(api):
    first = api.submit('failure source.wav', 'failing lyric')
    first_id = first['id']
    api.runs.wait_started(first_id)

    second = api.submit(
        '第二曲 (Live).mp3', 'second lyric', language='ja', aligner='tifa',
        audio_mode='vocals', vocals=('harmony.wav', b'fixture vocals'),
    )
    second_id = second['id']
    second_future = _future(second_id)
    document = api.queue()
    jobs = _by_id(document)
    assert jobs[second_id]['state'] == 'queued'
    assert jobs[second_id]['queue_position'] == 1

    api.runs.fail(first_id, 'fixture processing failure')
    _finish_run(api, first_id)
    api.runs.wait_started(second_id)

    second_call = api.runs.call(second_id)
    assert second_call['lyrics'] == lyrics('second lyric')
    assert second_call['language'] == 'ja'
    assert second_call['options']['aligner'] == 'tifa'
    assert second_call['options']['audio_mode'] == 'vocals'
    assert second_call['audio_path'] == api.data / second_id / 'source.mp3'
    assert second_call['vocals_path'] == api.data / second_id / 'stem.wav'
    assert second_future is not None

    api.runs.release(second_id)
    assert second_future.result(timeout=5) is None
    jobs = _by_id(api.queue())
    assert jobs[first_id]['state'] == 'failed'
    assert 'fixture processing failure' in jobs[first_id]['error']
    assert jobs[second_id]['state'] == 'complete'
    assert jobs[second_id]['result']['lyrics'] == lyrics('second lyric')
    assert (api.data / first_id / 'error.log').exists()
    assert (api.data / first_id / 'source.wav').read_bytes() == b'fixture audio failing lyric'
    assert (api.data / second_id / 'source.mp3').read_bytes() == b'fixture audio second lyric'
    assert (api.data / second_id / 'stem.wav').read_bytes() == b'fixture vocals'
    assert api.runs.max_active == 1


def test_uploading_reserves_capacity_before_it_becomes_queued(api):
    upload_started = threading.Event()
    upload_release = threading.Event()
    upload_calls = 0
    original_save_upload = server.save_upload

    async def stalled_upload(upload, target, max_bytes):
        nonlocal upload_calls
        upload_calls += 1
        upload_started.set()
        if not await asyncio.to_thread(upload_release.wait, 5):
            raise AssertionError('upload release missing')
        await original_save_upload(upload, target, max_bytes)

    api.monkeypatch.setattr(server, 'save_upload', stalled_upload)
    api.monkeypatch.setattr(server, 'MAX_QUEUE_JOBS', 1)
    box = {}

    def first_request():
        with TestClient(server.app) as sender:
            try:
                box['response'] = sender.post(
                    '/api/jobs',
                    data={'lyrics': lyrics('uploading lyric'), 'aligner': 'ctc'},
                    files={'audio': ('Uploading.wav', b'fixture audio', 'audio/wav')},
                    headers=headers(),
                )
            except BaseException as exc:  # surfaced in the test thread
                box['error'] = exc

    thread = threading.Thread(target=first_request, daemon=True)
    thread.start()
    try:
        assert upload_started.wait(5)
        assert not api.runs.calls
        assert not server.futures
        document = api.queue()
        assert (document['running'], document['waiting']) == (0, 0)
        assert [(job['state'], job['queue_position']) for job in document['jobs']] == [('uploading', None)]

        rejected = api.client.post(
            '/api/jobs',
            data={'lyrics': lyrics('capacity lyric'), 'aligner': 'ctc'},
            files={'audio': ('Over limit.wav', b'fixture audio', 'audio/wav')},
            headers=headers(),
        )
        assert rejected.status_code == 429
        assert upload_calls == 1
    finally:
        upload_release.set()
        thread.join(5)

    assert not thread.is_alive()
    assert 'error' not in box, box.get('error')
    response = box['response']
    assert response.status_code == 200, response.text
    job_id = response.json()['id']
    api.runs.wait_started(job_id)
    _finish_run(api, job_id)
    assert _by_id(api.queue())[job_id]['state'] == 'complete'


def test_refresh_keeps_this_startup_only_and_does_not_rerun_old_jobs(api):
    finished = api.submit('Finished.wav', 'finished lyric')
    finished_id = finished['id']
    api.runs.wait_started(finished_id)
    _finish_run(api, finished_id)
    run_count = len(api.runs.calls)
    assert _by_id(api.queue())[finished_id]['state'] == 'complete'
    assert _by_id(api.queue())[finished_id]['state'] == 'complete'

    server.jobs.clear()
    server.futures.clear()
    interrupted = {}
    before = {}
    for index, state in enumerate(('uploading', 'queued', 'running'), start=1):
        job_id = str(index) * 32
        folder = api.data / job_id
        folder.mkdir()
        job = {
            'id': job_id, 'state': state, 'name': f'interrupted {state}',
            'original_filename': f'interrupted {state}.wav', 'log': ['saved'],
            'created_at': index, 'language': 'en', 'aligner': 'ctc',
            'audio_mode': 'lead', 'syllables': True, 'english_splitter': 'syllabify',
        }
        job_store.save_job(folder, job)
        interrupted[job_id] = state
        before[job_id] = (folder / 'job.json').read_bytes()

    document = api.queue()
    jobs = _by_id(document)
    assert jobs == {}
    assert (document['running'], document['waiting']) == (0, 0)
    assert api.client.get(f'/api/jobs/{finished_id}').json()['state'] == 'complete'
    for job_id, original_state in interrupted.items():
        assert job_id not in jobs
        assert (api.data / job_id / 'job.json').read_bytes() == before[job_id]
        assert json.loads(before[job_id])['state'] == original_state
        detail = api.client.get(f'/api/jobs/{job_id}')
        assert detail.status_code == 200
        assert detail.json()['state'] == 'failed'
        assert (api.data / job_id / 'job.json').read_bytes() == before[job_id]

    assert len(api.runs.calls) == run_count
    assert server.futures == {}


def test_ttml_download_name_uses_original_stem_and_keeps_canonical_artifact(api):
    uploaded_name = '夜に駆ける (Live):feat*?.mp3'
    expected_stem = '夜に駆ける (Live)_feat__'
    response = api.submit(uploaded_name, 'download lyric')
    job_id = response['id']
    assert response['name'] == Path(uploaded_name).stem
    assert response['download_name'] == expected_stem + '.ttml'
    api.runs.wait_started(job_id)
    _finish_run(api, job_id)

    job = api.client.get(f'/api/jobs/{job_id}').json()
    assert job['original_filename'] == uploaded_name
    assert job['download_name'] == expected_stem + '.ttml'
    assert (api.data / job_id / 'lyrics.ttml').exists()
    assert not (api.data / job_id / (expected_stem + '.ttml')).exists()
    download = api.client.get(f'/api/jobs/{job_id}/files/ttml')
    assert download.status_code == 200
    assert _filename_from_content_disposition(
        download.headers['content-disposition'],
    ) == expected_stem + '.ttml'

    legacy_id = 'e' * 32
    legacy_folder = api.data / legacy_id
    legacy_folder.mkdir()
    (legacy_folder / 'lyrics.ttml').write_text('<tt>legacy</tt>', encoding='utf-8')
    job_store.save_job(legacy_folder, {
        'id': legacy_id, 'state': 'complete', 'name': 'Legacy Set.Final',
        'log': ['done'], 'created_at': 1,
    })
    legacy = api.client.get(f'/api/jobs/{legacy_id}/files/ttml')
    assert legacy.status_code == 200
    assert _filename_from_content_disposition(
        legacy.headers['content-disposition'],
    ) == 'Legacy Set.Final.ttml'
    assert (legacy_folder / 'lyrics.ttml').read_text(encoding='utf-8') == '<tt>legacy</tt>'


def test_llm_secret_is_held_by_queued_future_but_not_snapshots_or_api(api):
    secret = 'queue-secret-not-for-disk-or-http'
    sentinel = object()
    configs = []

    def llm_config(*args, **kwargs):
        configs.append((args, kwargs))
        return sentinel

    api.monkeypatch.setattr(server, 'LlmConfig', llm_config)
    blocker = api.submit('Blocker.wav', 'blocker lyric')
    blocker_id = blocker['id']
    api.runs.wait_started(blocker_id)

    queued = api.submit(
        'Secret Song.wav', 'secret lyric', aligner='llm_text', language='en',
        llm_base_url='https://example.invalid/v1', llm_model='fixture-model',
        llm_api_key=secret, llm_json_mode=True, llm_api_key_header='X-Key',
        llm_api_key_format='raw', llm_thinking_mode='enabled',
        llm_thinking_format='reasoning_effort', llm_reasoning_effort='low',
    )
    queued_id = queued['id']
    assert server.jobs[queued_id]['state'] == 'queued'
    assert configs and configs[0][0][2] == secret

    summary = _by_id(api.queue())[queued_id]
    detail = api.client.get(f'/api/jobs/{queued_id}')
    snapshot = (api.data / queued_id / 'job.json').read_text(encoding='utf-8')
    assert summary['queue_position'] == 1
    assert secret not in json.dumps(summary, ensure_ascii=False)
    assert secret not in detail.text
    assert secret not in snapshot
    assert 'llm_config' not in summary and 'llm_api_key' not in summary
    assert '"llm_config"' not in snapshot and '"llm_api_key"' not in snapshot

    queued_future = _future(queued_id)
    _finish_run(api, blocker_id)
    api.runs.wait_started(queued_id)
    assert api.runs.call(queued_id)['options']['llm_config'] is sentinel
    api.runs.release(queued_id)
    assert queued_future.result(timeout=5) is None
    assert secret not in (api.data / queued_id / 'job.json').read_text(encoding='utf-8')
    assert secret not in api.client.get('/api/jobs').text

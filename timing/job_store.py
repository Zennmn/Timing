"""Recoverable job-state snapshots, including transient Windows file locks."""
import json
import os
from pathlib import Path
import time
from uuid import uuid4

RETRY_DELAYS = (0, .02, .05, .1, .2)


def save_job(folder, job):
    folder = Path(folder)
    job['_state_revision'] = job.get('_state_revision', 0) + 1
    snapshot = json.dumps(job, ensure_ascii=False, indent=2)
    temp = folder / f'job.{uuid4().hex}.tmp'
    try:
        with temp.open('x', encoding='utf-8') as stream:
            stream.write(snapshot)
        for delay in RETRY_DELAYS:
            if delay:
                time.sleep(delay)
            try:
                os.replace(temp, folder / 'job.json')
                # Only our own completed checkpoint files; no recursive cleanup.
                for previous in folder.glob('job.state.*.json'):
                    try:
                        previous.unlink()
                    except OSError:
                        pass
                return 'primary'
            except PermissionError:
                continue
        # The existing primary may be held without delete sharing. A new name
        # lets us retain a complete snapshot without truncating the old file.
        checkpoint = folder / f'job.state.{time.time_ns()}.{uuid4().hex}.json'
        os.replace(temp, checkpoint)
        return 'checkpoint'
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def load_job(folder):
    folder = Path(folder)
    candidates = [folder / 'job.json', *folder.glob('job.state.*.json')]
    saved, last_error = [], None
    for path in candidates:
        try:
            job = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(job, dict) or job.get('id') != folder.name or job.get('state') not in {
                    'uploading', 'queued', 'running', 'complete', 'draft', 'failed', 'cancelled'}:
                continue
            revision = job.get('_state_revision', 0)
            if type(revision) is not int:
                continue
            saved.append((revision, path.stat().st_mtime_ns, job))
        except FileNotFoundError:
            continue
        except (OSError, ValueError) as exc:
            last_error = exc
    if saved:
        return max(saved, key=lambda row: row[:2])[2]
    if last_error:
        raise last_error
    return None

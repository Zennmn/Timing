"""One pinned local UniDic resource for every Japanese lexical reader.

Importing, readiness checks and runtime selection never download dictionaries.
The explicit installer hashes the files; cheap runtime checks verify its manifest
and sizes so UI status polling does not read the gigabyte dictionary repeatedly.
"""
import json
from pathlib import Path
import re

from .config import MODELS

VERSION = '2025.12'
RELEASE = '202512'
FOLDER = MODELS / 'unidic-cwj-202512'
ARCHIVE_URL = 'https://clrd.ninjal.ac.jp/unidic_archive/2512/unidic-cwj-202512.zip'
ARCHIVE_SHA256 = 'd94216b589d15d05c408ed59abc5259086703ebbac14e225b5314e4cd106c4db'
REQUIRED_FILES = ('sys.dic', 'unk.dic', 'matrix.bin', 'char.bin', 'dicrc',
                  'mecabrc', 'version', 'manifest.json')


def ready(folder=None):
    folder = Path(folder) if folder is not None else FOLDER
    try:
        if not all((folder / name).is_file() for name in REQUIRED_FILES):
            return False
        manifest = json.loads((folder / 'manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('version') != VERSION or manifest.get('release') != RELEASE or
                manifest.get('archive_sha256') != ARCHIVE_SHA256 or
                (folder / 'version').read_text(encoding='utf-8').strip() != VERSION):
            return False
        assets = manifest.get('assets', {})
        for name in REQUIRED_FILES[:-1]:
            entry = assets.get(name, {})
            size, digest = entry.get('bytes'), entry.get('sha256')
            if (type(size) is not int or size < 0 or (folder / name).stat().st_size != size or
                    not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest)):
                return False
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def dictionary_dir():
    if not ready():
        raise RuntimeError('日语 UniDic 2025.12 词典未准备完整，请运行 '
                           '.\\.venv\\Scripts\\python.exe -X utf8 scripts/download_unidic.py。')
    return FOLDER.resolve()


def make_tagger():
    import fugashi
    dictionary = dictionary_dir()
    return fugashi.Tagger(f'-r "{(dictionary / "mecabrc").as_posix()}" -d "{dictionary.as_posix()}"')


def status():
    return {'name': 'UniDic CWJ', 'version': VERSION, 'release': RELEASE, 'ready': ready()}

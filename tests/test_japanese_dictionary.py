import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from timing import japanese_dictionary as dictionary


def fixture_dictionary(folder):
    folder.mkdir(parents=True, exist_ok=True)
    assets = {}
    for name in dictionary.REQUIRED_FILES[:-1]:
        contents = dictionary.VERSION if name == 'version' else 'fixture-' + name
        path = folder / name
        path.write_text(contents, encoding='utf-8')
        data = path.read_bytes()
        assets[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    manifest = {'version': dictionary.VERSION, 'release': dictionary.RELEASE,
                'source': dictionary.ARCHIVE_URL, 'archive_sha256': dictionary.ARCHIVE_SHA256, 'assets': assets}
    (folder / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return manifest


def test_dictionary_selection_uses_the_verified_project_asset_and_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(dictionary, 'FOLDER', tmp_path / 'with spaces' / 'unidic')
    assert not dictionary.ready()
    with pytest.raises(RuntimeError, match='download_unidic.py'):
        dictionary.dictionary_dir()
    fixture_dictionary(dictionary.FOLDER)
    assert dictionary.ready()
    assert dictionary.dictionary_dir() == dictionary.FOLDER.resolve()
    (dictionary.FOLDER / 'sys.dic').unlink()
    assert not dictionary.ready()
    with pytest.raises(RuntimeError):
        dictionary.dictionary_dir()


@pytest.mark.parametrize('change', ['version', 'release', 'hash', 'size', 'invalid-json'])
def test_runtime_dictionary_check_rejects_incomplete_or_wrong_version_assets(tmp_path, change):
    manifest = fixture_dictionary(tmp_path)
    if change == 'version':
        manifest['version'] = '2.1.2'
    elif change == 'release':
        manifest['release'] = 'other'
    elif change == 'hash':
        manifest['archive_sha256'] = '0' * 64
    elif change == 'size':
        manifest['assets']['sys.dic']['bytes'] += 1
    (tmp_path / 'manifest.json').write_text('broken' if change == 'invalid-json' else json.dumps(manifest), encoding='utf-8')
    assert not dictionary.ready(tmp_path)


def test_text_tagger_delegates_to_the_shared_dictionary(monkeypatch):
    from timing.text import tagger
    expected = object()
    monkeypatch.setattr(dictionary, 'make_tagger', lambda: expected)
    tagger.cache_clear()
    try:
        assert tagger() is expected
    finally:
        tagger.cache_clear()


def test_status_reports_dictionary_readiness_for_four_model_groups(tmp_path, monkeypatch):
    from timing import server
    monkeypatch.setattr(dictionary, 'FOLDER', tmp_path)
    for name in ('separator_ready', 'ctc_ready', 'tifa_ready', 'bs_ready'):
        monkeypatch.setattr(server, name, lambda: True)
    with TestClient(server.app) as client:
        response = client.get('/api/status').json()
    assert not response['ready']
    assert response['japanese_dictionary'] == {'name': 'UniDic CWJ', 'version': '2025.12', 'release': '202512', 'ready': False}
    assert set(response['required_models']) == {'tifa', 'ctc', 'separator', 'bs_lead'}
    fixture_dictionary(tmp_path)
    assert server.status()['ready']


def test_installed_unidic_has_compatible_fields_and_local_phonemization():
    if not dictionary.ready():
        pytest.skip('full UniDic not installed')
    from timing.text import tagger, tokenize
    from timing.singing import phonemize
    from timing.ctc import text_units
    from timing.schema import Line
    words = list(tagger()('私は学校へ行く'))
    assert ''.join(word.surface for word in words) == '私は学校へ行く'
    assert all(word.feature.kana and word.feature.pron for word in words)
    assert next(word for word in words if word.surface == 'は').feature.pron == 'ワ'
    for text in ('今日も君と歩く', '青い箱をひらく', 'Tシャツを着た', '二人の未来', 'どうせ今日も遠い'):
        assert ''.join(unit.text for unit in tokenize(text, fine=False)) == text
        units, labels = text_units(text, language='ja')
        assert ''.join(unit.text for unit in units) == text and all(labels)
        units, *_ = phonemize([Line(text, language='ja')])
        assert ''.join(unit.text for unit in units) == text


def test_native_tifa_uses_the_same_dictionary_and_llm_readings_still_bypass_it(monkeypatch):
    if not dictionary.ready():
        pytest.skip('full UniDic not installed')
    from timing.config import MODELS
    if not (MODELS / 'tifa/deps/g2pflow/__init__.py').is_file():
        pytest.skip('native TIFA G2P not installed')
    import sys
    sys.path.insert(0, str(MODELS / 'tifa/deps'))
    from g2pflow.converters.japanese import JapaneseMecabConverter
    converter = JapaneseMecabConverter(dict_path=str(MODELS / 'tifa/dictionaries/japanese_dict_full.txt'),
                                       unidic_dir=str(dictionary.dictionary_dir()))
    assert Path(converter._get_tagger().dictionary_info[0]['filename']).resolve() == dictionary.dictionary_dir() / 'sys.dic'
    assert converter.convert('私は学校へ行く')
    from timing.llm_direct import compile_parts
    monkeypatch.setattr(dictionary, 'dictionary_dir', lambda: pytest.fail('LLM must not read lexical UniDic'))
    monkeypatch.setattr('timing.text.tagger', lambda: pytest.fail('LLM must not segment locally'))
    result = compile_parts('今日', [{'text': '今日', 'lang': 'ja', 'reading': 'キョー'}])
    assert result[0]['reading'] == 'キョー' and result[0]['groups']

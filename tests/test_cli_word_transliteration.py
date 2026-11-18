import sys
import types

import pytest

import timing
from timing import __main__ as cli


class _Plan:
    needs_llm = False
    syllables = True
    audio_mode = 'lead'
    english_splitter = 'syllabify'
    trust_line_timestamps = False


def _install_pipeline(monkeypatch, runs):
    module = types.ModuleType('timing.pipeline')

    def run(*args, **kwargs):
        runs.append((args, kwargs))
        args[2].mkdir()
        (args[2] / 'lyrics.ttml').write_text('<tt/>', encoding='utf-8')

    module.run = run
    monkeypatch.setitem(sys.modules, 'timing.pipeline', module)
    monkeypatch.setattr(timing, 'pipeline', module, raising=False)


def _run_cli(monkeypatch, tmp_path, *extra):
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'audio')
    lyrics = tmp_path / 'song.lrc'
    lyrics.write_text('[00:00]hi[00:01]', encoding='utf-8')
    output = tmp_path / 'song.ttml'
    plans, runs = [], []

    def plan_request(value, **kwargs):
        plans.append(kwargs)
        return _Plan()

    monkeypatch.setattr(cli, 'plan_request', plan_request)
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    _install_pipeline(monkeypatch, runs)
    monkeypatch.setattr(sys, 'argv', [
        'timing', str(audio), '--lyrics', str(lyrics), '--output', str(output),
        '--aligner', 'llm_text', *extra,
    ])
    cli.main()
    return plans, runs, output


def _export_cli(monkeypatch, tmp_path, *extra):
    lyrics = tmp_path / 'song.lrc'
    lyrics.write_text('[00:00]hi[00:01]', encoding='utf-8')
    prompt = tmp_path / 'prompt.txt'
    calls = []
    module = types.ModuleType('timing.llm_import')

    def export_prompt(value, audio_name='', **kwargs):
        calls.append((value, audio_name, kwargs))
        return {'prompt': 'fixture prompt'}

    module.export_prompt = export_prompt
    module.MAX_RESULT_BYTES = 16 * 1024 ** 2
    module.import_response = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, 'timing.llm_import', module)
    monkeypatch.setattr(timing, 'llm_import', module, raising=False)
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    monkeypatch.setattr(sys, 'argv', [
        'timing', '--lyrics', str(lyrics), '--export-llm-prompt', str(prompt), *extra,
    ])
    cli.main()
    return calls, prompt


def _import_cli(monkeypatch, tmp_path, *extra, fail=False, runs=None, plans=None):
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'audio')
    lyrics = tmp_path / 'song.lrc'
    lyrics.write_text('[00:00]hi[00:01]', encoding='utf-8')
    result = tmp_path / 'result.json'
    result.write_text('{}', encoding='utf-8')
    output = tmp_path / 'song.ttml'
    calls = []
    runs = [] if runs is None else runs
    plans = [] if plans is None else plans
    module = types.ModuleType('timing.llm_import')

    def import_response(value, response, **kwargs):
        calls.append((value, response, kwargs))
        if fail:
            raise ValueError('fixture rejected')
        return {'mode': 'llm_import'}

    module.import_response = import_response
    module.export_prompt = lambda *args, **kwargs: {'prompt': 'unused'}
    module.MAX_RESULT_BYTES = 16 * 1024 ** 2
    monkeypatch.setitem(sys.modules, 'timing.llm_import', module)
    monkeypatch.setattr(timing, 'llm_import', module, raising=False)
    monkeypatch.setattr(cli, 'initialize', lambda: None)

    def plan_request(value, **kwargs):
        plans.append(kwargs)
        return _Plan()

    monkeypatch.setattr(cli, 'plan_request', plan_request)
    _install_pipeline(monkeypatch, runs)
    monkeypatch.setattr(sys, 'argv', [
        'timing', str(audio), '--lyrics', str(lyrics), '--output', str(output),
        '--aligner', 'llm_import', '--llm-result', str(result), *extra,
    ])
    cli.main()
    return calls, runs, output


@pytest.mark.parametrize('extra, expected', [
    ((), False),
    (('--word-transliteration',), True),
])
def test_cli_forwards_word_transliteration_to_planner_and_pipeline(monkeypatch, tmp_path, extra, expected):
    plans, runs, output = _run_cli(monkeypatch, tmp_path, *extra)

    assert plans[0]['word_transliteration'] is expected
    assert runs[0][1]['word_transliteration'] is expected
    assert output.read_text(encoding='utf-8') == '<tt/>'


@pytest.mark.parametrize('extra, expected', [
    ((), False),
    (('--word-transliteration',), True),
])
def test_cli_forwards_word_transliteration_to_prompt_export(monkeypatch, tmp_path, extra, expected):
    calls, prompt = _export_cli(monkeypatch, tmp_path, *extra)

    assert calls[0][2]['word_transliteration'] is expected
    assert prompt.read_text(encoding='utf-8') == 'fixture prompt'


def test_cli_rejects_word_transliteration_outside_llm_routes(monkeypatch, tmp_path, capsys):
    audio = tmp_path / 'song.wav'
    audio.write_bytes(b'audio')
    lyrics = tmp_path / 'song.lrc'
    lyrics.write_text('[00:00]hi[00:01]', encoding='utf-8')
    monkeypatch.setattr(cli, 'initialize', lambda: None)
    _install_pipeline(monkeypatch, [])
    monkeypatch.setattr(sys, 'argv', [
        'timing', str(audio), '--lyrics', str(lyrics), '--aligner', 'tifa',
        '--word-transliteration',
    ])

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 1
    assert '仅适用于' in capsys.readouterr().err


@pytest.mark.parametrize('extra, expected', [
    ((), False),
    (('--word-transliteration',), True),
])
def test_cli_llm_import_preflight_receives_word_transliteration(monkeypatch, tmp_path, extra, expected):
    calls, runs, output = _import_cli(monkeypatch, tmp_path, *extra)

    assert calls[0][2]['word_transliteration'] is expected
    assert len(runs) == 1
    assert output.read_text(encoding='utf-8') == '<tt/>'


def test_cli_enabled_import_validation_stops_before_pipeline(monkeypatch, tmp_path, capsys):
    runs, plans = [], []
    with pytest.raises(SystemExit) as exc:
        _import_cli(monkeypatch, tmp_path, '--word-transliteration',
                    fail=True, runs=runs, plans=plans)

    assert exc.value.code == 1
    assert 'fixture rejected' in capsys.readouterr().err
    assert runs == [] and plans == []
    assert not (tmp_path / 'song.ttml').exists()

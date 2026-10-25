"""Isolated TIFA process: keep its generic lib/modules imports out of SOFA."""
import contextlib
import json
import re
import sys
import time
from pathlib import Path


def number_dictionary(clips, model_dir, output_dir):
    """Inject digit pronunciations without rewriting transcripts or model files."""
    from timing.numeric_lyrics import english_integer, english_numbers
    from g2pflow.converters.dictionary import load_pronunciation_dict
    texts = [Path(c["audio"]).with_suffix(".txt").read_text(encoding="utf-8") for c in clips]
    numbers = sorted({n for text in texts if english_numbers(text) for n in re.findall(r"\b[0-9]+\b", text)})
    if not numbers:
        return None
    lexicon = load_pronunciation_dict(model_dir / "dictionaries/ds_cmudict-07b.txt")
    entries = []
    for number in numbers:
        words = english_integer(number).split()
        phones = [p for word in words for p in lexicon[word][0]]
        entries.append(number + "\t" + " ".join(phones))
    path = output_dir / "english_numbers.txt"
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return path


SILENT_MARKS = ("ー", "゜")


def restore_silent_marks(source, texts):
    """Re-attach marks the Japanese frontend deletes as silent units.

    MeCab can split a prolonged-sound mark into its own token. The Japanese
    converter maps that token to an empty path and the framework drops silent
    words entirely, so the character vanishes from the model's word list: every
    later word then shifts against the lyric text and the line cannot be mapped
    back to characters. The mark carries no separate pronunciation, so it is
    appended to the preceding word, whose phonemes stay unchanged.
    """
    from timing.text import kept, canonical_apostrophes
    if not texts or any(not kept(char) for text in texts for char in text):
        return list(texts), set()
    wanted = [char for char in canonical_apostrophes(source) if kept(char)]
    merged = [[] for _ in texts]
    pointer = 0
    for index, text in enumerate(texts):
        for char in text:
            while pointer < len(wanted) and wanted[pointer].lower() != char.lower():
                if index and wanted[pointer] in SILENT_MARKS:
                    merged[index - 1].append(wanted[pointer])
                    pointer += 1
                else:
                    return list(texts), set()
            if pointer >= len(wanted):
                return list(texts), set()
            merged[index].append(char)
            pointer += 1
    while pointer < len(wanted) and wanted[pointer] in SILENT_MARKS:
        merged[-1].append(wanted[pointer])
        pointer += 1
    if pointer != len(wanted) or any(not chars for chars in merged):
        return list(texts), set()
    result = ["".join(chars) for chars in merged]
    return result, {index for index, (before, after) in enumerate(zip(texts, result)) if before != after}


def display_plan(text, choice, kana, overrides, fallback):
    from timing.text import tokenize
    from timing.singing import japanese_parts
    words = tokenize(text, fine=False)
    if len(words) != 1 or words[0].spoken != text:
        return fallback
    word = words[0]
    pronunciation = overrides.get(text) or word.source_pronunciation
    if len(word.reading) != len(pronunciation):
        return fallback
    pieces = japanese_parts(word, True)
    cursor, expected, plan = 0, [], []
    for surface, reading in pieces:
        actual = pronunciation[cursor:cursor+len(reading)]
        cursor += len(reading)
        groups = [g for w in kana._convert(actual) for g in w.readings[0].paths[0]]
        scripts = [g.script for g in groups]
        expected.extend(scripts)
        if not scripts and plan:
            plan[-1]["text"] += surface
            plan[-1]["flags"].append("shared_vowel_indivisible")
        else:
            plan.append({"text":surface, "group_count":len(scripts), "flags":[]})
    if cursor != len(pronunciation) or expected != choice["scripts"] or any(not p["group_count"] for p in plan):
        return fallback
    return plan


def display_plans(texts, selected, model_dir, japanese_overrides=None, shared_marks=()):
    from g2pflow.converters.japanese import JapaneseKanaConverter
    kana = JapaneseKanaConverter(dict_path=str(model_dir / "dictionaries/japanese_dict_full.txt"))
    overrides = japanese_overrides or {}
    plans = []
    for index, (text, choice) in enumerate(zip(texts, selected)):
        fallback = [{"text":text, "group_count":len(choice["scripts"]), "flags":["reading_mapping_indivisible"]}]
        if any(ph.startswith("en/") for ph in choice["phonemes"]):
            plan = None
        elif any(ph.startswith('zh/') for ph in choice['phonemes']):
            plan = [{'text':text, 'group_count':len(choice['scripts']), 'flags':['chinese_character']}]
        else:
            plan = display_plan(text, choice, kana, overrides, fallback)
        if index in shared_marks and plan:
            plan[-1]["flags"].append("prolonged_mark_shared")
        plans.append(plan)
    return plans


def transcripts(clips):
    """Read the transcript beside each clip, exactly as the G2P received it."""
    return {clip["id"]: Path(clip["audio"]).with_suffix(".txt").read_text(encoding="utf-8")
            for clip in clips}


def chinese_custom_words(source, texts, overrides):
    """Map manual phrase readings to their exact native word occurrences."""
    from timing.chinese import chinese_units, phrase_readings
    from timing.text import kept
    units = chinese_units(source)
    phrases = phrase_readings(units, overrides)
    manual, cursor = set(), 0
    for index, unit in enumerate(units):
        length = len(kept(unit.spoken).casefold())
        if unit.spoken in overrides or index in phrases:
            manual.update(range(cursor, cursor+length))
        cursor += length
    result, cursor = set(), 0
    for index, text in enumerate(texts):
        length = len(kept(text).casefold())
        if any(position in manual for position in range(cursor, cursor+length)):
            result.add(index)
        cursor += length
    return result


def main():
    model_dir, request_path = map(Path, sys.argv[1:3])
    root = Path(__file__).resolve().parent.parent
    sys.path[:0] = [str(model_dir / "deps"), str(model_dir / "runtime"), str(root)]
    import torch
    from timing.device import device, reset_peak_memory, peak_memory_gib
    from inference.api import load_inference_model, load_g2p_config
    from inference.data import AudioTextDataset
    from lib.path_traversal import materialize_paths
    from lib.config.schema import ConfigurationScope
    import yaml
    import unidic_lite

    request = json.loads(request_path.read_text(encoding="utf-8"))
    direct_mode = all('llm_direct_parts' in clip for clip in request['clips'])
    if direct_mode:
        from timing.llm_direct import write_pfml
        for clip in request['clips']:
            write_pfml(clip['audio'],clip['llm_direct_parts'])
    languages = request.get("languages", ["ja", "en"])
    if not languages or any(language not in ("ja", "en", "zh") for language in languages):
        raise ValueError("TIFA worker supports explicit zh/ja/en language scopes only")
    result_path = request_path.with_name("raw.json")
    with contextlib.redirect_stdout(sys.stderr):
        backend, vocabulary, config = load_inference_model(model_dir / "model.pt", scope=ConfigurationScope.FA)
    backend = backend.to(device()).eval()
    if device() == "mps":
        # MPS has no int64 scatter_reduce; TIFA's scoring uses it only on small
        # index tensors, so those calls run on the CPU.
        scatter_reduce = torch.Tensor.scatter_reduce

        def cpu_int64_scatter_reduce(self, dim, index, src, reduce, **kwargs):
            if self.device.type == "mps" and self.dtype == torch.int64:
                return scatter_reduce(self.cpu(), dim, index.cpu(), src.cpu(), reduce, **kwargs).to(self.device)
            return scatter_reduce(self, dim, index, src, reduce, **kwargs)

        torch.Tensor.scatter_reduce = cpu_int64_scatter_reduce
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text(encoding="utf-8"))["inference"]["g2p"]
    from timing.tifa_pronunciation import ensure_apostrophes_registered, APOSTROPHE_PREPROCESSOR_ID
    ensure_apostrophes_registered()
    cfg['preprocessors'].insert(0,{'id':APOSTROPHE_PREPROCESSOR_ID})
    converters = []
    for converter in cfg["converters"]:
        if converter['id'] == 'chinese-pinyin':
            if 'zh' not in languages:
                continue
            from timing.chinese_tifa import ensure_registered as ensure_chinese, CONVERTER_ID as CHINESE_ID
            ensure_chinese()
            converter = {'id':CHINESE_ID,'language':'zh','kwargs':{}}
        elif converter["id"] == "japanese-mecab":
            if "ja" not in languages:
                continue
            converter["kwargs"]["unidic_dir"] = unidic_lite.DICDIR
        elif converter.get("language") != "en" or "en" not in languages:
            continue
        for key in ("dict_path", "model_path"):
            value = converter.get("kwargs", {}).get(key)
            if value and value.startswith("@"):
                converter["kwargs"][key] = str(model_dir / value[1:])
        converters.append(converter)
    numeric = number_dictionary(request["clips"], model_dir, request_path.parent) if "en" in languages and not direct_mode else None
    sources = transcripts(request["clips"])
    if numeric:
        converters.insert(0, {"id": "dictionary", "language": "en", "kwargs": {"dict_path": str(numeric)}})
    if 'ja' in languages:
        from timing.tifa_pronunciation import ensure_loanwords_registered, LOANWORD_CONVERTER_ID
        ensure_loanwords_registered()
        converters.insert(0,{'id':LOANWORD_CONVERTER_ID,'language':'ja',
                             'kwargs':{'dict_path':str(model_dir/'dictionaries/japanese_dict_full.txt')}})
        from timing.tifa_pronunciation import ensure_numbers_registered, NUMBER_CONVERTER_ID
        ensure_numbers_registered()
        converters.insert(0,{'id':NUMBER_CONVERTER_ID,'language':'ja',
                             'kwargs':{'dict_path':str(model_dir/'dictionaries/japanese_dict_full.txt')}})
        if any(clip.get('japanese_audio_candidates') for clip in request['clips']):
            from timing.tifa_pronunciation import ensure_audio_registered, AUDIO_CONVERTER_ID
            ensure_audio_registered()
            converters.insert(0,{'id':AUDIO_CONVERTER_ID,'language':'ja',
                                'kwargs':{'dict_path':str(model_dir/'dictionaries/japanese_dict_full.txt')}})
    from timing.singing import pronunciation_overrides
    from timing.tifa_pronunciation import ensure_registered, prepare_configs
    overrides = {'ja':{},'en':{},'zh':{}} if direct_mode else pronunciation_overrides(request.get("pronunciations") or root / "dictionaries" / "pronunciations.json")
    for converter in converters:
        if converter['id'] == 'timing-mandarin':
            converter['kwargs']['overrides'] = overrides.get('zh', {})
    # A shared English digit dictionary must not claim numbers in zh clips.
    # Per-clip language scopes still exclude this converter for Japanese input.
    converters.sort(key=lambda entry: entry['id'] != 'timing-mandarin')
    override_configs = prepare_configs(overrides, languages, request_path.parent, model_dir)
    if override_configs:
        ensure_registered()
        for index, entry in enumerate(override_configs):
            converters.insert(index, entry)
    cfg["converters"] = converters
    if direct_mode:
        cfg = {'preprocessors':[],'converters':[]}
    custom = request_path.with_name("g2p.yaml")
    custom.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    dataset = AudioTextDataset(filemap={r["id"]: Path(r["audio"]) for r in request["clips"]},
                              g2p_config=load_g2p_config(custom), g2p_root=model_dir,
                              vocabulary=vocabulary, audio_sample_rate=backend.sample_rate,
                              language=languages, oov_handling="raise")
    started = time.monotonic()
    reset_peak_memory()
    results = []
    clips_by_id = {clip['id']:clip for clip in request['clips']}
    with torch.inference_mode():
        for index in range(len(dataset)):
            current_clip = clips_by_id[dataset.items[index][1]]
            dataset.language = current_clip.get('languages', languages)
            from timing.audio_readings import native_candidates
            # The native pipeline lowercases before converters run. Keep the
            # original lyric spelling in the audit, but match the same keys here.
            audio_candidates = {key.lower():value for key,value in
                                (current_clip.get('chinese_audio_candidates') or current_clip.get('japanese_audio_candidates',{})).items()}
            with native_candidates(audio_candidates):
                item = dataset[index]
            if item.get("skip"):
                raise ValueError(item.get("error") or item.get("warning") or "TIFA omitted an input")
            texts, shared_marks = restore_silent_marks(sources[item["identifier"]], item["texts"])
            batch = dataset.collate([item])
            batch = {key: value.to(device()) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            spec = backend.spectrogram(batch["waveform"], batch["duration"])
            scored = backend.score(spec, paths=batch["paths"], words=batch["words"],
                                   candidates=batch["candidates"], unit="levenshtein")
            tokens, owners, groups = materialize_paths(batch["paths"], batch["words"], batch["groups"], scored.choices)
            aligned = backend.align(spec, tokens=tokens, groups=groups, unit="second", skip_penalty=.5)
            spans = aligned.spans[0].cpu().tolist()
            selected = [candidates[choice-1] if choice > 0 else None
                        for candidates, choice in zip(item["lexicon"], scored.choices[0].tolist())]
            phones = [ph for word in selected if word for ph in word["phonemes"]]
            if any(word is None or not word["phonemes"] for word in selected):
                raise ValueError("TIFA selected an empty word pronunciation")
            active_languages = current_clip.get('languages', languages)
            japanese = overrides["ja"] if "ja" in active_languages else {}
            english = overrides["en"] if "en" in active_languages else {}
            chinese = overrides.get('zh', {}) if 'zh' in active_languages else {}
            chinese_manual = chinese_custom_words(sources[item['identifier']], item['texts'], chinese) if chinese else set()
            for word_index, (text, choice) in enumerate(zip(item["texts"], selected)):
                if text in japanese or text.lower() in english or word_index in chinese_manual:
                    choice["custom_pronunciation"] = True
                elif text in audio_candidates:
                    choice['llm_audio_reading_candidate'] = True
            if direct_mode:
                from timing.llm_direct import spoken
                direct_parts = current_clip['llm_direct_parts']
                if texts != [spoken(part['text']) for part in direct_parts] or len(selected)!=len(direct_parts):
                    raise ValueError('LLM 自主分段与原生 PFML 输入不一致。')
                for choice,part in zip(selected,direct_parts):
                    choice['llm_direct_candidate']=True
                    choice['llm_direct_reading']=part['reading']
                plans = [[{'text':text,'group_count':len(choice['scripts']),'flags':['llm_direct_candidate']}]
                         for text,choice in zip(texts,selected)]
            else:
                plans = display_plans(texts, selected, model_dir, japanese, shared_marks)
            row = {"id": item["identifier"], "texts": texts, "selected": selected,
                   "spans": spans, "phones": phones, "owners": owners[0].tolist(),
                   "groups": groups[0].tolist(), "agreement": float(aligned.agreement[0]),
                   "candidates": item["lexicon"], "frame_ms": backend.timestep*1000,
                   "display_plans":plans,
                   "skipped": [i for i, (a,b) in enumerate(spans) if b <= a]}
            if len(phones) != len(spans):
                raise ValueError("TIFA phone count mismatch")
            results.append(row)
            result_path.write_text(json.dumps({"results":results}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f'TIFA {index+1}/{len(dataset)} {item["identifier"]}', flush=True)
    result_path.write_text(json.dumps({"results":results, "languages":languages, "seconds":time.monotonic()-started,
                           "peak_torch_allocated_gib":peak_memory_gib(),
                           "pronunciation_overrides":{"ja":overrides["ja"] if "ja" in languages else {},
                                                      "zh":overrides.get('zh', {}) if 'zh' in languages else {},
                                                      "en":overrides["en"] if "en" in languages else {}}},
                          ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

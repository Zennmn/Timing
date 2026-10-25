"""Singing alignment from SOFA acoustic evidence and forced decoding.

SOFA runtime and checkpoints are pinned by scripts/download_sofa.py. Each
language uses its own acoustic head and its own pinned front-end dictionary
(Mandarin: the official opencpop-extension dictionary, never the Japanese
head or a TIFA fallback); the supplied phoneme sequence determines which head
supplies each state's emission. No duration division or zero-span repair is
performed. SOFA's original MIT license is retained with its runtime.
"""
import gc
import hashlib
import json
import math
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from .config import MODELS, ROOT
from .device import device, empty_cache
from .schema import Unit
from .text import kept, tokenize, SMALL
from .english_syllables import split_english, split_flags, prepare_llm_many, current_method, word_flags

SOFA_DIR = MODELS / "sofa"
OVERRIDES = ROOT / "dictionaries" / "pronunciations.json"
SOFA_CHECKPOINTS = {
    "ja": "ja/akm_ja_v001/model.ckpt",
    "en": "en/tgm_en_v100.ckpt",
    "zh": "zh/pretrained_mandarin_singing/v1.0.0_mandarin_singing.ckpt",
}
SOFA_ZH_DICTIONARY = "zh/opencpop-extension.txt"
SOFA_LANGUAGES = ("ja", "en", "zh")
SOFA_LABELS = {"ja": "日语", "en": "英语", "zh": "中文"}
VOWELS = {"aa", "ae", "ah", "ao", "aw", "ax", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}


def ready():
    return all((SOFA_DIR / p).is_file() for p in (
        "runtime/modules/task/forced_alignment.py", "en/tgm_en_v100.ckpt",
        "en/dictionary.txt", "ja/akm_ja_v001/model.ckpt", "ja/akm_ja_v001/dict.txt",
        SOFA_CHECKPOINTS["zh"], SOFA_ZH_DICTIONARY, "manifest.json"))


def pronunciation_overrides(path=None):
    source = Path(path) if path else OVERRIDES
    if not source.exists():
        return {"ja": {}, "en": {}, "zh": {}}
    rows = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, dict) or any(not isinstance(rows.get(lang, {}), dict) for lang in ("ja", "en", "zh")):
        raise ValueError("自定义读音表需包含 ja/en/zh 对象。")
    for word, reading in rows.get("ja", {}).items():
        if not isinstance(reading, str) or not re.fullmatch(r"[ぁ-ゖァ-ヺー]+", reading):
            raise ValueError(f"日语自定义读音「{word}」需填写假名。")
    for word, phones in rows.get("en", {}).items():
        if not isinstance(phones, list) or not phones or any(not isinstance(p, str) or not p.strip() for p in phones):
            raise ValueError(f"英语自定义读音「{word}」需填写音素数组。")
    chinese = {}
    for word, reading in rows.get("zh", {}).items():
        if not isinstance(word, str) or not word.strip():
            raise ValueError("中文自定义读音的词条应为非空文字。")
        if isinstance(reading, list) and all(isinstance(value, str) for value in reading):
            reading = " ".join(reading)
        if not isinstance(reading, str):
            raise ValueError(f"中文自定义读音「{word}」需填写拼音，音节间用空格分隔。")
        from .chinese import pinyin_groups
        pinyin_groups(reading, require_phones=False)
        chinese[word] = " ".join(reading.replace("’", " ").replace("'", " ").split())
    return {"ja": rows.get("ja", {}),
            "en": {word.lower(): phones for word, phones in rows.get("en", {}).items()},
            "zh": chinese}


@lru_cache(maxsize=1)
def japanese_dictionary():
    # Identity labels shared with SOFA; MMS must not need SOFA assets for kana.
    labels = "AP SP cl a i u e o N k ky s sh h hy g gy z zy d dy b by p py j f fy ch t ts ty n ny m my r ry y w"
    return {phone: [phone] for phone in labels.split()}


@lru_cache(maxsize=1)
def dictionary():
    rows = {}
    source = SOFA_DIR / "en/dictionary.txt"
    if not source.is_file():
        return rows  # english_parts can use the installed CMU dictionary.
    for row in source.read_text(encoding="utf-8").splitlines():
        if "\t" in row:
            word, phones = row.split("\t", 1)
            rows[word] = phones.split()
    return rows


@lru_cache(maxsize=1)
def mandarin_dictionary():
    # SOFA's own Mandarin dictionary; the aligner must not reuse the TIFA table.
    rows = {}
    source = SOFA_DIR / SOFA_ZH_DICTIONARY
    if not source.is_file():
        return rows
    for row in source.read_text(encoding="utf-8").splitlines():
        if "\t" in row:
            word, phones = row.split("\t", 1)
            rows[word.strip()] = phones.split()
    return rows


@lru_cache(maxsize=1)
def kana_converter():
    import pykakasi
    return pykakasi.kakasi()


def katakana(text):
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in text)


def kana_phones(reading):
    moras = []
    for c in katakana(reading):
        if c in SMALL and moras:
            moras[-1] += c
        else:
            moras.append(c)
    phones = []
    for mora in moras:
        if mora == "ン":
            phones.append("N")
        elif mora == "ッ":
            phones.append("cl")
        elif mora == "ー":
            if not phones or phones[-1] not in "aiueo":
                raise ValueError(f"長音の読みを解釈できません：{reading}")
            phones.append(phones[-1])
        else:
            roman = "".join(x["hepburn"] for x in kana_converter().convert(mora))
            if mora == "ヲ":
                roman = "o"
            parts = re.findall(r"ky|gy|sh|ch|ts|hy|ny|my|ry|by|py|fy|ty|dy|zy|[a-z]", roman)
            if not parts or "".join(parts) != roman:
                raise ValueError(f"无法转换日语读音：{reading}")
            phones.extend(parts)
    table = japanese_dictionary()
    missing = [phone for phone in phones if phone not in table]
    if missing:
        raise ValueError(f"日语读音「{reading}」包含配套音素表不支持的音素：{missing}")
    return [p for phone in phones for p in table[phone]]


def japanese_parts(unit, fine):
    reading = katakana(unit.reading or unit.spoken)
    if not fine or any(c.isdecimal() for c in unit.spoken):
        return [(unit.text, reading)]
    from .japanese_loanwords import loanword_parts, japanese_loanword_spans
    span = next(japanese_loanword_spans(unit.text), None)
    if span and kept(unit.text[span[0]:span[1]]) == unit.spoken:
        parts = loanword_parts(unit.text[span[0]:span[1]], reading)
        if parts:
            parts[0] = (unit.text[:span[0]] + parts[0][0], parts[0][1])
            parts[-1] = (parts[-1][0] + unit.text[span[1]:], parts[-1][1])
            return parts
    pieces = re.findall(r"[\u3400-\u9fff々〆〤]+|[\u3040-\u30ffー]", unit.spoken)
    merged = []
    for piece in pieces:
        if (piece in SMALL or piece == "ー") and merged:
            merged[-1] += piece
        else:
            merged.append(piece)
    cursor = 0
    result = []
    for index, piece in enumerate(merged):
        if re.search(r"[\u3400-\u9fff々]", piece):
            following = ""
            for later in merged[index + 1:]:
                if re.search(r"[\u3400-\u9fff々]", later):
                    break
                following += katakana(later)
            boundary = reading.find(following, cursor) if following else len(reading)
            if boundary <= cursor:
                return [(unit.text, reading)]
            sound = reading[cursor:boundary]
            cursor = boundary
        else:
            sound = katakana(piece)
            # は/へ as particles can have a contextual pronunciation.
            if len(merged) == 1:
                sound = reading
            if reading[cursor:cursor + len(sound)] != sound:
                return [(unit.text, reading)]
            cursor += len(sound)
        result.append((piece, sound))
    if cursor != len(reading) or not result:
        return [(unit.text, reading)]
    # Attach original spaces/punctuation without modifying lyric spelling.
    prefix = unit.text[:unit.text.find(unit.spoken)] if unit.spoken in unit.text else ""
    suffix = unit.text[len(prefix) + len(unit.spoken):]
    result[0] = (prefix + result[0][0], result[0][1])
    result[-1] = (result[-1][0] + suffix, result[-1][1])
    return result


def english_parts(unit, syllables, overrides=None):
    if unit.spoken.isascii() and unit.spoken.isdigit():
        from .numeric_lyrics import english_integer
        phones = []
        for word in english_integer(unit.spoken).split():
            phones.extend(english_parts(Unit(word, word), False, overrides)[0][1])
        return [(unit.text, phones, ["number_reading_candidate"])]
    word = unit.spoken.lower().replace("’", "'")
    custom = (overrides or {}).get(word)
    phones = custom or dictionary().get(word)
    if not phones:
        import cmudict
        alternatives = cmudict.dict().get(word)
        if alternatives:
            phones = [re.sub(r"\d", "", p).lower() for p in alternatives[0]]
    if not phones:
        raise ValueError(f"英文发音词典缺少「{unit.spoken}」，需要补充读音，未跳过歌词。")
    division = split_english(unit.spoken, tuple(phones)) if syllables else None
    if not division or len(division) <= 1:
        return [(unit.text, phones, (["custom_pronunciation"] if custom else []) +
                 (word_flags(unit.spoken,tuple(phones)) if syllables else []))]
    result = [(piece, phones[a:b], split_flags())
              for piece,a,b in division]
    if custom:
        for _, _, flags in result:
            flags.append("custom_pronunciation")
    position = unit.text.find(unit.spoken)
    prefix = unit.text[:position] if position >= 0 else ""
    suffix = unit.text[position+len(unit.spoken):] if position >= 0 else ""
    result[0] = (prefix + result[0][0], result[0][1], result[0][2])
    result[-1] = (result[-1][0] + suffix, result[-1][1], result[-1][2])
    return result


def mandarin_sofa_parts(unit, custom=None):
    """Mandarin phones from SOFA's own opencpop dictionary for one Han unit."""
    from .chinese import pinyin_groups
    reading = custom or unit.reading
    if not reading:
        raise ValueError(f"中文单位「{unit.spoken}」缺少拼音读音，无法使用 SOFA 中文模型。")
    groups = pinyin_groups(reading, require_phones=False)
    table = mandarin_dictionary()
    phones = []
    for group in groups:
        mapped = table.get(group["script"])
        if not mapped:
            raise ValueError(f"SOFA 中文声学词典没有拼音「{group['script']}」，请补充自定义读音。")
        phones.extend(mapped)
    flags = ["chinese_character"]
    if custom:
        flags.append("custom_pronunciation")
    if unit.spoken.isdigit():
        flags.append("number_reading_candidate")
    return [(unit.text, " ".join(group["script"] for group in groups), phones, flags)]


def mandarin_phrase_readings(units, overrides):
    """Explicit pinyin for a whole source phrase, split across its Han units."""
    from .chinese import phrase_readings
    return phrase_readings(units, (overrides or {}).get('zh', {}))


def japanese_acoustic_parts(word, fine, custom=None):
    """Keep display mapping from kana, but feed the actual UniDic pronunciation.

    E.g. どうせ is written ドウセ but pronounced ドーセ. Separate text
    units must not force an /u/ into a sung /o:/. Continuous identical vowels
    still have no independently identifiable boundary and are flagged.
    """
    written = word.reading or word.spoken
    spoken = custom or word.source_pronunciation or written
    parts = japanese_parts(Unit(word.text, word.spoken, written), fine)
    lengths = [len(kana_phones(reading)) for _, reading in parts]
    actual = kana_phones(spoken)
    flags = [flag for flag in word.flags if flag == 'number_reading_candidate']
    if custom:
        flags.append('custom_pronunciation')
    if katakana(spoken) != katakana(written):
        flags.append("spoken_pronunciation")
    if sum(lengths) != len(actual):
        # Do not invent a character/phone correspondence when lexical
        # pronunciation changes phone count. Retain the indivisible reading.
        return [(word.text, spoken, actual, flags + ["reading_mapping_indivisible"])], spoken
    result = []
    cursor = 0
    for (text, reading), count in zip(parts, lengths):
        phones = actual[cursor:cursor+count]
        local_flags = list(flags)
        if cursor and len(phones) == 1 and phones[0] in {"a", "i", "u", "e", "o"} and actual[cursor-1] == phones[0]:
            local_flags.append("shared_vowel_boundary_check")
        result.append((text, reading, phones, local_flags))
        cursor += count
    return result, spoken


def prepare_lines_llm(lines,overrides=None):
    if current_method() != 'llm':
        return
    overrides = pronunciation_overrides() if overrides is None else overrides
    from .chinese import chinese_units, is_chinese
    groups = []
    for line in lines:
        words = chinese_units(line.text) if is_chinese(line.language) else tokenize(line.text,fine=False)
        entries = [(word.spoken,tuple(english_parts(word,False,overrides['en'])[0][1]))
                   for word in words
                   if re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*",word.spoken)]
        groups.append((entries,line.text))
    prepare_llm_many(groups)


def phonemize(lines, syllables=True, fine=True):
    overrides = pronunciation_overrides()
    if syllables and fine:
        prepare_lines_llm(lines,overrides)
    from .chinese import chinese_units, is_chinese
    units, owners, phones, mapping = [], [], [("sp", "SP")], [-1]
    for line_index, line in enumerate(lines):
        chinese_line = is_chinese(line.language)
        words = chinese_units(line.text) if chinese_line else tokenize(line.text, fine=False)
        phrase_readings = mandarin_phrase_readings(words, overrides) if chinese_line else {}
        for index, word in enumerate(words):
            from .numeric_lyrics import english_numbers
            english = bool(re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", word.spoken)) or (
                not chinese_line and word.spoken.isascii() and word.spoken.isdigit() and
                (line.language == "en" or english_numbers(line.text)))
            spoken_pronunciation = ""
            if english:
                pieces = [(text, "", ph, flags) for text, ph, flags in english_parts(word, syllables and fine, overrides["en"])]
            elif chinese_line:
                custom = overrides.get("zh", {}).get(word.spoken) or phrase_readings.get(index)
                pieces = mandarin_sofa_parts(word, custom)
            else:
                custom = overrides["ja"].get(word.spoken)
                pieces, spoken_pronunciation = japanese_acoustic_parts(word, fine, custom)
            for text, reading, phonemes, flags in pieces:
                idx = len(units)
                units.append(Unit(text, kept(text), reading, flags=flags,
                                  source_pronunciation=spoken_pronunciation if not english else ""))
                owners.append(line_index)
                phones.extend(("en" if english else ("zh" if chinese_line else "ja"), ph) for ph in phonemes)
                mapping.extend([idx] * len(phonemes))
            phones.append(("sp", "SP")); mapping.append(-1)
        if "".join(unit.text for unit, owner in zip(units, owners) if owner == line_index) != line.text:
            raise ValueError(f"第 {line_index+1} 行读音映射未保留全部文字。")
    return units, owners, phones, mapping


class SingingAligner:
    def __init__(self, log=print):
        self.log = log
        self.native_models = {}

    def release(self):
        for model in self.native_models.values():
            model.cpu()
        self.native_models.clear()
        gc.collect()
        empty_cache()

    def _align_native(self, wave, lines, units, owners, sequence, unit_map, language):
        """Call upstream SOFA's complete single-language inference on one clip."""
        import torch
        from einops import repeat
        from modules.task.forced_alignment import LitForcedAlignmentTask
        if language not in self.native_models:
            self.log(f"加载 SOFA {SOFA_LABELS[language]}逐句模型…")
            checkpoint = SOFA_DIR / SOFA_CHECKPOINTS[language]
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = LitForcedAlignmentTask(**state["hyper_parameters"])
            model.load_state_dict(state["state_dict"])
            model = model.to(device()).eval()
            model.on_predict_start()
            self.native_models[language] = model
        model = self.native_models[language]
        phones = [phone for _, phone in sequence]
        missing = sorted({phone for phone in phones if phone not in model.vocab})
        if missing:
            raise ValueError(f"{language} 读音包含模型不支持的音素：{missing}")
        duration = len(wave) / 44100
        with torch.inference_mode():
            mel = model.get_melspec(torch.from_numpy(wave).to(device())).unsqueeze(0)
            mel = (mel-mel.mean()) / mel.std().clamp_min(1e-6)
            mel = repeat(mel, "B C T -> B C (T N)", N=model.melspec_config["scale_factor"])
            ph_seq, ph_bounds, word_seq, word_bounds, confidence, _, _ = model._infer_once(
                mel, duration, phones, [str(i) for i in range(len(units))], np.array(unit_map))
        if list(word_seq) != [str(i) for i in range(len(units))]:
            raise ValueError("SOFA 逐句解码未保留全部文字单位，未补造时间。")
        expected = [(phone, owner) for (_, phone), owner in zip(sequence, unit_map) if phone != "SP"]
        if list(ph_seq) != [phone for phone, _ in expected]:
            raise ValueError("SOFA 逐句解码音素路径与输入不一致。")
        phoneme_rows = []
        for (phone, owner), (start, end) in zip(expected, ph_bounds):
            start, end = max(0., float(start)), min(duration, float(end))
            phoneme_rows.append({"unit": owner, "language": language, "phoneme": phone, "start": start, "end": end})
            if end-start > .4 and phone not in VOWELS and phone not in {"a", "i", "u", "e", "o", "N", "m", "n", "ng", "s", "sh", "f", "v", "z"}:
                units[owner].flags.append("long_consonant_check")
        for unit, (start, end) in zip(units, word_bounds):
            unit.start, unit.end = round(max(0., float(start)), 3), round(min(duration, float(end)), 3)
            if unit.end-unit.start > 3:
                unit.flags.append("long_sustain_check")
            unit.flags = list(dict.fromkeys(unit.flags))
        for index, line in enumerate(lines):
            line.words = [unit for unit, owner in zip(units, owners) if owner == index]
            line.start, line.end = line.words[0].start, line.words[-1].end
        return {"engine": f"SOFA native {language} segment inference", "method": "native_monolingual",
                "confidence": float(confidence), "frame_ms": model.melspec_config["hop_length"] / (44100*model.melspec_config["scale_factor"])*1000,
                "runtime_commit": "584d6b9a57927843f85decebf4cbf03b9598125f", "phonemes": phoneme_rows}

    def align(self, samples, lines, syllables=True, cache_dir=None, fine=True, *, sample_rate=16000, native=False):
        import torch
        from scipy.signal import resample_poly
        if not ready():
            raise RuntimeError("SOFA 歌声模型尚未准备完整，请运行 setup.ps1。")
        sys.path.insert(0, str(SOFA_DIR / "runtime"))
        from modules.task.forced_alignment import LitForcedAlignmentTask
        from einops import repeat

        units, owners, sequence, unit_map = phonemize(lines, syllables, fine)
        languages = [lang for lang in SOFA_LANGUAGES if any(x[0] == lang for x in sequence)]
        vocab = {("sp", "SP"): 0}
        for phone in sequence:
            if phone not in vocab:
                vocab[phone] = len(vocab)
        factor = math.gcd(sample_rate, 44100)
        wave = np.asarray(samples, dtype=np.float32) if sample_rate == 44100 else resample_poly(np.asarray(samples, dtype=np.float32), 44100//factor, sample_rate//factor)
        if native and len(languages) == 1:
            return self._align_native(wave, lines, units, owners, sequence, unit_map, languages[0])
        frame_hop = 128
        frames = len(wave) // frame_hop
        duration = len(samples) / sample_rate
        model = None
        emissions = np.full((frames, len(vocab)), -np.inf, dtype=np.float32)
        blanks, edges = [], []
        # Acoustic cache is transcript independent; cache full softmax head logits.
        fingerprint = hashlib.sha256(str(sample_rate).encode()+np.asarray(samples, dtype=np.float32).tobytes()).hexdigest()
        try:
            for language in languages:
                checkpoint = SOFA_DIR / SOFA_CHECKPOINTS[language]
                self.log(f"SOFA {SOFA_LABELS[language]}歌声分析…")
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = LitForcedAlignmentTask(**state["hyper_parameters"])
                model.load_state_dict(state["state_dict"]); del state
                model = model.to(device()).eval()
                model.on_predict_start()
                missing = [phone[1] for phone in vocab if phone[0] == language and phone[1] not in model.vocab]
                if missing:
                    raise ValueError(f"{language} 读音表包含模型不支持的音素：{missing}")
                selected = {phone: model.vocab[phone[1]] for phone in vocab if phone[0] == language}
                native_ids = sorted(set(selected.values()) | {0})
                cache_file = Path(cache_dir) / f"sofa-{language}-{fingerprint[:16]}.npz" if cache_dir else None
                logits_all = None
                if cache_file and cache_file.exists():
                    with np.load(cache_file, allow_pickle=False) as cache:
                        if str(cache["fingerprint"]) == fingerprint and cache["logits"].shape == (frames, model.vocab["<vocab_size>"]):
                            logits_all, edge_all = cache["logits"], cache["edge"]
                if logits_all is None:
                    logits_all = np.empty((frames, model.vocab["<vocab_size>"]), dtype=np.float32)
                    edge_all = np.empty(frames, dtype=np.float32)
                    core, context = 6144, 512
                    for first in range(0, frames, core):
                        last = min(frames, first + core)
                        left, right = max(0, first - context), min(frames, last + context)
                        audio = torch.from_numpy(wave[left*frame_hop:right*frame_hop]).to(device())
                        with torch.inference_mode():
                            mel = model.get_melspec(audio).unsqueeze(0)
                            mel = (mel - mel.mean()) / mel.std().clamp_min(1e-6)
                            mel = repeat(mel, "B C T -> B C (T N)", N=4)
                            ph_logits, ph_edge, _ = model(mel.transpose(1, 2))
                        a, b = first - left, last - left
                        logits_all[first:last] = ph_logits[0, a:b].float().cpu().numpy()
                        edge_all[first:last] = ((ph_edge[0, a:b].float().sigmoid() - .1) / .8).clamp(0, 1).cpu().numpy()
                        self.log(f"SOFA {language} 已分析 {last*frame_hop/44100:.1f}/{duration:.1f} 秒")
                    if cache_file:
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(cache_file, fingerprint=np.array(fingerprint), logits=logits_all, edge=edge_all)
                sub = logits_all[:, native_ids]
                sub -= sub.max(axis=1, keepdims=True)
                sub -= np.log(np.exp(sub).sum(axis=1, keepdims=True))
                blanks.append(sub[:, native_ids.index(0)])
                for phone, native_id in selected.items():
                    emissions[:, vocab[phone]] = sub[:, native_ids.index(native_id)]
                edges.append(edge_all)
                # Keep the last lightweight module as decoder, release GPU weights.
                model.cpu(); empty_cache(); gc.collect()
            # Either expert can recognize a non-vocal frame. Requiring both
            # experts to call it silence stretched words into instrumental tails.
            emissions[:, 0] = np.maximum.reduce(blanks)
            edge = np.maximum.reduce(edges)
            edge_prob = np.clip(edge + np.concatenate(([0], edge[:-1])), 0, 1)
            self.log("按实际音素顺序解码当前片段时间轴…")
            ids = np.array([vocab[p] for p in sequence])
            with torch.inference_mode():
                indices, frame_indices, frame_confidence = model._decode(ids, emissions, edge_prob)
            edge_diff = np.concatenate((np.diff(edge), [0]))
            fractional = np.clip(edge_diff[frame_indices]/2, -.5, .5)
            times = np.concatenate((frame_indices + fractional, [frames])) * frame_hop / 44100
            spans = [[] for _ in units]
            phoneme_rows = []
            for i, sequence_index in enumerate(indices):
                owner = unit_map[sequence_index]
                if owner < 0:
                    continue
                start, end = max(0., float(times[i])), min(duration, float(times[i+1]))
                spans[owner].append((start, end))
                language, phone = sequence[sequence_index]
                if end-start > .4 and phone not in VOWELS and phone not in {"a", "i", "u", "e", "o", "N", "m", "n", "ng", "s", "sh", "f", "v", "z"}:
                    units[owner].flags.append("long_consonant_check")
                phoneme_rows.append({"unit": owner, "language": sequence[sequence_index][0],
                                     "phoneme": sequence[sequence_index][1], "start": start, "end": end})
            for unit, intervals in zip(units, spans):
                if not intervals:
                    raise ValueError(f"歌声对齐器未找到「{unit.text}」的音素路径，未伪造时间。")
                unit.start, unit.end = round(intervals[0][0], 3), round(intervals[-1][1], 3)
                if unit.end-unit.start > 3:
                    unit.flags.append("long_sustain_check")
                unit.flags = list(dict.fromkeys(unit.flags))
            for index, line in enumerate(lines):
                line.words = [u for u, owner in zip(units, owners) if owner == index]
                line.start, line.end = line.words[0].start, line.words[-1].end
            return {"engine": "SOFA bilingual acoustic forced alignment", "method": "experimental_bilingual", "frame_ms": frame_hop/44100*1000,
                    "runtime_commit": "584d6b9a57927843f85decebf4cbf03b9598125f", "phonemes": phoneme_rows}
        finally:
            if model is not None:
                model.cpu()
            gc.collect(); empty_cache()

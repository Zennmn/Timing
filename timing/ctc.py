"""Independent local multilingual CTC forced alignment; no duration repair."""
import gc
import json
import math
import re
from pathlib import Path

import numpy as np
from numba import njit

from .config import MODELS
from .device import require_gpu, empty_cache
from .schema import Unit
from .text import tokenize, kept
from .singing import japanese_acoustic_parts, pronunciation_overrides
from .numeric_lyrics import english_integer, english_numbers

CTC_DIR = MODELS / "ctc"


def ready():
    return all((CTC_DIR / name).is_file() for name in ["config.json", "model.safetensors", "preprocessor_config.json", "vocab.json", "manifest.json"])


@njit(cache=True)
def viterbi(log_probs, targets, blank=0):
    """Standard CTC state graph; every target is acoustically visited in order."""
    frames, _ = log_probs.shape
    if frames == 0:
        return np.empty(0, dtype=np.int64), -np.inf
    count = 2*len(targets)+1
    labels = np.full(count, blank, dtype=np.int64)
    labels[1::2] = targets
    previous = np.full(count, -np.inf, dtype=np.float32)
    back = np.zeros((frames, count), dtype=np.int8)
    previous[0] = log_probs[0, blank]
    if count > 1:
        previous[1] = log_probs[0, targets[0]]
    for t in range(1, frames):
        current = np.full(count, -np.inf, dtype=np.float32)
        for s in range(count):
            best, jump = previous[s], 0
            if s > 0 and previous[s-1] > best:
                best, jump = previous[s-1], 1
            if s > 1 and labels[s] != blank and labels[s] != labels[s-2] and previous[s-2] > best:
                best, jump = previous[s-2], 2
            current[s] = best + log_probs[t, labels[s]]
            back[t, s] = jump
        previous = current
    state = count-1
    if count > 1 and previous[count-2] > previous[state]:
        state = count-2
    score = previous[state]
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames-1, -1, -1):
        path[t] = state
        if t:
            state -= back[t, state]
    return path, score


def text_units(text, syllables=True, *, japanese_readings=None, language='mixed'):
    units, labels = [], []
    overrides = pronunciation_overrides()
    from .chinese import is_chinese, chinese_units, mandarin_parts, roman_label, phrase_readings
    if is_chinese(language):
        words = chinese_units(text)
        phrases = phrase_readings(words, overrides.get('zh', {}))
        for index, word in enumerate(words):
            if re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", word.spoken):
                english, roman = text_units(word.text, syllables)
                units.extend(english); labels.extend(roman)
            else:
                manual = overrides.get('zh', {}).get(word.spoken) or phrases.get(index)
                audio_reading = (japanese_readings or {}).get(word.spoken) if not manual else None
                parts = mandarin_parts(word, {word.spoken:manual or audio_reading} if manual or audio_reading else None, require_phones=False)
                for display, reading, phones, flags in parts:
                    flags = list(flags)
                    if audio_reading:
                        flags = [flag for flag in flags if flag != 'custom_pronunciation'] + ['llm_audio_reading_candidate']
                    units.append(Unit(display, kept(display), reading, flags=flags+['ctc_acoustic_alignment'], source_pronunciation=reading))
                    labels.append(roman_label(reading))
        if ''.join(unit.text for unit in units) != text:
            raise ValueError('中文 MMS 读音映射未保留原歌词。')
        return units, labels
    for word in tokenize(text, fine=False):
        if word.spoken.isascii() and word.spoken.isdigit() and english_numbers(text):
            reading = english_integer(word.spoken)
            units.append(Unit(word.text, word.spoken, reading, flags=["ctc_acoustic_alignment", "number_reading_candidate"]))
            labels.append(reading.replace(" ", ""))
        elif re.fullmatch(r"[A-Za-z]+(?:['’][A-Za-z]+)*", word.spoken):
            for unit in tokenize(word.text, fine=True, english_syllables=syllables):
                unit.flags.append("ctc_acoustic_alignment")
                units.append(unit)
                labels.append(unit.spoken.lower().replace("’", "'"))
        else:
            manual = overrides['ja'].get(word.spoken)
            audio_reading = (japanese_readings or {}).get(word.spoken) if not manual else None
            parts, pronunciation = japanese_acoustic_parts(word, True, manual or audio_reading)
            flat = [phone for _, _, phones, _ in parts for phone in phones]
            position = 0
            for display, reading, phones, flags in parts:
                if audio_reading:
                    flags = [f for f in flags if f!='custom_pronunciation']+['llm_audio_reading_candidate']
                roman = ""
                for phone in phones:
                    if phone == "cl":
                        following = flat[position+1] if position+1 < len(flat) else "t"
                        roman += "t" if following in {"ch", "ts"} else following[0].lower()
                    else:
                        roman += "n" if phone == "N" else phone.lower()
                    position += 1
                if not roman or not re.fullmatch("[a-z']+", roman):
                    raise ValueError(f"CTC 无法完整转写「{display}」的日语读音，未跳字。")
                units.append(Unit(display, kept(display), reading, flags=flags+["ctc_acoustic_alignment"], source_pronunciation=pronunciation))
                labels.append(roman)
    if "".join(unit.text for unit in units) != text:
        raise ValueError("CTC 读音映射未保留原歌词。")
    return units, labels


class CtcAligner:
    def __init__(self, log=print):
        self.log, self.model, self.extractor = log, None, None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCTC, AutoFeatureExtractor
        if not ready():
            raise RuntimeError("MMS CTC 模型未下载完整，请运行 scripts/download_ctc.py。")
        self.device = require_gpu("MMS 对齐")
        self.log("加载本地 MMS-300M 多语 CTC 对齐模型…")
        self.model = AutoModelForCTC.from_pretrained(str(CTC_DIR), local_files_only=True,
                                                    dtype=torch.float32, attn_implementation="sdpa").to(self.device).eval()
        self.extractor = AutoFeatureExtractor.from_pretrained(str(CTC_DIR), local_files_only=True)
        self.vocab = json.loads((CTC_DIR / "vocab.json").read_text(encoding="utf-8"))
        self.inverse = {value: key for key, value in self.vocab.items()}

    def release(self):
        self.model, self.extractor = None, None
        gc.collect()
        empty_cache()

    def align(self, samples, line, syllables=True, *, sample_rate=16000, prepared=None):
        import torch
        from scipy.signal import resample_poly
        self.load()
        from .audio_readings import primary_readings
        readings = primary_readings(line)
        if prepared is not None:
            units, labels = prepared
        else:
            options = {'japanese_readings':readings} if readings else {}
            if line.language in ('zh', 'zh-en'):
                options['language'] = line.language
            units, labels = text_units(line.text, syllables, **options)
        if len(units)!=len(labels) or ''.join(unit.text for unit in units)!=line.text or any(not label for label in labels):
            raise ValueError('CTC 显式分段未完整覆盖原歌词。')
        if not units:
            raise ValueError("没有可对齐文字。")
        chars = "".join(labels)
        missing = sorted(set(chars)-set(self.vocab))
        if missing:
            raise ValueError(f"CTC 字表不支持字符：{missing}")
        factor = math.gcd(sample_rate,16000)
        wave = np.asarray(samples,dtype=np.float32) if sample_rate == 16000 else resample_poly(samples,16000//factor,sample_rate//factor).astype("float32")
        if not 400 <= len(wave) <= 30*16000:
            raise ValueError("MMS 单句音频需在25毫秒到30秒之间，请检查 LRC 范围。")
        inputs = self.extractor(wave, sampling_rate=16000, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0].float()
            log_probs = logits.log_softmax(-1).cpu().numpy()
        targets = np.array([self.vocab[c] for c in chars], dtype=np.int64)
        path, score = viterbi(log_probs, targets, self.model.config.pad_token_id)
        if not np.isfinite(score):
            raise ValueError("CTC 未找到完整歌词路径，未补造时间。")
        # Convolutional receptive field has a fixed stride, rather than stretching
        # its last frame to the crop length. Bounds are model-frame boundaries.
        stride = int(np.prod(self.model.config.conv_stride))/16000
        char_spans = []
        for index in range(len(targets)):
            frames = np.flatnonzero(path == index*2+1)
            if not len(frames):
                raise ValueError("CTC 路径漏掉字符，未导出。")
            char_spans.append((int(frames[0]), int(frames[-1])+1))
        cursor = 0
        for unit, roman in zip(units, labels):
            spans = char_spans[cursor:cursor+len(roman)]
            unit.start, unit.end = round(spans[0][0]*stride,3), round(spans[-1][1]*stride,3)
            cursor += len(roman)
            if unit.end-unit.start <= stride+.001:
                unit.flags.append("ctc_single_frame_check")
        line.words = units
        line.start, line.end = units[0].start, units[-1].end
        greedy = []
        last = -1
        for index in log_probs.argmax(-1):
            if index != last and index != self.model.config.pad_token_id:
                greedy.append(self.inverse[int(index)])
            last = index
        return {"engine":"MMS-300M multilingual CTC", "frame_ms":stride*1000,
                "roman_labels":labels, "greedy_roman":"".join(greedy),
                "path_log_score":float(score),"mean_path_log_score":float(score/len(path)),
                "revision":"49402e9577b1158620820667c218cd494cc44486"}

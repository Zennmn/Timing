"""Global ordered-transcript localization; windows are acoustic, never averaged."""
import copy
import json
import math
from pathlib import Path

import numpy as np

from .ctc import text_units, viterbi
from .schema import Line

MAX_PATH_CELLS = 300_000_000
# A rejection floor for alternate CTC paths, not a calibrated confidence score.
MIN_ALTERNATIVE_POSTERIOR = .05


def align_transcript(lines, log_probs, vocab, blank, frame_seconds, duration, *, prepared=None):
    """Locate all occurrences jointly so repeated choruses stay in lyric order."""
    if (not lines or not math.isfinite(duration) or duration <= 0 or
            not math.isfinite(frame_seconds) or frame_seconds <= 0 or
            log_probs.ndim != 2 or not len(log_probs) or not np.isfinite(log_probs).all()):
        raise ValueError('整曲定位音频帧或时长无效。')
    if prepared is not None and len(prepared) != len(lines):
        raise ValueError('整曲显式分段数量与歌词行不一致。')
    supplied = prepared
    prepared, transcript = [], ''
    for index, line in enumerate(lines):
        units, labels = (copy.deepcopy(supplied[index]) if supplied is not None else
                         text_units(line.text, False, language=line.language))
        if (not units or len(units) != len(labels) or ''.join(unit.text for unit in units) != line.text
                or any(not isinstance(label, str) or not label for label in labels)):
            raise ValueError('整曲定位未能完整保留歌词和读音。')
        chars = ''.join(labels)
        prepared.append((units, labels, len(transcript), len(chars)))
        transcript += chars
    missing = set(transcript)-set(vocab)
    if missing:
        raise ValueError(f'MMS 字表不支持这些歌词读音字符：{sorted(missing)}。')
    if len(log_probs)*(2*len(transcript)+1) > MAX_PATH_CELLS:
        raise ValueError('整曲歌词定位路径过大，请提供逐句 LRC／TTML，或将歌曲分段处理。')
    targets = np.array([vocab[char] for char in transcript], dtype=np.int64)
    path, score = viterbi(log_probs, targets, blank)
    if not np.isfinite(score):
        raise ValueError('MMS 未找到覆盖完整歌词的整曲路径；请检查歌词或提供逐句时间。')
    # Aggregate visited character frames in a single pass, avoiding O(T * N)
    # repeated scans after the dynamic program.
    first = np.full(len(targets), len(path), dtype=np.int64)
    last = np.full(len(targets), -1, dtype=np.int64)
    greedy_support = np.zeros(len(targets), dtype=np.int64)
    greedy = log_probs.argmax(axis=1)
    token_scores = np.zeros(len(targets), dtype=np.float64)
    counts = np.zeros(len(targets), dtype=np.int64)
    peaks = np.full(len(targets), -np.inf, dtype=np.float64)
    for frame, state in enumerate(path):
        if state % 2:
            token = state//2
            first[token] = min(first[token], frame)
            last[token] = frame
            token_scores[token] += float(log_probs[frame, targets[token]])
            peaks[token] = max(peaks[token], float(log_probs[frame, targets[token]]))
            counts[token] += 1
            greedy_support[token] += int(greedy[frame] == targets[token])
    if np.any(last < first):
        raise ValueError('MMS 整曲定位漏掉歌词字符，未生成替代时间。')
    located, rows = [], []
    for index, (line, (units, labels, offset, size)) in enumerate(zip(lines, prepared)):
        support = int(np.count_nonzero(greedy_support[offset:offset+size]))
        peak = float(peaks[offset:offset+size].max())
        weak = support == 0
        # Forced alignment may select a credible non-argmax alternative. Keep
        # such a line for TIFA refinement, but reject effectively absent phones.
        if weak and peak < math.log(MIN_ALTERNATIVE_POSTERIOR):
            raise ValueError(f'第 {index+1} 行未找到可用的歌词发音证据，请检查歌词或提供逐句时间。')
        probe = copy.deepcopy(line)
        if weak:
            probe.flags.append('text_coarse_weak_evidence')
        cursor = offset
        for unit, label in zip(units, labels):
            right = cursor+len(label)
            unit.start = round(float(first[cursor])*frame_seconds, 3)
            unit.end = round(min(duration, float(last[right-1]+1)*frame_seconds), 3)
            if not unit.start < unit.end:
                raise ValueError(f'第 {index+1} 行粗定位边界无效。')
            cursor = right
        probe.words = units
        probe.start, probe.end = units[0].start, units[-1].end
        located.append(probe)
        rows.append({'line': index+1, 'start': probe.start, 'end': probe.end,
                     'characters': size, 'supported_characters': support,
                     'weak_evidence': weak, 'peak_target_log_probability': peak,
                     'mean_log_probability': float(token_scores[offset:offset+size].sum()/counts[offset:offset+size].sum()),
                     'units': [{'text': unit.text, 'start': unit.start, 'end': unit.end}
                               for unit in units]})
    return located, rows


def split_windows(located, duration, *, max_seconds=24, padding=.6):
    """Split long source lines only at intact, acoustically localized units."""
    if max_seconds <= 2*padding or padding < 0:
        raise ValueError('定位分段参数无效。')
    fragments, windows, source_indices = [], [], []
    for source_index, line in enumerate(located):
        chunks, current = [], []
        for unit in line.words:
            if unit.end-unit.start > max_seconds-2*padding:
                raise ValueError(f'第 {source_index+1} 行有无法安全分段的过长单元，请提供逐句时间。')
            if current and unit.end-current[0].start > max_seconds-2*padding:
                chunks.append(current)
                current = []
            current.append(unit)
        if current:
            chunks.append(current)
        for chunk in chunks:
            left = max(0, chunk[0].start-padding)
            right = min(duration, chunk[-1].end+padding)
            if not .1 <= right-left <= max_seconds+.001:
                raise ValueError('无法建立有效的声学句段，请提供逐句时间。')
            fragment = Line(''.join(unit.text for unit in chunk), left, right,
                            agent=line.agent, language=line.language,
                            flags=list(dict.fromkeys(line.flags + ['text_coarse_localization_check'])))
            fragments.append(fragment)
            source_indices.append(source_index)
            windows.append({'start': chunk[0].start, 'end': chunk[-1].end,
                            'source_line': source_index+1, 'range_start': left, 'range_end': right})
    return fragments, windows, source_indices


def locate(samples, rate, lines, engine, output_dir, log=print):
    probs, info = engine.emissions(samples, sample_rate=rate)
    located, rows = align_transcript(lines, probs, engine.vocab, engine.model.config.pad_token_id,
                                     info['frame_ms']/1000, len(samples)/rate)
    fragments, windows, owners = split_windows(located, len(samples)/rate)
    report = {**info, 'method': 'chunked_mms_global_ctc', 'lines': rows,
              'alternative_posterior_floor': MIN_ALTERNATIVE_POSTERIOR,
              'windows': windows, 'source_indices': owners,
              'note': '整曲有序歌词的声学粗定位，仅用于切片；不是人工逐句时间或 TIFA 精细边界。'}
    Path(output_dir, 'text_localization.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    log(f'整曲定位完成：{len(lines)} 行，{len(fragments)} 个 TIFA 句段。')
    return fragments, windows, owners, report

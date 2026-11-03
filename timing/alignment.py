"""Model-specific alignment routes; shared audio and request planning live elsewhere."""
import copy
import json
from .audio import decode, slice_audio
from .ctc import CtcAligner
from . import tifa


def align_llm_direct(lines, tracks, out_dir, report, log, *, syllables=True, direct_config=None):
    from .llm_direct import DirectSession
    session = DirectSession(direct_config, log, out_dir,
                            prompt_profile=report.get('llm_prompt_profile','fine'))
    source = (report.get('background_source') if any(line.role == 'bg' for line in lines) else
              'separated_lead' if report.get('audio_mode') == 'lead' else tracks.source)
    source = 'original_mix' if source == 'original' else source
    session.prepare(lines, tracks.working, tracks.rate, tracks.background, audio_source=source)
    report['llm_direct'] = session.report()
    align_llm_records(lines, tracks, out_dir, report, log, session.records)


def align_llm_import(lines, tracks, out_dir, report, log, *, syllables=True, direct_config=None):
    audit = report['llm_import']
    log(f'使用网页 LLM 导入结果：{len(audit["lines"])} 行；本次不请求任何 LLM。')
    align_llm_records(lines, tracks, out_dir, report, log, audit['lines'])


def align_llm_text(lines, tracks, out_dir, report, log, *, syllables=True, direct_config=None):
    from .llm_text import TextReadingSession
    session = TextReadingSession(direct_config, log, out_dir)
    session.prepare(lines)
    report['llm_text'] = session.report()
    align_llm_records(lines, tracks, out_dir, report, log, session.records)


def align_llm_records(lines, tracks, out_dir, report, log, records):
    from .llm_direct import detected_language, align as direct_align, locate as direct_locate
    working, sr = tracks.working, tracks.rate
    for line, record in zip(lines, records):
        line.language = detected_language(record['parts'])
    report['language'] = detected_language([part for record in records for part in record['parts']])
    report['alignment_backend'] = 'tifa'
    background_working, background_source = tracks.background, tracks.background_path
    singing_source = tracks.main_path
    trusted = report.get('trust_line_timestamps', False)
    if trusted:
        log('信任输入逐句时间轴：跳过 MMS 粗定位，直接使用逐句范围。')
        coarse = [{'start':line.start, 'end':line.end, 'method':'trusted_line_timestamps'} for line in lines]
    else:
        coarse = direct_locate(working, sr, lines, records, log, background_working)
    singing_audio, singing_rate = decode(singing_source, out_dir / 'singing.wav', rate=44100, floating=True)
    background_audio = None
    if any(line.role == 'bg' for line in lines):
        background_audio, _ = decode(background_source, out_dir / 'background_alignment.wav', rate=44100, floating=True)
    report['english_splitter'] = next((mode for mode in ('llm_text', 'llm_import') if mode in report), 'llm_direct')
    report['coarse_localization'] = coarse
    extra = {'trust_line_timestamps':True} if trusted else {}
    report['tifa'] = direct_align(singing_audio, singing_rate, lines, out_dir, records, log, background_audio, coarse, **extra)
    report['timestamp_grid_ms'] = 10


def align_tifa(lines, tracks, out_dir, report, log, *, syllables=True, direct_config=None):
    if any(line.role == 'main' and line.start is None for line in lines):
        return align_text_tifa(lines, tracks, out_dir, report, log, syllables=syllables)
    working, sr = tracks.working, tracks.rate
    singing_source = tracks.main_path
    background_working, background_source = tracks.background, tracks.background_path
    singer = None
    try:
        if not all(line.start is not None and line.end is not None for line in lines):
            raise ValueError("背景歌词需提供逐句时间范围。")
        trusted = report.get('trust_line_timestamps', False)
        if trusted:
            log('信任输入逐句时间轴：跳过 MMS 粗定位，直接使用逐句范围。')
            coarse_rows = [{'start':line.start, 'end':line.end, 'method':'trusted_line_timestamps'} for line in lines]
        else:
            coarse_rows = []
            singer = CtcAligner(log)
            try:
                for index,line in enumerate(lines):
                    left,right = line.start,line.end
                    selected = background_working if line.role == "bg" else working
                    # Keep the supplied anchors: CTC only suggests where to
                    # search for a following pause, never a fine-unit duration.
                    probe = copy.deepcopy(line)
                    log(f"定位句段 {index+1}/{len(lines)}：{line.text[:32]}")
                    singer.align(slice_audio(selected,left,right),probe,False)
                    coarse_rows.append({"start":left+probe.start,"end":left+probe.end})
            finally:
                singer.release()
                singer = None
        singing_audio,singing_rate = decode(singing_source,out_dir/"singing.wav",rate=44100,floating=True)
        background_audio = None
        if any(line.role == "bg" for line in lines):
            background_audio,_ = decode(background_source,out_dir/"background_alignment.wav",rate=44100,floating=True)
        report["coarse_localization"] = coarse_rows
        extra = {'trust_line_timestamps':True} if trusted else {}
        report["tifa"] = tifa.align(singing_audio,singing_rate,lines,out_dir,coarse_rows,log,
                                    syllables=syllables,background=background_audio, **extra)
        report["timestamp_grid_ms"] = 10
        return

    finally:
        if singer is not None and hasattr(singer, "release"):
            singer.release()


def align_ctc(lines, tracks, out_dir, report, log, *, syllables=True, direct_config=None):
    if any(line.role == 'main' and line.start is None for line in lines):
        return align_text_ctc(lines, tracks, out_dir, report, log, syllables=syllables)
    working, sr = tracks.working, tracks.rate
    background_working = tracks.background
    singer = None
    try:
        if not all(line.start is not None and line.end is not None for line in lines):
            raise ValueError("MMS 重叠背景歌词需要完整的逐句时间范围。")
        singer = CtcAligner(log)
        report['alignment_backend'] = 'mms'
        report["timestamp_grid_ms"] = 20
        report["ctc_note"] = "CTC 非空白状态定位发音证据，不等于完整持续音的起止；20ms 网格不代表20ms准确度。"
        for index, line in enumerate(lines):
            left, right = line.start, line.end
            log(f"MMS 逐句对齐 {index+1}/{len(lines)} [{left:.2f}–{right:.2f}]：{line.text[:38]}")
            selected = background_working if line.role == "bg" else working
            info = align_ctc_range(singer, selected, sr, line, syllables)
            report.setdefault("anchored_alignment", []).append({"line": index+1, "range_start": left, "range_end": right, **info})
        return

    finally:
        if singer is not None and hasattr(singer, "release"):
            singer.release()


def align_ctc_range(engine, samples, rate, line, syllables=True, *, prepared=None):
    """Return actual CTC edges in the song clock, preserving the supplied range."""
    from .ctc import text_units
    from .text_localization import align_transcript
    left, right = line.start, line.end
    if left is None or right is None or right <= left:
        raise ValueError('MMS 句段需要有效的开始和结束时间。')
    wave = samples[round(left*rate):round(right*rate)]
    probe = copy.deepcopy(line)
    if len(wave)/rate > 30:
        prepared = prepared if prepared is not None else text_units(line.text, syllables, language=line.language)
        probs, info = engine.emissions(wave, sample_rate=rate)
        located, rows = align_transcript([probe], probs, engine.vocab, engine.model.config.pad_token_id,
                                         info['frame_ms']/1000, len(wave)/rate, prepared=[prepared])
        probe = located[0]
        info = {**info, 'method': 'chunked_mms_ctc', 'evidence': rows[0]}
    else:
        info = engine.align(wave, probe, syllables, sample_rate=rate, prepared=prepared)
    for unit in probe.words:
        unit.start = round(unit.start+left, 3)
        unit.end = round(unit.end+left, 3)
        if unit.start < left-.001 or unit.end > right+.001 or unit.end <= unit.start:
            raise ValueError('MMS 模型边界越出逐句范围或无效，未导出。')
    if not probe.words or ''.join(unit.text for unit in probe.words) != line.text:
        raise ValueError('MMS 对齐结果未完整保留原歌词。')
    line.words = probe.words
    line.flags = list(dict.fromkeys(line.flags + probe.flags))
    line.start, line.end = line.words[0].start, line.words[-1].end
    return info


def align_text_ctc(lines, tracks, out_dir, report, log, *, syllables=True):
    """Align the complete ordered transcript on the native MMS frame grid."""
    from .ctc import text_units
    from .text_localization import align_transcript
    mains = [line for line in lines if line.role == 'main']
    prepared = [text_units(line.text, syllables, language=line.language) for line in mains]
    engine = CtcAligner(log)
    report.update(alignment_backend='mms', timestamp_grid_ms=20,
                  input_timing='automatically_aligned_text',
                  ctc_note='MMS 使用罗马化歌词和真实声学路径；20ms 网格不代表20ms准确度，持续音和收音需听音校对。')
    try:
        probs, info = engine.emissions(tracks.working, sample_rate=tracks.rate)
        located, rows = align_transcript(mains, probs, engine.vocab, engine.model.config.pad_token_id,
                                         info['frame_ms']/1000, len(tracks.working)/tracks.rate,
                                         prepared=prepared)
        for source, aligned in zip(mains, located):
            source.words = aligned.words
            source.start, source.end = aligned.start, aligned.end
            source.flags = list(dict.fromkeys(aligned.flags + ['text_ctc_global_alignment_check']))
        indices = {id(line): index+1 for index, line in enumerate(lines)}
        for row, line in zip(rows, mains):
            row['line'] = indices[id(line)]
        localization = {**info, 'method': 'chunked_mms_global_ctc_alignment', 'lines': rows,
                        'note': '纯文本整曲有序声学对齐；重复歌词按输入顺序保留，非平均分配时间，需听音校对。'}
        report['text_localization'] = localization
        (out_dir / 'text_localization.json').write_text(json.dumps(localization, ensure_ascii=False, indent=2), encoding='utf-8')
        for line in (line for line in lines if line.role == 'bg'):
            left, right = line.start, line.end
            info = align_ctc_range(engine, tracks.background, tracks.rate, line, syllables)
            report.setdefault('anchored_alignment', []).append({'line': indices[id(line)],
                                                               'range_start': left, 'range_end': right, **info})
    finally:
        engine.release()


def align_text_tifa(lines, tracks, out_dir, report, log, *, syllables=True):
    from .text_localization import locate
    mains = [line for line in lines if line.role == 'main']
    if len({line.agent for line in mains}) > 1:
        raise ValueError('多位主唱的纯文本歌词可能重叠，请分别处理，或提供逐句 LRC／TTML 时间。')
    engine = CtcAligner(log)
    try:
        fragments, windows, owners, localization = locate(tracks.working, tracks.rate, mains,
                                                         engine, out_dir, log)
    finally:
        engine.release()
    report['text_localization'] = localization
    report['input_timing'] = 'automatically_localized_text'
    audio, rate = decode(tracks.main_path, out_dir / 'singing.wav', rate=44100, floating=True)
    fine = tifa.align(audio, rate, fragments, out_dir, windows, log, syllables=syllables)
    for fragment, window, check in zip(fragments, windows, fine['checks']):
        if 'text_coarse_review_placeholder' in fragment.flags:
            # A failed fine pass keeps only the observed coarse edge union,
            # never the padding used to search for TIFA boundaries.
            fragment.start, fragment.end = window['start'], window['end']
            fragment.words[0].start, fragment.words[0].end = fragment.start, fragment.end
            check['range_start'], check['range_end'] = fragment.start, fragment.end
    for source_index, line in enumerate(mains):
        pieces = [fragment for fragment, owner in zip(fragments, owners) if owner == source_index]
        words = [word for piece in pieces for word in piece.words]
        if ''.join(word.text for word in words) != line.text or not words:
            raise ValueError('TIFA 纯文本分段未完整还原原歌词，未导出。')
        timed = [word for word in words if word.start is not None]
        if any(b.start < a.end-.001 for a,b in zip(timed,timed[1:])):
            raise ValueError(f'第 {source_index+1} 行分段边界冲突，请提供逐句时间后重试。')
        line.words = words
        line.start, line.end = pieces[0].start, pieces[-1].end
        line.flags = list(dict.fromkeys(line.flags + [flag for piece in pieces for flag in piece.flags]))
    line_indices = {id(line): index+1 for index, line in enumerate(lines)}
    for check, owner in zip(fine['checks'], owners):
        check['segment'] = check['line']
        check['line'] = line_indices[id(mains[owner])]
    # complete_results wrote fragment ranges before the coarse-only fallbacks
    # were trimmed above. Persist the final exported provenance as well.
    completion_path = out_dir / 'tifa' / 'completion.json'
    if completion_path.exists():
        completion = json.loads(completion_path.read_text(encoding='utf-8'))
        completion['checks'] = fine['checks']
        completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding='utf-8')
    report['tifa'] = fine
    backgrounds = [line for line in lines if line.role == 'bg']
    if backgrounds:
        background_dir = out_dir / 'background'
        background_dir.mkdir(exist_ok=True)
        background_report = {}
        align_tifa(backgrounds, tracks, background_dir, background_report, log, syllables=syllables)
        for check, line in zip(background_report['tifa']['checks'], backgrounds):
            check['line'] = line_indices[id(line)]
        report['background_alignment'] = background_report
    report['timestamp_grid_ms'] = 10

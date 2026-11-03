"""Alignment reports and TTML export; independent of model orchestration."""
import json
from .schema import serialize
from .ttml import export


def finish(lines, duration, language, title, out_dir, report, log):
    from .llm_splitter import current_session
    if current_session() is not None:
        report['llm_splitter'] = current_session().report()
        stats = report['llm_splitter']
        log(f"LLM 请求合计：{stats['requests']} 次（首发 {stats['initial_requests']} 次，修正 {stats['repair_requests']} 次），{stats['fallback_words']} 个词保留整词。")
    review, reading_notices = [], []
    for index, line in enumerate(lines):
        for word in line.words:
            if word.start is not None and word.end is not None and word.end-word.start < .04 and "short_unit_check" not in word.flags:
                word.flags.append("short_unit_check")
        reasons = set(line.flags + [flag for word in line.words for flag in word.flags])
        suspicious = reasons & {"short_unit_check", "long_sustain_check", "long_consonant_check", "model_timestamp_repaired", "shared_vowel_boundary_check", "ctc_single_frame_check", "tifa_crop_edge_check", "tifa_context_retry_check", "lrc_review_placeholder", "tifa_partial_alignment", "tifa_unresolved_unit", 'text_coarse_localization_check', 'text_coarse_review_placeholder', 'text_coarse_weak_evidence', 'text_ctc_global_alignment_check'}
        notices = reasons & {"number_reading_candidate", "english_llm_candidate", "english_llm_fallback", 'llm_direct_candidate', 'llm_direct_unresolved', 'llm_direct_coarse_fallback', 'llm_text_reading_candidate', 'llm_text_dictionary_fallback', 'llm_text_unresolved'}
        if notices:
            reading_notices.append({"line":index+1, "text":line.text, "reasons":sorted(notices)})
        if suspicious:
            review.append({"line": index+1, "text": line.text, "reasons": sorted(suspicious)})
    report["quality"] = {"status": "needs_review" if review or reading_notices else "unreviewed", "issues": review,
                         "reading_notices": reading_notices,
                         "note": "请对照歌曲检查时间；过短或过长的片段会列为检查提示。"}
    report["lines"] = serialize(lines)
    labels = {
        'short_unit_check':'有片段过短', 'long_sustain_check':'有片段持续较长',
        'long_consonant_check':'有辅音持续较长', 'model_timestamp_repaired':'模型修正过时间',
        'shared_vowel_boundary_check':'相邻片段共用元音', 'ctc_single_frame_check':'有片段只有一帧',
        'tifa_crop_edge_check':'时间靠近句段边缘', 'tifa_context_retry_check':'更换句段后重试过',
        'lrc_review_placeholder':'只有整句时间', 'tifa_partial_alignment':'部分字还缺时间',
        'tifa_unresolved_unit':'有字需要补时间', 'number_reading_candidate':'数字读音需检查',
        'english_llm_candidate':'使用接口拆分音节', 'english_llm_fallback':'部分词保留整词',
        'llm_direct_candidate':'使用接口提供的拆分',
        'llm_direct_unresolved':'接口拆分未通过', 'llm_direct_coarse_fallback':'句段定位使用原歌词',
        'llm_text_reading_candidate':'使用 LLM 文本读音（未听音）',
        'llm_text_dictionary_fallback':'LLM 文本读音未通过，使用本地词典',
        'llm_text_unresolved':'文本读音和本地词典均未通过',
        'text_coarse_localization_check':'句段来自整曲自动定位，需检查是否对应正确演唱位置',
        'text_coarse_review_placeholder':'未取得 TIFA 逐字边界，保留模型粗定位范围',
        'text_coarse_weak_evidence':'整曲定位的发音证据较弱，需重点核对句段',
        'text_ctc_global_alignment_check':'使用整曲 MMS 声学对齐，需核对重复演唱和持续音边界',
    }
    if report.get('aligner') == 'llm_import':
        labels['llm_direct_candidate'] = '使用网页 LLM 导入的拆分与读音'
        labels['llm_direct_unresolved'] = '网页 LLM 未提供本句拆分'
    if report.get('aligner') == 'llm_text':
        labels['llm_direct_candidate'] = '使用已校验的文本读音与拆分'
        labels['llm_direct_unresolved'] = '未取得有效文本读音与拆分'
    report['warnings'] = []
    for index in sorted({row['line'] for row in review + reading_notices}):
        reasons = set().union(*(set(row['reasons']) for row in review + reading_notices if row['line'] == index))
        report['warnings'].append(f"第 {index} 行：" + '；'.join(labels.get(reason, reason) for reason in sorted(reasons)) + '。')
    manual = [i+1 for i,line in enumerate(lines) if {'lrc_review_placeholder', 'text_coarse_review_placeholder'} & set(line.flags)]
    report["manual_timing_lines"] = manual
    partial = [{'line':i+1,'units':[{'index':j+1,'text':w.text}
                for j,w in enumerate(line.words) if 'tifa_unresolved_unit' in w.flags]}
               for i,line in enumerate(lines) if any('tifa_unresolved_unit' in w.flags for w in line.words)]
    report['partial_timing_lines'] = partial
    if partial:
        from .ttml import export_times
        report['export_placeholders'] = [
            {'line':i+1,'index':j+1,'text':w.text,'begin':export_times(line,j)[0],
             'end':export_times(line,j)[1],'kind':'zero_duration_edit_marker',
             'note':'仅用于兼容导入的待补轴占位，不是模型测得的字时间；JSON 原始时间仍为 null。'}
            for i,line in enumerate(lines) for j,w in enumerate(line.words) if 'tifa_unresolved_unit' in w.flags]
    partial_note = ''.join(f"第 {r['line']} 行的「{'、'.join(w['text'].strip() for w in r['units'])}」还缺时间，已放入零时长标记，需手动补齐；其他字的时间已保留。" for r in partial)
    if partial_note:
        report['warnings'].insert(0,partial_note)
    sentence_only = bool(lines) and len(manual) == len(lines)
    llm_error = (report.get('llm_direct') or report.get('llm_text') or {}).get('error')
    failure_note = ('接口验证失败（HTTP 401），请检查 API Key 和接口地址；刷新页面后需重新填写密钥。' if llm_error == 'HTTP 401'
                    else f'LLM 接口请求失败：{llm_error}。' if llm_error else '')
    report['sentence_only'] = sentence_only
    if manual:
        source = '模型粗定位范围或已有逐句范围' if report.get('text_localization') else '整句 LRC 时间'
        report["warnings"].insert(0, f"第 {', '.join(map(str,manual))} 行未获得完整逐字轴，仅保留{source}供人工补轴；其他有效单位已保留。")
    fallback = (report.get('llm_splitter') or {}).get('fallback_words',0)
    data = export(lines, duration, language, title)
    (out_dir / "lyrics.ttml").write_bytes(data)
    (out_dir / "alignment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    unresolved_count = sum(len(r['units']) for r in partial)
    log(f"完成：{len(lines)} 行，{sum(len(line.words) for line in lines)} 个单位。" +
        (f"其中 {unresolved_count} 个单位未获得时间，需人工补轴。" if unresolved_count else ""))
    return {"status": "complete", "source": report["source"], "aligner": report.get("aligner"), "lead_separated": report.get("lead_separation") is not None,
            "english_splitter": report.get("english_splitter"),
            "llm_fallback_words": fallback,
            "residual_includes_accompaniment":bool((report.get("lead_separation") or {}).get("residual_includes_accompaniment")), "lines": len(lines),
            "units": sum(len(line.words) for line in lines), "warnings": len(report["warnings"]), "report": report["warnings"],
            "review_lines": len(review),
            "reading_review_lines": len(reading_notices),
            "manual_timing_lines": manual,
            "partial_timing_lines": partial,
            "sentence_only": sentence_only,
            "llm_error": llm_error,
            "review_message": failure_note + (f"全曲 {len(lines)} 行只生成了整句时间，逐字时间需要手动补齐。" if sentence_only else
                              (f"{fallback} 个词未能拆成音节，已保留整词时间。" if fallback else "") + partial_note + (f"第 {', '.join(map(str,manual))} 行只保留整句时间，需要手动补上逐字时间。" if manual else "") +
                              (f"{len(reading_notices)} 行的读音或音节拆分需要检查。" if reading_notices else "") +
                              (f"{len(review)} 行的时间需要重点检查。" if review else "")) + '请对照歌曲检查后使用。'}

"""LRC-constrained joint Japanese/English singing alignment using native TIFA."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import MODELS, ROOT
from .schema import Unit
from .text import kept, canonical_apostrophes
from .english_syllables import split_english, split_flags, prepare_llm, prepare_llm_many, current_method, word_flags

FOLDER = MODELS / "tifa"


def ready():
    return all((FOLDER / p).is_file() for p in ("model.pt", "config.yaml", "vocabulary.json", "manifest.json",
                                               "runtime/inference/api.py", "deps/g2pflow/__init__.py",
                                               "dictionaries/japanese_dict_full.txt", "dictionaries/ds_cmudict-07b.txt",
                                               "dictionaries/ds-zh-pinyin-lite.txt"))


def restore_display(original, labels):
    """Locate every pronounced character, retaining all original punctuation/case."""
    normalized, positions = "", []
    for i, ch in enumerate(original):
        chars = canonical_apostrophes(kept(ch)).casefold()
        normalized += chars
        positions.extend([i]*len(chars))
    whole = canonical_apostrophes(kept(original)).casefold()
    if normalized != whole:
        # NFKC can compose multiple source characters into one display glyph.
        # Track the source start of the rewritten suffix, retaining the entire
        # original grapheme when native PFML returns its composed spelling.
        normalized, positions = '', []
        for i in range(len(original)):
            current = canonical_apostrophes(kept(original[:i+1])).casefold()
            shared = 0
            for left, right in zip(normalized, current):
                if left != right:
                    break
                shared += 1
            owner = positions[shared] if shared < len(positions) else i
            positions = positions[:shared] + [owner]*(len(current)-shared)
            normalized = current
    expected = "".join(canonical_apostrophes(kept(label)).casefold() for label in labels)
    if normalized != expected or not expected:
        raise ValueError("TIFA 原文映射不完整，未跳字或替换歌词。")
    starts, cursor = [], 0
    for label in labels:
        size = len(kept(label).casefold())
        if not size:
            raise ValueError("TIFA 空文字映射。")
        starts.append(positions[cursor])
        cursor += size
    if any(b <= a for a,b in zip(starts, starts[1:])):
        raise ValueError('TIFA 文字分段不可切开同一个组合字符。')
    starts[0] = 0
    return [original[a:b] for a,b in zip(starts, starts[1:]+[len(original)])]


def quiet_crop_end(samples, rate, coarse_end, limit, coarse_start):
    """Find a pause after lexical evidence; this selects context, never a word offset.

    CTC peaks can end inside a sustained vowel. Search AFTER those peaks and
    include 180ms of low energy plus context; do not clip at the CTC end.
    """
    if limit-coarse_end < 1.0:
        return limit, None
    step = max(1, round(rate*.01))
    peak_wave = samples[max(0,round(coarse_start*rate)):min(len(samples),round(coarse_end*rate))]
    if not len(peak_wave):
        return limit, None
    blocks = peak_wave[:len(peak_wave)//step*step].reshape(-1,step)
    if not len(blocks):
        return limit, None
    peak = np.sqrt(np.mean(blocks.astype("float64")**2,axis=1)).max()
    threshold = max(peak*10**(-23/20), 10**(-65/20))
    first = max(0,round((coarse_end+.08)*rate))
    last = min(len(samples),round(limit*rate))
    run = 0
    for pos in range(first,last-step+1,step):
        rms = np.sqrt(np.mean(samples[pos:pos+step].astype("float64")**2))
        run = run+1 if rms < threshold else 0
        if run >= 18:
            end = min(limit,(pos+step)/rate+.15)
            return end, {"method":"lead_energy_pause_after_ctc", "threshold_db":float(20*np.log10(threshold)),
                         "coarse_end":coarse_end, "crop_end":end, "note":"句段上下文，不是字尾标注。"}
    return limit, None


def map_result(line, row, offset, syllables=True, *, partial=False):
    spans = np.asarray(row["spans"], dtype=float)
    if spans.shape != (len(row["phones"]), 2):
        raise ValueError("TIFA 音素边界数量无效。")
    invalid = set(row.get("skipped", [])) | {i for i,(a,b) in enumerate(spans)
                                              if not np.isfinite([a,b]).all() or b <= a}
    if invalid and not partial:
        raise ValueError("TIFA 存在漏音素／无效边界，未补造时间。")
    displays = restore_display(line.text, row["texts"])
    if syllables:
        prepare_llm([(kept(text),tuple(p.split('/',1)[-1] for p in choice['phonemes']))
                     for text,choice,plan in zip(displays,row['selected'],row['display_plans']) if plan is None],line.text)
    units = []
    for word_id, (text, candidate, plan) in enumerate(zip(displays,row["selected"],row["display_plans"]),1):
        indices = [i for i, owner in enumerate(row["owners"]) if owner == word_id]
        if len(indices) != len(candidate["phonemes"]):
            raise ValueError("TIFA 音素与原文计数不一致。")
        if plan is None:
            phones = [p.split("/",1)[-1] for p in candidate["phonemes"]]
            division = split_english(kept(text), tuple(phones)) if syllables else None
            if division and len(division) > 1:
                parts = restore_display(text,[p for p,_,_ in division])
                segments = [(part,indices[a:b],split_flags())
                            for part,(_,a,b) in zip(parts,division)]
            else:
                segments = [(text,indices,["english_word_level"] + (word_flags(kept(text),tuple(phones)) if syllables else []))]
        else:
            parts = restore_display(text,[p["text"] for p in plan])
            group_ids = list(dict.fromkeys(row["groups"][i] for i in indices))
            segments, cursor = [], 0
            for part, item in zip(parts,plan):
                count = item["group_count"]
                owners = set(group_ids[cursor:cursor+count]);cursor += count
                segments.append((part,[i for i in indices if row["groups"][i] in owners],item["flags"]))
            if cursor != len(group_ids):
                raise ValueError("TIFA 日语读音分组映射不一致。")
        for part, owned, flags in segments:
            if not owned:
                raise ValueError("TIFA 原文字段没有音素，未平均分时间。")
            start,end = spans[owned[0],0]+offset, spans[owned[-1],1]+offset
            if partial and (invalid.intersection(owned) or not np.isfinite([start,end]).all()
                            or round(float(start),3) >= round(float(end),3)):
                units.append(Unit(part,kept(part),flags=list(flags)+["tifa_unresolved_unit"]))
                continue
            local_flags = list(flags)+["tifa_native_alignment"]
            if candidate.get("custom_pronunciation"):
                local_flags.append("custom_pronunciation")
            if candidate.get('llm_direct_candidate') and 'llm_direct_candidate' not in local_flags:
                local_flags.append('llm_direct_candidate')
            if any(c.isdecimal() for c in kept(part)):
                local_flags.append("number_reading_candidate")
            if end-start > 2.5:
                local_flags.append("long_sustain_check")
            reading = candidate.get('llm_direct_reading','')
            pronunciation = ' '.join(reading) if isinstance(reading,list) else reading
            units.append(Unit(part,kept(part), start=round(float(start),3),end=round(float(end),3),flags=local_flags,
                              source_pronunciation=pronunciation))
    if "".join(u.text for u in units) != line.text:
        raise ValueError("TIFA 文字未完整保留。")
    timed = [u for u in units if u.start is not None]
    if not timed:
        raise ValueError("TIFA 本句没有可保留的有效单位。")
    line.words = units
    line.start = units[0].start if units[0].start is not None else min(line.start,timed[0].start)
    line.end = units[-1].end if units[-1].end is not None else max(line.end,timed[-1].end)
    if len(timed) != len(units):
        line.flags.append("tifa_partial_alignment")


def target_result(row, start, end):
    """Select complete semantic words of a line from a contextual alignment."""
    cursor, chosen = 0, []
    for i,text in enumerate(row["texts"]):
        next_cursor=cursor+len(kept(text).casefold())
        if cursor < end and next_cursor > start:
            if cursor < start or next_cursor > end:
                raise ValueError("TIFA 跨句分词无法无损拆开，未分摊时间。")
            chosen.append(i)
        cursor=next_cursor
    remap={old+1:new+1 for new,old in enumerate(chosen)}
    indices=[i for i,owner in enumerate(row["owners"]) if owner in remap]
    result={**row}
    for key in ("texts","selected","display_plans"):
        result[key]=[row[key][i] for i in chosen]
    for key in ("spans","phones","groups"):
        result[key]=[row[key][i] for i in indices]
    result["owners"]=[remap[row["owners"][i]] for i in indices]
    result["skipped"]=[i for i,(a,b) in enumerate(result["spans"]) if b<=a]
    return result


def run_worker(folder, clips, log=print, pronunciations=None):
    request = folder / "request.json"
    active = set()
    for clip in clips:
        if 'llm_direct_parts' in clip:
            from .llm_direct import acoustic_languages
            active.update(acoustic_languages(clip['llm_direct_parts']))
        active.update(clip.get('languages', []))
    payload = {"clips": clips}
    if active:
        payload['languages'] = [lang for lang in ('ja','zh','en') if lang in active]
    if pronunciations is not None:
        payload["pronunciations"] = str(pronunciations)
    request.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    env = os.environ.copy(); env["PYTHONUTF8"] = "1"; env["HF_HUB_OFFLINE"] = "1"
    with (folder / "worker.log").open("w", encoding="utf-8") as error:
        proc = subprocess.Popen([sys.executable, "-X", "utf8", str(ROOT / "timing/tifa_worker.py"),
                                 str(FOLDER), str(request)], stdout=subprocess.PIPE, stderr=error,
                                text=True, encoding="utf-8", env=env)
        for message in proc.stdout:
            if message.startswith("TIFA "): log(message.strip())
        if proc.wait():
            detail = (folder / "worker.log").read_text(encoding="utf-8")[-1200:]
            raise RuntimeError("TIFA 运行失败：" + detail)
    return json.loads((folder / "raw.json").read_text(encoding="utf-8"))


def map_clip(line, row, clip, syllables, *, partial=False):
    return _map_clip(line,row,clip,syllables,partial=partial)


def _map_clip(line, row, clip, syllables, *, partial=False):
    """Validate rounded display boundaries too, before accepting a prediction."""
    probe = copy.deepcopy(line)
    row = target_result(row, clip["target_start"], clip["target_end"])
    if syllables and current_method() == 'llm':
        map_result(probe,row,clip['start'],False,partial=partial)
        validate_clip(probe,line,clip)
    map_result(probe, row, clip["start"], syllables,partial=partial)
    validate_clip(probe,line,clip)
    return probe, row


def validate_clip(probe,line,clip):
    if line.end is not None and probe.start >= line.end:
        raise ValueError("TIFA 将本句起音对齐到下一句之后，需要局部重试。")
    last = clip["start"] - .001
    for word in probe.words:
        if "tifa_unresolved_unit" in word.flags and word.start is None and word.end is None:
            continue
        if not (np.isfinite([word.start, word.end]).all() and
                last <= word.start < word.end <= clip["end"] + .001):
            raise ValueError("TIFA 字边界无效或次序冲突。")
        last = word.end


def prepare_clips_llm(lines,clips,rows):
    """Prefetch all valid target pronunciations, including a batch of retries."""
    if current_method() != 'llm':
        return
    groups = []
    for line,clip in zip(lines,clips):
        if clip['id'] not in rows:
            continue
        try:
            # Validate native paths/anchors using whole English words first;
            # no network calls and no fabricated fine boundaries in this pass.
            _,row = map_clip(line,rows[clip['id']],clip,False)
            displays = restore_display(line.text,row['texts'])
            entries = [(kept(text),tuple(p.split('/',1)[-1] for p in choice['phonemes']))
                       for text,choice,plan in zip(displays,row['selected'],row['display_plans']) if plan is None]
            groups.append((entries,line.text))
        except ValueError:
            continue
    prepare_llm_many(groups)


def complete_results(samples, rate, lines, folder, clips, raw, log=print, *, syllables=True, background=None,
                      pronunciations=None):
    """Retry failed targets only; unresolved lyrics remain explicit LRC blocks."""
    rows = {r["id"]: r for r in raw["results"]}
    if len(rows) != len(lines) or len(clips) != len(lines):
        raise ValueError("TIFA 未返回全部歌词行，未导出。")
    anchors = copy.deepcopy(lines)
    if syllables:
        prepare_clips_llm(anchors,clips,rows)
    selected, failures = {}, {}
    for i, (line, clip) in enumerate(zip(anchors, clips)):
        try:
            probe, row = map_clip(line, rows[clip["id"]], clip, syllables)
            selected[i] = (probe, row, clip, "initial")
        except ValueError as exc:
            failures[i] = str(exc)
    previous = {}
    conflicts = set()
    for i, (probe, _, _, _) in selected.items():
        if probe.role != "main":
            continue
        prior = previous.get(probe.agent)
        if prior is not None and probe.start < selected[prior][0].start:
            conflicts.update((prior, i))
        previous[probe.agent] = i
    for i in conflicts:
        selected.pop(i)
        failures[i] = "TIFA 相邻歌词的起始时间次序冲突。"
    retries, attempts, retry_rows = [], [], {}
    retry_folder = folder / "retry"
    if failures:
        retry_folder.mkdir(exist_ok=True)
        for i in failures:
            line = anchors[i]
            log(f"第 {i+1} 行「{line.text}」精细对齐失败，重试局部句段…")
            wave = background if line.role == "bg" and background is not None else samples
            for attempt, pad in enumerate((.12, .35, 0), 1):
                a = round(max(0, line.start-pad)*rate)
                b = round(min(len(wave)/rate, line.end+pad)*rate)
                if not .1 <= (b-a)/rate <= 60:
                    continue
                name = f"line_{i+1:03}_retry_{attempt}"
                audio = retry_folder / (name + ".wav")
                sf.write(audio, wave[a:b], rate, subtype="FLOAT")
                audio.with_suffix(".txt").write_text("".join(ch if kept(ch) else " " for ch in line.text), encoding="utf-8")
                retries.append({"id": name, "audio": str(audio.resolve()), "start": a/rate, "end": b/rate,
                                "gate": None, "line_index": i, "target_start": 0,
                                "target_end": len(kept(line.text).casefold())})
                retries[-1]['languages'] = ['zh','en'] if line.language in ('zh','zh-en') else ['ja','en']
                if 'llm_direct_parts' in clips[i]:
                    from .llm_direct import target_parts, write_pfml, acoustic_languages
                    parts=target_parts(clips[i]['llm_direct_parts'],clips[i]['target_start'],clips[i]['target_end'])
                    retries[-1]['llm_direct_parts']=parts
                    retries[-1]['languages']=acoustic_languages(parts)
                    write_pfml(audio,parts)
        if retries:
            try:
                retry_raw = run_worker(retry_folder, retries, log, pronunciations=pronunciations)
            except (RuntimeError, ValueError) as exc:
                log("局部重试未完成，将保留失败句的整句 LRC 时间供人工补轴。")
                attempts.append({"valid": False, "reason": str(exc), "stage": "retry_worker"})
                retry_raw = {"results": []}
            retry_rows = {r["id"]: r for r in retry_raw["results"]}
            if syllables:
                prepare_clips_llm([anchors[c['line_index']] for c in retries],retries,retry_rows)
            for clip in retries:
                i = clip["line_index"]
                if clip["id"] not in retry_rows:
                    continue
                try:
                    probe, row = map_clip(anchors[i], retry_rows[clip["id"]], clip, syllables)
                    attempts.append({"id": clip["id"], "line": i+1, "valid": True})
                    if i not in selected:
                        probe.flags.append("tifa_context_retry_check")
                        selected[i] = (probe, row, clip, "retry")
                except ValueError as exc:
                    attempts.append({"id": clip["id"], "line": i+1, "valid": False, "reason": str(exc)})
    # Full predictions and their retries take priority. Only salvage individual
    # display units after those fail; never invent a duration for a missing phone.
    partials = {}
    candidates = [(i, rows[clip['id']], clip) for i,clip in enumerate(clips) if i not in selected]
    candidates.extend((clip['line_index'], retry_rows[clip['id']], clip)
                      for clip in retries if clip['line_index'] not in selected and clip['id'] in retry_rows)
    for i, source_row, clip in candidates:
        if i in conflicts:
            continue
        try:
            probe,row = map_clip(anchors[i],source_row,clip,syllables,partial=True)
            if "tifa_partial_alignment" not in probe.flags:
                continue
            # Keep one coherent native prediction; do not mix windows' boundaries.
            score = sum(len(kept(w.text)) for w in probe.words if w.start is not None)
            if i not in partials or score > partials[i][0]:
                partials[i] = (score,probe,row,clip)
        except (ValueError,TypeError):
            continue
    for i,(_,probe,row,clip) in partials.items():
        # Preserve the line-order guard used for full predictions as well.
        if probe.role == 'main' and any(other.agent == probe.agent and other.role == 'main' and
                ((j < i and other.start > probe.start) or (j > i and other.start < probe.start))
                for j,(other,*_) in selected.items()):
            continue
        selected[i] = (probe,row,clip,'partial')
        unresolved = [w.text.strip() for w in probe.words if 'tifa_unresolved_unit' in w.flags]
        log(f"第 {i+1} 行保留其他有效单位；「{'、'.join(unresolved)}」未获得时间，需人工补轴。")
    checks = []
    for i, line in enumerate(lines):
        if i not in selected:
            anchor = anchors[i]
            automatic = 'text_coarse_localization_check' in anchor.flags
            flag = 'text_coarse_review_placeholder' if automatic else 'lrc_review_placeholder'
            line.start, line.end = round(anchor.start, 3), round(anchor.end, 3)
            line.flags.append(flag)
            line.words = [Unit(line.text, kept(line.text), start=line.start, end=line.end,
                               flags=[flag])]
            source = '模型粗定位范围' if automatic else '整句 LRC 时间'
            log(f"第 {i+1} 行「{line.text}」需人工补逐字轴：保留{source}后继续导出。")
            checks.append({"line": i+1, "mode": flag, "reason": failures[i],
                           "range_start": line.start, "range_end": line.end,
                           "note": "仅为模型粗定位范围，未获得 TIFA 逐字边界。" if automatic else "仅为输入逐句时间占位，未获得模型逐字边界。"})
            continue
        probe, row, clip, mode = selected[i]
        line.start, line.end, line.words, line.flags = probe.start, probe.end, probe.words, probe.flags
        edge = line.start-clip["start"] < .03 or clip["end"]-line.end < .03
        if edge: line.flags.append("tifa_crop_edge_check")
        checks.append({"line": i+1, "mode": mode, "id": clip["id"], "range_start": clip["start"],
                       "range_end": clip["end"], "gate": clip.get("gate"), "crop_edge": edge,
                       "agreement": row["agreement"], "note": "agreement 为模型内部指标，不是准确率。"})
        if mode == 'partial':
            checks[-1]['unresolved_units'] = [
                {'index':j+1,'text':w.text,'reason':'音素缺失或单位边界无效，未补造时间。'}
                for j,w in enumerate(line.words) if 'tifa_unresolved_unit' in w.flags]
    (folder / "completion.json").write_text(json.dumps({"checks": checks, "attempts": attempts}, ensure_ascii=False, indent=2), encoding="utf-8")
    return checks


def align(samples, rate, lines, output_dir, coarse_rows, log=print, *, syllables=True, background=None,
          pronunciations=None, trust_line_timestamps=False):
    if not ready():
        raise RuntimeError("TIFA 未配置完整，请运行 scripts/download_tifa.py。")
    manifest = json.loads((FOLDER/"manifest.json").read_text(encoding="utf-8"))
    for relative,asset in manifest["assets"].items():
        path=FOLDER/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=asset["sha256"]:
            raise RuntimeError(f"TIFA 文件校验失败：{relative}")
    folder = output_dir/"tifa"
    folder.mkdir(exist_ok=True)
    clips=[]
    for index,(line,coarse) in enumerate(zip(lines,coarse_rows)):
        wave = background if line.role == "bg" and background is not None else samples
        neighbours = [index]
        # The final lexical line may be followed by untranscribed adlibs.
        # Give it a pause-localized utterance instead of a long left context
        # that can pull the final vowel into those other vocal events.
        if line.role == "main" and index < len(lines)-1:
            neighbours=[i for i in range(max(0,index-1),min(len(lines),index+2))
                        if lines[i].role=="main" and lines[i].agent==line.agent and
                        (lines[i].language in ('zh','zh-en')) == (line.language in ('zh','zh-en'))]
        if lines[neighbours[-1]].end-lines[neighbours[0]].start > 59:
            neighbours=[index]
        first,last=neighbours[0],neighbours[-1]
        left = max(0,lines[first].start-.12)
        # LRC marks a next-line onset, not a verified previous-vowel offset.
        # Retain a little crossfade/reverb context rather than truncating at it.
        context_limit = min(len(wave)/rate, lines[last].end+.35)
        if trust_line_timestamps:
            right, gate = context_limit, None
        else:
            right, gate = quiet_crop_end(wave,rate,coarse_rows[last]["end"],context_limit,coarse_rows[last]["start"])
        if not .1 <= right-left <= 60:
            raise ValueError("TIFA 单句范围需在0.1到60秒之间。")
        a,b=round(left*rate),round(right*rate)
        name=f"line_{index+1:03}"
        audio=folder/(name+".wav")
        sf.write(audio,wave[a:b],rate,subtype="FLOAT")
        # Punctuation stays in display; the upstream G2P's ASCII filter does
        # not remove Japanese corner quotes and can reject an otherwise valid line.
        text="\n".join(lines[i].text for i in neighbours)
        spoken = "".join(ch if kept(ch) else " " for ch in text)
        audio.with_suffix(".txt").write_text(spoken,encoding="utf-8")
        target_start=sum(len(kept(lines[i].text).casefold()) for i in neighbours if i<index)
        target_end=target_start+len(kept(line.text).casefold())
        clips.append({"id":name,"audio":str(audio.resolve()),"start":a/rate,"end":b/rate,"gate":gate,
                      "context_lines":[i+1 for i in neighbours],"target_start":target_start,"target_end":target_end})
        clips[-1]['languages'] = ['zh','en'] if line.language in ('zh','zh-en') else ['ja','en']
    log("TIFA：选择匹配歌声的读音并对齐字词…")
    raw=run_worker(folder,clips,log,pronunciations=pronunciations)
    checks=complete_results(samples,rate,lines,folder,clips,raw,log,syllables=syllables,background=background,
                            pronunciations=pronunciations)
    return {"model":"TIFA-1.0-ST","runtime_revision":manifest["runtime_revision"],"frame_ms":10,
            "seconds":raw["seconds"],"peak_torch_allocated_gib":raw["peak_torch_allocated_gib"],"checks":checks,
            "note":"原生音素边界；10ms 网格不等于10ms准确度。英文音节映射为候选，连续长元音保留不可分组。"}

// Timing upload form: pure request-state rules.
// Loaded as a classic script before app.js and reused verbatim by
// tests/test_web_ui.cjs. Keep this file free of DOM, network, storage and
// timer references so it can run in a bare node:vm context.
//
// The enums and normalization mirror the shared backend plan in
// timing/request_plan.py:
//   ALIGNERS      auto / tifa / ctc / llm_text / llm_direct / llm_import
//   AUDIO_MODES   lead / vocals / source
//   LANGUAGES     mixed / ja / en / auto / zh / zh-en
//                 (zh is Mandarin)
//   SPLITTERS     syllabify / pyphen / llm
//   PROMPT_PROFILES fine / original (audio LLM prompt profiles)
// and the audio-route naming mirrors timing/sources.py. Han text only counts
// as Chinese when the local route's song or line language is zh / zh-en;
// everywhere else Han keeps the historical Japanese reading. LLM routes do
// not consume either source of language metadata.
'use strict';

const ALIGNERS = ['auto', 'tifa', 'ctc', 'llm_text', 'llm_direct', 'llm_import'];
const ANCHORED_ALIGNERS = ['llm_text', 'llm_direct', 'llm_import'];
const AUDIO_MODES = ['lead', 'vocals', 'source'];
const LANGUAGES = ['mixed', 'ja', 'en', 'auto', 'zh', 'zh-en'];
const REQUIRED_MODELS = ['ctc', 'tifa', 'bs_lead', 'separator'];
const PROMPT_PROFILES = ['fine', 'original'];
const EXPECTED_API_VERSION = 16;
const MIN_API_VERSION = EXPECTED_API_VERSION;
const AGENT_ID = /^v[1-9]\d{0,8}$/;

function serviceCompatibility(status) {
  return Boolean(status && status.app === 'Timing' &&
    Number.isInteger(status.api_version) && status.api_version >= MIN_API_VERSION);
}

const KANA = /[\u3040-\u30ff]/;
const JAPANESE_MARKS = /[々〆〤]/;
const HAN = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;
const LATIN = /[A-Za-zÀ-ÖØ-öø-ÿ]/;
const LRC = /^\[(\d+):(\d+(?:\.\d+)?)\](.*)$/;
const LRC_END = /\[(\d+):(\d+(?:\.\d+)?)\]$/;
const LRC_INLINE_TIME = /[\[<]\s*-?\d+:/;

function lrcSeconds(minutes, seconds) {
  return Number(minutes) * 60 + Number(seconds);
}

function isTtmlXml(raw) {
  const text = String(raw || '').replace(/^\uFEFF/, '').trimStart();
  return text.startsWith('<');
}

function analyzeLyrics(raw, language, llmRoute = false) {
  const mode = llmRoute ? 'auto' : (LANGUAGES.includes(language) ? language : 'mixed');
  const text = String(raw || '').replace(/^\uFEFF/, '').trim();
  const result = { empty: !text, error: '', kind: 'text', count: 0, anchored: false, partial: false, hasEnglish: false, hasJapanese: false, hasChinese: false, hasBackground: false, maxKnownLineSeconds: 0, language: mode, effectiveLanguages: [], needsTtmlConversion: false };
  if (!text) { result.error = '未填写歌词：请粘贴或导入歌词。'; return result; }
  if (isTtmlXml(text)) {
    result.kind = 'ttml';
    result.needsTtmlConversion = true;
    result.error = 'TTML 歌词需要先转换为逐句歌词；转换完成后才能提交。';
    return result;
  }
  let rows = []; let json = false;
  const first = text.split(/\r?\n/)[0];
  const metadata = /^\[(?:ar|ti|al|by|offset|length):/i.test(first);
  const section = /^\[[A-Za-z\u3040-\u30ff\u3400-\u9fff][A-Za-z\u3040-\u30ff\u3400-\u9fff 0-9_-]*\]$/.test(first);
  if (text.startsWith('{') || (text.startsWith('[') && !LRC.test(first) && !metadata && !section)) {
    try {
      const parsed = JSON.parse(text);
      if (!Array.isArray(parsed)) { result.error = 'JSON 歌词应为数组，每一项是一行歌词。'; return result; }
      json = true; rows = parsed; result.kind = 'json';
    } catch { result.error = 'JSON 歌词格式不完整，请检查括号、引号和逗号。'; return result; }
  }
  if (!json) {
    for (const source of text.split(/\r?\n/)) {
      const line = source.trim();
      if (!line || /^\[(?:ar|ti|al|by|offset|length):/i.test(line)) continue;
      const m = line.match(LRC);
      if (m) {
        let body = m[3].trim();
        if (LRC.test(body)) { result.error = '一行包含多个连续时间标签，请展开为独立歌词行。'; return result; }
        const tail = LRC_END.exec(body);
        let end;
        if (tail) {
          end = lrcSeconds(tail[1], tail[2]);
          body = body.slice(0, tail.index).trimEnd();
        }
        if (LRC_INLINE_TIME.test(body)) { result.error = 'LRC 每行仅支持 [开始]歌词 或 [开始]歌词[结束]；请检查时间标签。'; return result; }
        if (body) {
          const row = { text: body, start: lrcSeconds(m[1], m[2]), role: 'main' };
          if (end !== undefined) row.end = end;
          rows.push(row);
        }
      } else if (!(line.startsWith('[') && line.endsWith(']'))) rows.push({ text: line, role: 'main' });
    }
    if (rows.some(row => row.start != null)) result.kind = 'lrc';
  }
  result.count = rows.length;
  if (!rows.length) { result.error = '没有可对齐的歌词，请填写歌词。'; return result; }
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    if (!row || typeof row !== 'object' || typeof row.text !== 'string' || !/[\p{L}\p{N}]/u.test(row.text)) { result.error = `第 ${i + 1} 行没有可演唱文字。`; return result; }
    if (!['main', 'bg'].includes(row.role === undefined ? 'main' : row.role) || !LANGUAGES.includes(row.language === undefined ? 'auto' : row.language)) { result.error = `第 ${i + 1} 行的角色或语言标记无效。`; return result; }
    const agent = row.agent === undefined ? 'v1' : row.agent;
    if (typeof agent !== 'string' || !AGENT_ID.test(agent)) { result.error = `第 ${i + 1} 行的歌手 ID 无效；须为 v1 这类格式：v 后跟 1–9 开头、最多 9 位的数字（如 v1、v3、v1000）。`; return result; }
    if (['translation', 'roman'].some(key => key in row && typeof row[key] !== 'string')) { result.error = `第 ${i + 1} 行的 translation 和 roman 必须是文字。`; return result; }
    const start = row.start != null, end = row.end != null;
    if (!json && start && !Number.isFinite(row.start)) { result.error = `第 ${i + 1} 行的 LRC 开始时间必须是有限数字。`; return result; }
    if (json && (start !== end || (start && (!Number.isFinite(row.start) || !Number.isFinite(row.end) || row.start < 0 || row.end <= row.start)))) { result.error = `第 ${i + 1} 行需提供有效的 start / end 起止时间。`; return result; }
    if (json && start) result.maxKnownLineSeconds = Math.max(result.maxKnownLineSeconds, row.end - row.start);
    if (!json && start && end) {
      if (!Number.isFinite(row.end)) { result.error = `第 ${i + 1} 行的 LRC 结束时间必须是有限数字。`; return result; }
      if (row.end <= row.start) { result.error = `第 ${i + 1} 行的 LRC 结束时间必须晚于开始时间。`; return result; }
      result.maxKnownLineSeconds = Math.max(result.maxKnownLineSeconds, row.end - row.start);
    } else if (!json && start) {
      const next = rows.slice(i + 1).find(item => item.start != null);
      if (next && next.start <= row.start) { result.error = `第 ${i + 1} 行之后的 LRC 时间必须递增。`; return result; }
      if (next) result.maxKnownLineSeconds = Math.max(result.maxKnownLineSeconds, next.start - row.start);
    }
    if ((row.role ?? 'main') === 'bg') { result.hasBackground = true; if (!start || !Number.isInteger(row.parent) || row.parent < 0 || row.parent >= rows.length || (rows[row.parent]?.role ?? 'main') !== 'main') { result.error = `第 ${i + 1} 行背景歌词需带起止时间，并通过 parent 指向主唱行。`; return result; } }
  }
  const mains = rows.filter(row => (row.role ?? 'main') === 'main');
  if (!mains.length) { result.error = '至少需要一行主歌词。'; return result; }
  let previousMainStart = -1;
  for (const row of mains) { if (row.start != null) { if (row.start < previousMainStart) { result.error = '主唱歌词请按开始时间排序；重叠演唱可以保留。'; return result; } previousMainStart = row.start; } }
  const starts = mains.filter(row => row.start != null).length;
  result.partial = starts > 0 && starts < mains.length;
  if (result.partial) result.error = '逐句时间不完整：请给每句补全时间，或统一使用纯文本歌词。';
  result.anchored = rows.every(row => row.start != null && (json ? row.end != null : true));
  for (const row of rows) {
    const singing = String(row.text).normalize('NFKC');
    if (llmRoute) { result.effectiveLanguages.push('auto'); continue; }
    const declared = row.language === undefined ? 'auto' : row.language;
    const effective = declared === 'auto' ? (mode === 'auto' ? 'mixed' : mode) : declared;
    result.effectiveLanguages.push(effective);
    const chinese = effective === 'zh' || effective === 'zh-en';
    if (KANA.test(singing) || JAPANESE_MARKS.test(singing)) result.hasJapanese = true;
    if (HAN.test(singing)) { if (chinese) result.hasChinese = true; else result.hasJapanese = true; }
    if (LATIN.test(singing)) result.hasEnglish = true;
    if (effective === 'ja') result.hasJapanese = true;
    if (effective === 'en') result.hasEnglish = true;
    if (chinese) result.hasChinese = true;
  }
  return result;
}

function resolveOptions(input) {
  const lyrics = input.lyrics || {};
  const requestedLanguage = LANGUAGES.includes(input.language) ? input.language : 'mixed';
  const requestedAligner = ALIGNERS.includes(input.aligner) ? input.aligner : 'auto';
  const aligner = requestedAligner;
  const engine = aligner === 'auto' ? 'tifa' : aligner;
  const direct = engine === 'llm_direct';
  const audioPromptMode = direct || engine === 'llm_import';
  const textLlm = engine === 'llm_text';
  const directLike = audioPromptMode || textLlm;
  const showImport = engine === 'llm_import';
  const language = directLike ? 'auto' : requestedLanguage;
  const trustEligible = Boolean(engine !== 'ctc' && lyrics.anchored && !lyrics.partial && !lyrics.needsTtmlConversion);
  const trustLineTimestamps = trustEligible && Boolean(input.trustLineTimestamps);
  let error = lyrics.error || (lyrics.empty ? '未填写歌词：请粘贴或导入歌词。' : '') || (lyrics.partial ? '逐句时间不完整：请给每句补全时间，或统一使用纯文本歌词。' : '');
  if (ANCHORED_ALIGNERS.includes(engine) && !lyrics.anchored && !error) error = '当前方案需要逐句时间：请补全每句时间，或改用“自动 / TIFA”。';
  const valid = !error;
  const showEnglish = !directLike && Boolean(lyrics.hasEnglish);
  const syllables = showEnglish && Boolean(input.syllables);
  const allowLlmSplitter = !['ctc', 'llm_text', 'llm_direct'].includes(engine);
  const requested = typeof input.splitter === 'string' ? input.splitter : 'syllabify';
  const splitter = syllables && !(requested === 'llm' && !allowLlmSplitter) ? requested : 'syllabify';
  const showSplitter = showEnglish && syllables;
  const llm = valid && (direct || textLlm || (syllables && splitter === 'llm'));
  const promptProfile = PROMPT_PROFILES.includes(input.promptProfile) ? input.promptProfile : 'fine';
  const audioMode = AUDIO_MODES.includes(input.audioMode) ? input.audioMode : 'lead';
  const hasVocals = Boolean(input.hasVocals);
  const source = audioMode === 'lead' ? 'separated_vocals' : audioMode === 'vocals' ? (hasVocals ? 'provided_vocals' : 'separated_vocals') : (hasVocals ? 'provided_vocals' : 'original');
  const models = REQUIRED_MODELS.filter(name => name !== 'ctc' || !trustLineTimestamps);
  return { aligner, engine, language, showLanguage: !directLike, valid, error, showEnglish, showSplitter, allowLlmSplitter, syllables, splitter, llm, sendsAudio: llm && direct, showImport, showPromptProfile: audioPromptMode, promptReady: showImport && valid, promptProfile, audioMode, source, trustEligible, trustLineTimestamps, models, hasBackground: Boolean(lyrics.hasBackground) };
}

// Keep submission policy beside option resolution so every LLM route sends the
// automatic language value without stale selections or hidden helper flags.
function submissionFields(plan) {
  const llmRoute = ['llm_text', 'llm_direct', 'llm_import'].includes(plan.engine);
  const fields = {
    language: llmRoute ? 'auto' : plan.language,
    aligner: plan.aligner,
    audio_mode: plan.audioMode,
    syllables: String(Boolean(plan.syllables)),
    english_splitter: plan.splitter,
    trust_line_timestamps: String(Boolean(plan.trustLineTimestamps)),
  };
  if (['llm_direct', 'llm_import'].includes(plan.engine)) {
    fields.llm_prompt_profile = PROMPT_PROFILES.includes(plan.promptProfile) ? plan.promptProfile : 'fine';
  }
  return fields;
}

function disclosureText(plan) {
  if (plan.engine === 'llm_import') return '只生成本地提示词并接收你带回的 JSON；提示词不附带预设歌曲或行语言信息，由你选择的模型自行判断语言。不发送音频、歌词或凭据到网页 LLM。下载后仍需试听校对。';
  if (plan.engine === 'llm_direct') return '整首歌和全部歌词只发送一次' + (plan.hasBackground ? '，并附带含主唱与背景的完整人声' : '') + '；不附带预设歌曲或行语言信息，由模型自行判断语言。下载后仍需试听校对。';
  if (plan.engine === 'llm_text') return '全部逐行歌词会发送到接口，不附带预设歌曲或行语言信息或音频；由模型自行判断语言。下载后先试听校对。';
  if (plan.sendsAudio) return '音频片段和歌词会发送到所选接口；下载后先试听校对。';
  if (plan.llm) return '只发送英文词和所在句，不发送音频；下载后先试听校对。';
  return '音频和歌词都在本机处理；下载后先试听校对。';
}
// Composite song-language name for the lyrics status line, e.g. 中英混唱.
function describeLanguages(plan) {
  if (plan.language === 'auto' && plan.effectiveLanguages?.length && plan.effectiveLanguages.every(language => language === 'auto')) return '待 LLM 识别语言';
  const present = [['hasChinese', '中文', '中'], ['hasJapanese', '日语', '日'],
    ['hasEnglish', '英语', '英']].filter(([key]) => plan[key]);
  if (present.length === 1) return present[0][1];
  return present.length > 1 ? present.map(([, , short]) => short).join('') + '混唱' : '';
}

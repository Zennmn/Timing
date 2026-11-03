'use strict';

/*
 * Timing upload-form state tests.
 *
 * Scope: web/ui-logic.js (the pure request-state module, loaded here as an
 * external file) plus the static wiring in web/index.html, web/styles.css and
 * web/app.js. Everything runs offline: no browser, no server, no models and no
 * network access.
 *
 * Run from the repository root:
 *   node --test tests/test_web_ui.cjs
 * (the project shell wrapper is fine too: rtk node --test tests/test_web_ui.cjs)
 *
 * Contract under test
 *   analyzeLyrics(raw, language, llmRoute) -> {empty,error,kind,count,anchored,partial,
 *                          hasEnglish,hasJapanese,hasChinese,hasBackground,
 *                          language,effectiveLanguages}
 *   resolveOptions({lyrics,language,aligner,syllables,splitter,promptProfile,
 *                   hasVocals,audioMode}) -> {aligner,engine,language,showLanguage,
 *                   valid,error,showEnglish,showSplitter,allowLlmSplitter,
 *                   syllables,splitter,llm,sendsAudio,showImport,showPromptProfile,promptReady,
 *                   promptProfile,audioMode,source,models,hasBackground}
 *   submissionFields(plan) -> the browser payload fields owned by the UI.
 *   serviceCompatibility(status) -> boolean for Timing with API 15 or newer.
 *
 * Expected values mirror the shared backend rules:
 *   timing/request_plan.py  ALIGNERS / AUDIO_MODES / SPLITTERS / LANGUAGES
 *                           (mixed / ja / en / auto / zh / zh-en), required
 *                           lyrics, auto -> tifa for plain and anchored input,
 *                           MMS without the LLM
 *                           splitter, text-only LLM splitter. zh is Mandarin;
 *                           Han text only reads as Chinese under zh / zh-en.
 *   timing/sources.py       source naming for the lead / vocals / source
 *                           routes, including imported vocal stems.
 *   timing/text.py          TXT/LRC/JSON recognition, LRC metadata tags and
 *                           JSON role:"bg" background lines.
 */

const { test, describe } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const ROOT = path.join(__dirname, '..');
const WEB = path.join(ROOT, 'web');
const UI_LOGIC = path.join(WEB, 'ui-logic.js');
const APP_JS = path.join(WEB, 'app.js');
const INDEX_HTML = path.join(WEB, 'index.html');
const STYLES_CSS = path.join(WEB, 'styles.css');

const REQUIRED_MODELS = ['ctc', 'tifa', 'bs_lead', 'separator'];

/* ------------------------------------------------------------------ *
 * Harness: evaluate the external pure module in a bare node:vm context.
 * ------------------------------------------------------------------ */

function loadPureSource(source, filename) {
  const context = vm.createContext({});
  vm.runInContext(source, context, { filename: filename });
  return (name) => vm.runInContext(
    'typeof ' + name + ' === "undefined" ? undefined : ' + name,
    context,
    { filename: filename + '.lookup.js' }
  );
}

function loadPureFile(file) {
  return loadPureSource(fs.readFileSync(file, 'utf8'), path.basename(file));
}

function loadFromFile() {
  const lookup = loadPureFile(UI_LOGIC);
  const missing = ['analyzeLyrics', 'resolveOptions', 'submissionFields', 'disclosureText', 'describeLanguages', 'serviceCompatibility']
    .filter((name) => typeof lookup(name) !== 'function');
  if (missing.length) throw new Error(path.basename(UI_LOGIC) + ' does not define ' + missing.join(', ') + '.');
  return {
    analyzeLyrics: lookup('analyzeLyrics'),
    resolveOptions: lookup('resolveOptions'),
    submissionFields: lookup('submissionFields'),
    disclosureText: lookup('disclosureText'),
    describeLanguages: lookup('describeLanguages'),
    serviceCompatibility: lookup('serviceCompatibility'),
  };
}

const html = fs.readFileSync(INDEX_HTML, 'utf8');
const appSource = fs.readFileSync(APP_JS, 'utf8');
const uiSource = fs.readFileSync(UI_LOGIC, 'utf8');

function scriptSrcs(source) {
  return Array.from(source.matchAll(/<script\b[^>]*\bsrc\s*=\s*["']([^"']+)["']/gi), (match) => match[1].split('?')[0]);
}

function optionValues(source, id) {
  const select = new RegExp('<select\\b[^>]*\\bid\\s*=\\s*["\']' + id + '["\'][^>]*>([\\s\\S]*?)</select>', 'i').exec(source);
  return select ? Array.from(select[1].matchAll(/<option\b[^>]*\bvalue\s*=\s*["']([^"']*)["']/gi), (match) => match[1]) : null;
}

function selectBlock(source, id) {
  const start = source.indexOf('id="' + id + '"');
  if (start < 0) return null;
  const open = source.lastIndexOf('<select', start);
  const close = source.indexOf('</select>', start);
  return open < 0 || close < 0 ? null : source.slice(open, close);
}

function optionLabels(source, id) {
  const block = selectBlock(source, id);
  return block === null ? null : block.split('<option').slice(1)
    .map((part) => part.slice(part.indexOf('>') + 1, part.indexOf('</option>')).trim());
}

/* ------------------------------------------------------------------ *
 * Fixtures
 * ------------------------------------------------------------------ */

const JA_TEXT = '光の中へ\n走り出す\n夜が明ける';
const EN_TEXT = 'Run into the light\nWe are moving on';
const MIXED_TEXT = '光の中へ\nRun into the light';
const KO_TEXT = '빛 속으로\n달려가\n밤이 밝아와';
const ZH_TEXT = '今天很好\n明天更好';
const ZH_EN_TEXT = '今天很好\nWe are moving on';
const FULL_LRC = '[00:12.34]光の中へ\n[00:18.00]Run into the light';
const PARTIAL_LRC = '光の中へ\n[00:18.00]Run into the light';
const META_ONLY_LRC = '[ar:测试歌手]\n[ti:歌词]';
const META_LRC = '[ar:测试歌手]\n[ti:歌词]\n[al:专辑]\n[by:上传者]\n[offset:120]\n[length:03:20]\n' +
  '[00:12.34]光の中へ\n[00:18.00]走り出す';
const BROKEN_JSON = '[{"text":"光の中へ","start":12.34,"end":18.0},';
const JSON_WITHOUT_TIMES = '[{"text":"光の中へ"},{"text":"Run into the light"}]';
const JSON_MISSING_END = '[{"text":"光の中へ","start":12.34,"end":18.0},{"text":"走り出す","start":18.0}]';
const JSON_REVERSED_TIMES = '[{"text":"光の中へ","start":18.0,"end":12.34}]';
const DOUBLE_STAMP_LRC = '[00:10.00][00:20.00]同じ行に二つの時間タグ';
const JSON_LYRICS = '[{"text":"光の中へ","start":12.34,"end":18.0},' +
  '{"text":"Run into the light","start":18.0,"end":24.5}]';
const JSON_BACKGROUND = '[{"text":"光の中へ","start":12.34,"end":18.0,"role":"main"},' +
  '{"text":"oh oh","start":13.0,"end":17.0,"role":"bg","parent":0}]';
const TTML_XML = '<?xml version="1.0" encoding="UTF-8"?>' +
  '<tt xmlns="http://www.w3.org/ns/ttml" xml:lang="ja"><body><div begin="12.34s" end="18s">' +
  '<p begin="12.34s" end="18s">光の中へ</p></div></body></tt>';
const TTML_NORMALIZED = '[{"text":"光の中へ","start":12.34,"end":18.0}]';

// A lyrics descriptor shaped like the analyzeLyrics() result.
function lyrics(overrides) {
  return Object.assign({
    empty: false,
    error: '',
    kind: 'lrc',
    count: 4,
    anchored: true,
    partial: false,
    hasEnglish: true,
    hasJapanese: true,
    hasBackground: false,
  }, overrides);
}

// A saved-preferences object shaped like the form state.
function preferences(overrides) {
  return Object.assign({
    lyrics: lyrics(),
    language: 'mixed',
    aligner: 'auto',
    syllables: true,
    splitter: 'syllabify',
    promptProfile: 'fine',
    hasVocals: false,
    audioMode: 'lead',
    trustLineTimestamps: false,
  }, overrides);
}

function deferred() {
  let resolve, reject;
  const promise = new Promise((ok, fail) => { resolve = ok; reject = fail; });
  return { promise, resolve, reject };
}

function fakeResponse(status, body) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

function fakeClassList() {
  const values = new Set();
  return {
    add: (...names) => names.forEach((name) => values.add(name)),
    remove: (...names) => names.forEach((name) => values.delete(name)),
    contains: (name) => values.has(name),
    toggle(name, force) {
      const visible = force === undefined ? !values.has(name) : Boolean(force);
      if (visible) values.add(name); else values.delete(name);
      return visible;
    },
  };
}

function fakeElement(id) {
  const listeners = new Map();
  const element = {
    id, tagName: 'DIV', value: '', checked: false, disabled: false, required: false,
    hidden: false, files: [], options: [], textContent: '', title: '', className: '',
    scrollTop: 0, scrollHeight: 0, open: false, type: 'text', classList: fakeClassList(), children: [],
    addEventListener(type, handler) {
      if (!listeners.has(type)) listeners.set(type, []);
      listeners.get(type).push(handler);
    },
    async dispatch(type, extra = {}) {
      for (const handler of listeners.get(type) || []) {
        await handler(Object.assign({ type, target: element, preventDefault() {} }, extra));
      }
    },
    querySelector(selector) {
      const match = /^option\[value=["'](.+)["']\]$/.exec(selector);
      return match ? element.options.find((option) => option.value === match[1]) || null : null;
    },
    closest: () => null,
    append(...children) { if (!element.children) element.children = []; element.children.push(...children); },
    replaceChildren(...children) { element.children = children; },
    setAttribute(name, value) { if (!element.attributes) element.attributes = {}; element.attributes[name] = String(value); },
    removeAttribute(name) { if (element.attributes) delete element.attributes[name]; },
    setCustomValidity() {},
    checkValidity: () => true,
    reportValidity: () => true,
    focus() {},
    click() {},
  };
  return element;
}

async function makeAppHarness(statusOverrides = {}, initialStorage = {}, queueOptions = {}) {
  const elements = new Map();
  const timers = new Map();
  const ttmlRequests = [], statusRequests = [];
  const queueRequests = [], submitRequests = [], cancelRequests = [], detailRequests = [];
  let nextTimer = 1;
  let ttmlHandler = async () => fakeResponse(500, { detail: 'unexpected TTML request' });
  let queueHandler = queueOptions.queue || (async () => fakeResponse(200, { jobs: [], running: 0, waiting: 0, limit: 50 }));
  let submitHandler = queueOptions.submit || (async () => fakeResponse(500, { detail: 'unexpected submit request' }));
  let cancelHandler = queueOptions.cancel || (async () => fakeResponse(500, { detail: 'unexpected cancel request' }));
  let detailHandler = queueOptions.detail || (async () => fakeResponse(404, { detail: 'unexpected detail request' }));

  const element = (id) => {
    if (!elements.has(id)) elements.set(id, fakeElement(id));
    return elements.get(id);
  };
  const setOptions = (id, values) => {
    const target = element(id);
    target.tagName = 'SELECT';
    target.options = values.map((value) => ({ value, textContent: value, disabled: false, hidden: false }));
  };
  setOptions('aligner', optionValues(html, 'aligner'));
  setOptions('audio-mode', optionValues(html, 'audio-mode'));
  setOptions('language', optionValues(html, 'language'));
  setOptions('english-splitter', ['syllabify', 'pyphen', 'llm']);
  setOptions('llm-prompt-profile', ['fine', 'original']);
  element('aligner').value = 'auto';
  element('audio-mode').value = 'lead';
  element('language').value = 'mixed';
  element('english-splitter').value = 'syllabify';
  element('llm-prompt-profile').value = 'fine';
  element('syllables').checked = true;
  element('llm-thinking-mode').value = 'default';
  element('llm-thinking-format').value = 'reasoning_effort';
  element('llm-api-key-header').value = 'Authorization';
  element('llm-api-key-format').value = 'auto';

  const document = {
    getElementById: element,
    querySelector: (selector) => selector === 'meta[name="timing-token"]' ? { content: 'test-token' } : null,
    createElement: (tag) => fakeElement('created-' + tag),
    createRange: () => ({ selectNodeContents() {} }),
    execCommand: () => true,
  };
  const storageValues = new Map(Object.entries(initialStorage));
  const context = vm.createContext({
    document,
    window: { getSelection: () => ({ removeAllRanges() {}, addRange() {} }) },
    localStorage: {
      getItem: (key) => storageValues.has(key) ? storageValues.get(key) : null,
      setItem: (key, value) => storageValues.set(key, String(value)),
      removeItem: (key) => storageValues.delete(key),
    },
    navigator: {},
    fetch: async (url, init = {}) => {
      if (url === '/api/status') {
        statusRequests.push(init);
        return fakeResponse(200, { app: 'Timing', api_version: 16, version: 'test', queued: 0,
          models: { ctc: true, tifa: true, bs_lead: true, separator: true, singing: false, sofa: false }, ...statusOverrides });
      }
      if (url === '/api/jobs') {
        if (init.method === 'POST') { submitRequests.push(init); return submitHandler(init); }
        queueRequests.push(init); return queueHandler(init);
      }
      const match = /^\/api\/jobs\/([^/]+)(\/cancel)?$/.exec(url);
      if (match) {
        const id = decodeURIComponent(match[1]);
        if (match[2]) { cancelRequests.push({ id, init }); return cancelHandler(id, init); }
        detailRequests.push({ id, init }); return detailHandler(id, init);
      }
      if (url === '/api/lyrics/ttml') {
        ttmlRequests.push({ init, body: init.body });
        return ttmlHandler(init, ttmlRequests.length);
      }
      throw new Error('unexpected request: ' + url);
    },
    setTimeout: (handler, delay) => {
      const id = nextTimer++;
      timers.set(id, { handler, delay });
      return id;
    },
    clearTimeout: (id) => timers.delete(id),
    FormData, Blob, TextEncoder, TextDecoder,
    URL: { createObjectURL: () => 'blob:test', revokeObjectURL() {} },
  });

  vm.runInContext(uiSource, context, { filename: 'ui-logic.js' });
  vm.runInContext(appSource, context, { filename: 'app.js' });
  const settle = async (rounds = 8) => { for (let i = 0; i < rounds; i++) await Promise.resolve(); };
  await settle();
  return {
    setTtmlHandler(handler) { ttmlHandler = handler; },
    setQueueHandler(handler) { queueHandler = handler; },
    setSubmitHandler(handler) { submitHandler = handler; },
    setCancelHandler(handler) { cancelHandler = handler; },
    setDetailHandler(handler) { detailHandler = handler; },
    element,
    dispatch: (id, type) => element(id).dispatch(type),
    runNextTimer() {
      const next = timers.entries().next();
      assert.equal(next.done, false, 'expected a pending lyric timer');
      const [id, timer] = next.value;
      timers.delete(id);
      return Promise.resolve(timer.handler());
    },
    timerCount: () => timers.size,
    settle,
    storageValues,
    ttmlRequests,
    statusRequests,
    queueRequests,
    submitRequests,
    cancelRequests,
    detailRequests,
  };
}

/* ------------------------------------------------------------------ *
 * Boot
 * ------------------------------------------------------------------ */

let loadedApi = null;
let loadError = null;
try {
  loadedApi = loadFromFile();
} catch (error) {
  loadError = error;
}

describe('shipped files', () => {
  test('web/ui-logic.js loads in a bare sandbox and is required by this harness', () => {
    const lookup = loadPureSource('function sampleFunction() { return 42; }', 'sample.js');
    assert.equal(lookup('sampleFunction')(), 42);
    assert.equal(loadError, null, 'web/ui-logic.js must load: ' + (loadError && loadError.message));
    assert.equal(typeof loadedApi.analyzeLyrics, 'function');
    assert.equal(typeof loadedApi.resolveOptions, 'function');
  });

  test('web/ui-logic.js stays free of DOM, storage, network and timer references', () => {
    assert.ok(
      !/\b(document|window|localStorage|fetch|setTimeout|setInterval|FormData)\b/.test(uiSource),
      'ui-logic.js must stay a pure module so it can be tested without a browser'
    );
  });

  test('web/ui-logic.js exports the shipped language contract', () => {
    const lookup = loadPureFile(UI_LOGIC);
    assert.deepStrictEqual([...lookup('LANGUAGES')], ['mixed', 'ja', 'en', 'auto', 'zh', 'zh-en']);
  });

  test('web/ui-logic.js exports the TIFA aligner and four-model readiness contract', () => {
    const lookup = loadPureFile(UI_LOGIC);
    assert.deepStrictEqual([...lookup('ALIGNERS')], ['auto', 'tifa', 'ctc', 'llm_text', 'llm_direct', 'llm_import']);
    assert.deepStrictEqual([...lookup('ANCHORED_ALIGNERS')], ['llm_text', 'llm_direct', 'llm_import']);
    assert.deepStrictEqual([...lookup('REQUIRED_MODELS')], REQUIRED_MODELS);
  });

  test('serviceCompatibility requires the trusted-timestamp API 16 minimum', () => {
    const lookup = loadPureFile(UI_LOGIC);
    assert.equal(lookup('EXPECTED_API_VERSION'), 16);
    assert.equal(lookup('MIN_API_VERSION'), 16);
    const compatible = lookup('serviceCompatibility');
    assert.equal(compatible({ app: 'Timing', api_version: 15 }), false);
    assert.equal(compatible({ app: 'Timing', api_version: 16 }), true);
    assert.equal(compatible({ app: 'Timing', api_version: 17 }), true, 'newer APIs must remain compatible');
    assert.equal(compatible({ app: 'Other', api_version: 16 }), false);
    assert.equal(compatible({ app: 'Timing', api_version: '16' }), false);
    assert.equal(compatible(null), false);
  });

  test('web/index.html contains no inline script blocks', () => {
    assert.ok(
      !/<script\b(?![^>]*\bsrc\s*=)[^>]*>[\s\S]*?<\/script>/i.test(html),
      'form logic must live in external files, not inline <script> blocks'
    );
  });

  test('web/index.html loads the external ui-logic and app scripts in order', () => {
    const srcs = scriptSrcs(html);
    assert.ok(srcs.includes('/static/ui-logic.js'), 'missing ui-logic.js script tag: ' + srcs.join(', '));
    assert.ok(srcs.includes('/static/app.js'), 'missing app.js script tag: ' + srcs.join(', '));
    assert.ok(srcs.indexOf('/static/ui-logic.js') < srcs.indexOf('/static/app.js'), 'ui-logic.js must load before app.js');
  });

  test('web/index.html links styles.css and the stylesheet carries the form styles', () => {
    assert.ok(html.includes('/static/styles.css'), 'stylesheet link missing');
    const css = fs.readFileSync(STYLES_CSS, 'utf8');
    assert.ok(css.length > 4000, 'styles.css looks truncated: ' + css.length + ' bytes');
    assert.ok(css.includes('.lyrics-box'), 'styles.css is missing the form styles');
  });

  test('the request token ships exactly once, through the timing-token meta tag', () => {
    assert.ok(/<meta\s+name="timing-token"\s+content="__TIMING_TOKEN__">/i.test(html), 'timing-token meta tag missing');
    assert.equal(html.split('__TIMING_TOKEN__').length - 1, 1, 'the token placeholder must appear exactly once');
  });

  test('web/app.js compiles as a classic script', () => {
    assert.doesNotThrow(() => new vm.Script(appSource, { filename: 'app.js' }));
  });

  test('web/app.js reads the token meta and submits audio_mode without the removed fields', () => {
    assert.ok(appSource.includes('timing-token'), 'app.js must read the token meta tag');
    assert.match(appSource, /Object\.entries\(submissionFields\(current\)\)/,
      'payload policy must come from the tested ui-logic helper');
    assert.ok(appSource.includes("data.set('lyrics', $('lyrics').value);"),
      'lyrics must be submitted from the textarea verbatim');
    assert.ok(!/data\.set\('(?:separate|lead)'/.test(appSource), 'the removed separate/lead fields must not be sent');
  });

  test('web/app.js uses the shared service compatibility check', () => {
    assert.match(appSource, /!serviceCompatibility\(status\)/);
  });

  test('older API responses are rejected instead of using a stale cached frontend', async () => {
    for (const apiVersion of [10, 12, 14]) {
      const app = await makeAppHarness({ api_version: apiVersion });
      assert.equal(app.element('submit').disabled, true);
      assert.match(app.element('model-status').textContent, /版本不匹配.*Ctrl\+F5/);
      assert.equal(app.element('retry-models').classList.contains('hidden'), false);
    }
  });

  test('a saved SOFA preference is migrated to automatic TIFA selection', async () => {
    const app = await makeAppHarness({}, { 'timing-aligner': 'sofa' });
    assert.equal(app.element('aligner').value, 'auto');
    assert.equal(app.storageValues.get('timing-aligner'), 'auto');
  });

  test('saved Korean language preferences migrate to mixed', async () => {
    for (const saved of ['ko', 'ko-en']) {
      const app = await makeAppHarness({}, { 'timing-language': saved });
      assert.equal(app.element('language').value, 'mixed');
      assert.equal(app.storageValues.get('timing-language'), 'mixed');
    }
  });

  test('a supported language preference persists', async () => {
    const app = await makeAppHarness();
    app.element('language').value = 'en';
    await app.dispatch('language', 'change');
    assert.equal(app.storageValues.get('timing-language'), 'en');
  });

  test('four ready model groups ignore retired singing and SOFA readiness flags', async () => {
    const app = await makeAppHarness();
    app.element('lyrics').value = 'Run into the light\nWe are moving on';
    app.element('audio').files = [{ name: 'song.wav', size: 100 }];
    await app.dispatch('lyrics', 'input');
    assert.equal(app.element('submit').disabled, false);
    assert.match(app.element('model-status').textContent, /本地模型已就绪/);
    assert.doesNotMatch(app.element('model-status').textContent, /SOFA/);

    const missing = await makeAppHarness({ models: { ctc: true, tifa: false, bs_lead: true, separator: true } });
    assert.equal(missing.element('submit').disabled, true);
    assert.match(missing.element('model-status').textContent, /缺少 TIFA.*全部 4 个模型/);
  });

  test('the page exposes the song-language selector as a visible lyrics setting', () => {
    assert.deepStrictEqual(optionValues(html, 'language'), ['mixed', 'ja', 'en', 'zh', 'zh-en']);
    assert.ok(html.includes('for="language"'), 'the language selector needs a visible label');
    assert.ok(!html.includes('语言提示'), 'the language selector must not fold back into a details hint');
    const before = html.slice(0, html.indexOf('id="language"'));
    assert.equal(before.split('<details').length, before.split('</details>').length, 'the language selector must sit outside any details element');
    assert.ok(!selectBlock(html, 'language').includes('selected'), 'mixed stays the default first option');
  });

  test('the language options carry the mixed and Mandarin labels', () => {
    assert.deepStrictEqual(optionLabels(html, 'language'), ['自动识别（日／英）', '日语', '英语', '中文（普通话）', '中英混唱']);
  });

  test('the language hint explains Mandarin selection instead of an export-only tag', () => {
    assert.ok(html.includes('中文歌曲或中英混唱，请手动选择。'), 'the hint must explain when to select a language manually');
    assert.ok(html.includes('中文（普通话）'), 'Chinese must be named as Mandarin');
    assert.ok(!html.includes('此选项用于导出语言标记'), 'the old export-only language hint must be gone');
  });

  test('the page does not advertise or route Korean lyrics', () => {
    assert.doesNotMatch(html, /韩语|韩英混唱|韩文/, 'Korean choices and routing copy must be removed');
    assert.doesNotMatch(appSource, /hasKorean|localKoreanMms|koreanMms/, 'Korean routing state must be removed');
    assert.doesNotMatch(uiSource, /hasKorean/, 'Korean must not remain a positive capability field');
  });

  test('llm_text hides the local language setting and reports model-detected language', async () => {
    const app = await makeAppHarness();
    app.element('aligner').value = 'llm_text';
    app.element('language').value = 'zh';
    app.element('lyrics').value = '[00:01]今天很好[00:03]';
    app.element('audio').files = [{ name: 'song.wav', size: 100 }];
    await app.dispatch('lyrics', 'input');

    assert.equal(app.element('language-settings').classList.contains('hidden'), true);
    assert.match(app.element('lyrics-status').textContent, /待 LLM 识别语言/);
    assert.match(app.element('mode-hint').textContent, /不附带歌曲或行语言信息/);
    assert.match(app.element('mode-hint').textContent, /模型自行判断语言/);
  });

  test('lyrics are not blocked based on written language', async () => {
    const app = await makeAppHarness();
    app.element('lyrics').value = '빛 속으로\nRun into the light';
    app.element('audio').files = [{ name: 'korean.wav', size: 100 }];
    await app.dispatch('lyrics', 'input');

    assert.doesNotMatch(app.element('lyrics-status').textContent, /当前仅支持/);
    assert.equal(app.element('language-settings').classList.contains('hidden'), false);
    assert.equal(app.element('submit').disabled, false);
    assert.doesNotMatch(app.element('run-disclosure').textContent, /韩|MMS CTC/);
  });

  test('the retired audio assistants are absent from the page', () => {
    assert.ok(!html.includes('llm-audio-reading'), 'the audio-reading checkbox must be removed');
    assert.ok(!html.includes('llm-audio-split'), 'the audio-split checkbox must be removed');
    assert.ok(!html.includes('听音校对读音'), 'the removed audio-reading copy must be gone');
  });

  test('the shipped assets carry the current version with optional cache busters', () => {
    const stamps = Array.from(html.matchAll(/\/static\/(?:styles\.css|ui-logic\.js|app\.js)\?v=([^"'&]+)/g),
      (match) => match[1]);
    assert.equal(stamps.length, 3, 'all three static assets must be versioned');
    assert.deepStrictEqual([...new Set(stamps.map((stamp) => stamp.split('-')[0]))], ['0.22.0'],
      'static assets must use the trusted-timestamp version');
    for (const stamp of stamps) {
      assert.match(stamp, /^0\.22\.0(?:-[0-9A-Za-z._-]+)?$/, 'cache busters must keep the current asset version');
    }
  });

  test('the lyrics help documents ordinary and explicit-end LRC', () => {
    assert.ok(html.includes('id="lrc-help"'), 'the LRC help note is missing');
    assert.ok(html.includes('[00:17.250]'), 'the start-time example is missing');
    assert.ok(html.includes('[00:21.120]'), 'the explicit-end example is missing');
    assert.ok(html.includes('明确每行结尾'), 'the help must explain explicit ends');
  });

  test('the removed ASR model group and draft copy are gone from the shipped files', () => {
    const combined = html + appSource + uiSource;
    assert.ok(!/['"]asr['"]/.test(appSource + uiSource), 'no asr model group may remain');
    assert.ok(!/听写/.test(combined), 'no ASR dictation copy may remain');
    assert.ok(!/生成歌词草稿/.test(combined), 'no draft-generation copy may remain');
    assert.ok(!/Qwen/i.test(combined), 'the retired Qwen references are gone');
  });

  test('the audio-mode selector exposes exactly lead, vocals and source', () => {
    assert.deepStrictEqual(optionValues(html, 'audio-mode'), ['lead', 'vocals', 'source']);
  });

  test('the aligner selector exposes TIFA, MMS and LLM modes without retired SOFA', () => {
    assert.deepStrictEqual(optionValues(html, 'aligner'), ['auto', 'llm_text', 'llm_direct', 'llm_import', 'tifa', 'ctc']);
    assert.equal(optionValues(html, 'aligner')[0], 'auto');
    assert.ok(!/sofa/i.test(selectBlock(html, 'aligner')), 'SOFA must not remain a public aligner option');
    assert.doesNotMatch(optionLabels(html, 'aligner')[4], /韩/);
    assert.match(optionLabels(html, 'aligner')[5], /支持纯文本/);
    for (const index of [1, 2, 3]) assert.match(optionLabels(html, 'aligner')[index], /需逐句时间/);
  });

  test('the Japanese prompt profile is a visible direct-like selector with two labels', () => {
    assert.ok(html.includes('id="prompt-profile-settings" class="local-settings hidden"'));
    assert.ok(html.includes('for="llm-prompt-profile">日语拆分</label>'));
    assert.deepStrictEqual(optionValues(html, 'llm-prompt-profile'), ['fine', 'original']);
    assert.deepStrictEqual(optionLabels(html, 'llm-prompt-profile'), ['细拆（默认）', '适度合并（旧版）']);
    assert.match(appSource, /show\('prompt-profile-settings', current\.showPromptProfile\)/);
  });

  test('LLM direct is presented as one full-track request with manual fallback', () => {
    assert.match(optionLabels(html, 'aligner')[2], /^LLM 整曲识别/);
    assert.ok(html.includes('id="language-settings"'), 'the language settings need a hideable container');
    assert.ok(uiSource.includes('整首歌和全部歌词只发送一次'), 'the disclosure must name the single full-track request');
    assert.ok(appSource.includes('不会自动重试'), 'the hint must explain the manual fallback after an invalid response');
  });

  test('the web import flow is a visible three-step local prompt and result handoff', () => {
    assert.match(optionLabels(html, 'aligner')[3], /^网页 LLM（导入结果，需逐句时间）/);
    for (const id of ['llm-import-panel', 'llm-import-audio-name', 'generate-prompt', 'copy-prompt',
      'download-prompt', 'llm-prompt', 'llm-result', 'llm-result-file', 'validate-llm-result']) {
      assert.ok(html.includes('id="' + id + '"'), 'missing web import control #' + id);
    }
    const panel = html.slice(html.indexOf('id="llm-import-panel"'), html.indexOf('id="english-settings"'));
    assert.equal((panel.match(/<li>/g) || []).length, 3, 'the handoff must stay at three steps');
    assert.ok(panel.includes('完整音频'), 'step one must name the full recording');
    assert.ok(panel.includes('同起点、同时长'), 'step one must preserve the optional-vocal alignment rule');
    assert.ok(panel.includes('检查结果'), 'the optional local check must be visible');
    assert.ok(!panel.includes('llm-api-key'), 'the import panel must not expose provider credentials');
  });

  test('web import requests use only the local prompt/check endpoints and current textarea', () => {
    assert.ok(appSource.includes("fetch('/api/llm/prompt'"));
    assert.ok(appSource.includes("fetch('/api/llm/import/validate'"));
    assert.match(appSource, /data\.set\('llm_result_file',\s*new Blob\(\[text\]/);
    assert.match(appSource, /MAX_LLM_RESULT_BYTES\s*=\s*16 \* 1024 \*\* 2/);
    assert.ok(appSource.includes("$('llm-result').value"), 'the current textarea value must win');
    assert.ok(!/storage\.(?:get|set)\([^\n]*(?:llm-result|llm-prompt(?!-profile))/.test(appSource), 'prompt and result must stay out of storage');
    assert.ok(!/llm-result[^\n]*innerHTML|innerHTML[^\n]*llm-result/.test(appSource), 'LLM result must never use HTML rendering');
  });

  test('web import prompt/check requests discard stale responses and send no audio or credentials', () => {
    const generate = appSource.slice(appSource.indexOf('async function generatePrompt()'), appSource.indexOf('async function copyPrompt()'));
    const check = appSource.slice(appSource.indexOf('async function checkImportResult()'), appSource.indexOf('function link('));
    assert.ok(generate.includes("data.set('lyrics', lyrics)"));
    assert.ok(generate.includes("data.set('audio_name', audio.name)"));
    assert.ok(generate.includes("data.set('llm_prompt_profile', profile)"));
    assert.ok(!generate.includes("data.set('audio'"));
    assert.ok(!generate.includes('llmData('));
    assert.ok(!check.includes('llmData('));
    assert.match(generate, /sequence !== promptSequence \|\| revision !== lyricsRevision \|\| profile !== \$\('llm-prompt-profile'\)\.value/);
    assert.match(check, /sequence !== checkSequence \|\| revision !== lyricsRevision \|\| resultRevisionAtStart !== resultRevision/);
    assert.ok(appSource.includes("invalidateImportBinding('歌词已更改"), 'lyrics edits must invalidate the prompt and check state');
  });

  test('profile changes persist, discard stale request state and keep the pasted result', () => {
    const handler = appSource.slice(
      appSource.indexOf("$('llm-prompt-profile').addEventListener('change'"),
      appSource.indexOf("$('english-splitter').addEventListener('change'")
    );
    assert.ok(appSource.includes("storage.get('llm-prompt-profile')"));
    assert.ok(handler.includes("storage.set('llm-prompt-profile', preferredPromptProfile)"));
    assert.match(handler, /\+\+promptSequence; \+\+checkSequence/);
    assert.match(handler, /promptText = ''/);
    assert.ok(!handler.includes("$('llm-result').value ="));
    assert.ok(handler.includes('已保留当前结果文本'));
    assert.ok(handler.includes('只影响之后生成的提示词'));
  });

  test('the page carries the background-lyrics note for the lead route', () => {
    assert.ok(html.includes('id="bg-note"'), 'bg-note element missing');
  });

  test('the page keeps the optional vocals import and its explicit harmonies copy', () => {
    assert.ok(html.includes('id="vocals"'), 'optional vocals input missing');
    assert.ok(html.includes('保留主唱和和声'), 'the full-vocals option must spell out that harmonies are kept');
  });

  test('the lyrics picker accepts TTML and converts it locally', () => {
    assert.match(html, /id="lyrics_file"[^>]*accept="[^"]*\.ttml/);
    assert.ok(appSource.includes("fetch('/api/lyrics/ttml'"));
    assert.ok(appSource.includes("data.set('lyrics', xml)"));
    assert.ok(appSource.includes("headers: { 'X-Timing-Token': token }"));
    assert.match(appSource, /ttmlConverting \? '正在转换 TTML 歌词…'/);
    assert.match(appSource, /\$\('generate-prompt'\)\.disabled = busy \|\| promptBusy \|\| importing \|\| ttmlConverting/);
  });
});

describe('TTML conversion wiring', () => {
  test('a missing installed Japanese dictionary blocks submission and explains repair', async () => {
    const app = await makeAppHarness({ japanese_dictionary: { version: '2025.12', ready: false } });
    app.element('lyrics').value = '[00:01]青い箱[00:03]';
    app.element('audio').files = [{ name: 'song.wav', size: 100 }];
    await app.dispatch('lyrics', 'input');
    assert.equal(app.element('submit').disabled, true);
    assert.match(app.element('model-status').textContent, /日语词典未就绪/);
    assert.equal(app.element('retry-models').classList.contains('hidden'), false);
  });

  for (const [aligner, language, text] of [
    ['llm_text', 'ja', '青い箱'], ['llm_direct', 'ja', '青い箱'], ['llm_import', 'ja', '青い箱'],
    ['auto', 'en', 'blue box'], ['auto', 'zh', '蓝色的盒子'], ['auto', 'zh-en', '蓝色的 box'],
  ]) {
    test(`a missing Japanese dictionary allows ${aligner} / ${language} when no lexical Japanese reading is needed`, async () => {
      const app = await makeAppHarness({ ready: false, japanese_dictionary: { version: '2025.12', ready: false } });
      app.element('aligner').value = aligner;
      app.element('language').value = language;
      app.element('lyrics').value = `[00:01]${text}[00:03]`;
      app.element('audio').files = [{ name: 'song.wav', size: 100 }];
      app.element('llm-result').value = '{"lines":[]}';
      await app.dispatch('lyrics', 'input');
      assert.equal(app.element('submit').disabled, false);
      assert.doesNotMatch(app.element('model-status').textContent, /日语词典未就绪/);
    });
  }

  test('a pasted TTML conversion is requested once and a late reply cannot replace newer text', async () => {
    const app = await makeAppHarness();
    const response = deferred();
    app.setTtmlHandler(() => response.promise);
    const lyrics = app.element('lyrics');
    lyrics.value = TTML_XML;
    await app.dispatch('lyrics', 'input');

    assert.equal(app.element('lyrics-status').textContent, '正在转换 TTML 歌词…');
    assert.equal(app.timerCount(), 1);
    const conversion = app.runNextTimer();
    await app.settle();
    assert.equal(app.ttmlRequests.length, 1);
    assert.equal(app.ttmlRequests[0].body.get('lyrics'), TTML_XML);
    assert.equal(app.ttmlRequests[0].init.headers['X-Timing-Token'], 'test-token');

    await app.dispatch('syllables', 'change');
    assert.equal(app.ttmlRequests.length, 1, 'unrelated UI renders must not repeat the conversion');
    lyrics.value = '手动输入的新歌词';
    await app.dispatch('lyrics', 'input');
    response.resolve(fakeResponse(200, { lyrics: TTML_NORMALIZED, lines: 1 }));
    await conversion;
    await app.settle();

    assert.equal(lyrics.value, '手动输入的新歌词');
    assert.match(app.element('lyrics-status').textContent, /已识别纯文本 · 1 行/);
    assert.equal(app.element('lyrics-status').title, '');
  });

  test('clearing lyrics cancels a pending pasted TTML conversion', async () => {
    const app = await makeAppHarness();
    const response = deferred();
    app.setTtmlHandler(() => response.promise);
    app.element('lyrics').value = TTML_XML;
    await app.dispatch('lyrics', 'input');
    const conversion = app.runNextTimer();
    await app.settle();
    await app.element('clear-lyrics').dispatch('click');
    response.resolve(fakeResponse(200, { lyrics: TTML_NORMALIZED, lines: 1 }));
    await conversion;
    await app.settle();

    assert.equal(app.element('lyrics').value, '');
    assert.equal(app.element('lyrics-file-note').textContent, '');
  });

  test('a .ttml file is converted to normalized JSON before entering the textarea', async () => {
    const app = await makeAppHarness();
    app.setTtmlHandler(async () => fakeResponse(200, { lyrics: TTML_NORMALIZED, lines: 1 }));
    const bytes = new TextEncoder().encode(TTML_XML);
    app.element('lyrics_file').files = [{ name: 'Song.TTML', size: bytes.byteLength, arrayBuffer: async () => bytes.buffer }];
    app.element('lyrics').value = '旧的纯歌词';

    await app.dispatch('lyrics_file', 'change');

    assert.equal(app.ttmlRequests.length, 1);
    assert.equal(app.ttmlRequests[0].body.get('lyrics'), TTML_XML);
    assert.equal(app.element('lyrics').value, TTML_NORMALIZED);
    assert.equal(app.element('lyrics-file-note').textContent,
      '已从 Song.TTML 导入 TTML 并转换为逐句歌词（1 行）；文件内语言声明已忽略，提交以当前文本框为准。');
    assert.match(app.element('lyrics-status').textContent, /已识别逐句时间 · 1 行/);
  });

  test('the trust control follows anchors, model choice and converted TTML spans', async () => {
    const app = await makeAppHarness();
    app.setTtmlHandler(async () => fakeResponse(200, { lyrics: TTML_NORMALIZED, lines: 1 }));
    const lyricsInput = app.element('lyrics');
    const trust = app.element('trust-line-timestamps');

    lyricsInput.value = 'plain lyrics';
    await app.dispatch('lyrics', 'input');
    assert.equal(trust.disabled, true, 'plain text cannot skip MMS coarse localization');
    assert.equal(trust.checked, false);

    lyricsInput.value = '[00:01.00]First line\n[00:03.00]Second line';
    await app.dispatch('lyrics', 'input');
    assert.equal(trust.disabled, false, 'start-only LRC has inferred sentence spans');
    trust.checked = true;
    await app.dispatch('trust-line-timestamps', 'change');
    assert.equal(trust.checked, true);

    app.element('aligner').value = 'ctc';
    await app.dispatch('aligner', 'change');
    assert.equal(trust.disabled, true, 'standalone MMS always performs its own fine alignment');
    assert.equal(trust.checked, false);

    app.element('aligner').value = 'tifa';
    await app.dispatch('aligner', 'change');
    assert.equal(trust.disabled, false);
    assert.equal(trust.checked, false, 'switching through MMS clears the opt-in');

    lyricsInput.value = TTML_XML;
    await app.dispatch('lyrics', 'input');
    assert.equal(trust.disabled, true, 'raw TTML must be converted first');
    const conversion = app.runNextTimer();
    await conversion;
    await app.settle();
    assert.equal(lyricsInput.value, TTML_NORMALIZED);
    assert.equal(trust.disabled, false, 'converted TTML carries the required start and end spans');
    trust.checked = true;
    await app.dispatch('trust-line-timestamps', 'change');
    assert.equal(trust.checked, true);
  });

  test('a trusted TIFA route does not require a missing standalone MMS model', async () => {
    const app = await makeAppHarness({ models: { ctc: false, tifa: true, bs_lead: true, separator: true,
      singing: false, sofa: false } });
    app.element('audio').files = [{ name: 'song.wav', size: 100 }];
    app.element('lyrics').value = '[00:01.00]First line\n[00:03.00]Second line';
    await app.dispatch('lyrics', 'input');
    assert.equal(app.element('submit').disabled, true);
    assert.match(app.element('model-status').textContent, /MMS/);

    app.element('trust-line-timestamps').checked = true;
    await app.dispatch('trust-line-timestamps', 'change');
    assert.equal(app.element('submit').disabled, false, 'trusted TIFA must not require CTC readiness');
    assert.doesNotMatch(app.element('model-status').textContent, /MMS/);

    app.element('aligner').value = 'ctc';
    await app.dispatch('aligner', 'change');
    assert.equal(app.element('trust-line-timestamps').disabled, true);
    assert.equal(app.element('submit').disabled, true, 'standalone MMS still requires CTC');
    assert.match(app.element('model-status').textContent, /MMS/);
  });

  test('editing during a file read prevents the older file result from taking over', async () => {
    const app = await makeAppHarness();
    const bytes = deferred();
    app.setTtmlHandler(async () => fakeResponse(200, { lyrics: TTML_NORMALIZED, lines: 1 }));
    app.element('lyrics_file').files = [{ name: 'raced.ttml', size: 100, arrayBuffer: () => bytes.promise }];
    const importing = app.dispatch('lyrics_file', 'change');
    await app.settle();
    app.element('lyrics').value = '用户先写入的新歌词';
    await app.dispatch('lyrics', 'input');
    bytes.resolve(new TextEncoder().encode(TTML_XML).buffer);
    await importing;
    await app.settle();

    assert.equal(app.element('lyrics').value, '用户先写入的新歌词');
    assert.equal(app.ttmlRequests.length, 0, 'a stale file read must not call the conversion endpoint');
  });

  test('an HTTP 400 TTML error keeps prior text and surfaces the backend detail', async () => {
    const app = await makeAppHarness();
    app.setTtmlHandler(async () => fakeResponse(400, { detail: '不支持的 TTML 时间表达式' }));
    const bytes = new TextEncoder().encode('<tt><body/></tt>');
    app.element('lyrics_file').files = [{ name: 'bad.ttml', size: bytes.byteLength, arrayBuffer: async () => bytes.buffer }];
    app.element('lyrics').value = '保留的旧歌词';

    await app.dispatch('lyrics_file', 'change');

    assert.equal(app.element('lyrics').value, '保留的旧歌词');
    assert.equal(app.element('lyrics-status').textContent, '不支持的 TTML 时间表达式');
    assert.equal(app.element('lyrics-file-note').textContent, '');
  });
});

describe('queue workflow wiring', () => {
  const FIRST = 'a'.repeat(32);
  const SECOND = 'b'.repeat(32);
  const DONE = 'c'.repeat(32);
  const FAILED = 'd'.repeat(32);

  const makeJob = (id, state, overrides = {}) => ({
    id,
    state,
    name: id.slice(0, 1) + '.wav',
    original_filename: id.slice(0, 1) + '.wav',
    download_name: id.slice(0, 1) + '.ttml',
    queue_position: null,
    log: [],
    result: null,
    error: null,
    ...overrides,
  });
  const queueCard = (app, id) => app.element('queue-list').children
    .find((card) => card.attributes?.['data-job-id'] === id);
  const queueActions = (app, id) => {
    const card = queueCard(app, id);
    return (card?.children || []).find((child) => child.className === 'queue-actions')?.children || [];
  };
  const queueState = (app, id) => queueCard(app, id)?.children?.[0]?.children?.[1]?.textContent;

  test('an expanded running log survives repeated queue and detail polls and completion', async () => {
    let snapshot = [makeJob(FIRST, 'running', { name: 'first.wav', log: ['正在加载模型'] })];
    const app = await makeAppHarness({}, {}, {
      queue: async () => fakeResponse(200, {
        jobs: snapshot,
        running: snapshot.filter((job) => job.state === 'running').length,
        waiting: 0,
        limit: 50,
      }),
      detail: async (id) => fakeResponse(200, snapshot.find((job) => job.id === id) || makeJob(id, 'running')),
    });
    await app.settle();

    const logDetails = app.element('log-details');
    assert.equal(logDetails.open, false);
    logDetails.open = true;

    for (const log of ['正在分离人声', '正在对齐歌词']) {
      snapshot = [makeJob(FIRST, 'running', { name: 'first.wav', log: [log] })];
      const queueCount = app.queueRequests.length;
      const detailCount = app.detailRequests.length;

      await app.runNextTimer();
      await app.settle(30);

      assert.equal(app.queueRequests.length, queueCount + 1);
      assert.equal(app.detailRequests.length, detailCount + 1);
      assert.equal(app.element('log').textContent, log);
      assert.equal(logDetails.open, true);
    }

    snapshot = [makeJob(FIRST, 'complete', {
      name: 'first.wav', log: ['处理完成'], result: { lines: 1 },
    })];
    await app.runNextTimer();
    await app.settle(30);

    assert.equal(app.element('state').textContent, 'TTML 已生成');
    assert.equal(logDetails.open, true);
  });

  test('a failed log opens on first observation and stays closed after manual collapse', async () => {
    let snapshot = [
      makeJob(FIRST, 'running', { name: 'first.wav', log: ['正在处理'] }),
      makeJob(SECOND, 'queued', { name: 'second.wav', queue_position: 1 }),
    ];
    const app = await makeAppHarness({}, {}, {
      queue: async () => fakeResponse(200, {
        jobs: snapshot,
        running: snapshot.filter((job) => job.state === 'running').length,
        waiting: snapshot.filter((job) => job.state === 'queued').length,
        limit: 50,
      }),
      detail: async (id) => fakeResponse(200, snapshot.find((job) => job.id === id) || makeJob(id, 'running')),
    });
    await app.settle();

    const logDetails = app.element('log-details');
    assert.equal(logDetails.open, false);
    snapshot = [
      makeJob(FIRST, 'failed', { name: 'first.wav', log: ['模型失败'], error: '音源读取失败' }),
      makeJob(SECOND, 'running', { name: 'second.wav' }),
    ];
    await app.runNextTimer();
    await app.settle(30);

    assert.equal(queueState(app, FIRST), '处理失败');
    assert.equal(logDetails.open, true, 'the first failed snapshot must open the log');
    logDetails.open = false;

    for (let poll = 0; poll < 2; poll++) {
      const queueCount = app.queueRequests.length;
      const detailCount = app.detailRequests.length;
      await app.runNextTimer();
      await app.settle(30);

      assert.equal(app.queueRequests.length, queueCount + 1);
      assert.equal(app.detailRequests.length, detailCount + 1);
      assert.equal(logDetails.open, false, 'a user-collapsed failure must stay collapsed');
    }
  });

  test('switching jobs restores each log choice and ignores a late detail reply', async () => {
    const firstDetail = deferred();
    let holdFirstDetail = true;
    const jobs = [
      makeJob(FIRST, 'running', { name: 'first.wav' }),
      makeJob(SECOND, 'complete', { name: 'second.wav', result: { lines: 1 } }),
    ];
    const app = await makeAppHarness({}, {}, {
      queue: async () => fakeResponse(200, { jobs, running: 1, waiting: 0, limit: 50 }),
      detail: (id) => {
        const job = jobs.find((candidate) => candidate.id === id);
        if (id === FIRST && holdFirstDetail) {
          holdFirstDetail = false;
          return firstDetail.promise.then(() => fakeResponse(200, job));
        }
        return Promise.resolve(fakeResponse(200, job));
      },
    });
    await app.settle();

    const logDetails = app.element('log-details');
    assert.equal(logDetails.open, false);
    logDetails.open = true;

    await queueActions(app, SECOND)[0].dispatch('click');
    await app.settle();
    assert.equal(app.element('job-info').textContent, 'second.wav · ' + SECOND.slice(0, 8));
    assert.equal(logDetails.open, false);
    logDetails.open = true;

    firstDetail.resolve();
    await app.settle(30);
    assert.equal(app.element('job-info').textContent, 'second.wav · ' + SECOND.slice(0, 8));
    assert.equal(logDetails.open, true, 'a late response for the old job must not replace the current choice');

    await queueActions(app, FIRST)[0].dispatch('click');
    await app.settle();
    assert.equal(app.element('job-info').textContent, 'first.wav · ' + FIRST.slice(0, 8));
    assert.equal(logDetails.open, true);

    logDetails.open = false;
    await queueActions(app, SECOND)[0].dispatch('click');
    await app.settle();
    assert.equal(logDetails.open, true, 'the second job remembers its own expanded choice');

    logDetails.open = false;
    await queueActions(app, FIRST)[0].dispatch('click');
    await app.settle();
    assert.equal(logDetails.open, false, 'the first job remembers its own collapsed choice');
  });

  test('each upload unlocks the form and sends its own audio, lyrics and options', async () => {
    const jobs = [];
    const app = await makeAppHarness({}, {}, {
      queue: async () => fakeResponse(200, {
        jobs,
        running: jobs.filter((job) => job.state === 'running').length,
        waiting: 0,
        limit: 50,
      }),
      submit: async (init) => {
        const audio = init.body.get('audio');
        const job = makeJob(jobs.length ? SECOND : FIRST, 'running', {
          name: audio.name,
          original_filename: audio.name,
        });
        jobs.push(job);
        return fakeResponse(200, job);
      },
      detail: async (id) => fakeResponse(200, jobs.find((job) => job.id === id) || makeJob(id, 'running')),
    });

    app.element('audio').files = [new File(['first'], 'first song.flac', { type: 'audio/flac' })];
    app.element('lyrics').value = 'Run into the light';
    await app.dispatch('lyrics', 'input');
    assert.equal(app.element('submit').disabled, false);
    await app.dispatch('form', 'submit');
    await app.settle();

    assert.equal(app.submitRequests.length, 1);
    const first = app.submitRequests[0].body;
    assert.equal(first.get('audio').name, 'first song.flac');
    assert.equal(first.get('lyrics'), 'Run into the light');
    assert.equal(first.get('language'), 'mixed');
    assert.equal(first.get('aligner'), 'auto');
    assert.equal(first.get('audio_mode'), 'lead');
    assert.equal(app.element('submit').disabled, false);
    assert.equal(app.element('submit-label').textContent, '加入队列');
    assert.equal(app.element('task-inputs').disabled, false);
    assert.equal(queueState(app, FIRST), '正在处理…');
    assert.match(app.element('queue-message').textContent, /可以继续添加下一首/);

    app.element('audio').files = [new File(['second'], '第二首.測試.mp3', { type: 'audio/mpeg' })];
    app.element('lyrics').value = 'Walking home tonight';
    app.element('language').value = 'en';
    app.element('aligner').value = 'tifa';
    app.element('audio-mode').value = 'vocals';
    app.element('syllables').checked = false;
    await app.dispatch('form', 'submit');
    await app.settle();

    assert.equal(app.submitRequests.length, 2);
    const second = app.submitRequests[1].body;
    assert.equal(second.get('audio').name, '第二首.測試.mp3');
    assert.equal(second.get('lyrics'), 'Walking home tonight');
    assert.equal(second.get('language'), 'en');
    assert.equal(second.get('aligner'), 'tifa');
    assert.equal(second.get('audio_mode'), 'vocals');
    assert.equal(second.get('syllables'), 'false');
    assert.equal(queueState(app, SECOND), '正在处理…');
    assert.equal(app.element('queue-summary').textContent, '2 处理中 · 0 等待 · 上限 50');
  });

  test('the trust opt-in starts fresh and snapshots separately for each queued song', async () => {
    const jobs = [];
    const app = await makeAppHarness({}, { 'timing-trust-line-timestamps': 'true' }, {
      queue: async () => fakeResponse(200, {
        jobs,
        running: jobs.filter((job) => job.state === 'running').length,
        waiting: jobs.filter((job) => job.state === 'queued').length,
        limit: 50,
      }),
      submit: async (init) => {
        const audio = init.body.get('audio');
        const job = makeJob(jobs.length ? SECOND : FIRST, jobs.length ? 'queued' : 'running', {
          name: audio.name,
          original_filename: audio.name,
        });
        jobs.push(job);
        return fakeResponse(200, job);
      },
      detail: async (id) => fakeResponse(200, jobs.find((job) => job.id === id) || makeJob(id, 'running')),
    });
    const trust = app.element('trust-line-timestamps');
    app.element('audio').files = [new File(['first'], 'first.flac', { type: 'audio/flac' })];
    app.element('lyrics').value = '[00:01.00]First line\n[00:03.00]Second line';
    await app.dispatch('lyrics', 'input');
    assert.equal(trust.checked, false, 'trust must not be restored from browser storage');

    trust.checked = true;
    await app.dispatch('trust-line-timestamps', 'change');
    await app.dispatch('form', 'submit');
    await app.settle();
    assert.equal(app.submitRequests[0].body.get('trust_line_timestamps'), 'true');

    app.element('audio').files = [new File(['second'], 'second.mp3', { type: 'audio/mpeg' })];
    trust.checked = false;
    await app.dispatch('trust-line-timestamps', 'change');
    await app.dispatch('form', 'submit');
    await app.settle();
    assert.equal(app.submitRequests.length, 2);
    assert.equal(app.submitRequests[0].body.get('trust_line_timestamps'), 'true', 'the first FormData must stay unchanged');
    assert.equal(app.submitRequests[1].body.get('trust_line_timestamps'), 'false');
  });

  test('page boot restores every current job and replaces the list with the server snapshot', async () => {
    let snapshot = [
      makeJob(FIRST, 'running', { name: 'first.wav' }),
      makeJob(SECOND, 'queued', { name: 'second.wav', queue_position: 1 }),
    ];
    let polls = 0;
    const app = await makeAppHarness({}, {}, {
      queue: async () => {
        polls += 1;
        if (polls > 1) snapshot = [snapshot[1]];
        return fakeResponse(200, {
          jobs: snapshot,
          running: polls === 1 ? 1 : 0,
          waiting: 1,
          limit: 50,
        });
      },
      detail: async (id) => fakeResponse(200, snapshot.find((job) => job.id === id) || makeJob(id, 'queued')),
    });
    await app.settle();

    assert.equal(app.element('queue-list').children.length, 2);
    assert.equal(queueState(app, FIRST), '正在处理…');
    assert.equal(queueState(app, SECOND), '等待 · 第 1 位');
    assert.equal(app.element('queue-summary').textContent, '1 处理中 · 1 等待 · 上限 50');

    await app.runNextTimer();
    await app.settle(30);
    assert.equal(app.element('queue-list').children.length, 1);
    assert.equal(queueCard(app, FIRST), undefined);
    assert.ok(queueCard(app, SECOND));
    assert.equal(app.element('queue-summary').textContent, '0 处理中 · 1 等待 · 上限 50');
  });

  test('a late detail response cannot replace a newer selection', async () => {
    const replies = new Map([[FIRST, deferred()], [SECOND, deferred()]]);
    const app = await makeAppHarness({}, {}, {
      queue: async () => fakeResponse(200, {
        jobs: [makeJob(FIRST, 'complete'), makeJob(SECOND, 'complete')],
        running: 0,
        waiting: 0,
        limit: 50,
      }),
      detail: (id) => replies.get(id)?.promise || Promise.resolve(fakeResponse(404, { detail: 'missing' })),
    });
    await app.settle();
    assert.ok(app.detailRequests.some((request) => request.id === FIRST));

    await queueActions(app, SECOND)[0].dispatch('click');
    await app.settle();
    assert.ok(app.detailRequests.some((request) => request.id === SECOND));

    replies.get(SECOND).resolve(fakeResponse(200, makeJob(SECOND, 'complete', {
      name: 'second.wav', download_name: 'second.ttml',
    })));
    await app.settle(30);
    assert.equal(app.element('job-info').textContent, 'second.wav · ' + SECOND.slice(0, 8));
    assert.equal(app.element('links').children[0].href, '/api/jobs/' + SECOND + '/files/ttml');

    replies.get(FIRST).resolve(fakeResponse(200, makeJob(FIRST, 'complete', { name: 'first.wav' })));
    await app.settle(30);
    assert.equal(app.element('job-info').textContent, 'second.wav · ' + SECOND.slice(0, 8));
    assert.match(queueCard(app, SECOND).className, /\bselected\b/);
  });

  test('only a queued job offers cancellation, and cancelling it leaves other cards alone', async () => {
    const running = makeJob(FIRST, 'running', { name: 'running.wav' });
    const done = makeJob(DONE, 'complete', { name: 'done.wav' });
    let queued = makeJob(SECOND, 'queued', { name: 'queued.wav', queue_position: 1 });
    let snapshot = [queued, running, done];
    const app = await makeAppHarness({}, {}, {
      queue: async () => fakeResponse(200, {
        jobs: snapshot,
        running: 1,
        waiting: snapshot.filter((job) => job.state === 'queued').length,
        limit: 50,
      }),
      cancel: async (id) => {
        assert.equal(id, SECOND);
        queued = makeJob(SECOND, 'cancelled', { name: 'queued.wav', log: ['已从等待队列移除。'] });
        snapshot = [queued, running, done];
        return fakeResponse(200, queued);
      },
      detail: async (id) => fakeResponse(200, snapshot.find((job) => job.id === id) || makeJob(id, 'cancelled')),
    });
    await app.settle();

    const cancel = queueActions(app, SECOND).find((action) => action.className === 'queue-cancel');
    assert.ok(cancel);
    assert.equal(queueActions(app, FIRST).some((action) => action.className === 'queue-cancel'), false);
    assert.equal(queueActions(app, DONE).some((action) => action.className === 'queue-cancel'), false);

    await cancel.dispatch('click');
    await app.settle(30);

    assert.equal(app.cancelRequests.length, 1);
    assert.equal(app.cancelRequests[0].id, SECOND);
    assert.equal(queueState(app, SECOND), '已取消');
    assert.equal(queueActions(app, SECOND).some((action) => action.className === 'queue-cancel'), false);
    assert.equal(queueState(app, FIRST), '正在处理…');
    assert.equal(queueState(app, DONE), '已完成');
    assert.equal(app.element('queue-summary').textContent, '1 处理中 · 0 等待 · 上限 50');
  });

  test('TTML links use the exact Unicode and dotted download_name from the server', async () => {
    const downloadName = '夜の歌.ver.2.ttml';
    const job = makeJob(FIRST, 'complete', {
      name: '夜の歌.ver.2',
      original_filename: '夜の歌.ver.2.mp3',
      download_name: downloadName,
    });
    const app = await makeAppHarness({}, {}, {
      queue: async () => fakeResponse(200, { jobs: [job], running: 0, waiting: 0, limit: 50 }),
      detail: async () => fakeResponse(200, job),
    });
    await app.settle(20);

    const queueLink = queueActions(app, FIRST).find((action) => action.textContent === '下载 TTML');
    assert.ok(queueLink);
    assert.equal(queueLink.download, downloadName);
    assert.equal(queueLink.href, '/api/jobs/' + FIRST + '/files/ttml');
    assert.notEqual(queueLink.download, 'lyrics.ttml');

    const detailLink = app.element('links').children.find((action) => action.textContent === '下载 TTML');
    assert.ok(detailLink);
    assert.equal(detailLink.download, downloadName);
  });

  test('a failed song leaves the rest queued, and a queue network error still retries', async () => {
    const running = makeJob(FIRST, 'running', { name: 'running.wav' });
    const failed = makeJob(FAILED, 'failed', { name: 'failed.wav', error: '音源读取失败' });
    const queued = makeJob(SECOND, 'queued', { name: 'queued.wav', queue_position: 1 });
    let snapshot = [running, failed, queued];
    let failNext = false;
    const app = await makeAppHarness({}, {}, {
      queue: async () => {
        if (failNext) { failNext = false; throw new Error('网络中断'); }
        return fakeResponse(200, {
          jobs: snapshot,
          running: snapshot.filter((job) => job.state === 'running').length,
          waiting: snapshot.filter((job) => job.state === 'queued').length,
          limit: 50,
        });
      },
      detail: async (id) => fakeResponse(200, snapshot.find((job) => job.id === id) || makeJob(id, 'running')),
    });
    await app.settle();

    assert.equal(app.element('queue-list').children.length, 3);
    assert.equal(queueState(app, FAILED), '处理失败');
    assert.equal(queueCard(app, FAILED).children.find((child) => child.className === 'queue-error').textContent,
      '音源读取失败');
    assert.equal(queueState(app, SECOND), '等待 · 第 1 位');
    assert.ok(queueActions(app, SECOND).some((action) => action.className === 'queue-cancel'));
    assert.equal(app.element('queue-summary').textContent, '1 处理中 · 1 等待 · 上限 50');

    failNext = true;
    await app.runNextTimer();
    await app.settle(30);
    assert.equal(app.element('queue-list').children.length, 3);
    assert.match(app.element('queue-message').textContent, /网络中断，正在重试/);
    assert.equal(app.element('queue-message').classList.contains('warning'), true);
    assert.ok(app.timerCount() >= 1);

    snapshot = [makeJob(FIRST, 'complete', { name: 'running.wav' }), failed,
      makeJob(SECOND, 'complete', { name: 'queued.wav' })];
    await app.runNextTimer();
    await app.settle(30);
    assert.equal(app.element('queue-message').textContent, '');
    assert.equal(app.element('queue-message').classList.contains('warning'), false);
    assert.equal(app.element('queue-summary').textContent, '0 处理中 · 0 等待 · 上限 50');
    assert.equal(queueState(app, FAILED), '处理失败');
    assert.equal(app.timerCount(), 0);
  });
});

function defineContractTests(api) {
  const analyze = (raw, language) => language === undefined ? api.analyzeLyrics(raw) : api.analyzeLyrics(raw, language);
  const analyzeLlm = (raw) => api.analyzeLyrics(raw, 'zh-en', true);
  const resolve = (overrides) => api.resolveOptions(preferences(overrides));

  describe('analyzeLyrics', () => {
    test('empty input is a blocking error, not a parse failure', () => {
      const info = analyze('');
      assert.equal(info.empty, true);
      assert.equal(info.error, '未填写歌词：请粘贴或导入歌词。');
      assert.equal(info.count, 0);
      assert.equal(info.anchored, false);
      assert.equal(info.partial, false);
      assert.equal(info.hasEnglish, false);
      assert.equal(info.hasJapanese, false);
      assert.equal(info.hasChinese, false);
      assert.ok(!('hasKorean' in info));
      assert.equal(info.hasBackground, false);
    });

    test('any leading angle-bracket input is delegated instead of counted as lyric tags', () => {
      for (const source of [TTML_XML, '<!-- timing -->\n<tt xmlns="http://www.w3.org/ns/ttml"><body/></tt>',
        '<tt/>', '<tt', '<html>', '\uFEFF\n  <tt']) {
        const info = analyze(source);
        assert.equal(info.empty, false);
        assert.equal(info.kind, 'ttml');
        assert.equal(info.needsTtmlConversion, true);
        assert.equal(info.count, 0);
        assert.equal(info.anchored, false);
        assert.equal(info.hasJapanese, false);
        assert.match(info.error, /TTML.*转换/);
      }
      assert.equal(api.resolveOptions(preferences({ lyrics: analyze(TTML_XML), aligner: 'auto' })).valid, false);
    });

    test('plain Japanese text is text and unanchored', () => {
      const info = analyze(JA_TEXT);
      assert.equal(info.empty, false);
      assert.equal(info.error, '');
      assert.equal(info.kind, 'text');
      assert.equal(info.count, 3);
      assert.equal(info.anchored, false);
      assert.equal(info.partial, false);
      assert.equal(info.hasJapanese, true);
      assert.equal(info.hasEnglish, false);
    });

    test('plain English text is text and unanchored', () => {
      const info = analyze(EN_TEXT);
      assert.equal(info.kind, 'text');
      assert.equal(info.count, 2);
      assert.equal(info.hasEnglish, true);
      assert.equal(info.hasJapanese, false);
    });

    test('mixed Japanese and English lines report both languages', () => {
      const info = analyze(MIXED_TEXT);
      assert.equal(info.hasJapanese, true);
      assert.equal(info.hasEnglish, true);
      assert.equal(info.count, 2);
    });

    test('written script does not add a preflight language restriction', () => {
      for (const source of [KO_TEXT, KO_TEXT.normalize('NFD'), 'ㅎㅏㄴㄱㅡㄹ', '\u{ffa1}']) {
        const info = analyze(source);
        assert.equal(info.error, '');
        assert.ok(!('hasKorean' in info), 'Korean must not remain a positive capability field');
      }
    });

    test('a full LRC is anchored with both languages detected', () => {
      const info = analyze(FULL_LRC);
      assert.equal(info.error, '');
      assert.equal(info.kind, 'lrc');
      assert.equal(info.count, 2);
      assert.equal(info.anchored, true);
      assert.equal(info.partial, false);
      assert.equal(info.hasJapanese, true);
      assert.equal(info.hasEnglish, true);
    });

    test('a dual-tag LRC keeps explicit line ends, including the final line', () => {
      const info = analyze([
        '[00:17.250]青い箱　ひらく[00:21.120]',
        '[00:21.120]小さな紙の船[00:25.100]',
        '[00:25.100]ゆっくり流れてく[00:28.970]',
      ].join('\n'));
      assert.equal(info.error, '');
      assert.equal(info.kind, 'lrc');
      assert.equal(info.count, 3);
      assert.equal(info.anchored, true);
      assert.equal(info.partial, false);
      assert.ok(Math.abs(info.maxKnownLineSeconds - 3.98) < 1e-9, String(info.maxKnownLineSeconds));
    });

    test('explicit ends may leave gaps, overlap the next start and finish the song', () => {
      const info = analyze([
        '[00:01.00]First[00:02.00]',
        '[00:05.00]Second[00:07.00]',
        '[00:06.00]Inferred',
        '[00:09.00]Final[00:12.00]',
      ].join('\n'));
      assert.equal(info.error, '');
      assert.equal(info.count, 4);
      assert.equal(info.anchored, true);
      assert.equal(info.partial, false);
      assert.equal(info.maxKnownLineSeconds, 3);
    });

    test('an inferred end needs a later start while explicit equal starts can overlap', () => {
      assert.ok(analyze('[00:01.00]First\n[00:01.00]Second').error);
      const info = analyze('[00:01.00]First[00:02.00]\n[00:01.00]Second[00:03.00]');
      assert.equal(info.error, '');
      assert.equal(info.count, 2);
      assert.equal(info.anchored, true);
    });

    test('LRC strips only timestamp-like end tags and keeps generic bracket text', () => {
      const info = analyze('[00:01.00][Chorus]');
      assert.equal(info.error, '');
      assert.equal(info.count, 1);
      assert.equal(info.hasEnglish, true);
    });

    test('blank timestamp rows remain ignored', () => {
      const info = analyze('[00:01.00]\n[00:02.00]Hello');
      assert.equal(info.error, '');
      assert.equal(info.count, 1);
      assert.equal(info.anchored, true);
    });

    test('extra LRC timestamps and word tags are rejected', () => {
      for (const raw of [
        '[00:01.00][00:02.00]Text',
        '[00:01.00]Hello [00:02.00]world',
        '[00:01.00]Hello<00:02.00>world',
        '[00:01.00]Hello[00:02.000',
        '[00:01.00]Hello[00:02.bad]',
        '[00:01.00]Hello[-00:02.00]',
      ]) assert.ok(analyze(raw).error, raw);
    });

    test('LRC stamps reject nonfinite values and nonpositive ranges', () => {
      const huge = '9'.repeat(400);
      for (const raw of [
        `[${huge}:00.000]Hello`,
        `[00:01.000]Hello[${huge}:00.000]`,
        '[00:05.000]Hello[00:05.000]',
        '[00:05.000]Hello[00:04.999]',
      ]) assert.ok(analyze(raw).error, raw);
    });

    test('a partly stamped LRC is partial', () => {
      const info = analyze(PARTIAL_LRC);
      assert.equal(info.count, 2);
      assert.equal(info.partial, true);
    });

    test('LRC metadata tags are not lyric lines', () => {
      const info = analyze(META_LRC);
      assert.equal(info.kind, 'lrc');
      assert.equal(info.count, 2);
      assert.equal(info.anchored, true);
      assert.equal(info.partial, false);
    });

    test('a metadata-only LRC reports no alignable lyrics', () => {
      const info = analyze(META_ONLY_LRC);
      assert.equal(info.count, 0);
      assert.equal(info.error, '没有可对齐的歌词，请填写歌词。');
    });

    test('malformed JSON is reported instead of read as lyrics', () => {
      const info = analyze(BROKEN_JSON);
      assert.ok(info.error, 'expected a JSON format error');
      assert.equal(info.count, 0);
    });

    test('JSON objects and invalid optional metadata are rejected before upload', () => {
      for (const raw of ['{"text":"Hello"}', '{broken', '[{"text":"Hello","role":null}]', '[{"text":"Hello","translation":false}]']) {
        assert.ok(analyze(raw).error, raw);
      }
    });

    test('JSON singer IDs accept v3, v1000 and the nine-digit boundary', () => {
      const raw = JSON.stringify([
        { text: 'First', start: 0, end: 1, agent: 'v3' },
        { text: 'Second', start: 1, end: 2, agent: 'v1000' },
        { text: 'Third', start: 2, end: 3, agent: 'v999999999' },
      ]);
      const info = analyze(raw);
      assert.equal(info.error, '');
      assert.equal(info.anchored, true);
      assert.equal(info.count, 3);
    });

    test('JSON singer IDs reject zero-leading, malformed and overlong values with the allowed format', () => {
      for (const agent of ['v0', 'v01', 'v', 'V3', 'v3x', 'v1000000000', 3, null]) {
        const info = analyze(JSON.stringify([{ text: 'Line', start: 0, end: 1, agent }]));
        assert.match(info.error, /歌手 ID 无效.*v1.*v3.*v1000/);
      }
    });

    test('main JSON rows must be chronological while overlapping singing stays supported', () => {
      assert.ok(analyze('[{"text":"Later","start":5,"end":7},{"text":"Earlier","start":1,"end":3}]').error);
      assert.equal(analyze('[{"text":"First","start":0,"end":5},{"text":"Overlap","start":2,"end":4}]').error, '');
    });

    test('MMS accepts long known sentence ranges and globally ordered pure text', () => {
      for (const raw of [
        '[{"text":"Long line","start":0,"end":31}]',
        '[00:00]Long line\n[00:31]Next',
        '[00:00]Long line[00:31]',
      ]) {
        const info = analyze(raw);
        assert.equal(info.error, '');
        assert.equal(resolve({ lyrics: info, aligner: 'ctc' }).valid, true);
        assert.equal(resolve({ lyrics: info, aligner: 'tifa' }).valid, true);
      }
      const boundary = analyze('[00:00]Thirty seconds[00:30]');
      assert.equal(boundary.error, '');
      assert.equal(resolve({ lyrics: boundary, aligner: 'ctc' }).valid, true);
      const plain = analyze('Run into the light\nWe are moving on');
      assert.equal(resolve({ lyrics: plain, aligner: 'ctc' }).valid, true);
    });

    test('JSON rows need a valid start and end pair', () => {
      assert.ok(analyze(JSON_MISSING_END).error, 'a start without an end is incomplete');
      assert.ok(analyze(JSON_REVERSED_TIMES).error, 'end must come after start');
    });

    test('JSON without times is valid but unanchored', () => {
      const info = analyze(JSON_WITHOUT_TIMES);
      assert.equal(info.error, '');
      assert.equal(info.kind, 'json');
      assert.equal(info.count, 2);
      assert.equal(info.anchored, false);
      assert.equal(info.partial, false);
    });

    test('a JSON line array is json and carries its own times', () => {
      const info = analyze(JSON_LYRICS);
      assert.equal(info.error, '');
      assert.equal(info.kind, 'json');
      assert.equal(info.count, 2);
      assert.equal(info.anchored, true);
      assert.equal(info.partial, false);
      assert.equal(info.hasJapanese, true);
      assert.equal(info.hasEnglish, true);
    });

    test('JSON role bg lines set hasBackground', () => {
      const info = analyze(JSON_BACKGROUND);
      assert.equal(info.kind, 'json');
      assert.equal(info.hasBackground, true);
      assert.equal(info.count, 2);
    });

    test('background rows still require their own explicit start and end', () => {
      const info = analyze('[{"text":"Main","start":0,"end":2},' +
        '{"text":"oh","role":"bg","parent":0}]');
      assert.match(info.error, /背景歌词需带起止时间/);
      assert.equal(info.anchored, false);
    });

    test('text and JSON without background lines do not set hasBackground', () => {
      assert.equal(analyze(JA_TEXT).hasBackground, false);
      assert.equal(analyze(FULL_LRC).hasBackground, false);
      assert.equal(analyze(JSON_LYRICS).hasBackground, false);
    });

    test('two time tags on one LRC line is an error, not a lyric', () => {
      const info = analyze(DOUBLE_STAMP_LRC);
      assert.ok(info.error, 'expected a non-empty error for an unexpandable LRC line');
    });

    test('the default language keeps the historical JA/EN reading, Han included', () => {
      const japanese = analyze(JA_TEXT);
      assert.equal(japanese.language, 'mixed');
      assert.equal(japanese.hasJapanese, true);
      assert.equal(japanese.hasChinese, false);
      const hanOnly = analyze(ZH_TEXT);
      assert.equal(hanOnly.hasJapanese, true, 'Han alone still reads as Japanese under mixed');
      assert.equal(hanOnly.hasChinese, false);
      assert.deepStrictEqual([...hanOnly.effectiveLanguages], ['mixed', 'mixed']);
    });

    test('zh mode reads Han text as Mandarin Chinese', () => {
      const info = analyze(ZH_TEXT, 'zh');
      assert.equal(info.language, 'zh');
      assert.equal(info.error, '');
      assert.equal(info.hasChinese, true);
      assert.equal(info.hasJapanese, false);
      assert.equal(info.hasEnglish, false);
      assert.deepStrictEqual([...info.effectiveLanguages], ['zh', 'zh']);
    });

    test('zh-en separates Chinese from English and keeps kana Japanese', () => {
      const info = analyze(ZH_EN_TEXT, 'zh-en');
      assert.equal(info.hasChinese, true);
      assert.equal(info.hasEnglish, true);
      assert.equal(info.hasJapanese, false);
      const kana = analyze('今天很好\nまた明日', 'zh-en');
      assert.equal(kana.hasChinese, true);
      assert.equal(kana.hasJapanese, true);
    });

    test('explicit JSON row languages override the song language', () => {
      const info = analyze('[{"text":"今天很好","language":"zh"},{"text":"We are moving on","language":"en"}]', 'mixed');
      assert.equal(info.hasChinese, true);
      assert.equal(info.hasEnglish, true);
      assert.equal(info.hasJapanese, false);
      assert.deepStrictEqual([...info.effectiveLanguages], ['zh', 'en']);
    });

    test('auto rows inherit the song language and unknown row values are rejected', () => {
      const info = analyze('[{"text":"今天很好"},{"text":"We are moving on"}]', 'zh-en');
      assert.deepStrictEqual([...info.effectiveLanguages], ['zh-en', 'zh-en']);
      assert.equal(info.hasChinese, true);
      assert.equal(info.hasEnglish, true);
      assert.equal(analyze('[{"text":"Hello","language":"auto"}]').error, '');
      assert.equal(analyze('[{"text":"빛"}]').error, '');
      assert.ok(analyze('[{"text":"Hello","language":"xx"}]').error, 'unknown row languages must be rejected');
    });

    test('an unknown language argument falls back to mixed and auto follows mixed', () => {
      assert.equal(analyze(JA_TEXT, 'garbage').language, 'mixed');
      assert.equal(analyze(JA_TEXT, 'garbage').hasJapanese, true);
      assert.equal(analyze(JA_TEXT, 'auto').language, 'auto');
      assert.equal(analyze(JA_TEXT, 'auto').hasJapanese, true);
      assert.equal(analyze(JA_TEXT, 'auto').hasChinese, false);
    });

    test('LLM routes leave language pending without inferring Han before the model responds', () => {
      const info = analyzeLlm('今天很好\nまた明日\nHello');
      assert.equal(info.language, 'auto');
      assert.equal(info.hasChinese, false);
      assert.equal(info.hasJapanese, false);
      assert.equal(info.hasEnglish, false);
      assert.deepStrictEqual([...info.effectiveLanguages], ['auto', 'auto', 'auto']);
      const rows = analyzeLlm('[{"text":"今天很好","language":"zh","start":0,"end":1},' +
        '{"text":"Run into the light","language":"en","start":1,"end":2}]');
      assert.equal(rows.hasChinese, false);
      assert.equal(rows.hasEnglish, false);
      assert.deepStrictEqual([...rows.effectiveLanguages], ['auto', 'auto']);
    });

    test('LLM routes leave the input language decision to the model', () => {
      const info = analyzeLlm('[{"text":"빛","start":0,"end":1}]');
      assert.equal(info.error, '');
      assert.deepStrictEqual([...info.effectiveLanguages], ['auto']);
    });

    test('invalid row-language metadata keeps the shared JSON schema on LLM routes', () => {
      const info = analyzeLlm('[{"text":"Hello","language":"xx","start":0,"end":1}]');
      assert.match(info.error, /角色或语言标记无效/);
    });
  });

  describe('resolveOptions: empty lyrics', () => {
    test('empty lyrics block submission and normalize every helper off', () => {
      const result = resolve({
        lyrics: lyrics({ empty: true, kind: '', count: 0, anchored: false, hasJapanese: false, hasEnglish: false }),
        aligner: 'tifa',
      });
      assert.equal(result.valid, false);
      assert.ok(result.error);
      assert.equal(result.engine, 'tifa');
      assert.equal(result.llm, false);
      assert.equal(result.sendsAudio, false);
      assert.deepStrictEqual([...result.models], REQUIRED_MODELS);
    });

    test('empty input with a saved llm_direct is refused end to end', () => {
      const result = api.resolveOptions(preferences({
        lyrics: analyze(''),
        aligner: 'llm_direct',
        splitter: 'llm',
      }));
      assert.equal(result.valid, false);
      assert.ok(result.error);
      assert.equal(result.llm, false);
      assert.equal(result.sendsAudio, false);
    });
  });

  describe('resolveOptions: engine choice', () => {
    test('auto with a full LRC resolves to tifa', () => {
      const result = resolve({ aligner: 'auto', lyrics: lyrics({ anchored: true, partial: false }) });
      assert.equal(result.valid, true);
      assert.equal(result.engine, 'tifa');
    });

    test('auto with plain text resolves to tifa', () => {
      const result = resolve({ aligner: 'auto', lyrics: lyrics({ kind: 'txt', anchored: false, partial: false }) });
      assert.equal(result.valid, true);
      assert.equal(result.engine, 'tifa');
    });

    test('explicit aligners keep their own engine', () => {
      const anchored = lyrics({ anchored: true, partial: false });
      assert.equal(resolve({ aligner: 'tifa', lyrics: anchored }).engine, 'tifa');
      assert.equal(resolve({ aligner: 'ctc', lyrics: anchored }).engine, 'ctc');
      assert.equal(resolve({ aligner: 'llm_text', lyrics: anchored }).engine, 'llm_text');
      assert.equal(resolve({ aligner: 'llm_direct', lyrics: anchored }).engine, 'llm_direct');
      assert.equal(resolve({ aligner: 'llm_import', lyrics: anchored }).engine, 'llm_import');
    });

    test('plain text is valid for TIFA and MMS while every LLM mode keeps anchor guards', () => {
      const plain = lyrics({ kind: 'txt', anchored: false, partial: false });
      const tifa = resolve({ aligner: 'tifa', lyrics: plain });
      assert.equal(tifa.valid, true);
      assert.equal(tifa.engine, 'tifa');
      assert.equal(tifa.error, '');
      const mms = resolve({ aligner: 'ctc', lyrics: plain });
      assert.equal(mms.valid, true);
      assert.equal(mms.engine, 'ctc');
      assert.equal(mms.error, '');
      for (const aligner of ['llm_text', 'llm_direct', 'llm_import']) {
        const result = resolve({ aligner: aligner, lyrics: plain });
        assert.equal(result.valid, false, aligner + ' must not be submitted without anchors');
        assert.match(result.error, /逐句时间/, aligner + ' must explain why it is unavailable');
      }
    });

    test('a stale sofa preference normalizes to automatic TIFA', () => {
      const result = resolve({ aligner: 'sofa', lyrics: lyrics({ kind: 'txt', anchored: false, partial: false }) });
      assert.equal(result.aligner, 'auto');
      assert.equal(result.valid, true);
      assert.equal(result.engine, 'tifa');
      assert.ok(!result.error);
    });

    test('partial anchors are rejected before the job is submitted', () => {
      for (const aligner of ['auto', 'tifa', 'llm_import']) {
        const result = resolve({ aligner: aligner, lyrics: lyrics({ anchored: false, partial: true }) });
        assert.equal(result.valid, false, aligner + ' must reject partly anchored lyrics');
        assert.ok(result.error);
      }
    });
  });

  describe('resolveOptions: trusted sentence timestamps', () => {
    test('is opt-in for fully anchored TIFA-backed routes and removes only MMS readiness', () => {
      for (const aligner of ['auto', 'tifa', 'llm_text', 'llm_direct', 'llm_import']) {
        const plan = resolve({ aligner, trustLineTimestamps: true });
        assert.equal(plan.trustEligible, true, aligner);
        assert.equal(plan.trustLineTimestamps, true, aligner);
        assert.deepStrictEqual([...plan.models], REQUIRED_MODELS.filter(name => name !== 'ctc'), aligner);
        assert.equal(api.submissionFields(plan).trust_line_timestamps, 'true', aligner);
      }

      const defaultPlan = resolve({ aligner: 'tifa' });
      assert.equal(defaultPlan.trustLineTimestamps, false);
      assert.deepStrictEqual([...defaultPlan.models], REQUIRED_MODELS);
      assert.equal(api.submissionFields(defaultPlan).trust_line_timestamps, 'false');

      const plain = resolve({ aligner: 'tifa', trustLineTimestamps: true,
        lyrics: lyrics({ kind: 'txt', anchored: false, partial: false }) });
      assert.equal(plain.trustEligible, false);
      assert.equal(plain.trustLineTimestamps, false);
      assert.equal(api.submissionFields(plain).trust_line_timestamps, 'false');

      const mms = resolve({ aligner: 'ctc', trustLineTimestamps: true });
      assert.equal(mms.trustEligible, false);
      assert.equal(mms.trustLineTimestamps, false);
      assert.deepStrictEqual([...mms.models], REQUIRED_MODELS);
      assert.equal(api.submissionFields(mms).trust_line_timestamps, 'false');
    });

    test('accepts start-only LRC and converted start/end TTML, never raw or incomplete TTML', () => {
      const startOnlyLrc = analyze(FULL_LRC);
      assert.equal(startOnlyLrc.anchored, true);
      assert.equal(resolve({ lyrics: startOnlyLrc, trustLineTimestamps: true }).trustLineTimestamps, true);

      const rawTtml = analyze(TTML_XML);
      assert.equal(rawTtml.needsTtmlConversion, true);
      assert.equal(resolve({ lyrics: rawTtml, trustLineTimestamps: true }).trustLineTimestamps, false);

      const convertedTtml = analyze(TTML_NORMALIZED);
      assert.equal(convertedTtml.anchored, true);
      assert.equal(resolve({ lyrics: convertedTtml, trustLineTimestamps: true }).trustLineTimestamps, true);

      const incompleteTtml = analyze(JSON_MISSING_END);
      assert.ok(incompleteTtml.error);
      assert.equal(resolve({ lyrics: incompleteTtml, trustLineTimestamps: true }).trustLineTimestamps, false);
    });
  });

  describe('resolveOptions: LLM text mode', () => {
    test('llm_text discloses the whole lyrics text without claiming an audio request', () => {
      const result = resolve({
        aligner: 'llm_text',
        language: 'ja',
        lyrics: lyrics({ anchored: true, partial: false, hasEnglish: true }),
      });
      assert.equal(api.disclosureText(result), '全部逐行歌词会发送到接口，不附带预设歌曲或行语言信息或音频；由模型自行判断语言。下载后先试听校对。');
      assert.doesNotMatch(api.disclosureText(result), /全部已校验段落/);
      assert.doesNotMatch(api.disclosureText(result), /英文词和所在句/);
    });

    test('every LLM route submits automatic language without a Korean proxy disclosure', () => {
      for (const aligner of ['llm_text', 'llm_direct', 'llm_import']) {
        const result = resolve({
          aligner,
          lyrics: lyrics({ anchored: true, partial: false }),
        });
        const disclosure = api.disclosureText(result);
        assert.equal(api.submissionFields(result).language, 'auto', aligner);
        assert.doesNotMatch(disclosure, /韩/, aligner);
        assert.match(disclosure, /试听校对/, aligner);
        assert.doesNotMatch(disclosure, /全部已校验段落/, aligner);
      }
    });

    test('llm_text hides the selected song language, submits auto and never sends audio', () => {
      const result = resolve({
        aligner: 'llm_text',
        language: 'zh',
        lyrics: lyrics({ anchored: true, partial: false, hasEnglish: true }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.engine, 'llm_text');
      assert.equal(result.valid, true);
      assert.equal(result.language, 'auto');
      assert.equal(result.showLanguage, false);
      assert.equal(result.showEnglish, false);
      assert.equal(result.showSplitter, false);
      assert.equal(result.llm, true);
      assert.equal(result.sendsAudio, false);
      assert.equal(result.showPromptProfile, false);
      assert.deepStrictEqual({ ...api.submissionFields(result) }, {
        language: 'auto', aligner: 'llm_text', audio_mode: 'lead',
        syllables: 'false', english_splitter: 'syllabify', trust_line_timestamps: 'false',
      });
    });

    test('llm_text requires fully anchored lyrics', () => {
      const result = resolve({ aligner: 'llm_text', lyrics: lyrics({ kind: 'txt', anchored: false, partial: false }) });
      assert.equal(result.valid, false);
      assert.match(result.error, /逐句时间/);
    });

    test('llm_direct hides the local splitter options and uses automatic language', () => {
      const result = resolve({
        aligner: 'llm_direct',
        lyrics: lyrics({ anchored: true, partial: false }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.engine, 'llm_direct');
      assert.equal(result.showEnglish, false);
      assert.equal(result.showSplitter, false);
      assert.equal(result.llm, true);
      assert.equal(result.sendsAudio, true);
      assert.equal(result.language, 'auto');
      assert.equal(result.showLanguage, false);
      assert.equal(result.showPromptProfile, true);
      assert.equal(result.promptProfile, 'fine');
      assert.deepStrictEqual({ ...api.submissionFields(result) }, {
        language: 'auto',
        aligner: 'llm_direct',
        audio_mode: 'lead',
        syllables: 'false',
        english_splitter: 'syllabify',
        trust_line_timestamps: 'false',
        llm_prompt_profile: 'fine',
      });
    });

    test('web import is direct-like but never requires or sends API settings', () => {
      const result = resolve({
        aligner: 'llm_import',
        lyrics: lyrics({ anchored: true, partial: false }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.engine, 'llm_import');
      assert.equal(result.valid, true);
      assert.equal(result.showImport, true);
      assert.equal(result.showPromptProfile, true);
      assert.equal(result.promptProfile, 'fine');
      assert.equal(result.promptReady, true);
      assert.equal(result.language, 'auto');
      assert.equal(result.showLanguage, false);
      assert.equal(result.showEnglish, false);
      assert.equal(result.showSplitter, false);
      assert.equal(result.llm, false);
      assert.equal(result.sendsAudio, false);
      assert.deepStrictEqual({ ...api.submissionFields(result) }, {
        language: 'auto',
        aligner: 'llm_import',
        audio_mode: 'lead',
        syllables: 'false',
        english_splitter: 'syllabify',
        trust_line_timestamps: 'false',
        llm_prompt_profile: 'fine',
      });
      assert.equal(resolve({ aligner: 'llm_import', lyrics: lyrics({ anchored: false, kind: 'txt' }) }).promptReady, false);
    });

    test('prompt profiles are visible only for direct-like routes and normalize to fine', () => {
      const direct = resolve({ aligner: 'llm_direct' });
      const imported = resolve({ aligner: 'llm_import', promptProfile: 'original' });
      assert.equal(direct.showPromptProfile, true);
      assert.equal(direct.promptProfile, 'fine');
      assert.equal(imported.showPromptProfile, true);
      assert.equal(imported.promptProfile, 'original');
      assert.equal(api.submissionFields(imported).llm_prompt_profile, 'original');
      assert.equal(resolve({ aligner: 'llm_direct', promptProfile: 'unknown' }).promptProfile, 'fine');
      for (const aligner of ['auto', 'tifa', 'ctc', 'llm_text']) {
        const result = resolve({ aligner, promptProfile: 'original' });
        assert.equal(result.showPromptProfile, false, aligner + ' must hide the prompt profile');
        assert.ok(!('llm_prompt_profile' in api.submissionFields(result)), aligner + ' must not submit a prompt profile');
      }
    });
  });

  describe('resolveOptions: Mandarin and language plumbing', () => {
    const lookup = loadPureFile(UI_LOGIC);

    test('the resolved plan echoes the provided song language', () => {
      assert.equal(resolve({ language: 'zh' }).language, 'zh');
      assert.equal(resolve({ language: 'zh-en' }).language, 'zh-en');
      assert.equal(resolve({ language: 'ko' }).language, 'mixed');
      assert.equal(resolve({ language: 'ko-en' }).language, 'mixed');
      assert.equal(resolve({}).language, 'mixed');
      assert.equal(resolve({ language: 'garbage' }).language, 'mixed');
    });

    test('Chinese rows do not switch on the English syllable settings', () => {
      const result = resolve({
        language: 'zh',
        aligner: 'tifa',
        lyrics: lyrics({ anchored: true, partial: false, hasEnglish: false, hasJapanese: false, hasChinese: true }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.showEnglish, false);
      assert.equal(result.showSplitter, false);
      assert.equal(result.llm, false);
    });

    test('describeLanguages names the shipped combinations', () => {
      const describeLanguages = lookup('describeLanguages');
      assert.equal(describeLanguages({ hasChinese: true, hasJapanese: false, hasEnglish: false }), '中文');
      assert.equal(describeLanguages({ hasChinese: false, hasJapanese: true, hasEnglish: false }), '日语');
      assert.equal(describeLanguages({ hasChinese: false, hasJapanese: false, hasEnglish: true }), '英语');
      assert.equal(describeLanguages({ hasChinese: true, hasJapanese: false, hasEnglish: true }), '中英混唱');
      assert.equal(describeLanguages({ hasChinese: false, hasJapanese: true, hasEnglish: true }), '日英混唱');
      assert.equal(describeLanguages({ hasChinese: true, hasJapanese: true, hasEnglish: false }), '中日混唱');
      assert.equal(describeLanguages({ hasChinese: true, hasJapanese: true, hasEnglish: true }), '中日英混唱');
      assert.equal(describeLanguages({ hasKorean: true }), '');
      assert.equal(describeLanguages({ language: 'auto', effectiveLanguages: ['auto'] }), '待 LLM 识别语言');
      assert.equal(describeLanguages({}), '');
    });

    test('unknown language values normalize to mixed without a Korean route', () => {
      const result = resolve({ language: 'ko', aligner: 'tifa', lyrics: lyrics({ hasEnglish: true }) });
      assert.equal(result.valid, true);
      assert.equal(result.language, 'mixed');
      assert.equal(result.engine, 'tifa');
      assert.equal(api.submissionFields(result).aligner, 'tifa');
      assert.equal(api.submissionFields(result).language, 'mixed');
      assert.ok(!('hasKorean' in result));
    });

    test('submission fields preserve local routing and force every LLM route to auto', () => {
      const normal = resolve({ language: 'zh-en', aligner: 'auto' });
      assert.equal(normal.language, 'zh-en');
      assert.equal(normal.showLanguage, true);
      assert.equal(api.submissionFields(normal).aligner, 'auto');
      assert.equal(api.submissionFields(normal).language, 'zh-en');

      const unknown = resolve({ language: 'ko-en', aligner: 'auto' });
      assert.equal(api.submissionFields(unknown).aligner, 'auto');
      assert.equal(api.submissionFields(unknown).language, 'mixed');

      for (const aligner of ['llm_text', 'llm_direct', 'llm_import']) {
        const llm = resolve({ language: 'zh', aligner });
        assert.equal(llm.language, 'auto', aligner);
        assert.equal(llm.showLanguage, false, aligner);
        assert.equal(api.submissionFields(llm).language, 'auto', aligner);
        assert.equal(api.submissionFields({ ...llm, language: 'zh' }).language, 'auto', aligner + ' submission guard');
      }
    });

  });

  describe('resolveOptions: English syllable settings', () => {
    test('no English means no English settings and no API requirement', () => {
      const result = resolve({
        aligner: 'auto',
        lyrics: lyrics({ kind: 'txt', anchored: false, partial: false, hasEnglish: false, hasJapanese: true }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.showEnglish, false);
      assert.equal(result.showSplitter, false);
      assert.equal(result.llm, false);
      assert.equal(result.sendsAudio, false);
      assert.equal(result.valid, true);
    });

    test('English lyrics with the llm splitter require the text API but no audio', () => {
      const result = resolve({
        aligner: 'tifa',
        lyrics: lyrics({ anchored: true, partial: false, hasEnglish: true }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.showEnglish, true);
      assert.equal(result.showSplitter, true);
      assert.equal(result.allowLlmSplitter, true);
      assert.equal(result.llm, true);
      assert.equal(result.sendsAudio, false);
    });

    test('turning syllables off drops the splitter without needing the API', () => {
      const result = resolve({
        aligner: 'tifa',
        lyrics: lyrics({ anchored: true, partial: false, hasEnglish: true }),
        syllables: false,
        splitter: 'llm',
      });
      assert.equal(result.showEnglish, true);
      assert.equal(result.showSplitter, false);
      assert.equal(result.llm, false);
      assert.equal(result.sendsAudio, false);
    });

    test('MMS cannot use the llm splitter, so it falls back to syllabify', () => {
      const result = resolve({
        aligner: 'ctc',
        lyrics: lyrics({ anchored: true, partial: false, hasEnglish: true }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.valid, true, 'the fallback must still be a submittable combination');
      assert.equal(result.splitter, 'syllabify');
      assert.equal(result.allowLlmSplitter, false);
      assert.equal(result.llm, false);
      assert.equal(result.sendsAudio, false);
    });

    test('plain TIFA keeps language, syllable and text-API controls in sync', () => {
      const result = resolve({
        language: 'en',
        aligner: 'tifa',
        lyrics: lyrics({ kind: 'txt', anchored: false, partial: false, hasEnglish: true, hasJapanese: false }),
        syllables: true,
        splitter: 'llm',
      });
      assert.equal(result.valid, true);
      assert.equal(result.language, 'en');
      assert.equal(result.showLanguage, true);
      assert.equal(result.showEnglish, true);
      assert.equal(result.showSplitter, true);
      assert.equal(result.allowLlmSplitter, true);
      assert.equal(result.splitter, 'llm');
      assert.equal(result.llm, true);
      assert.equal(result.sendsAudio, false);
    });
  });

  describe('resolveOptions: audio route and required models', () => {
    test('audioMode defaults to lead and unknown values normalize to lead', () => {
      assert.equal(resolve({}).audioMode, 'lead');
      assert.equal(resolve({ audioMode: 'garbage' }).audioMode, 'lead');
      assert.equal(resolve({ audioMode: 'vocals' }).audioMode, 'vocals');
      assert.equal(resolve({ audioMode: 'source' }).audioMode, 'source');
    });

    test('source mirrors timing/sources.py for every mode and stem combination', () => {
      assert.equal(resolve({ audioMode: 'lead', hasVocals: false }).source, 'separated_vocals');
      assert.equal(resolve({ audioMode: 'lead', hasVocals: true }).source, 'separated_vocals');
      assert.equal(resolve({ audioMode: 'vocals', hasVocals: false }).source, 'separated_vocals');
      assert.equal(resolve({ audioMode: 'vocals', hasVocals: true }).source, 'provided_vocals');
      assert.equal(resolve({ audioMode: 'source', hasVocals: false }).source, 'original');
      assert.equal(resolve({ audioMode: 'source', hasVocals: true }).source, 'provided_vocals');
    });

    test('every state lists exactly the four required model groups in order', () => {
      const states = [
        {},
        { aligner: 'tifa' },
        { aligner: 'ctc', syllables: false },
        { aligner: 'sofa', audioMode: 'source' },
        { aligner: 'llm_direct' },
        { aligner: 'llm_import' },
        { hasVocals: true, audioMode: 'vocals' },
      ];
      for (const overrides of states) {
        const result = resolve(overrides);
        assert.deepStrictEqual([...result.models], REQUIRED_MODELS, JSON.stringify(overrides) + ' must require all four groups');
      }
      assert.notEqual(resolve({}).models, resolve({}).models, 'each result must carry a fresh models array');
    });

    test('the background flag passes through for the lead-route note', () => {
      assert.equal(resolve({ lyrics: lyrics({ hasBackground: true }) }).hasBackground, true);
      assert.equal(resolve({}).hasBackground, false);
    });
  });

  describe('resolveOptions: result plumbing', () => {
    test('every documented field is present with the right type', () => {
      const result = resolve({});
      const flags = ['showLanguage', 'valid', 'showEnglish', 'showSplitter', 'allowLlmSplitter', 'syllables',
        'llm', 'sendsAudio', 'showImport',
        'showPromptProfile', 'promptReady', 'trustEligible', 'trustLineTimestamps', 'hasBackground'];
      for (const key of flags) {
        assert.equal(typeof result[key], 'boolean', key + ' must be a boolean');
      }
      assert.equal(typeof result.engine, 'string');
      assert.ok(result.engine.length > 0, 'engine must name the resolved aligner');
      assert.equal(typeof result.language, 'string');
      assert.equal(typeof result.splitter, 'string');
      assert.equal(typeof result.promptProfile, 'string');
      assert.equal(typeof result.source, 'string');
      assert.equal(typeof result.audioMode, 'string');
      assert.equal(typeof result.error, 'string');
      assert.ok(Array.isArray(result.models), 'models must be an array');
      for (const model of result.models) {
        assert.equal(typeof model, 'string');
      }
    });

    test('draft-era fields are gone from the result', () => {
      const result = resolve({});
      for (const key of ['draft', 'direct', 'separate', 'lead', 'directLead', 'audioReading', 'audioSplit', 'showAudioReading', 'showAudioSplit']) {
        assert.ok(!(key in result), key + ' must not exist any more');
      }
    });

    test('the saved preferences object is never modified', () => {
      const input = preferences({ aligner: 'ctc', syllables: true, splitter: 'llm' });
      const snapshot = structuredClone(input);
      const result = api.resolveOptions(input);
      assert.deepStrictEqual(input, snapshot, 'resolveOptions must not write back into the form state');
      assert.equal(input.splitter, 'llm', 'the user preference is kept even when it is not usable');
      assert.notEqual(result, input);
    });

    test('a frozen preferences object can be resolved without throwing', () => {
      const input = Object.freeze(preferences({ aligner: 'ctc', splitter: 'llm', hasVocals: true }));
      const result = api.resolveOptions(input);
      assert.equal(result.valid, true);
    });
  });

  describe('resolveOptions with real analyzeLyrics output', () => {
    test('mixed anchored LRC on auto runs on tifa with the text API', () => {
      const info = analyze(FULL_LRC);
      const result = api.resolveOptions(preferences({
        lyrics: info,
        aligner: 'auto',
        syllables: true,
        splitter: 'llm',
        hasVocals: false,
        audioMode: 'lead',
      }));
      assert.equal(result.valid, true);
      assert.equal(result.engine, 'tifa');
      assert.equal(result.llm, true);
      assert.equal(result.sendsAudio, false);
      assert.equal(result.source, 'separated_vocals');
      assert.deepStrictEqual([...result.models], REQUIRED_MODELS);
    });

    test('the same text pasted without times runs on tifa with the text API only', () => {
      const info = analyze(MIXED_TEXT);
      const result = api.resolveOptions(preferences({ lyrics: info, aligner: 'auto', splitter: 'llm' }));
      assert.equal(result.valid, true);
      assert.equal(result.engine, 'tifa');
      assert.equal(result.llm, true);
      assert.equal(result.sendsAudio, false);
    });

    test('a partly anchored LRC is refused before submission', () => {
      const info = analyze(PARTIAL_LRC);
      const result = api.resolveOptions(preferences({ lyrics: info, aligner: 'auto' }));
      assert.equal(result.valid, false);
      assert.ok(result.error);
    });

    test('a JSON background track keeps the four groups and reports hasBackground', () => {
      const info = analyze(JSON_BACKGROUND);
      assert.equal(info.hasBackground, true);
      const result = api.resolveOptions(preferences({ lyrics: info, aligner: 'tifa' }));
      assert.equal(result.hasBackground, true);
      assert.deepStrictEqual([...result.models], REQUIRED_MODELS);
    });

    test('empty input with a saved llm_direct is blocked instead of drafted', () => {
      const info = analyze('');
      const result = api.resolveOptions(preferences({
        lyrics: info,
        aligner: 'llm_direct',
        splitter: 'llm',
      }));
      assert.equal(result.valid, false);
      assert.ok(result.error);
      assert.equal(result.llm, false);
      assert.equal(result.sendsAudio, false);
      assert.deepStrictEqual([...result.models], REQUIRED_MODELS);
    });
  });
}

if (loadError) {
  test('web/ui-logic.js exposes analyzeLyrics and resolveOptions', () => {
    throw loadError;
  });
} else {
  defineContractTests(loadedApi);
}

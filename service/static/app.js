'use strict';

/*
 * Интерфейс проверяющего. Чистый JS без сборки.
 *
 * Контракт API (не менять):
 *   POST /api/v1/analyze   multipart/form-data, поле file; заголовок X-API-Key  -> {job_id, status}
 *   GET  /api/v1/jobs/{id} заголовок X-API-Key                                  -> статус + результат
 */

const API = '/api/v1';
const MAX_MB = 200;                       // лимит сервера (MAX_UPLOAD_MB); проверяем до отправки
const RETRY_DELAYS_MS = [2000, 5000, 10000];
const POLL_MS = 2000;
const GIVE_UP_MS = 10 * 60 * 1000;        // сервер обязан ответить за 5 минут; 10 — с запасом
const HISTORY_MAX = 5;

const LS = { key: 'kk_api_key', active: 'kk_active_job', history: 'kk_history' };

const $ = (id) => document.getElementById(id);
const dom = {
  status: $('status'),
  apiKey: $('apiKey'), apiKeyToggle: $('apiKeyToggle'),
  uploadCard: $('uploadCard'), drop: $('drop'), fileInput: $('fileInput'), chooseBtn: $('chooseBtn'),
  dropEmpty: $('dropEmpty'), dropFile: $('dropFile'), fileName: $('fileName'), fileSize: $('fileSize'),
  fileRemove: $('fileRemove'), uploadNote: $('uploadNote'), analyzeBtn: $('analyzeBtn'),
  progressCard: $('progressCard'), progressBar: $('progressBar'), progressFill: $('progressFill'),
  progressStage: $('progressStage'), progressPct: $('progressPct'), progressElapsed: $('progressElapsed'),
  progressJob: $('progressJob'),
  errorCard: $('errorCard'), errorTitle: $('errorTitle'), errorText: $('errorText'), errorNewBtn: $('errorNewBtn'),
  result: $('result'), verdict: $('verdict'), verdictTitle: $('verdictTitle'), verdictText: $('verdictText'),
  downloadBtn: $('downloadBtn'), newBtn: $('newBtn'),
  eventsSection: $('eventsSection'), events: $('events'),
  historyCard: $('historyCard'), history: $('history'),
  box: $('box'), boxImg: $('boxImg'), boxCaption: $('boxCaption'),
  boxClose: $('boxClose'), boxPrev: $('boxPrev'), boxNext: $('boxNext'),
};

const state = {
  file: null,
  busy: false,
  job: null,              // {job_id, filename, started_at}
  pollTimer: null,
  elapsedTimer: null,
  retry: 0,
  resultText: null,       // сырой текст ответа — «Скачать JSON» отдаёт его байт в байт
  result: null,
  gallery: { frames: [], index: 0, opener: null },
};

// ---------------------------------------------------------------- утилиты

function fmtSize(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} КБ`;
  return `${(bytes / 1024 / 1024).toFixed(1)} МБ`;
}
function fmtMmSs(sec) {
  const s = Math.max(0, Math.floor(sec));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
function fmtTimecode(sec) {
  const s = Math.max(0, Number(sec) || 0);
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, '0')}:${(s - m * 60).toFixed(1).padStart(4, '0')}`;
}
function fmtWhen(ts) {
  const d = new Date(ts);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}
function lsGet(key, fallback) {
  try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); }
  catch (e) { return fallback; }
}
function lsSet(key, value) {
  try { if (value == null) localStorage.removeItem(key); else localStorage.setItem(key, JSON.stringify(value)); }
  catch (e) { /* localStorage может быть недоступен — интерфейс работает без него */ }
}
function setStatus(text) { dom.status.textContent = text || ''; }

// ---------------------------------------------------------------- ключ

function apiKey() { return dom.apiKey.value.trim(); }

function initKey() {
  dom.apiKey.value = lsGet(LS.key, '');
  dom.apiKey.addEventListener('input', () => lsSet(LS.key, apiKey()));
  dom.apiKeyToggle.addEventListener('click', () => {
    const show = dom.apiKey.type === 'password';
    dom.apiKey.type = show ? 'text' : 'password';
    dom.apiKeyToggle.setAttribute('aria-pressed', String(show));
    dom.apiKeyToggle.setAttribute('aria-label', show ? 'Скрыть ключ' : 'Показать ключ');
  });
}

// ---------------------------------------------------------------- файл

function isVideo(file) {
  const n = file.name.toLowerCase();
  return n.endsWith('.mov') || n.endsWith('.mp4');
}

function note(text) { dom.uploadNote.textContent = text || ''; }

function setFile(file) {
  if (!file) return;
  if (!isVideo(file)) { note('Нужен видеофайл .mov или .mp4.'); return; }
  if (file.size > MAX_MB * 1024 * 1024) {
    note(`Файл ${fmtSize(file.size)} больше ${MAX_MB} МБ — сервер его отклонит (код 413). Файл не отправлен.`);
    return;
  }
  state.file = file;
  dom.fileName.textContent = file.name;
  dom.fileSize.textContent = fmtSize(file.size);
  dom.dropEmpty.hidden = true;
  dom.dropFile.hidden = false;
  note('');
  dom.analyzeBtn.disabled = state.busy;
  dom.analyzeBtn.focus();
}

function clearFile() {
  state.file = null;
  dom.fileInput.value = '';
  dom.dropEmpty.hidden = false;
  dom.dropFile.hidden = true;
  dom.analyzeBtn.disabled = true;
}

function initFile() {
  dom.chooseBtn.addEventListener('click', () => dom.fileInput.click());
  dom.drop.addEventListener('click', (e) => {
    if (state.busy || e.target.closest('button')) return;
    if (!state.file) dom.fileInput.click();
  });
  dom.fileInput.addEventListener('change', () => setFile(dom.fileInput.files && dom.fileInput.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => dom.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    if (!state.busy) dom.drop.classList.add('drop--drag');
  }));
  ['dragleave', 'drop'].forEach((ev) => dom.drop.addEventListener(ev, (e) => {
    e.preventDefault();
    dom.drop.classList.remove('drop--drag');
  }));
  dom.drop.addEventListener('drop', (e) => {
    if (state.busy) return;
    setFile(e.dataTransfer.files && e.dataTransfer.files[0]);
  });
  dom.fileRemove.addEventListener('click', () => { clearFile(); dom.chooseBtn.focus(); });
}

// ---------------------------------------------------------------- ошибки

class ApiError extends Error {
  constructor(title, text, fatal = true) { super(title); this.title = title; this.text = text; this.fatal = fatal; }
}

function describeHttp(status, bodyText) {
  let detail = '';
  try { detail = JSON.parse(bodyText).detail || ''; } catch (e) { detail = ''; }
  if (status === 401) return new ApiError('Неверный API-ключ', 'Сервер ответил 401. Проверьте ключ в поле вверху страницы и повторите.');
  if (status === 400) return new ApiError('Файл отклонён', `Сервер ответил 400: ${detail || 'ожидается видеофайл .mov'}.`);
  if (status === 413) return new ApiError('Файл слишком большой', `Сервер ответил 413: ${detail || `файл больше ${MAX_MB} МБ`}.`);
  if (status === 404) return new ApiError('Задание не найдено', 'Сервер ответил 404: задание с таким номером не найдено — возможно, сервис был перезапущен.');
  return new ApiError('Ошибка сервера', `Код ответа ${status}${detail ? `: ${detail}` : ''}.`, status < 500);
}

function showError(title, text) {
  hideProgress();
  dom.errorTitle.textContent = title;
  dom.errorText.textContent = text || '';
  dom.errorCard.hidden = false;
  setStatus(`${title}. ${text || ''}`);
  dom.errorNewBtn.focus();
}

// ---------------------------------------------------------------- отправка

async function startAnalysis() {
  if (!state.file || state.busy) return;
  if (!apiKey()) {
    note('Введите API-ключ в поле вверху страницы.');
    dom.apiKey.focus();
    return;
  }
  dom.errorCard.hidden = true;
  hideResult();
  setBusy(true);
  showProgress({ progress: 0, stage: 'отправка файла' });
  setStatus('Отправка файла…');

  const body = new FormData();
  body.append('file', state.file);
  try {
    const r = await fetch(`${API}/analyze`, { method: 'POST', headers: { 'X-API-Key': apiKey() }, body });
    if (!r.ok) throw describeHttp(r.status, await r.text());
    const data = await r.json();
    state.job = { job_id: data.job_id, filename: state.file.name, started_at: Date.now() };
    lsSet(LS.active, state.job);
    addHistory({ ...state.job, verdict: null });
    dom.progressJob.textContent = data.job_id;
    startPolling();
  } catch (e) {
    setBusy(false);
    if (e instanceof ApiError) showError(e.title, e.text);
    else showError('Нет соединения с сервером', 'Файл не отправлен. Проверьте сеть и повторите.');
  }
}

// ---------------------------------------------------------------- опрос

function startPolling() {
  stopPolling();
  state.retry = 0;
  tickElapsed();
  state.elapsedTimer = setInterval(tickElapsed, 1000);
  poll();
}

function stopPolling() {
  clearTimeout(state.pollTimer);
  clearInterval(state.elapsedTimer);
  state.pollTimer = null;
  state.elapsedTimer = null;
}

function tickElapsed() {
  if (!state.job) return;
  dom.progressElapsed.textContent = fmtMmSs((Date.now() - state.job.started_at) / 1000);
}

function schedule(ms) { state.pollTimer = setTimeout(poll, ms); }

async function poll() {
  if (!state.job) return;
  if (Date.now() - state.job.started_at > GIVE_UP_MS) {
    finish();
    showError('Сервер не ответил', 'Обработка не завершилась за 10 минут. Задание можно открыть позже из списка последних заданий.');
    return;
  }
  let r;
  let text;
  try {
    r = await fetch(`${API}/jobs/${encodeURIComponent(state.job.job_id)}`, { headers: { 'X-API-Key': apiKey() } });
    text = await r.text();
  } catch (e) {
    return reconnect();                                  // сеть: пробуем снова 2 → 5 → 10 с
  }
  if (r.status >= 500) return reconnect();
  if (!r.ok) {
    finish();
    const err = describeHttp(r.status, text);
    showError(err.title, err.text);
    return;
  }
  reconnected();
  let job;
  try { job = JSON.parse(text); } catch (e) { return reconnect(); }

  if (job.status === 'done') {
    finish();
    state.resultText = text;
    state.result = job;
    updateHistory(job.job_id, { verdict: job.low_coverage ? 'low_coverage' : job.verdict });
    renderResult(job);
  } else if (job.status === 'failed') {
    finish();
    updateHistory(job.job_id, { verdict: 'failed' });
    showError('Обработка завершилась с ошибкой', job.error || 'Сервер не указал причину.');
  } else {
    showProgress(job);
    schedule(POLL_MS);
  }
}

function reconnect() {
  const delay = RETRY_DELAYS_MS[Math.min(state.retry, RETRY_DELAYS_MS.length - 1)];
  state.retry += 1;
  dom.progressCard.classList.add('progress--reconnect');
  dom.progressStage.textContent = `переподключение… (повтор через ${delay / 1000} с)`;
  setStatus('Нет связи с сервером, переподключение.');
  schedule(delay);
}

function reconnected() {
  if (state.retry) setStatus('Связь восстановлена.');
  state.retry = 0;
  dom.progressCard.classList.remove('progress--reconnect');
}

function finish() {
  stopPolling();
  setBusy(false);
  hideProgress();
  lsSet(LS.active, null);
}

// ---------------------------------------------------------------- экраны

function setBusy(busy) {
  state.busy = busy;
  dom.drop.classList.toggle('drop--busy', busy);
  dom.analyzeBtn.disabled = busy || !state.file;
  dom.chooseBtn.disabled = busy;
  dom.fileRemove.disabled = busy;
}

function showProgress(job) {
  const pct = Math.round(Math.min(1, Math.max(0, Number(job.progress) || 0)) * 100);
  dom.progressCard.hidden = false;
  dom.progressFill.style.width = `${pct}%`;
  dom.progressBar.setAttribute('aria-valuenow', String(pct));
  dom.progressPct.textContent = `${pct} %`;
  const stage = job.stage || 'обработка';
  if (dom.progressStage.textContent !== stage) {
    dom.progressStage.textContent = stage;
    setStatus(`${stage[0].toUpperCase()}${stage.slice(1)}, ${pct} %.`);
  }
  if (state.job) dom.progressJob.textContent = state.job.job_id;
}

function hideProgress() {
  dom.progressCard.hidden = true;
  dom.progressCard.classList.remove('progress--reconnect');
}

function hideResult() {
  dom.result.hidden = true;
  dom.eventsSection.hidden = true;
  dom.events.replaceChildren();
  state.result = null;
  state.resultText = null;
}

function resetAll() {
  stopPolling();
  state.job = null;
  lsSet(LS.active, null);
  dom.errorCard.hidden = true;
  hideResult();
  hideProgress();
  setBusy(false);
  clearFile();
  setStatus('');
  renderHistory();
  dom.chooseBtn.focus();
}

// ---------------------------------------------------------------- результат

function renderResult(job) {
  dom.result.hidden = false;
  const violations = Array.isArray(job.violations) ? job.violations : [];
  const dur = job.video && typeof job.video.duration_sec === 'number' ? `Длительность видео ${fmtMmSs(job.video.duration_sec)}.` : '';
  const frames = job.frames_analyzed != null && job.frames_total_sampled != null
    ? `Разобрано кадров: ${job.frames_analyzed} из ${job.frames_total_sampled}.` : '';

  dom.verdict.className = 'verdict';
  if (job.low_coverage) {
    dom.verdict.classList.add('verdict--warn');
    dom.verdictTitle.textContent = 'Анализ неполный';
    dom.verdictText.textContent = `Сервис успел разобрать слишком мало кадров (лимит времени), вердикт недостоверен — повторите загрузку позже. ${frames} ${dur}`.trim();
  } else if (job.verdict === 'violation') {
    dom.verdict.classList.add('verdict--bad');
    dom.verdictTitle.textContent = violations.length === 1
      ? 'Обнаружено нарушение'
      : `Обнаружены нарушения: ${violations.length}`;
    dom.verdictText.textContent = `Каждое событие ниже — с таймкодом, кадрами и пунктом норматива. Окончательную оценку даёт проверяющий. ${dur}`.trim();
  } else {
    dom.verdict.classList.add('verdict--ok');
    dom.verdictTitle.textContent = 'Нарушений не обнаружено';
    dom.verdictText.textContent = `Признаков нарушений в ролике не найдено. ${frames} ${dur}`.trim();
  }
  setStatus(`Готово. ${dom.verdictTitle.textContent}.`);

  if (violations.length) {
    dom.eventsSection.hidden = false;
    dom.events.replaceChildren(...violations.map(renderEvent));
  }
  dom.downloadBtn.focus();
}

function renderEvent(v, i) {
  const li = el('li', 'event');
  li.appendChild(el('div', 'event__time', fmtTimecode(v.timecode_sec)));
  li.appendChild(el('h3', 'event__type', v.type || 'Событие'));
  if (v.description) li.appendChild(el('p', 'event__desc', v.description));
  if (v.rule && v.rule.doc_title) li.appendChild(renderRule(v.rule));
  const frames = Array.isArray(v.evidence_frames) ? v.evidence_frames : [];
  if (frames.length) li.appendChild(renderFrames(frames, i));
  return li;
}

function renderRule(rule) {
  const box = el('div', 'rule');
  const head = el('div', 'rule__head');
  head.appendChild(el('span', null, rule.clause ? `Норматив, ${rule.clause}` : 'Норматив'));
  if (rule.verified === false) head.appendChild(el('span', 'badge', 'трактовка'));
  box.appendChild(head);
  const doc = [rule.doc_title, rule.doc_requisites, rule.edition].filter(Boolean).join(' — ');
  box.appendChild(el('p', 'rule__doc', doc));
  if (rule.quote) box.appendChild(el('blockquote', 'rule__quote', `«${rule.quote}»`));
  if (rule.verified === false) {
    // Пояснение из нормативной базы остаётся в JSON; в интерфейсе — только бейдж
    // и одна строка: служебный текст с именами полей проверяющему не нужен.
    box.appendChild(el('p', 'rule__note', 'Трактовка, а не дословный запрет: оценку даёт проверяющий.'));
  }
  if (rule.url) {
    const a = el('a', 'rule__link', 'Текст акта на adilet.zan.kz');
    a.href = rule.url; a.target = '_blank'; a.rel = 'noopener';
    box.appendChild(a);
  }
  return box;
}

function renderFrames(frames, eventIndex) {
  const ul = el('ul', 'frames');
  ul.setAttribute('aria-label', 'Кадры-доказательства');
  frames.forEach((f, k) => {
    const li = el('li');
    const btn = el('button', 'frame');
    btn.type = 'button';
    btn.setAttribute('aria-label', `Открыть кадр ${k + 1} из ${frames.length}, ${fmtTimecode(f.timestamp_sec)}`);
    const img = el('img');
    img.src = `data:image/jpeg;base64,${f.image_base64}`;
    img.alt = `Кадр на ${fmtTimecode(f.timestamp_sec)}`;
    img.loading = 'lazy';
    btn.appendChild(img);
    btn.appendChild(el('span', 'frame__time', fmtTimecode(f.timestamp_sec)));
    btn.addEventListener('click', () => openBox(frames, k, btn, eventIndex));
    li.appendChild(btn);
    ul.appendChild(li);
  });
  return ul;
}

// ---------------------------------------------------------------- просмотр кадра

function openBox(frames, index, opener) {
  state.gallery = { frames, index, opener };
  dom.box.hidden = false;
  showBoxFrame();
  dom.boxClose.focus();
}

function showBoxFrame() {
  const { frames, index } = state.gallery;
  const f = frames[index];
  dom.boxImg.src = `data:image/jpeg;base64,${f.image_base64}`;
  dom.boxImg.alt = `Кадр на ${fmtTimecode(f.timestamp_sec)}`;
  dom.boxCaption.textContent = `${fmtTimecode(f.timestamp_sec)} · кадр ${index + 1} из ${frames.length}`;
  dom.boxPrev.disabled = index === 0;
  dom.boxNext.disabled = index === frames.length - 1;
}

function stepBox(delta) {
  const g = state.gallery;
  const next = g.index + delta;
  if (next < 0 || next >= g.frames.length) return;
  g.index = next;
  showBoxFrame();
}

function closeBox() {
  dom.box.hidden = true;
  dom.boxImg.src = '';
  const opener = state.gallery.opener;
  state.gallery = { frames: [], index: 0, opener: null };
  if (opener && document.contains(opener)) opener.focus();
}

function initBox() {
  dom.boxClose.addEventListener('click', closeBox);
  dom.boxPrev.addEventListener('click', () => stepBox(-1));
  dom.boxNext.addEventListener('click', () => stepBox(1));
  dom.box.addEventListener('click', (e) => { if (e.target === dom.box) closeBox(); });
  document.addEventListener('keydown', (e) => {
    if (dom.box.hidden) return;
    if (e.key === 'Escape') { e.preventDefault(); closeBox(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); stepBox(-1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); stepBox(1); }
    else if (e.key === 'Tab') {
      // фокус остаётся внутри окна просмотра
      const items = [dom.boxClose, dom.boxPrev, dom.boxNext].filter((b) => !b.disabled);
      const i = items.indexOf(document.activeElement);
      const n = e.shiftKey ? (i <= 0 ? items.length - 1 : i - 1) : (i === items.length - 1 ? 0 : i + 1);
      e.preventDefault();
      items[n].focus();
    }
  });
}

// ---------------------------------------------------------------- JSON

function downloadJson() {
  if (!state.resultText) return;
  const blob = new Blob([state.resultText], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = el('a');
  a.href = url;
  a.download = `analysis-${(state.result && state.result.job_id) || 'result'}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- история

function getHistory() { return lsGet(LS.history, []).filter((h) => h && h.job_id); }

function addHistory(item) {
  const list = [item, ...getHistory().filter((h) => h.job_id !== item.job_id)].slice(0, HISTORY_MAX);
  lsSet(LS.history, list);
  renderHistory();
}

function updateHistory(jobId, patch) {
  const list = getHistory().map((h) => (h.job_id === jobId ? { ...h, ...patch } : h));
  lsSet(LS.history, list);
  renderHistory();
}

function renderHistory() {
  const list = getHistory();
  dom.historyCard.hidden = list.length === 0;
  dom.history.replaceChildren(...list.map((h) => {
    const li = el('li', 'history__item');
    li.appendChild(el('span', 'history__name', h.filename || h.job_id));
    li.appendChild(el('span', 'history__when', fmtWhen(h.started_at)));
    const v = h.verdict;
    const badge = v === 'violation' ? ['history__verdict--bad', 'нарушение']
      : v === 'no_violation' ? ['history__verdict--ok', 'нарушений нет']
        : v === 'low_coverage' ? ['history__verdict--run', 'анализ неполный']
          : v === 'failed' ? ['history__verdict--bad', 'ошибка']
            : ['history__verdict--run', 'в обработке'];
    li.appendChild(el('span', `history__verdict ${badge[0]}`, badge[1]));
    const btn = el('button', 'btn btn--ghost btn--small', 'Открыть');
    btn.type = 'button';
    btn.disabled = state.busy;
    btn.addEventListener('click', () => resumeJob(h));
    li.appendChild(btn);
    return li;
  }));
}

function resumeJob(h) {
  if (state.busy) return;
  if (!apiKey()) { note('Введите API-ключ в поле вверху страницы.'); dom.apiKey.focus(); return; }
  dom.errorCard.hidden = true;
  hideResult();
  note('');
  state.job = { job_id: h.job_id, filename: h.filename, started_at: h.started_at || Date.now() };
  lsSet(LS.active, state.job);
  setBusy(true);
  showProgress({ progress: 0, stage: 'запрос состояния задания' });
  dom.progressJob.textContent = h.job_id;
  startPolling();
}

// ---------------------------------------------------------------- запуск

function initEvents() {
  dom.analyzeBtn.addEventListener('click', startAnalysis);
  dom.newBtn.addEventListener('click', resetAll);
  dom.errorNewBtn.addEventListener('click', resetAll);
  dom.downloadBtn.addEventListener('click', downloadJson);
}

function init() {
  initKey();
  initFile();
  initBox();
  initEvents();
  renderHistory();
  // После обновления страницы продолжаем опрос текущего задания.
  const active = lsGet(LS.active, null);
  if (active && active.job_id) {
    state.job = active;
    setBusy(true);
    showProgress({ progress: 0, stage: 'восстановление после обновления страницы' });
    dom.progressJob.textContent = active.job_id;
    startPolling();
  }
}

document.addEventListener('DOMContentLoaded', init);

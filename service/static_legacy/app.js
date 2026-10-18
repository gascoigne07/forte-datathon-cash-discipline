'use strict';

/**
 * Фронтенд сервиса анализа видео кассовой зоны (ForteBank Datathon).
 * Без сборки и фреймворков — чистый vanilla JS, работает открытием index.html.
 *
 * Контракт backend-API (не менять):
 *   POST /api/v1/analyze  — multipart/form-data, поле file (.mov), заголовок X-API-Key
 *   GET  /api/v1/jobs/{id} — опрос статуса, тот же заголовок
 */

// ===================================================================
// Константы
// ===================================================================

const API_BASE = '/api/v1';
const LS_API_KEY = 'forte_video_api_key'; // ключ в localStorage
const POLL_INTERVAL_MS = 2000;            // опрос раз в 2 секунды
const POLL_TIMEOUT_MS = 6 * 60 * 1000;    // общий таймаут опроса — 6 минут
const MAX_FILE_SIZE_HINT_MB = 200;       // ориентир для подсказки пользователю (реальный лимит задаёт backend/413)

// ===================================================================
// Состояние приложения
// ===================================================================

const state = {
  selectedFile: null,     // File — выбранный пользователем .mov
  isProcessing: false,    // идёт отправка или опрос — блокирует повторную отправку
  jobId: null,
  pollTimer: null,        // id setInterval опроса
  pollStartedAt: null,    // Date.now() момента начала опроса — для таймера и таймаута
  elapsedTimer: null,     // id setInterval, обновляющего "прошедшее время"
  lastResult: null,       // последний успешный job-ответ (status=done) — для экспорта JSON
};

// ===================================================================
// Ссылки на DOM-элементы
// ===================================================================

const dom = {
  apiKeyInput: document.getElementById('apiKeyInput'),
  apiKeyToggle: document.getElementById('apiKeyToggle'),
  apiKeyEyeIcon: document.getElementById('apiKeyEyeIcon'),

  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  dropzoneEmpty: document.getElementById('dropzoneEmpty'),
  dropzoneFile: document.getElementById('dropzoneFile'),
  fileName: document.getElementById('fileName'),
  fileSize: document.getElementById('fileSize'),
  fileRemoveBtn: document.getElementById('fileRemoveBtn'),

  analyzeBtn: document.getElementById('analyzeBtn'),
  uploadNote: document.getElementById('uploadNote'),

  progressSection: document.getElementById('progressSection'),
  progressFill: document.getElementById('progressFill'),
  progressStage: document.getElementById('progressStage'),
  progressPercent: document.getElementById('progressPercent'),
  progressElapsed: document.getElementById('progressElapsed'),
  progressJobId: document.getElementById('progressJobId'),

  errorSection: document.getElementById('errorSection'),
  errorTitle: document.getElementById('errorTitle'),
  errorText: document.getElementById('errorText'),

  resultSection: document.getElementById('resultSection'),
  verdictBlock: document.getElementById('verdictBlock'),
  timelineCard: document.getElementById('timelineCard'),
  timelineTrack: document.getElementById('timelineTrack'),
  timelineDuration: document.getElementById('timelineDuration'),
  violationsList: document.getElementById('violationsList'),

  downloadJsonBtn: document.getElementById('downloadJsonBtn'),
  copyJsonBtn: document.getElementById('copyJsonBtn'),
  newAnalysisBtn: document.getElementById('newAnalysisBtn'),

  lightbox: document.getElementById('lightbox'),
  lightboxImg: document.getElementById('lightboxImg'),
  lightboxClose: document.getElementById('lightboxClose'),
};

// ===================================================================
// Утилиты форматирования
// ===================================================================

/** Байты → человекочитаемый размер ("12.4 МБ"). */
function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} Б`;
  const units = ['КБ', 'МБ', 'ГБ'];
  let value = bytes / 1024;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(1)} ${units[unitIndex]}`;
}

/** Секунды → "мм:сс" (для общих таймеров без долей секунды). */
function formatMinutesSeconds(totalSeconds) {
  const safe = Math.max(0, Math.floor(totalSeconds));
  const minutes = Math.floor(safe / 60);
  const seconds = safe % 60;
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

/** Секунды с долями → "мм:сс.д" — формат таймкода нарушения, как требует ТЗ. */
function formatTimecode(totalSeconds) {
  const safe = Math.max(0, totalSeconds);
  const minutes = Math.floor(safe / 60);
  const seconds = safe - minutes * 60;
  // одна цифра после запятой, секунды с ведущим нулём ("03.1")
  const secondsStr = seconds.toFixed(1).padStart(4, '0');
  return `${String(minutes).padStart(2, '0')}:${secondsStr}`;
}

/** Доля 0..1 → проценты для отображения ("42%"). */
function formatPercent(fraction) {
  return `${Math.round((fraction || 0) * 100)}%`;
}

/** Экранирование текста при вставке в innerHTML (описания/типы — произвольный текст от backend). */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text == null ? '' : String(text);
  return div.innerHTML;
}

// ===================================================================
// API-ключ: хранение в localStorage + маскирование поля
// ===================================================================

function loadApiKey() {
  const saved = localStorage.getItem(LS_API_KEY) || '';
  dom.apiKeyInput.value = saved;
}

function saveApiKey() {
  localStorage.setItem(LS_API_KEY, dom.apiKeyInput.value.trim());
}

function getApiKey() {
  return dom.apiKeyInput.value.trim();
}

/** Переключение видимости символов ключа (type=password ↔ type=text). */
function toggleApiKeyVisibility() {
  const isHidden = dom.apiKeyInput.type === 'password';
  dom.apiKeyInput.type = isHidden ? 'text' : 'password';
}

function initApiKeyField() {
  loadApiKey();
  dom.apiKeyInput.addEventListener('input', saveApiKey);
  dom.apiKeyToggle.addEventListener('click', toggleApiKeyVisibility);
}

// ===================================================================
// Выбор файла: drag&drop + клик + валидация расширения
// ===================================================================

function isMovFile(file) {
  return file && file.name.toLowerCase().endsWith('.mov');
}

/** Применить выбранный файл: показать имя/размер, включить кнопку отправки. */
function setSelectedFile(file) {
  if (!file) return;

  if (!isMovFile(file)) {
    showUploadNote('Поддерживается только формат .mov', true);
    return;
  }

  state.selectedFile = file;
  dom.fileName.textContent = file.name;
  dom.fileSize.textContent = formatFileSize(file.size);

  dom.dropzoneEmpty.hidden = true;
  dom.dropzoneFile.hidden = false;

  showUploadNote('');
  updateAnalyzeButtonState();
}

function clearSelectedFile() {
  state.selectedFile = null;
  dom.fileInput.value = '';
  dom.dropzoneEmpty.hidden = false;
  dom.dropzoneFile.hidden = true;
  updateAnalyzeButtonState();
}

function showUploadNote(text, isWarning) {
  dom.uploadNote.textContent = text || '';
  dom.uploadNote.style.color = isWarning ? 'var(--color-danger)' : '';
}

/** Кнопка "Отправить" активна только если файл выбран и сейчас ничего не обрабатывается. */
function updateAnalyzeButtonState() {
  dom.analyzeBtn.disabled = !state.selectedFile || state.isProcessing;
}

/** Блокировка/разблокировка зоны загрузки на время обработки — запрет повторной отправки. */
function setDropzoneDisabled(disabled) {
  dom.dropzone.classList.toggle('dropzone--disabled', disabled);
  dom.dropzone.setAttribute('aria-disabled', String(disabled));
  dom.fileRemoveBtn.disabled = disabled;
}

function initFileSelection() {
  // Клик по зоне открывает системный диалог выбора файла.
  dom.dropzone.addEventListener('click', () => {
    if (!state.isProcessing) dom.fileInput.click();
  });
  dom.dropzone.addEventListener('keydown', (event) => {
    if ((event.key === 'Enter' || event.key === ' ') && !state.isProcessing) {
      event.preventDefault();
      dom.fileInput.click();
    }
  });

  dom.fileInput.addEventListener('change', () => {
    const file = dom.fileInput.files && dom.fileInput.files[0];
    if (file) setSelectedFile(file);
  });

  // Drag & drop.
  ['dragenter', 'dragover'].forEach((eventName) => {
    dom.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      if (!state.isProcessing) dom.dropzone.classList.add('dropzone--drag');
    });
  });
  ['dragleave', 'drop'].forEach((eventName) => {
    dom.dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dom.dropzone.classList.remove('dropzone--drag');
    });
  });
  dom.dropzone.addEventListener('drop', (event) => {
    if (state.isProcessing) return;
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) setSelectedFile(file);
  });

  // Удаление выбранного файла (не всплываем до клика по dropzone).
  dom.fileRemoveBtn.addEventListener('click', (event) => {
    event.stopPropagation();
    clearSelectedFile();
  });
}

// ===================================================================
// Отправка файла на анализ: POST /api/v1/analyze
// ===================================================================

async function startAnalysis() {
  if (!state.selectedFile || state.isProcessing) return;

  const apiKey = getApiKey();
  if (!apiKey) {
    showErrorBlock('Не указан API-ключ', 'Введите ключ доступа в поле вверху страницы.');
    return;
  }

  hideErrorBlock();
  hideResultBlock();
  enterProcessingMode();

  const formData = new FormData();
  formData.append('file', state.selectedFile);

  try {
    const response = await fetch(`${API_BASE}/analyze`, {
      method: 'POST',
      headers: { 'X-API-Key': apiKey },
      body: formData,
    });

    if (response.status === 401) {
      throw new ApiError('Неверный API-ключ', 'Проверьте значение ключа и повторите отправку.');
    }
    if (response.status === 400) {
      throw new ApiError('Файл отклонён сервером', 'Проверьте, что это корректный видеофайл .mov.');
    }
    if (response.status === 413) {
      throw new ApiError('Файл слишком большой', 'Уменьшите размер видео и попробуйте снова.');
    }
    if (!response.ok) {
      throw new ApiError('Сервер вернул ошибку', `Код ответа: ${response.status}`);
    }

    const data = await response.json();
    state.jobId = data.job_id;
    dom.progressJobId.textContent = state.jobId;

    startPolling();
  } catch (error) {
    handleRequestError(error, 'Не удалось отправить файл на анализ');
    exitProcessingMode();
  }
}

/** Внутренний класс ошибки с заголовком+текстом, чтобы единообразно показывать в errorSection. */
class ApiError extends Error {
  constructor(title, text) {
    super(title);
    this.title = title;
    this.text = text;
  }
}

/** Единая обработка сетевых/API ошибок на любом этапе запроса. */
function handleRequestError(error, fallbackTitle) {
  if (error instanceof ApiError) {
    showErrorBlock(error.title, error.text);
    return;
  }
  // TypeError от fetch обычно означает обрыв сети/CORS/сервер недоступен.
  if (error instanceof TypeError) {
    showErrorBlock('Нет соединения с сервером', 'Проверьте подключение к внутренней сети и повторите попытку.');
    return;
  }
  showErrorBlock(fallbackTitle, error.message || 'Неизвестная ошибка');
}

// ===================================================================
// Опрос статуса: GET /api/v1/jobs/{id}, раз в 2 секунды, таймаут 6 минут
// ===================================================================

function startPolling() {
  state.pollStartedAt = Date.now();
  tickElapsedTime(); // сразу показать 00:00, не ждать первый интервал

  state.elapsedTimer = setInterval(tickElapsedTime, 1000);
  state.pollTimer = setInterval(pollJobStatus, POLL_INTERVAL_MS);

  // Первый опрос — сразу, не дожидаясь полного интервала.
  pollJobStatus();
}

function stopPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  if (state.elapsedTimer) clearInterval(state.elapsedTimer);
  state.pollTimer = null;
  state.elapsedTimer = null;
}

function tickElapsedTime() {
  const elapsedSec = (Date.now() - state.pollStartedAt) / 1000;
  dom.progressElapsed.textContent = formatMinutesSeconds(elapsedSec);
}

async function pollJobStatus() {
  // Таймаут общего времени ожидания — если backend завис дольше 6 минут.
  if (Date.now() - state.pollStartedAt > POLL_TIMEOUT_MS) {
    stopPolling();
    showErrorBlock('Обработка не завершилась', 'Превышено время ожидания ответа от сервера (6 минут).');
    exitProcessingMode();
    return;
  }

  try {
    const response = await fetch(`${API_BASE}/jobs/${state.jobId}`, {
      headers: { 'X-API-Key': getApiKey() },
    });

    if (response.status === 401) {
      stopPolling();
      throw new ApiError('Неверный API-ключ', 'Ключ стал недействителен во время обработки.');
    }
    if (!response.ok) {
      throw new ApiError('Сервер вернул ошибку', `Код ответа: ${response.status}`);
    }

    const job = await response.json();
    updateProgressUi(job);

    if (job.status === 'done') {
      stopPolling();
      state.lastResult = job;
      exitProcessingMode();
      renderResult(job);
    } else if (job.status === 'failed') {
      stopPolling();
      exitProcessingMode();
      showErrorBlock('Обработка завершилась с ошибкой', job.error || 'Backend не указал причину.');
    }
    // status === 'processing' — просто продолжаем опрос по интервалу.
  } catch (error) {
    stopPolling();
    handleRequestError(error, 'Не удалось получить статус обработки');
    exitProcessingMode();
  }
}

function updateProgressUi(job) {
  const fraction = typeof job.progress === 'number' ? job.progress : 0;
  dom.progressFill.style.width = formatPercent(fraction);
  dom.progressPercent.textContent = formatPercent(fraction);
  dom.progressStage.textContent = job.stage || 'Обработка…';
}

// ===================================================================
// Переключение режимов экрана: загрузка / обработка / результат / ошибка
// ===================================================================

function enterProcessingMode() {
  state.isProcessing = true;
  updateAnalyzeButtonState();
  setDropzoneDisabled(true);

  dom.progressSection.hidden = false;
  dom.progressFill.style.width = '0%';
  dom.progressPercent.textContent = '0%';
  dom.progressStage.textContent = 'Отправка файла на сервер…';
  dom.progressElapsed.textContent = '00:00';
  dom.progressJobId.textContent = '—';
}

function exitProcessingMode() {
  state.isProcessing = false;
  updateAnalyzeButtonState();
  setDropzoneDisabled(false);
  dom.progressSection.hidden = true;
}

function showErrorBlock(title, text) {
  dom.errorTitle.textContent = title;
  dom.errorText.textContent = text || '';
  dom.errorSection.hidden = false;
}

function hideErrorBlock() {
  dom.errorSection.hidden = true;
}

function hideResultBlock() {
  dom.resultSection.hidden = true;
  dom.violationsList.innerHTML = '';
  dom.timelineTrack.innerHTML = '';
}

/** Полный сброс к начальному состоянию — кнопка "Новый анализ". */
function resetToUploadScreen() {
  stopPolling();
  state.jobId = null;
  state.lastResult = null;
  hideErrorBlock();
  hideResultBlock();
  exitProcessingMode();
  clearSelectedFile();
}

// ===================================================================
// Рендер результата: вердикт, таймлайн, карточки нарушений
// ===================================================================

function renderResult(job) {
  // Предупреждение, если модель успела посмотреть мало кадров: вердикт в этом
  // случае недостоверен, и проверяющий должен это видеть, а не угадывать.
  if (job.low_coverage) {
    showErrorBlock('Неполный анализ',
      'Модель успела разобрать только ' + (job.frames_analyzed || 0) +
      ' кадров из-за лимита времени. Вердикт недостоверен — повторите загрузку.');
  }
  dom.resultSection.hidden = false;
  renderVerdict(job);

  const violations = Array.isArray(job.violations) ? job.violations : [];
  if (job.verdict === 'violation' && violations.length > 0) {
    renderTimeline(job, violations);
    renderViolationCards(violations);
  } else {
    dom.timelineCard.hidden = true;
  }
}

function renderVerdict(job) {
  const duration = job.video && typeof job.video.duration_sec === 'number'
    ? job.video.duration_sec
    : null;
  const durationText = duration != null
    ? `Длительность видео: ${formatMinutesSeconds(duration)}`
    : '';

  if (job.verdict === 'no_violation') {
    dom.verdictBlock.className = 'verdict verdict--ok';
    dom.verdictBlock.innerHTML = `
      <div class="verdict__icon" aria-hidden="true">✓</div>
      <div>
        <div class="verdict__title">Нарушение не обнаружено</div>
        <div class="verdict__subtitle">${escapeHtml(durationText)}</div>
      </div>
    `;
    return;
  }

  const count = Array.isArray(job.violations) ? job.violations.length : 0;
  dom.verdictBlock.className = 'verdict verdict--violation';
  dom.verdictBlock.innerHTML = `
    <div class="verdict__icon" aria-hidden="true">⚠</div>
    <div>
      <div class="verdict__title">Обнаружены признаки возможных нарушений (${count})</div>
      <div class="verdict__subtitle">${escapeHtml(durationText)}</div>
    </div>
  `;
}

/** Горизонтальная шкала длиной video.duration_sec с метками на месте каждого нарушения. */
function renderTimeline(job, violations) {
  const duration = job.video && typeof job.video.duration_sec === 'number'
    ? job.video.duration_sec
    : Math.max(...violations.map((v) => v.timecode_sec), 1);

  dom.timelineCard.hidden = false;
  dom.timelineDuration.textContent = formatMinutesSeconds(duration);
  dom.timelineTrack.innerHTML = '';

  violations.forEach((violation, index) => {
    const positionPercent = duration > 0
      ? Math.min(100, Math.max(0, (violation.timecode_sec / duration) * 100))
      : 0;

    const mark = document.createElement('button');
    mark.type = 'button';
    mark.className = 'timeline__mark';
    mark.style.left = `${positionPercent}%`;
    mark.title = `${violation.type} — ${formatTimecode(violation.timecode_sec)}`;
    mark.innerHTML = `
      <span class="timeline__mark-dot"></span>
      <span class="timeline__mark-label">${formatMinutesSeconds(violation.timecode_sec)}</span>
    `;
    mark.addEventListener('click', () => scrollToViolationCard(index));

    dom.timelineTrack.appendChild(mark);
  });
}

function scrollToViolationCard(index) {
  const card = document.getElementById(`violation-card-${index}`);
  if (card) card.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function renderViolationCards(violations) {
  dom.violationsList.innerHTML = '';
  violations.forEach((violation, index) => {
    dom.violationsList.appendChild(buildViolationCard(violation, index));
  });
}

/** Собрать одну карточку нарушения со всеми требуемыми по ТЗ полями. */
function buildViolationCard(violation, index) {
  const card = document.createElement('article');
  card.className = 'violation-card';
  card.id = `violation-card-${index}`;

  const confidencePercent = Math.round((violation.confidence || 0) * 100);
  const rule = violation.rule || {};

  card.innerHTML = `
    <div class="violation-card__header">
      <h3 class="violation-card__type">${escapeHtml(violation.type)}</h3>
      <span class="violation-card__timecode">
        ${formatTimecode(violation.timecode_sec)}
        <small>(${violation.timecode_sec.toFixed(1)} с)</small>
      </span>
    </div>

    <div class="violation-card__confidence">
      <div class="confidence__label">
        <span>Уверенность модели</span>
        <span class="confidence__value">${confidencePercent}%</span>
      </div>
      <div class="confidence__bar">
        <div class="confidence__bar-fill" style="width:${confidencePercent}%"></div>
      </div>
    </div>

    <p class="violation-card__description">${escapeHtml(violation.description)}</p>

    ${buildRuleBoxHtml(rule)}

    ${buildEvidenceFramesHtml(violation.evidence_frames)}
  `;

  // Клики по кадрам-доказательствам навешиваем отдельно (нужен доступ к image_base64 из замыкания).
  attachEvidenceFrameHandlers(card, violation.evidence_frames);

  return card;
}

/** Блок "Норматив" — документ, реквизиты, пункт, цитата, ссылка. */
function buildRuleBoxHtml(rule) {
  if (!rule || !rule.doc_title) return '';

  return `
    <div class="rule-box">
      <div class="rule-box__label">Норматив</div>
      <div class="rule-box__title">${escapeHtml(rule.doc_title)}</div>
      ${rule.doc_requisites ? `<div class="rule-box__requisites">${escapeHtml(rule.doc_requisites)}</div>` : ''}
      ${rule.clause ? `<div class="rule-box__clause">${escapeHtml(rule.clause)}${rule.edition ? ` — ${escapeHtml(rule.edition)}` : ''}</div>` : ''}
      ${rule.quote ? `<blockquote class="rule-box__quote">«${escapeHtml(rule.quote)}»</blockquote>` : ''}
      ${rule.verified === false ? `
        <div class="rule-box__unverified">
          Трактовка, а не дословный запрет${rule.note ? `: ${escapeHtml(rule.note)}` : '.'}
        </div>
      ` : ''}
      ${rule.url ? `
        <a class="rule-box__link" href="${escapeHtml(rule.url)}" target="_blank" rel="noopener">
          Открыть документ на adilet.zan.kz ↗
        </a>
      ` : ''}
    </div>
  `;
}

/** Ряд превью кадров-доказательств (base64 JPEG). */
function buildEvidenceFramesHtml(frames) {
  if (!Array.isArray(frames) || frames.length === 0) return '';

  const framesHtml = frames.map((frame, frameIndex) => `
    <div class="evidence-frame" data-frame-index="${frameIndex}">
      <img src="data:image/jpeg;base64,${frame.image_base64}" alt="Кадр-доказательство на ${formatTimecode(frame.timestamp_sec)}">
      <span class="evidence-frame__time">${formatTimecode(frame.timestamp_sec)}</span>
    </div>
  `).join('');

  return `
    <div class="evidence-frames-block">
      <div class="evidence-frames__label">Кадры-доказательства</div>
      <div class="evidence-frames">${framesHtml}</div>
    </div>
  `;
}

/** Клик по кадру открывает его крупно в лайтбоксе. */
function attachEvidenceFrameHandlers(cardEl, frames) {
  if (!Array.isArray(frames)) return;
  const frameElements = cardEl.querySelectorAll('.evidence-frame');
  frameElements.forEach((el, frameIndex) => {
    el.addEventListener('click', () => openLightbox(frames[frameIndex].image_base64));
  });
}

// ===================================================================
// Лайтбокс — простое увеличенное окно просмотра кадра (без библиотек)
// ===================================================================

function openLightbox(base64) {
  dom.lightboxImg.src = `data:image/jpeg;base64,${base64}`;
  dom.lightbox.hidden = false;
}

function closeLightbox() {
  dom.lightbox.hidden = true;
  dom.lightboxImg.src = '';
}

function initLightbox() {
  dom.lightboxClose.addEventListener('click', closeLightbox);
  dom.lightbox.addEventListener('click', (event) => {
    if (event.target === dom.lightbox) closeLightbox(); // клик по фону вне картинки
  });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !dom.lightbox.hidden) closeLightbox();
  });
}

// ===================================================================
// Экспорт результата: скачать JSON / скопировать в буфер обмена
// ===================================================================

function downloadResultJson() {
  if (!state.lastResult) return;

  const blob = new Blob([JSON.stringify(state.lastResult, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const link = document.createElement('a');
  link.href = url;
  link.download = `analysis-${state.lastResult.job_id || 'result'}.json`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
}

async function copyResultJson() {
  if (!state.lastResult) return;
  const text = JSON.stringify(state.lastResult, null, 2);

  try {
    await navigator.clipboard.writeText(text);
    flashButtonText(dom.copyJsonBtn, 'Скопировано ✓');
  } catch (error) {
    // Clipboard API может быть недоступен (например, не-HTTPS контекст внутри сети) — запасной путь.
    flashButtonText(dom.copyJsonBtn, 'Не удалось скопировать');
  }
}

/** Временно показать сообщение в кнопке, затем вернуть исходный текст. */
function flashButtonText(button, tempText) {
  const original = button.textContent;
  button.textContent = tempText;
  setTimeout(() => { button.textContent = original; }, 1800);
}

// ===================================================================
// Инициализация приложения
// ===================================================================

function initEventListeners() {
  dom.analyzeBtn.addEventListener('click', startAnalysis);
  dom.newAnalysisBtn.addEventListener('click', resetToUploadScreen);
  dom.downloadJsonBtn.addEventListener('click', downloadResultJson);
  dom.copyJsonBtn.addEventListener('click', copyResultJson);
}

function init() {
  initApiKeyField();
  initFileSelection();
  initLightbox();
  initEventListeners();
}

document.addEventListener('DOMContentLoaded', init);

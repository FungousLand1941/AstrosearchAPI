// AstroSearch Sky Explorer -- single-page UI (vanilla ES module, no build step).
//
// Every server call goes through ENDPOINTS below; tests/test_imaging_ui.py checks
// that each one is part of the published API contract and that no other API path
// is referenced. Optional endpoints that are missing (404 "Not Found", 405, 501)
// render as "unavailable" instead of failing the page.

const ENDPOINTS = Object.freeze({
  searchStream: { method: 'GET', path: '/api/v1/search/stream' },
  search: { method: 'POST', path: '/api/v1/search' },
  sed: { method: 'GET', path: '/api/v1/sed' },
  sedByName: { method: 'POST', path: '/api/v1/sed' },
  lightcurves: { method: 'GET', path: '/api/v1/lightcurves' },
  solarSystem: { method: 'GET', path: '/api/v1/solar-system' },
  cutoutSurveys: { method: 'GET', path: '/api/v1/cutouts/surveys' },
  cutoutStack: { method: 'GET', path: '/api/v1/cutouts/stack' },
  cutouts: { method: 'GET', path: '/api/v1/cutouts' },
  aiQuery: { method: 'POST', path: '/api/v1/ai/query' },
  aiExplain: { method: 'POST', path: '/api/v1/ai/explain' },
  citations: { method: 'GET', path: '/api/v1/citations' },
  limits: { method: 'GET', path: '/api/v1/limits' },
});

const ALADIN_SRC = 'https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js';
const DEFAULT_SURVEY = 'CDS/P/DSS2/color';
// Used only when /api/v1/cutouts/surveys is unavailable; IDs verified in the CDS MocServer.
const FALLBACK_SURVEYS = [
  { hips_id: 'CDS/P/NVSS', label: 'NVSS 1.4 GHz', wavelength: '1.4 GHz' },
  { hips_id: 'CDS/P/allWISE/color', label: 'AllWISE W1/W2/W4', wavelength: '2.75–27.9 µm' },
  { hips_id: 'CDS/P/2MASS/color', label: '2MASS J/H/Ks', wavelength: '1.15–2.3 µm' },
  { hips_id: 'CDS/P/PanSTARRS/DR1/color-z-zg-g', label: 'Pan-STARRS1 g/z', wavelength: '394–951 nm' },
  { hips_id: 'CDS/P/SDSS9/color', label: 'SDSS DR9', wavelength: '378–839 nm' },
  { hips_id: 'CDS/P/DSS2/color', label: 'DSS2 colour', wavelength: '400–600 nm' },
  { hips_id: 'CDS/P/GALEXGR6_7/color', label: 'GALEX FUV/NUV', wavelength: '134–283 nm' },
  { hips_id: 'CDS/P/RASS', label: 'ROSAT All-Sky Survey', wavelength: '0.1–2.4 keV' },
];
// Colour-blind-aware categorical palette (Okabe-Ito + Tableau extensions), readable on dark and light.
const PALETTE = ['#56b4e9', '#e69f00', '#009e73', '#f0e442', '#cc79a7', '#d55e00', '#0072b2',
  '#9edae5', '#ff9da7', '#b07aa1', '#8cd17d', '#f28e2b', '#76b7b2', '#edc948'];
const STORAGE = { theme: 'astrosearch.theme', key: 'astrosearch.apiKey', stream: 'astrosearch.stream' };
// Server-side cone limits of the SED and light-curve services (sed.MAX_RADIUS_ARCSEC and
// timedomain.MAX_LIGHTCURVE_RADIUS_ARCSEC); wider searches are clamped for those panels.
const SED_MAX_RADIUS_ARCSEC = 60;
const LIGHTCURVE_MAX_RADIUS_ARCSEC = 60;
// The search radius bound until GET /api/v1/limits answers: the server's default
// API_MAX_RADIUS_ARCSEC. The server's own value replaces it (loadLimits), so the check and its
// message follow the deployment; the server still checks every request.
const DEFAULT_MAX_RADIUS_ARCSEC = 1800;

const state = {
  controller: null,
  query: null,
  target: null,
  record: null,
  groups: new Map(),
  catalogStatus: new Map(),
  plannedCatalogs: 0,
  aladin: null,
  aladinReady: null,
  skyCatalogs: new Map(),
  skySources: new Map(),
  selectionOverlay: null,
  selectedKey: null,
  charts: { sed: null, lc: null },
  sedData: null,
  lcData: null,
  lcView: { unit: null, folded: false },
  bibtex: '',
  surveys: [],
  maxRadiusArcsec: DEFAULT_MAX_RADIUS_ARCSEC,
};

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

const $ = (sel, root = document) => root.querySelector(sel);

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key === 'text') el.textContent = value;
    else if (key === 'style' && typeof value === 'object') Object.assign(el.style, value);
    else if (key.startsWith('on') && typeof value === 'function') el.addEventListener(key.slice(2), value);
    else if (key === 'dataset') Object.assign(el.dataset, value);
    else el.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

// Every URL is resolved against the document base, so the UI also works under a
// sub-path (e.g. behind a proxy at /astro/) and never hard-codes an origin.
function apiBase() {
  return globalThis.document?.baseURI || globalThis.location?.href || 'http://localhost/';
}

function safeUrl(url) {
  try {
    const parsed = new URL(url, apiBase());
    return parsed.protocol === 'http:' || parsed.protocol === 'https:' ? parsed.href : null;
  } catch {
    return null;
  }
}

function link(href, text) {
  const url = safeUrl(href);
  return url ? h('a', { href: url, target: '_blank', rel: 'noopener' }, text) : document.createTextNode(text);
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || value === '') return '—';
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  if (num !== 0 && (Math.abs(num) >= 1e5 || Math.abs(num) < 1e-3)) return num.toExponential(digits - 1);
  return Number(num.toPrecision(digits)).toString();
}

function fmtDeg(value, digits = 6) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : '—';
}

function sexagesimal(ra, dec) {
  const pad = (n, w = 2, d = 0) => n.toFixed(d).padStart(w + (d ? d + 1 : 0), '0');
  let hours = ((ra % 360) + 360) % 360 / 15;
  let hh = Math.floor(hours); let mm = Math.floor((hours - hh) * 60);
  let ss = (hours - hh - mm / 60) * 3600;
  if (ss >= 59.995) { ss = 0; mm += 1; }
  if (mm >= 60) { mm = 0; hh = (hh + 1) % 24; }
  const sign = dec < 0 ? '−' : '+';
  const adec = Math.abs(dec);
  let dd = Math.floor(adec); let dm = Math.floor((adec - dd) * 60);
  let ds = (adec - dd - dm / 60) * 3600;
  if (ds >= 59.95) { ds = 0; dm += 1; }
  if (dm >= 60) { dm = 0; dd += 1; }
  return `${pad(hh)}h${pad(mm)}m${pad(ss, 2, 2)}s ${sign}${pad(dd)}°${pad(dm)}′${pad(ds, 2, 1)}″`;
}

const SUPERSCRIPT = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
function powerOfTen(value) {
  if (!(value > 0)) return '';
  const exp = Math.log10(value);
  const rounded = Math.round(exp);
  if (Math.abs(exp - rounded) > 1e-6) return '';
  return `10${String(rounded).split('').map((c) => SUPERSCRIPT[c] ?? c).join('')}`;
}

function colorFor(name) {
  let hash = 0;
  for (const ch of String(name)) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return PALETTE[hash % PALETTE.length];
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function stripTags(value) {
  return String(value ?? '').replace(/[<>]/g, '');
}

function nowMjd() {
  return Date.now() / 86400000 + 40587;
}

let toastTimer = null;
function toast(message) {
  const el = $('#toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3500);
}

function storageGet(key, fallback = null) {
  try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode: settings last for this page only */ }
}

// ---------------------------------------------------------------------------
// Coordinate parsing (decimal degrees or sexagesimal, ICRS)
// ---------------------------------------------------------------------------

// Numeric reading of the text with the reason it is not a valid position (or null
// when the text is not numeric at all, i.e. an object name).
function readCoordinates(text) {
  const t = String(text).trim().replace(/−/g, '-').replace(/[,;]+/g, ' ').replace(/\s+/g, ' ');
  const dec = t.match(/^([+-]?\d+(?:\.\d+)?)\s*(?:d|deg|°)?\s+([+-]?\d+(?:\.\d+)?)\s*(?:d|deg|°)?$/i);
  if (dec) {
    const ra = Number(dec[1]); const de = Number(dec[2]);
    if (!(ra >= 0 && ra < 360)) return { error: `RA must be in [0, 360) degrees (got ${dec[1]}).` };
    if (!(de >= -90 && de <= 90)) return { error: `Dec must be in [-90, +90] degrees (got ${dec[2]}).` };
    return { ra, dec: de };
  }
  if (!/^[\d\s:hHmMsSdD°'′"″.+-]+$/.test(t)) return null;
  const s = t.replace(/[hHmMsSdD°:'′"″]/g, ' ').replace(/\s+/g, ' ').trim();
  const sx = s.match(/^(\d{1,2}) (\d{1,2}) (\d{1,2}(?:\.\d+)?) ([+-]?)(\d{1,2}) (\d{1,2}) (\d{1,2}(?:\.\d+)?)$/);
  if (!sx) return null;
  const [hh, mm, ss] = [Number(sx[1]), Number(sx[2]), Number(sx[3])];
  const [dd, dm, ds] = [Number(sx[5]), Number(sx[6]), Number(sx[7])];
  if (hh >= 24 || mm >= 60 || ss >= 60) return { error: 'RA must be 0-23 h, 0-59 m, 0-59.99 s.' };
  if (dd > 90 || dm >= 60 || ds >= 60) return { error: 'Dec must be at most 90°, with 0-59 arcmin and 0-59.99 arcsec.' };
  const ra = 15 * (hh + mm / 60 + ss / 3600);
  const de = (sx[4] === '-' ? -1 : 1) * (dd + dm / 60 + ds / 3600);
  return de >= -90 && de <= 90 ? { ra, dec: de } : { error: 'Dec must be in [-90, +90] degrees.' };
}

export function parseCoordinates(text) {
  const read = readCoordinates(text);
  return read && !read.error ? read : null;
}

// Search input from the search box: coordinates when the text is numeric (a range error
// is reported, never sent to the name resolver), otherwise an object name.
function parseSearchInput(text, radius) {
  const clean = String(text ?? '').trim();
  if (!clean) throw new Error('Enter an object name or coordinates.');
  const r = Number(radius);
  const max = state.maxRadiusArcsec;
  if (!(r > 0 && r <= max)) throw new Error(`Radius must be greater than 0 and at most ${max} arcsec.`);
  const read = readCoordinates(clean);
  if (read?.error) throw new Error(read.error);
  return read ? { text: clean, radius: r, ra: read.ra, dec: read.dec } : { text: clean, radius: r, name: clean };
}

// ---------------------------------------------------------------------------
// API layer
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, detail, endpoint, retryAfter = null) {
    super(detail || `HTTP ${status}`);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
    this.endpoint = endpoint;
    this.retryAfter = retryAfter;
  }

  get unavailable() {
    // A missing route answers FastAPI's default {"detail": "Not Found"}; 405/501 mean
    // the method or feature is not implemented; status 0 is a network failure.
    return this.status === 0 || this.status === 405 || this.status === 501
      || (this.status === 404 && (!this.detail || this.detail === 'Not Found'));
  }
}

function retryAfterSeconds(response) {
  const value = response.headers?.get?.('Retry-After');
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, Math.round(seconds));
  const date = Date.parse(value);
  return Number.isFinite(date) ? Math.max(0, Math.round((date - Date.now()) / 1000)) : null;
}

async function apiError(response, endpoint) {
  return new ApiError(response.status, await errorDetail(response), endpoint, retryAfterSeconds(response));
}

// How a failed call is shown: 'aborted' (ignored), 'unavailable' (endpoint missing),
// 'temporarily-unavailable' (503 with Retry-After: the name resolver or an archive is down for
// now), 'unconfigured' (503 without Retry-After, e.g. AI without a key), 'rate-limited' (429) or 'error'.
function errorKind(err) {
  if (err?.name === 'AbortError') return 'aborted';
  if (err instanceof ApiError) {
    if (err.unavailable) return 'unavailable';
    if (err.status === 503) return err.retryAfter != null ? 'temporarily-unavailable' : 'unconfigured';
    if (err.status === 429) return 'rate-limited';
  }
  return 'error';
}

function errorMessage(err, what) {
  switch (errorKind(err)) {
    case 'unavailable': return `${what} is unavailable on this server.`;
    case 'unconfigured': return `${what} is not configured: ${err.detail}`;
    case 'temporarily-unavailable':
      return `${what} is temporarily unavailable, retry in ${err.retryAfter} s${err.detail ? ` (${err.detail})` : ''}.`;
    case 'rate-limited':
      return `${what} is rate limited by the server${err.retryAfter != null ? `; retry in ${err.retryAfter} s` : ''}.`;
    default: return `${what} failed: ${err?.detail || err?.message || err}`;
  }
}

function apiHeaders(extra = {}) {
  const headers = { Accept: 'application/json', ...extra };
  const key = storageGet(STORAGE.key, '');
  if (key) headers['X-API-Key'] = key;
  return headers;
}

function endpointUrl(name, params) {
  const endpoint = ENDPOINTS[name];
  const url = new URL(endpoint.path.replace(/^\//, ''), apiBase());
  for (const [key, value] of Object.entries(params || {})) {
    if (value !== null && value !== undefined && value !== '') url.searchParams.set(key, String(value));
  }
  return url;
}

async function errorDetail(response) {
  try {
    const body = await response.json();
    if (typeof body?.detail === 'string') return body.detail;
    if (Array.isArray(body?.detail)) return body.detail.map((d) => `${(d.loc || []).slice(1).join('.')}: ${d.msg}`).join('; ');
    return JSON.stringify(body).slice(0, 300);
  } catch {
    return response.statusText;
  }
}

async function api(name, { params, body, signal } = {}) {
  const endpoint = ENDPOINTS[name];
  let response;
  try {
    response = await fetch(endpointUrl(name, params), {
      method: endpoint.method,
      headers: apiHeaders(body ? { 'Content-Type': 'application/json' } : {}),
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    throw new ApiError(0, `Network error: ${err.message}`, name);
  }
  if (!response.ok) throw await apiError(response, name);
  return response.json();
}

async function fetchBlob(url, signal) {
  // Only image URLs of this API's cutout endpoint (as listed by /cutouts/stack) are fetched.
  const target = safeUrl(url);
  if (!target || !new URL(target).pathname.endsWith(ENDPOINTS.cutouts.path)) throw new ApiError(0, 'Invalid cutout URL', 'cutouts');
  const response = await fetch(target, { headers: apiHeaders({ Accept: 'image/*,application/fits' }), signal });
  if (!response.ok) throw await apiError(response, 'cutouts');
  return { blob: await response.blob(), headers: response.headers };
}

// Server-Sent Events parser (WHATWG HTML "event stream interpretation": CRLF, CR or LF end a line).
async function* sseEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    // A chunk may end between the CR and the LF of one CRLF: hold that CR back until the
    // next chunk shows whether an LF follows, so it is not read as two line breaks. At the
    // end of the stream a trailing CR is a line break of its own.
    let held = '';
    if (!done) {
      buffer += value;
      if (buffer.endsWith('\r')) { held = '\r'; buffer = buffer.slice(0, -1); }
    }
    buffer = buffer.replace(/\r\n?/g, '\n');
    let idx;
    while ((idx = buffer.indexOf('\n\n')) >= 0) {
      const block = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      let event = 'message';
      const data = [];
      for (const line of block.split('\n')) {
        if (!line || line.startsWith(':')) continue;
        const colon = line.indexOf(':');
        const field = colon < 0 ? line : line.slice(0, colon);
        const val = colon < 0 ? '' : line.slice(colon + 1).replace(/^ /, '');
        if (field === 'event') event = val;
        else if (field === 'data') data.push(val);
      }
      if (!data.length) continue;
      let payload = data.join('\n');
      try { payload = JSON.parse(payload); } catch { /* plain-text event */ }
      yield { event, data: payload };
    }
    if (done) break; // an unterminated last block is incomplete and is dropped (as EventSource does)
    buffer += held;
  }
}

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

function setPanel(el, stateName, ...content) {
  el.dataset.state = stateName;
  el.setAttribute('aria-busy', stateName === 'loading' ? 'true' : 'false');
  el.replaceChildren(...content.flat().filter((c) => c !== null && c !== undefined && c !== false && c !== ''));
}

function panelLoading(el) {
  setPanel(el, 'loading');
}

function note(text, kind = '') {
  const badge = kind === 'unavailable' ? h('span', { class: 'badge warn' }, 'unavailable')
    : kind === 'busy' ? h('span', { class: 'badge warn' }, 'rate limited')
    : kind === 'error' ? h('span', { class: 'badge err' }, 'error')
      : kind === 'empty' ? h('span', { class: 'badge' }, 'no data') : null;
  return h('p', { class: `state-note ${kind === 'error' ? 'error' : ''}` }, badge, h('span', {}, text));
}

function panelError(el, err, what) {
  const kind = errorKind(err);
  if (kind === 'aborted') return;
  const transient = kind === 'rate-limited' || kind === 'temporarily-unavailable';
  const panelState = kind === 'error' ? 'error' : transient ? 'rate-limited' : 'unavailable';
  const badge = kind === 'error' ? 'error' : transient ? 'busy' : 'unavailable';
  setPanel(el, panelState, note(errorMessage(err, what), badge));
}

// ---------------------------------------------------------------------------
// Theme & settings
// ---------------------------------------------------------------------------

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $('#theme-toggle');
  btn.setAttribute('aria-label', theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  if (state.sedData) renderSedChart();
  if (state.lcData) renderLightcurveChart();
}

function initSettings() {
  applyTheme(storageGet(STORAGE.theme, 'dark') === 'light' ? 'light' : 'dark');
  $('#theme-toggle').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    storageSet(STORAGE.theme, next);
    applyTheme(next);
  });
  const dialog = $('#settings-dialog');
  $('#settings-toggle').addEventListener('click', () => {
    $('#api-key').value = storageGet(STORAGE.key, '');
    $('#use-stream').checked = storageGet(STORAGE.stream, 'true') !== 'false';
    if (typeof dialog.showModal === 'function') dialog.showModal();
  });
  dialog.addEventListener('close', () => {
    if (dialog.returnValue !== 'save') return;
    storageSet(STORAGE.key, $('#api-key').value.trim());
    storageSet(STORAGE.stream, $('#use-stream').checked ? 'true' : 'false');
    toast('Settings saved');
  });
}

// ---------------------------------------------------------------------------
// Sky viewer (Aladin Lite v3)
// ---------------------------------------------------------------------------

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.async = true;
    script.onload = resolve;
    script.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.append(script);
  });
}

function initAladin() {
  state.aladinReady = (async () => {
    const status = $('#aladin-status');
    try {
      await loadScript(ALADIN_SRC);
      await window.A.init;
      status.remove();
      state.aladin = window.A.aladin('#aladin', {
        survey: DEFAULT_SURVEY,
        fov: 0.5,
        target: '187.70593 +12.39112',
        cooFrame: 'ICRS',
        showReticle: true,
        showCooGridControl: true,
        showSimbadPointerControl: true,
        showFullscreenControl: true,
        showLayersControl: true,
        showContextMenu: true,
      });
      state.aladin.on('objectClicked', (object) => {
        const key = object?.data?.key;
        if (key) selectSource(key, { fromSky: true });
      });
      return state.aladin;
    } catch (err) {
      status.textContent = `Sky viewer unavailable (${err?.message || err}). Aladin Lite needs WebGL2 and network access to aladin.cds.unistra.fr.`;
      return null;
    }
  })();
}

async function setSurvey(hipsId) {
  const aladin = await state.aladinReady;
  if (!aladin) return;
  try {
    aladin.setBaseImageLayer(hipsId);
  } catch (err) {
    toast(`Could not switch survey: ${err?.message || err}`);
  }
  const select = $('#survey-select');
  if ([...select.options].some((o) => o.value === hipsId)) select.value = hipsId;
}

async function loadSurveyList() {
  let surveys;
  try {
    surveys = await api('cutoutSurveys');
  } catch {
    surveys = FALLBACK_SURVEYS;
  }
  state.surveys = surveys;
  const select = $('#survey-select');
  select.replaceChildren(...surveys.map((s) => h('option', { value: s.hips_id }, `${s.label} · ${s.wavelength}`)));
  select.value = surveys.some((s) => s.hips_id === DEFAULT_SURVEY) ? DEFAULT_SURVEY : surveys[0]?.hips_id;
  select.addEventListener('change', () => setSurvey(select.value));
}

async function skyShowTarget(target, radiusArcsec) {
  const aladin = await state.aladinReady;
  if (!aladin || !window.A) return;
  const A = window.A;
  aladin.removeLayers();
  state.skyCatalogs.clear();
  state.skySources.clear();
  aladin.gotoRaDec(target.ra, target.dec);
  aladin.setFoV(Math.max((radiusArcsec * 10) / 3600, 2 / 60));
  const ring = A.graphicOverlay({ name: 'Search radius', color: cssVar('--accent') || '#7aa2ff', lineWidth: 2 });
  aladin.addOverlay(ring);
  ring.add(A.circle(target.ra, target.dec, radiusArcsec / 3600));
  state.selectionOverlay = A.graphicOverlay({ name: 'Selection', color: '#ffd479', lineWidth: 3 });
  aladin.addOverlay(state.selectionOverlay);
}

async function skyAddSources(members) {
  const aladin = await state.aladinReady;
  if (!aladin || !window.A) return;
  const A = window.A;
  for (const m of members) {
    if (!Number.isFinite(Number(m.ra)) || !Number.isFinite(Number(m.dec))) continue;
    const key = sourceKey(m);
    if (state.skySources.has(key)) continue;
    let catalog = state.skyCatalogs.get(m.catalog);
    if (!catalog) {
      catalog = A.catalog({ name: m.catalog, color: colorFor(m.catalog), sourceSize: 14, shape: 'circle', onClick: 'showPopup' });
      aladin.addCatalog(catalog);
      state.skyCatalogs.set(m.catalog, catalog);
    }
    const source = A.source(Number(m.ra), Number(m.dec), {
      key,
      name: stripTags(m.source_id),
      catalog: stripTags(m.catalog),
      separation_arcsec: fmt(m.separation_arcsec),
      probability: fmt(m.confidence),
    });
    catalog.addSources([source]);
    state.skySources.set(key, source);
  }
  renderLegend();
}

async function skyAddSolarSystem(objects) {
  const aladin = await state.aladinReady;
  if (!aladin || !window.A || !objects.length) return;
  const A = window.A;
  const catalog = A.catalog({ name: 'Solar system', color: '#ffd479', sourceSize: 16, shape: 'cross', onClick: 'showPopup' });
  aladin.addCatalog(catalog);
  catalog.addSources(objects.filter((o) => Number.isFinite(Number(o.ra)) && Number.isFinite(Number(o.dec)))
    .map((o) => A.source(Number(o.ra), Number(o.dec), { name: stripTags(o.name), type: stripTags(o.type), v_mag: fmt(o.v_mag) })));
  state.skyCatalogs.set('Solar system', catalog);
  renderLegend();
}

function renderLegend() {
  const legend = $('#sky-legend');
  legend.replaceChildren(...[...state.skyCatalogs.entries()].map(([name, catalog]) => {
    const btn = h('button', { type: 'button', class: 'chip', 'aria-pressed': 'true', title: `Show or hide ${name}` },
      h('span', { class: 'swatch', style: { background: name === 'Solar system' ? '#ffd479' : colorFor(name) } }), name);
    btn.addEventListener('click', () => {
      const visible = btn.getAttribute('aria-pressed') === 'true';
      if (visible) catalog.hide(); else catalog.show();
      btn.setAttribute('aria-pressed', visible ? 'false' : 'true');
    });
    return btn;
  }));
}

async function skyHighlight(member) {
  const aladin = await state.aladinReady;
  if (!aladin || !window.A || !state.selectionOverlay) return;
  state.selectionOverlay.removeAll();
  state.selectionOverlay.add(window.A.circle(Number(member.ra), Number(member.dec), 2.5 / 3600));
  aladin.gotoRaDec(Number(member.ra), Number(member.dec));
}

// ---------------------------------------------------------------------------
// Search (SSE stream with POST fallback)
// ---------------------------------------------------------------------------

function sourceKey(member) {
  return `${member.catalog}|${member.source_id}`;
}

function progressStart(label) {
  const box = $('#progress');
  box.hidden = false;
  box.dataset.mode = 'indeterminate';
  box.setAttribute('aria-busy', 'true');
  $('#progress-label').textContent = label;
  $('#progress-count').textContent = '';
  $('#progress-bar').style.width = '';
  $('#catalog-chips').replaceChildren();
}

function progressUpdate() {
  const done = state.catalogStatus.size;
  const planned = state.plannedCatalogs;
  const box = $('#progress');
  if (planned > 0) {
    box.dataset.mode = 'determinate';
    const pct = Math.min(100, Math.round((100 * done) / planned));
    $('#progress-bar').style.width = `${pct}%`;
    box.querySelector('.progress-track').setAttribute('aria-valuenow', String(pct));
    $('#progress-count').textContent = `${done} / ${planned} catalogs`;
  } else {
    $('#progress-count').textContent = `${done} catalog${done === 1 ? '' : 's'} answered`;
  }
}

function progressCatalog(info) {
  const name = info.catalog;
  const status = info.status || 'ok';
  state.catalogStatus.set(name, info);
  let chip = [...$('#catalog-chips').children].find((li) => li.dataset.catalog === name);
  if (!chip) {
    chip = h('li', { dataset: { catalog: name } });
    $('#catalog-chips').append(chip);
  }
  chip.dataset.status = status === 'ok' || status === 'success' ? 'ok' : status;
  const count = info.count ?? info.matched_count ?? info.row_count;
  const extra = status === 'ok' || status === 'success'
    ? `${count ?? 0}${info.elapsed_ms != null ? ` · ${Math.round(info.elapsed_ms)} ms` : ''}`
    : status;
  chip.replaceChildren(h('span', { class: 'dot', style: { color: colorFor(name) } }), `${name} `, h('span', { class: 'muted' }, extra));
  chip.title = info.message || info.error_type || `${name}: ${extra}`;
  progressUpdate();
}

function progressDone(record) {
  const box = $('#progress');
  box.dataset.mode = 'determinate';
  box.setAttribute('aria-busy', 'false');
  $('#progress-bar').style.width = '100%';
  const failures = record?.failures?.length || 0;
  const groups = record?.crossmatch_groups?.length || 0;
  $('#progress-label').textContent = record
    ? `Done: ${groups} object${groups === 1 ? '' : 's'} from ${record.catalogs_queried} catalogs${failures ? `, ${failures} failed` : ''}`
    : 'Search stopped';
}

function buildSearchInput() {
  const text = $('#q').value.trim();
  if (!text) $('#q').focus();
  return parseSearchInput(text, $('#radius').value);
}

async function runSearch(input, advanced = null) {
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const { signal } = controller;
  Object.assign(state, { query: input, target: null, record: null, bibtex: '' });
  state.groups.clear();
  state.catalogStatus.clear();
  state.plannedCatalogs = 0;
  state.selectedKey = null;
  $('#download-json').disabled = true;
  $('#copy-bibtex').disabled = true;
  const params = new URLSearchParams({ q: input.text, r: String(input.radius) });
  history.replaceState(null, '', `?${params}`);
  progressStart(input.name ? `Resolving “${input.name}” and querying catalogs…` : 'Querying catalogs…');
  setPanel($('#results-panel'), 'loading');
  renderTargetPending(input);
  for (const id of ['#sed-panel', '#lc-panel', '#cutout-panel', '#sso-panel', '#cite-panel']) panelLoading($(id));
  panelLoading($('#explain-panel'));
  loadExplain(input, signal);

  if (input.ra !== undefined) onTargetKnown({ ra: input.ra, dec: input.dec }, signal);

  const body = advanced || (input.name ? { name: input.name, radius_arcsec: input.radius } : { ra: input.ra, dec: input.dec, radius_arcsec: input.radius });
  let record = null;
  try {
    const stream = storageGet(STORAGE.stream, 'true') !== 'false' && !advanced && 'TextDecoderStream' in globalThis;
    ({ record } = await searchRecord({
      input, body, stream, signal,
      onEvent: (event, data) => onStreamEvent(event, data, signal),
      onFallback: () => { $('#progress-label').textContent = 'Querying catalogs (waiting for the full result)…'; },
    }));
  } catch (err) {
    if (err.name === 'AbortError') return;
    const message = errorKind(err) === 'error' ? `Search failed: ${err.detail || err.message}` : errorMessage(err, 'The search service');
    progressDone(null);
    $('#progress-label').textContent = message;
    setPanel($('#results-panel'), 'error', note(message, 'error'));
    for (const id of ['#sed-panel', '#lc-panel', '#cutout-panel', '#sso-panel', '#cite-panel']) {
      if ($(id).dataset.state === 'loading') setPanel($(id), 'idle', note('Waiting for a successful search.'));
    }
    renderTargetError(err);
    return;
  }
  if (signal.aborted) return;
  handleRecord(record, signal);
}

// Run a search: the SSE stream first (when enabled), then POST /api/v1/search whenever
// the stream is missing, not an event stream, unreachable, or ends without 'done'.
// Returns { record, via: 'stream' | 'post' }. Bad input (422) and rate limiting (429)
// on the stream are real answers and are thrown instead of retried.
async function searchRecord({ input, body, stream = true, signal, onEvent = () => {}, onFallback = () => {} }) {
  if (stream) {
    const record = await streamSearch(input, signal, onEvent);
    if (record) return { record, via: 'stream' };
    if (signal?.aborted) throw new DOMException('Search aborted', 'AbortError');
  }
  onFallback();
  return { record: await api('search', { body, signal }), via: 'post' };
}

async function streamSearch(input, signal, onEvent = () => {}) {
  const params = input.name
    ? { name: input.name, radius_arcsec: input.radius }
    : { ra: input.ra, dec: input.dec, radius_arcsec: input.radius };
  let response;
  try {
    response = await fetch(endpointUrl('searchStream', params), { headers: apiHeaders({ Accept: 'text/event-stream' }), signal });
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    return null;
  }
  const type = response.headers.get('content-type') || '';
  if (!response.ok || !type.includes('text/event-stream')) {
    if (response.status === 422 || response.status === 429) throw await apiError(response, 'searchStream');
    return null; // endpoint missing or not streaming: the caller falls back to POST /api/v1/search
  }
  for await (const { event, data } of sseEvents(response)) {
    if (signal?.aborted) return null;
    if (event === 'error') throw new ApiError(data?.status || 502, data?.detail || String(data), 'searchStream');
    if (event === 'done') return data?.record || data;
    onEvent(event, data);
  }
  return null;
}

function onStreamEvent(event, data, signal) {
  if (event === 'start' || event === 'plan') {
    const planned = data?.catalogs || data?.catalogs_planned;
    if (Array.isArray(planned)) state.plannedCatalogs = planned.length;
    const target = data?.target || data?.resolved;
    if (target && Number.isFinite(Number(target.ra))) onTargetKnown(targetInfo(target), signal);
    progressUpdate();
  } else if (event === 'catalog') {
    progressCatalog(data);
    if (Array.isArray(data.sources) && data.sources.length) skyAddSources(data.sources);
  } else if (event === 'group') {
    state.groups.set(data.group_id, data);
    scheduleResultsRender();
  }
}

let renderTimer = null;
function scheduleResultsRender() {
  clearTimeout(renderTimer);
  renderTimer = setTimeout(() => renderResults([...state.groups.values()], null), 120);
}

function handleRecord(record, signal) {
  state.record = record;
  progressDone(record);
  for (const [name, result] of Object.entries(record.catalog_results || {})) {
    if (!state.catalogStatus.has(name)) {
      progressCatalog({ catalog: name, status: result.status === 'failed' ? 'failed' : 'ok', count: result.matched_count, elapsed_ms: result.elapsed_ms, message: result.message });
    }
  }
  if (!state.target) onTargetKnown(targetInfo(record.target), signal);
  renderTarget(record);
  const members = (record.crossmatch_groups || []).flatMap((g) => g.members || []);
  skyAddSources(members);
  renderResults(record.crossmatch_groups || [], record);
  $('#download-json').disabled = false;
  loadCitations(record, signal);
}

// Position plus, when the search resolved them, the proper motion, its epoch and the parallax
// (the cutout stack and the light curves use them to follow fast stars such as Barnard's star
// to each survey's epoch).
function targetInfo(source) {
  const target = { ra: Number(source.ra), dec: Number(source.dec) };
  for (const key of ['pm_ra_masyr', 'pm_dec_masyr', 'epoch', 'parallax_mas']) {
    if (source[key] !== null && source[key] !== undefined && Number.isFinite(Number(source[key]))) target[key] = Number(source[key]);
  }
  return target;
}

// The search target as the loaders see it: position (+ proper motion when known) plus the
// searched name, so the cutout stack can ask the server to resolve it (with proper motion)
// when the first target event carries none.
function knownTarget(target, query) {
  const name = target.name ?? query?.name;
  return name ? { ...target, name } : { ...target };
}

function onTargetKnown(target, signal) {
  if (state.target) return;
  const known = knownTarget(target, state.query);
  state.target = known;
  const radius = state.query?.radius ?? 10;
  skyShowTarget(known, radius);
  loadSed(known, radius, signal);
  loadLightcurves(known, radius, signal);
  loadCutouts(known, signal);
  loadSolarSystem(known, radius, signal);
}

// ---------------------------------------------------------------------------
// Target card
// ---------------------------------------------------------------------------

function renderTargetPending(input) {
  const label = input.name ? input.name : `${fmtDeg(input.ra)} ${fmtDeg(input.dec)}`;
  $('#target-body').replaceChildren(h('p', { class: 'target-name' }, label), h('p', { class: 'muted' }, 'Searching…'));
}

function renderTargetError(err) {
  $('#target-body').replaceChildren(note(err.detail || err.message, 'error'));
}

function renderTarget(record) {
  const resolved = record.resolved_object;
  const ra = Number(record.target.ra); const dec = Number(record.target.dec);
  const title = resolved?.canonical_name || state.query.name || `${fmtDeg(ra, 5)} ${fmtDeg(dec, 5)}`;
  const nSources = (record.crossmatch_groups || []).reduce((n, g) => n + (g.members?.length || 0), 0);
  const kv = h('dl', { class: 'kv' },
    h('dt', {}, 'RA, Dec (ICRS)'), h('dd', {}, `${fmtDeg(ra)}°, ${fmtDeg(dec)}°`),
    h('dt', {}, 'Sexagesimal'), h('dd', {}, sexagesimal(ra, dec)),
    record.target.epoch ? [h('dt', {}, 'Epoch'), h('dd', {}, `J${record.target.epoch}`)] : null,
    resolved?.object_type ? [h('dt', {}, 'Type (SIMBAD)'), h('dd', {}, resolved.object_type)] : null,
    resolved?.redshift != null ? [h('dt', {}, 'Redshift'), h('dd', {}, fmt(resolved.redshift, 5))] : null,
    resolved?.aliases?.length ? [h('dt', {}, 'Aliases'), h('dd', {}, resolved.aliases.slice(0, 6).join(' · '))] : null,
    h('dt', {}, 'Search radius'), h('dd', {}, `${record.provenance?.query_radius_arcsec ?? state.query.radius}″`),
  );
  const stats = h('div', { class: 'stats' },
    stat(record.catalogs_queried, 'catalogs queried'),
    stat((record.crossmatch_groups || []).length, 'physical objects'),
    stat(nSources, 'matched sources'),
    stat((record.failures || []).length, 'failed catalogs'),
  );
  const bands = Object.entries(record.counterparts || {}).map(([wave, list]) => h('span', { class: 'badge' }, `${wave} ${list.length}`));
  $('#target-body').replaceChildren(...[h('p', { class: 'target-name' }, title), kv, stats, bands.length ? h('div', { class: 'row-actions' }, bands) : null].filter(Boolean));
}

function stat(value, label) {
  return h('div', { class: 'stat' }, h('b', {}, String(value ?? 0)), h('span', {}, label));
}

// ---------------------------------------------------------------------------
// Results table (grouped by physical object)
// ---------------------------------------------------------------------------

function groupProbability(group) {
  if (Number.isFinite(Number(group.probability))) return Number(group.probability);
  const values = (group.members || []).map((m) => Number(m.confidence)).filter(Number.isFinite);
  return values.length ? Math.max(...values) : null;
}

function renderResults(groups, record) {
  const panel = $('#results-panel');
  const sorted = [...groups].sort((a, b) => (groupProbability(b) ?? 0) - (groupProbability(a) ?? 0)
    || minSep(a) - minSep(b));
  const members = sorted.reduce((n, g) => n + (g.members?.length || 0), 0);
  $('#results-meta').textContent = groups.length ? `${groups.length} objects · ${members} sources` : '';
  if (!groups.length) {
    const failures = record?.failures?.length ? ` ${record.failures.length} catalog(s) failed.` : '';
    setPanel(panel, record ? 'empty' : 'loading', record ? note(`No catalog sources within the search radius.${failures}`, 'empty') : '');
    return;
  }
  const tbody = h('tbody');
  for (const group of sorted) {
    const prob = groupProbability(group);
    tbody.append(h('tr', { class: 'group-row' }, h('th', { colspan: '6', scope: 'rowgroup' },
      h('div', { class: 'group-title' },
        h('span', {}, group.group_id),
        h('span', { class: 'badge' }, `${group.members?.length || 0} sources`),
        h('span', { class: 'muted small' }, (group.catalogs || []).join(', ')),
        (group.wavelengths || []).map((w) => h('span', { class: 'badge ok' }, w)),
        prob != null ? h('span', { class: 'small' }, `best match ${Math.round(prob * 100)}%`) : null))));
    const rows = [...(group.members || [])].sort((a, b) => Number(a.separation_arcsec) - Number(b.separation_arcsec));
    for (const m of rows) {
      const key = sourceKey(m);
      const p = Number(m.confidence);
      const tr = h('tr', { class: 'member', tabindex: '0', dataset: { key }, 'aria-label': `${m.catalog} ${m.source_id}` },
        h('td', {}, h('span', { class: 'swatch', style: { background: colorFor(m.catalog) } }), ' ', m.catalog),
        h('td', { class: 'srcid' }, String(m.source_id)),
        h('td', {}, String(m.metadata?.wavelength || '')),
        h('td', { class: 'num' }, fmt(m.separation_arcsec)),
        h('td', {}, Number.isFinite(p) ? h('div', { class: 'prob', title: `match probability ${p.toFixed(3)}` },
          h('div', { class: 'prob-bar' }, h('i', { style: { width: `${Math.round(p * 100)}%` } })), `${Math.round(p * 100)}%`) : '—'),
        h('td', { class: 'num mono' }, `${fmtDeg(m.ra, 5)} ${fmtDeg(m.dec, 5)}`));
      tr.addEventListener('click', () => selectSource(key));
      tr.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); selectSource(key); } });
      tbody.append(tr);
    }
  }
  const table = h('table', {},
    h('caption', { class: 'visually-hidden' }, 'Catalog sources grouped by physical object'),
    h('thead', {}, h('tr', {},
      h('th', { scope: 'col' }, 'Catalog'), h('th', { scope: 'col' }, 'Source ID'), h('th', { scope: 'col' }, 'Band'),
      h('th', { scope: 'col', class: 'num' }, 'Sep (″)'), h('th', { scope: 'col' }, 'Probability'),
      h('th', { scope: 'col', class: 'num' }, 'RA Dec (°)'))),
    tbody);
  setPanel(panel, 'ready', h('div', { class: 'table-wrap' }, table));
}

function minSep(group) {
  const seps = (group.members || []).map((m) => Number(m.separation_arcsec)).filter(Number.isFinite);
  return seps.length ? Math.min(...seps) : Infinity;
}

function findMember(key) {
  const groups = state.record?.crossmatch_groups || [...state.groups.values()];
  for (const g of groups) for (const m of g.members || []) if (sourceKey(m) === key) return m;
  return null;
}

function selectSource(key, { fromSky = false } = {}) {
  state.selectedKey = key;
  for (const row of document.querySelectorAll('tr.member.selected')) row.classList.remove('selected');
  const row = [...document.querySelectorAll('tr.member')].find((r) => r.dataset.key === key);
  if (row) {
    row.classList.add('selected');
    if (fromSky) row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  const member = findMember(key);
  if (member && !fromSky) skyHighlight(member);
}

// ---------------------------------------------------------------------------
// SED
// ---------------------------------------------------------------------------

function nuFnu(p) {
  if (Number.isFinite(Number(p.nu_fnu_erg_s_cm2))) return Number(p.nu_fnu_erg_s_cm2);
  const nu = Number(p.frequency_hz); const f = Number(p.flux_jy);
  return Number.isFinite(nu) && Number.isFinite(f) ? nu * f * 1e-23 : NaN; // 1 Jy = 1e-23 erg s^-1 cm^-2 Hz^-1
}

// A panel's service radius: the search radius, capped at that service's own limit.
function serviceRadius(radius, max) {
  const r = Number(radius);
  return r > max ? { radius: max, clamped: true } : { radius: r, clamped: false };
}

function clampNote(requested, used, service) {
  return h('p', { class: 'small muted' },
    `The ${service} service accepts at most ${used}″, so ${used}″ was used instead of the ${requested}″ search radius.`);
}

async function loadSed(target, radius, signal) {
  const panel = $('#sed-panel');
  panelLoading(panel);
  const cone = serviceRadius(radius, SED_MAX_RADIUS_ARCSEC);
  try {
    // A name search goes to POST /sed with the name: the service resolves it with Sesame and
    // moves the position by the proper motion to each catalogue's epoch (e.g. Barnard's star).
    const name = state.query?.name;
    state.sedData = name
      ? await api('sedByName', { body: { name, radius_arcsec: cone.radius }, signal })
      : await api('sed', { params: { ra: target.ra, dec: target.dec, radius_arcsec: cone.radius }, signal });
    state.sedData.clampNote = cone.clamped ? { requested: Number(radius), used: cone.radius } : null;
    renderSed();
  } catch (err) {
    state.sedData = null;
    panelError(panel, err, 'The SED service');
  }
}

function renderSed() {
  const data = state.sedData;
  const panel = $('#sed-panel');
  const points = (data.points || []).filter((p) => Number(p.wavelength_um) > 0 && nuFnu(p) > 0);
  const children = [];
  if (points.length) {
    children.push(window.Chart
      ? h('div', { class: 'chart-box' }, h('canvas', { id: 'sed-canvas', role: 'img', 'aria-label': `SED with ${points.length} photometric points` }))
      : sedTable(points));
  } else {
    children.push(note('No photometry found for this position.', 'empty'));
  }
  const cls = data.classification;
  if (cls) {
    const scores = Object.entries(cls.scores || {}).sort((a, b) => b[1] - a[1]);
    children.push(h('div', { class: 'class-card' },
      h('div', {}, h('span', { class: 'class-label' }, cls.label || 'unknown'),
        cls.confidence != null ? h('span', { class: 'muted small' }, `  confidence ${Math.round(cls.confidence * 100)}%`) : null),
      scores.map(([k, v]) => h('div', { class: 'score-row' }, h('span', {}, k),
        h('div', { class: 'prob-bar' }, h('i', { style: { width: `${Math.round(Math.max(0, Math.min(1, v)) * 100)}%` } })),
        h('span', { class: 'small' }, `${Math.round(v * 100)}%`))),
      cls.evidence?.length ? h('ul', { class: 'evidence' }, cls.evidence.map((e) => h('li', {}, e))) : null));
  }
  const z = data.redshift;
  if (z && z.value != null) {
    children.push(h('p', { class: 'small' }, h('b', {}, `z = ${fmt(z.value, 4)}`),
      z.error != null ? ` ± ${fmt(z.error, 2)}` : '', ` (${z.kind || 'unknown'}${z.source ? `, ${z.source}` : ''})`));
  }
  if (data.clampNote) children.push(clampNote(data.clampNote.requested, data.clampNote.used, 'SED'));
  setPanel(panel, 'ready', ...children);
  if (points.length && window.Chart) renderSedChart();
}

function sedTable(points) {
  return h('div', { class: 'table-wrap' }, h('table', {},
    h('thead', {}, h('tr', {}, ['Band', 'Facility', 'λ (µm)', 'Fν (Jy)', 'νFν'].map((t) => h('th', { scope: 'col' }, t)))),
    h('tbody', {}, points.map((p) => h('tr', {}, h('td', {}, p.band), h('td', {}, p.facility),
      h('td', { class: 'num' }, fmt(p.wavelength_um)), h('td', { class: 'num' }, `${p.is_upper_limit ? '<' : ''}${fmt(p.flux_jy)}`),
      h('td', { class: 'num' }, fmt(nuFnu(p))))))));
}

function renderSedChart() {
  const canvas = $('#sed-canvas');
  if (!canvas || !window.Chart) return;
  state.charts.sed?.destroy();
  const points = (state.sedData.points || []).filter((p) => Number(p.wavelength_um) > 0 && nuFnu(p) > 0);
  const byFacility = new Map();
  for (const p of points) {
    const key = p.facility || p.catalog || 'other';
    if (!byFacility.has(key)) byFacility.set(key, []);
    const y = nuFnu(p);
    const rel = Number(p.flux_err_jy) / Number(p.flux_jy);
    const err = Number.isFinite(rel) && rel > 0 && !p.is_upper_limit ? y * rel : null;
    byFacility.get(key).push({ x: Number(p.wavelength_um), y, yLo: err ? Math.max(y - err, y * 1e-3) : null, yHi: err ? y + err : null, ul: !!p.is_upper_limit, p });
  }
  const text = cssVar('--text'); const grid = cssVar('--border');
  state.charts.sed = new window.Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [...byFacility.entries()].map(([name, data]) => ({
        label: name, data, borderColor: colorFor(name), backgroundColor: colorFor(name),
        pointRadius: 4.5, pointHoverRadius: 7,
        pointStyle: (ctx) => (ctx.raw?.ul ? 'triangle' : 'circle'),
        rotation: (ctx) => (ctx.raw?.ul ? 180 : 0),
      })),
    },
    options: chartOptions({
      x: { type: 'logarithmic', title: 'Wavelength (µm)' },
      y: { type: 'logarithmic', title: 'νFν (erg s⁻¹ cm⁻²)' },
      text, grid,
      tooltip: (ctx) => {
        const p = ctx.raw.p;
        return `${p.facility || ''} ${p.band || ''}: ${p.is_upper_limit ? '< ' : ''}${fmt(p.flux_jy)} Jy at ${fmt(p.wavelength_um)} µm`;
      },
    }),
    plugins: [errorBarsPlugin],
  });
}

const errorBarsPlugin = {
  id: 'errorBars',
  afterDatasetsDraw(chart) {
    const { ctx } = chart;
    const yScale = chart.scales.y;
    chart.data.datasets.forEach((ds, i) => {
      const meta = chart.getDatasetMeta(i);
      if (meta.hidden) return;
      ctx.save();
      ctx.strokeStyle = ds.borderColor;
      ctx.lineWidth = 1;
      ds.data.forEach((pt, j) => {
        const el = meta.data[j];
        if (!el || pt.yLo == null || pt.yHi == null) return;
        const y1 = yScale.getPixelForValue(pt.yLo); const y2 = yScale.getPixelForValue(pt.yHi);
        ctx.beginPath(); ctx.moveTo(el.x, y1); ctx.lineTo(el.x, y2);
        ctx.moveTo(el.x - 3, y1); ctx.lineTo(el.x + 3, y1); ctx.moveTo(el.x - 3, y2); ctx.lineTo(el.x + 3, y2);
        ctx.stroke();
      });
      ctx.restore();
    });
  },
};

function chartOptions({ x, y, text, grid, tooltip, reverseY = false }) {
  const axis = (spec, reverse = false) => ({
    type: spec.type,
    reverse,
    min: spec.min, max: spec.max,
    title: { display: true, text: spec.title, color: text },
    grid: { color: grid },
    ticks: {
      color: text,
      maxRotation: 0,
      autoSkip: true,
      callback: spec.type === 'logarithmic' ? (v) => powerOfTen(v) : undefined,
    },
  });
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? false : { duration: 250 },
    parsing: false,
    scales: { x: axis(x), y: axis(y, reverseY) },
    plugins: {
      legend: { labels: { color: text, boxWidth: 10, usePointStyle: true } },
      tooltip: { callbacks: { label: tooltip } },
    },
  };
}

// ---------------------------------------------------------------------------
// Light curves
// ---------------------------------------------------------------------------

async function loadLightcurves(target, radius, signal) {
  const panel = $('#lc-panel');
  panelLoading(panel);
  const cone = serviceRadius(Math.max(radius, 1), LIGHTCURVE_MAX_RADIUS_ARCSEC);
  try {
    state.lcData = await api('lightcurves', { params: lightcurveParams(target, cone.radius), signal });
    state.lcData.clampNote = cone.clamped ? { requested: Number(radius), used: cone.radius } : null;
    const units = [...new Set((state.lcData.series || []).map((s) => s.unit))];
    state.lcView = { unit: units.includes('mag') ? 'mag' : units[0] || null, folded: false };
    renderLightcurves();
  } catch (err) {
    state.lcData = null;
    panelError(panel, err, 'The light-curve service');
  }
}

// Light-curve query for a target: its position and, when known, the epoch of that position with
// the proper motion (each survey's cone then follows a fast star to that survey's epoch, as the
// cutout stack does) and the parallax (it widens the per-epoch identity tolerance of nearby stars).
// A proper motion without the epoch of the position cannot be applied and is not sent.
function lightcurveParams(target, radius) {
  const params = { ra: target.ra, dec: target.dec, radius_arcsec: radius, surveys: 'ztf,neowise,gaia' };
  const finite = (value) => value !== null && value !== undefined && Number.isFinite(Number(value));
  if (finite(target.epoch) && finite(target.pm_ra_masyr) && finite(target.pm_dec_masyr)) {
    Object.assign(params, { epoch: Number(target.epoch), pm_ra_masyr: Number(target.pm_ra_masyr),
      pm_dec_masyr: Number(target.pm_dec_masyr) });
  }
  const plx = Number(target.parallax_mas);
  if (finite(target.parallax_mas) && plx >= 0 && plx < 1000) params.parallax_mas = plx;
  return params;
}

function renderLightcurves() {
  const data = state.lcData;
  const panel = $('#lc-panel');
  const series = (data.series || []).filter((s) => (s.points || []).length);
  const clamp = data.clampNote ? clampNote(data.clampNote.requested, data.clampNote.used, 'light-curve') : null;
  if (!series.length) {
    setPanel(panel, 'empty', note('No time-series photometry found for this position.', 'empty'), clamp);
    return;
  }
  const units = [...new Set(series.map((s) => s.unit))];
  const toolbar = h('div', { class: 'chart-toolbar' });
  if (units.length > 1) {
    toolbar.append(h('div', { class: 'seg', role: 'group', 'aria-label': 'Units' }, units.map((u) => h('button', {
      type: 'button', 'aria-pressed': String(state.lcView.unit === u),
      onclick: () => { state.lcView.unit = u; renderLightcurves(); },
    }, u === 'mag' ? 'Magnitudes' : 'Flux'))));
  }
  const period = data.period;
  if (period && Number(period.best_period_days) > 0) {
    toolbar.append(h('div', { class: 'seg', role: 'group', 'aria-label': 'Time axis' },
      h('button', { type: 'button', 'aria-pressed': String(!state.lcView.folded), onclick: () => { state.lcView.folded = false; renderLightcurves(); } }, 'Time'),
      h('button', { type: 'button', 'aria-pressed': String(state.lcView.folded), onclick: () => { state.lcView.folded = true; renderLightcurves(); } }, 'Phase-folded')));
    toolbar.append(h('span', { class: 'small' }, `P = ${fmt(period.best_period_days, 6)} d · power ${fmt(period.power)} · FAP ${fmt(period.false_alarm_probability)}${period.series ? ` (${period.series})` : ''}`));
  }
  const children = [toolbar];
  children.push(window.Chart
    ? h('div', { class: 'chart-box' }, h('canvas', { id: 'lc-canvas', role: 'img', 'aria-label': 'Light curve' }))
    : note('Charting library unavailable; metrics are listed below.', 'unavailable'));
  const perSeries = data.variability?.per_series || {};
  const metricRows = Object.entries(perSeries).map(([name, metrics]) => h('tr', {}, h('th', { scope: 'row' }, name),
    h('td', {}, Object.entries(metrics || {}).filter(([, v]) => typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string')
      .slice(0, 10).map(([k, v]) => `${k}=${typeof v === 'number' ? fmt(v) : v}`).join(' · '))));
  if (metricRows.length) {
    children.push(h('details', {}, h('summary', {}, 'Variability metrics'),
      h('div', { class: 'table-wrap' }, h('table', { class: 'metrics' }, h('tbody', {}, metricRows)))));
  }
  if (clamp) children.push(clamp);
  setPanel(panel, 'ready', ...children);
  if (window.Chart) renderLightcurveChart();
}

// Phase in [0, 1) of an epoch for period P and reference epoch t0 (all in days).
function foldPhase(mjd, period, t0) {
  return ((((Number(mjd) - t0) / period) % 1) + 1) % 1;
}

// One reference epoch for every series: the earliest epoch of the whole light-curve set.
// The service reports all epochs as BMJD_TDB, so a single t0 keeps the bands of one
// periodic source in phase with each other (a per-series t0 would shift each band by an
// arbitrary fraction of a cycle).
function lightcurveEpoch(series) {
  let t0 = Infinity;
  for (const s of series || []) for (const p of s.points || []) {
    const t = Number(p.mjd);
    if (Number.isFinite(t) && t < t0) t0 = t;
  }
  return Number.isFinite(t0) ? t0 : null;
}

// Chart points per series for the current view (time axis, or folded on one shared t0).
function lightcurveDatasets(lcData, view) {
  const { unit, folded } = view;
  const period = Number(lcData?.period?.best_period_days);
  const all = (lcData?.series || []).filter((s) => (s.points || []).length);
  const t0 = lightcurveEpoch(all);
  const fold = folded && period > 0 && t0 !== null;
  const datasets = all.filter((s) => s.unit === unit).map((s) => {
    const data = [];
    for (const p of s.points) {
      const v = Number(p.value); const e = Number(p.error);
      if (!Number.isFinite(v) || !Number.isFinite(Number(p.mjd))) continue;
      const base = { y: v, yLo: Number.isFinite(e) ? v - e : null, yHi: Number.isFinite(e) ? v + e : null, flag: p.flag, mjd: Number(p.mjd) };
      if (fold) {
        const phase = foldPhase(p.mjd, period, t0);
        data.push({ ...base, x: phase }, { ...base, x: phase + 1 });
      } else {
        data.push({ ...base, x: Number(p.mjd) });
      }
    }
    return { label: `${s.survey} ${s.band}`, data };
  });
  return { datasets, folded: fold, period, t0 };
}

function renderLightcurveChart() {
  const canvas = $('#lc-canvas');
  if (!canvas || !window.Chart || !state.lcData) return;
  state.charts.lc?.destroy();
  const { unit } = state.lcView;
  const { datasets: raw, folded, period, t0 } = lightcurveDatasets(state.lcData, state.lcView);
  const text = cssVar('--text'); const grid = cssVar('--border');
  const datasets = raw.map(({ label, data }) => {
    const color = colorFor(label);
    return {
      label, data, borderColor: color, backgroundColor: (ctx) => (ctx.raw?.flag ? 'transparent' : color),
      pointRadius: 3, pointHoverRadius: 6, pointBorderWidth: 1,
    };
  });
  state.charts.lc = new window.Chart(canvas, {
    type: 'scatter',
    data: { datasets },
    options: chartOptions({
      x: folded
        ? { type: 'linear', title: `Phase (P = ${fmt(period, 6)} d, T₀ = BMJD ${fmt(t0, 7)})`, min: 0, max: 2 }
        : { type: 'linear', title: 'BMJD (TDB)' },
      y: { type: 'linear', title: unit === 'mag' ? 'Magnitude' : 'Flux' },
      reverseY: unit === 'mag',
      text, grid,
      tooltip: (ctx) => `${ctx.dataset.label}: ${fmt(ctx.raw.y, 4)}${ctx.raw.yHi != null ? ` ± ${fmt(ctx.raw.yHi - ctx.raw.y, 2)}` : ''} at MJD ${fmt(ctx.raw.mjd, 7)}${ctx.raw.flag ? ` (flag ${ctx.raw.flag})` : ''}`,
    }),
    plugins: [errorBarsPlugin],
  });
}

// ---------------------------------------------------------------------------
// Multi-wavelength cutouts
// ---------------------------------------------------------------------------

const objectUrls = [];
// One strip at a time: each loadCutouts() aborts the previous strip's requests, and a
// generation counter drops answers that arrive after a newer strip started (FoV changes).
const cutoutStrip = { controller: null, generation: 0 };

// Stack query for a target: its proper motion when known (panels then follow the target
// to each survey's epoch), else the name (the server resolves it, with proper motion,
// through Sesame), else the bare position.
function cutoutStackParams(target, fov) {
  const pm = [target.pm_ra_masyr, target.pm_dec_masyr].map(Number);
  const hasPm = target.pm_ra_masyr != null && target.pm_dec_masyr != null && pm.every(Number.isFinite);
  if (!hasPm && target.name) return { name: target.name, fov_arcmin: fov };
  const params = { ra: target.ra, dec: target.dec, fov_arcmin: fov };
  if (hasPm) {
    Object.assign(params, { pm_ra_masyr: pm[0], pm_dec_masyr: pm[1] });
    if (Number.isFinite(Number(target.epoch)) && target.epoch != null) params.epoch = Number(target.epoch);
  }
  return params;
}

async function loadCutouts(target, signal) {
  const panel = $('#cutout-panel');
  cutoutStrip.controller?.abort();
  const controller = new AbortController();
  cutoutStrip.controller = controller;
  const generation = ++cutoutStrip.generation;
  if (signal?.aborted) controller.abort();
  else signal?.addEventListener?.('abort', () => controller.abort(), { once: true });
  const stripSignal = controller.signal;
  const current = () => generation === cutoutStrip.generation && !stripSignal.aborted;
  panelLoading(panel);
  while (objectUrls.length) URL.revokeObjectURL(objectUrls.pop());
  let stack;
  try {
    stack = await api('cutoutStack', { params: cutoutStackParams(target, $('#cutout-fov').value), signal: stripSignal });
  } catch (err) {
    if (current()) panelError(panel, err, 'The cutout service');
    return;
  }
  if (!current()) return;
  if (!stack.panels?.length) {
    setPanel(panel, 'empty', note('No survey covers this position.', 'empty'));
    return;
  }
  const strip = h('div', { class: 'strip', role: 'list' });
  const children = [strip];
  const pm = stack.proper_motion;
  if (pm) {
    const rate = Math.hypot(pm.pm_ra_masyr, pm.pm_dec_masyr) / 1000;
    children.push(h('p', { class: 'small muted' },
      `Proper motion ${fmt(rate, 3)}″/yr: each panel is centred on the target's position at that survey's mean epoch.`));
  }
  if (stack.coverage_error) children.push(h('p', { class: 'small muted' }, `Footprint check unavailable: ${stack.coverage_error}`));
  setPanel(panel, 'ready', ...children);
  const pending = [];
  for (const p of stack.panels) {
    const frame = h('div', { class: 'frame' }, h('span', { class: 'muted small' }, 'Loading…'));
    const moved = p.epoch != null && Number(p.offset_arcsec) > 0;
    const fig = h('figure', { class: 'cutout', role: 'listitem', dataset: { survey: p.survey } }, frame,
      h('figcaption', {}, h('b', {}, p.label), h('span', { class: 'muted' }, p.wavelength),
        moved ? h('span', { class: 'muted small' }, `Epoch ${fmt(p.epoch, 5)}: ${fmt(p.offset_arcsec, 3)}″ from the catalogue position`) : null,
        p.note ? h('span', { class: 'small panel-note' }, p.note) : null,
        h('span', { class: 'links' },
          h('button', { type: 'button', onclick: () => setSurvey(p.hips_id), title: `Show ${p.label} in the sky viewer` }, 'View in sky'),
          fitsButton(p))));
    strip.append(fig);
    const where = moved ? `centred on the target's expected position at epoch ${fmt(p.epoch, 5)}` : 'centred on the target';
    pending.push(fetchBlob(p.url, stripSignal).then(({ blob, headers }) => {
      if (!current()) return;
      const url = URL.createObjectURL(blob);
      objectUrls.push(url);
      const status = cutoutStatus(headers, p);
      frame.replaceChildren(...[
        h('img', { src: url, alt: `${p.label} (${p.wavelength}) cutout, ${stack.fov_arcmin} arcmin field ${where}`, loading: 'lazy', width: String(stack.size_px), height: String(stack.size_px) }),
        h('span', { class: 'crosshair', 'aria-hidden': 'true' }),
        status.message ? h('span', { class: 'overlay-note', title: status.detail || status.message }, status.message) : null,
      ].filter(Boolean));
      if (status.kind !== 'ok') fig.classList.add(status.kind);
    }).catch((err) => {
      if (err.name === 'AbortError' || !current()) return;
      frame.replaceChildren(h('span', { class: 'small', style: { color: 'var(--err)', padding: '8px', textAlign: 'center' } },
        ['rate-limited', 'temporarily-unavailable'].includes(errorKind(err)) ? errorMessage(err, 'This cutout') : err.detail || err.message));
    }));
  }
  await Promise.all(pending);
}

// What the image of one panel shows: 'ok', 'degraded' (hips2fits returned a blank image
// that is a rendering failure, not the footprint), 'nodata' (blank, and the survey's MOC
// confirms it does not reach the field) or 'unconfirmed' (blank, footprint not checkable).
// X-Cutout-Blank: no-data | unconfirmed | rendering-failure; JPEG blanks carry it too.
function cutoutStatus(headers, panel) {
  const degraded = headers?.get?.('X-Cutout-Degraded');
  if (degraded) return { kind: 'degraded', message: 'Rendering failed upstream', detail: degraded };
  const blank = headers?.get?.('X-Cutout-Blank');
  if (blank === 'no-data' || panel.in_coverage === false) return { kind: 'nodata', message: 'No survey data here' };
  const coverage = headers?.get?.('X-Cutout-Coverage');
  const empty = coverage !== null && coverage !== undefined && Number(coverage) === 0;
  if (blank || empty) {
    return { kind: 'unconfirmed', message: 'Blank image; footprint could not be checked',
      detail: 'The image has no survey pixels, but whether the survey covers this field could not be checked: it may be empty sky coverage or an upstream rendering failure.' };
  }
  return { kind: 'ok', message: null };
}

// FITS link of a panel: single-band survey pixel values, from the survey itself or, for a
// colour composite (8-bit RGB display planes), from a single-band band of it that covers
// the position. The label states the pixel units; only physical units count as calibrated
// (DSS2 red is photographic density, 2MASS/AllWISE are DN, Pan-STARRS1 counts).
function fitsButton(p) {
  if (!p.fits_url) {
    return h('button', { type: 'button', disabled: true, title: p.fits_note || `${p.label} has no single-band FITS survey` }, 'No FITS');
  }
  const own = p.fits_survey === p.survey;
  const units = p.fits_pixel_units || 'unknown units';
  const values = `single-band survey pixel values: ${units}${p.fits_calibrated ? '' : ', not flux-calibrated'}`;
  const title = own ? `Download ${p.label} FITS (${values})`
    : `Download ${p.fits_label} FITS (${values}). ${p.label} itself is a colour composite, not survey data.`
      + (p.fits_note ? ` ${p.fits_note}` : '');
  return h('button', { type: 'button', onclick: () => downloadFits(p), title },
    `${own ? 'FITS' : `FITS: ${p.fits_label}`} (${units})`);
}

async function downloadFits(panel) {
  try {
    const { blob } = await fetchBlob(panel.fits_url);
    const href = URL.createObjectURL(blob);
    const a = h('a', { href, download: `${panel.fits_survey}_${fmtDeg(state.target.ra, 5)}_${fmtDeg(state.target.dec, 5)}.fits` });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 10000);
  } catch (err) {
    toast(`FITS download failed: ${err.detail || err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Solar-system objects
// ---------------------------------------------------------------------------

async function loadSolarSystem(target, radius, signal) {
  const panel = $('#sso-panel');
  panelLoading(panel);
  const searchRadius = Math.max(radius, 600);
  const epoch = nowMjd();
  try {
    const data = await api('solarSystem', { params: { ra: target.ra, dec: target.dec, radius_arcsec: searchRadius, epoch_mjd: epoch.toFixed(5) }, signal });
    const objects = data.objects || [];
    if (!objects.length) {
      setPanel(panel, 'empty', note(`No known asteroids or comets within ${searchRadius / 60}′ at MJD ${epoch.toFixed(3)}.`, 'empty'));
      return;
    }
    setPanel(panel, 'ready', h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, ['Name', 'Type', 'Sep (″)', 'V (mag)'].map((t, i) => h('th', { scope: 'col', class: i > 1 ? 'num' : null }, t)))),
      h('tbody', {}, objects.map((o) => h('tr', {}, h('td', {}, o.name), h('td', {}, o.type || ''),
        h('td', { class: 'num' }, fmt(o.separation_arcsec)), h('td', { class: 'num' }, fmt(o.v_mag))))))),
    h('p', { class: 'small muted' }, `Positions at MJD ${epoch.toFixed(3)} within ${searchRadius / 60}′.`));
    skyAddSolarSystem(objects);
  } catch (err) {
    panelError(panel, err, 'The solar-system service');
  }
}

// ---------------------------------------------------------------------------
// Explain (AI) & citations
// ---------------------------------------------------------------------------

async function loadExplain(input, signal) {
  const panel = $('#explain-panel');
  panelLoading(panel);
  try {
    const body = input.name ? { name: input.name } : { ra: input.ra, dec: input.dec };
    const data = await api('aiExplain', { body, signal });
    const paragraphs = String(data.summary || '').split(/\n{2,}/).filter(Boolean).map((t) => h('p', { class: 'explain-summary' }, t));
    const facts = Object.entries(data.facts || {}).filter(([, v]) => v !== null && v !== undefined && v !== '');
    const cites = data.citations || [];
    setPanel(panel, 'ready',
      ...(paragraphs.length ? paragraphs : [note('No summary returned.', 'empty')]),
      facts.length ? h('dl', { class: 'kv' }, facts.slice(0, 16).flatMap(([k, v]) => [h('dt', {}, k.replace(/_/g, ' ')), h('dd', {}, typeof v === 'object' ? JSON.stringify(v) : String(v))])) : null,
      cites.length ? h('ol', { class: 'cite-list' }, cites.map((c) => h('li', {},
        link(c.url || (c.bibcode ? `https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(c.bibcode)}/abstract` : ''), c.title || c.bibcode || 'reference'),
        c.year ? ` (${c.year})` : '', c.bibcode ? h('span', { class: 'muted small' }, ` ${c.bibcode}`) : null))) : null);
  } catch (err) {
    panelError(panel, err, 'The explain service');
  }
}

function contributingCatalogs(record) {
  const fromResults = Object.entries(record.catalog_results || {})
    .filter(([, r]) => r.status !== 'failed' && Number(r.matched_count ?? r.row_count ?? 0) > 0).map(([name]) => name);
  const fromGroups = (record.crossmatch_groups || []).flatMap((g) => g.catalogs || []);
  return [...new Set([...fromResults, ...fromGroups])].sort();
}

async function loadCitations(record, signal) {
  const panel = $('#cite-panel');
  const catalogs = contributingCatalogs(record);
  if (!catalogs.length) {
    setPanel(panel, 'empty', note('No catalog contributed sources, so there is nothing to cite yet.', 'empty'));
    return;
  }
  panelLoading(panel);
  try {
    const data = await api('citations', { params: { catalogs: catalogs.join(',') }, signal });
    state.bibtex = data.bibtex || '';
    const acks = (data.acknowledgements || []).map((a) => (typeof a === 'string' ? h('li', {}, a)
      : h('li', {}, a.catalog ? h('b', {}, `${a.catalog}: `) : null, a.text || a.acknowledgement || a.citation || JSON.stringify(a))));
    setPanel(panel, 'ready',
      acks.length ? h('ul', { class: 'cite-list' }, acks) : null,
      state.bibtex ? h('details', {}, h('summary', {}, `BibTeX (${(state.bibtex.match(/^@/gm) || []).length} entries)`), h('pre', { class: 'code' }, state.bibtex)) : null,
      state.bibtex ? h('div', { class: 'row-actions' }, h('button', { type: 'button', class: 'btn btn-ghost btn-small', onclick: downloadBibtex }, 'Download .bib')) : null);
    $('#copy-bibtex').disabled = !state.bibtex;
  } catch (err) {
    if (err.name === 'AbortError') return;
    // Fall back to the per-catalog citations carried in the search record itself.
    const items = catalogs.map((name) => {
      const r = record.catalog_results?.[name] || {};
      return h('li', {}, h('b', {}, `${name}: `), r.citation || r.acknowledgement || 'no citation recorded');
    });
    setPanel(panel, 'ready', note(`The citations service is ${err instanceof ApiError && err.unavailable ? 'unavailable' : `failing (${err.detail || err.message})`}; showing citations recorded with the search.`, 'unavailable'),
      h('ul', { class: 'cite-list' }, items));
  }
}

function downloadBibtex() {
  const href = URL.createObjectURL(new Blob([state.bibtex], { type: 'application/x-bibtex' }));
  const a = h('a', { href, download: 'astrosearch_citations.bib' });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 10000);
}

// ---------------------------------------------------------------------------
// Ask the sky (natural language -> AdvancedQuery)
// ---------------------------------------------------------------------------

const SEARCH_FIELDS = ['object_types', 'spectral_types', 'morphology', 'max_results', 'min_confidence', 'catalogs',
  'time_period', 'spatial_constraints', 'search_mode', 'min_radius_arcsec', 'proper_motion', 'adaptive_radius',
  'min_distance_pc', 'max_distance_pc'];

function searchBodyFromAdvanced(aq) {
  const target = aq.target || {};
  const body = { radius_arcsec: Number(aq.radius_arcsec) > 0 ? Number(aq.radius_arcsec) : Number($('#radius').value) || 10 };
  if (Number.isFinite(Number(target.ra)) && Number.isFinite(Number(target.dec)) && !(aq.use_resolved_name && aq.resolved_name)) {
    body.ra = Number(target.ra);
    body.dec = Number(target.dec);
  } else if (aq.resolved_name) {
    body.name = aq.resolved_name;
  } else {
    return null;
  }
  if (Array.isArray(aq.profiles) && aq.profiles.length) body.profile = aq.profiles[0];
  for (const key of SEARCH_FIELDS) if (aq[key] !== null && aq[key] !== undefined) body[key] = aq[key];
  return body;
}

async function runAsk(text) {
  const panel = $('#ask-panel');
  panelLoading(panel);
  $('#ask-btn').disabled = true;
  try {
    const data = await api('aiQuery', { body: { text } });
    const aq = data.advanced_query || {};
    const searchBody = searchBodyFromAdvanced(aq);
    setPanel(panel, 'ready',
      data.explanation ? h('p', {}, data.explanation) : null,
      data.plan?.length ? h('ol', { class: 'plan-list' }, data.plan.map((s) => h('li', {}, s))) : null,
      h('details', {}, h('summary', {}, 'Structured query'), h('pre', { class: 'code' }, JSON.stringify(aq, null, 2))),
      data.adql ? h('details', {}, h('summary', {}, 'ADQL'), h('pre', { class: 'code' }, data.adql)) : null,
      h('div', { class: 'row-actions' }, searchBody
        ? h('button', {
          type: 'button', class: 'btn btn-primary btn-small',
          onclick: () => {
            const label = searchBody.name || `${fmtDeg(searchBody.ra, 5)} ${fmtDeg(searchBody.dec, 5)}`;
            $('#q').value = label;
            $('#radius').value = String(searchBody.radius_arcsec);
            const input = searchBody.name
              ? { text: label, radius: searchBody.radius_arcsec, name: searchBody.name }
              : { text: label, radius: searchBody.radius_arcsec, ra: searchBody.ra, dec: searchBody.dec };
            runSearch(input, searchBody);
          },
        }, 'Run this query')
        : h('span', { class: 'muted small' }, 'The question has no sky position, so it cannot be run as a cone search.')));
  } catch (err) {
    panelError(panel, err, 'The AI query service');
  } finally {
    $('#ask-btn').disabled = false;
  }
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

// The server's search limits (GET /api/v1/limits, public): the radius field and
// parseSearchInput follow API_MAX_RADIUS_ARCSEC. Without the endpoint the default stays.
async function loadLimits() {
  try {
    const limits = await api('limits');
    const max = Number(limits?.max_radius_arcsec);
    if (Number.isFinite(max) && max > 0) state.maxRadiusArcsec = max;
  } catch {
    // keep DEFAULT_MAX_RADIUS_ARCSEC; the server still rejects a wider cone with a 422
  }
  const field = typeof document !== 'undefined' ? $('#radius') : null;
  if (field) field.max = String(state.maxRadiusArcsec);
  return state.maxRadiusArcsec;
}

function init() {
  initSettings();
  initAladin();
  loadSurveyList();
  loadLimits();

  $('#search-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    try {
      runSearch(buildSearchInput());
    } catch (err) {
      toast(err.message);
    }
  });
  for (const chip of document.querySelectorAll('.chip.example')) {
    chip.addEventListener('click', () => {
      $('#q').value = chip.dataset.q;
      if (chip.dataset.r) $('#radius').value = chip.dataset.r;
      $('#search-form').requestSubmit();
    });
  }
  $('#ask-form').addEventListener('submit', (ev) => {
    ev.preventDefault();
    const text = $('#ask-text').value.trim();
    if (text) runAsk(text); else $('#ask-text').focus();
  });
  $('#ask-text').addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) $('#ask-form').requestSubmit();
  });
  $('#cutout-fov').addEventListener('change', () => {
    if (state.target) loadCutouts(state.target, state.controller?.signal);
  });
  $('#download-json').addEventListener('click', () => {
    if (!state.record) return;
    const href = URL.createObjectURL(new Blob([JSON.stringify(state.record, null, 2)], { type: 'application/json' }));
    const a = h('a', { href, download: 'astrosearch_result.json' });
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 10000);
  });
  $('#copy-bibtex').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(state.bibtex);
      toast('BibTeX copied to the clipboard');
    } catch {
      toast('Clipboard unavailable; use Download .bib instead');
    }
  });
  document.addEventListener('keydown', (ev) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
    if (ev.key === '/' && !typing) { ev.preventDefault(); $('#q').focus(); }
  });

  const params = new URLSearchParams(window.location.search);
  if (params.get('q')) {
    $('#q').value = params.get('q');
    if (Number(params.get('r')) > 0) $('#radius').value = params.get('r');
    try { runSearch(buildSearchInput()); } catch (err) { toast(err.message); }
  }
}

// Exported for tests: node imports this module without a DOM (init only runs in a
// browser); tests/test_imaging_ui.py drives these under a minimal DOM stand-in.
export {
  ENDPOINTS, sseEvents, powerOfTen, sexagesimal, searchBodyFromAdvanced, nuFnu,
  ApiError, errorKind, errorMessage, panelError, searchRecord, streamSearch, parseSearchInput,
  foldPhase, lightcurveEpoch, lightcurveDatasets, serviceRadius, loadCutouts, cutoutStatus, endpointUrl,
  cutoutStackParams, lightcurveParams, targetInfo, fitsButton, knownTarget, onTargetKnown, state,
  SED_MAX_RADIUS_ARCSEC, LIGHTCURVE_MAX_RADIUS_ARCSEC, DEFAULT_MAX_RADIUS_ARCSEC, loadLimits,
};

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
}

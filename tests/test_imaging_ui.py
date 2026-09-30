"""Static web UI (web/): served at '/', asset integrity, JS syntax, accessibility basics, API contract."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import imaging

WEB = Path(__file__).resolve().parents[1] / "web"
INDEX = (WEB / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "app.js").read_text(encoding="utf-8")
CSS = (WEB / "styles.css").read_text(encoding="utf-8")
NODE = shutil.which("node")

# The API contract the UI is built against (method, path).
API_CONTRACT: set[tuple[str, str]] = {
    ("GET", "/api/v1/sed"), ("POST", "/api/v1/sed"),
    ("GET", "/api/v1/lightcurves"),
    ("GET", "/api/v1/solar-system"),
    ("GET", "/api/v1/cutouts"), ("GET", "/api/v1/cutouts/surveys"), ("GET", "/api/v1/cutouts/stack"),
    ("POST", "/api/v1/ai/query"), ("POST", "/api/v1/ai/explain"),
    ("POST", "/api/v1/provenance/replay"), ("GET", "/api/v1/citations"),
    ("GET", "/api/v1/alerts"), ("POST", "/api/v1/alerts/poll"), ("GET", "/api/v1/alerts/{id}"),
    ("POST", "/api/v1/search"), ("GET", "/api/v1/search/stream"),
}
# Hosts the page may load code from (task: CDN libs only from jsdelivr/cdnjs, plus Aladin Lite from CDS).
ALLOWED_SCRIPT_HOSTS = {"cdn.jsdelivr.net", "cdnjs.cloudflare.com"}
ALADIN_URL = "https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


def parsed() -> list[tuple[str, dict[str, str | None]]]:
    parser = PageParser()
    parser.feed(INDEX)
    return parser.tags


def endpoints_in_app_js() -> dict[str, tuple[str, str]]:
    block = re.search(r"const ENDPOINTS = Object\.freeze\(\{(.*?)\}\);", APP_JS, re.DOTALL)
    assert block, "app.js must declare every API call in ENDPOINTS"
    entries = re.findall(r"(\w+):\s*\{\s*method:\s*'(GET|POST)',\s*path:\s*'([^']+)'\s*\}", block.group(1))
    assert entries
    return {name: (method, path) for name, method, path in entries}


def ui_app() -> FastAPI:
    app = FastAPI()
    added = imaging.mount_ui(app)
    # Routes registered AFTER the UI must still win for their own paths.
    app.include_router(imaging.router)

    @app.get("/api/v1/ping")
    async def ping() -> dict[str, str]:
        return {"pong": "yes"}

    @app.get("/vo/tap/availability")
    async def availability() -> dict[str, bool]:
        return {"available": True}

    app.state.added = added
    return app


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


def test_ui_is_served_at_root_with_correct_types() -> None:
    client = TestClient(ui_app())
    root = client.get("/")
    assert root.status_code == 200
    assert root.headers["content-type"].startswith("text/html")
    assert root.headers["cache-control"] == "no-cache"
    assert "<title>AstroSearch Sky Explorer</title>" in root.text
    assert client.get("/index.html").text == root.text
    js = client.get("/app.js")
    assert js.status_code == 200 and js.headers["content-type"].startswith("text/javascript")
    css = client.get("/styles.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")


def test_ui_answers_head_requests() -> None:
    """Uptime and link checkers send HEAD; the UI routes must not answer 405."""
    client = TestClient(ui_app())
    for path in ("/", "/index.html", "/app.js", "/styles.css"):
        head = client.head(path)
        assert head.status_code == 200, path
        assert head.content == b""
        assert int(head.headers["content-length"]) == len(client.get(path).content)


def test_ui_does_not_shadow_api_or_vo_routes() -> None:
    client = TestClient(ui_app())
    assert client.get("/api/v1/ping").json() == {"pong": "yes"}
    assert client.get("/vo/tap/availability").json() == {"available": True}
    assert client.get("/api/v1/cutouts/surveys").status_code == 200
    missing = client.get("/api/v1/does-not-exist")
    assert missing.status_code == 404 and missing.json() == {"detail": "Not Found"}
    assert client.get("/vo/nothing").status_code == 404
    assert client.get("/not-a-file.js").status_code == 404
    assert client.get("/../imaging.py").status_code == 404


def test_mount_ui_lists_only_web_files_and_is_idempotent(tmp_path: Path) -> None:
    app = ui_app()
    assert set(app.state.added) == {"/", "/index.html", "/app.js", "/styles.css"}
    assert imaging.mount_ui(app) == []
    (tmp_path / "index.html").write_text("<!doctype html><title>x</title>", encoding="utf-8")
    (tmp_path / ".secret.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.py").write_text("print()", encoding="utf-8")
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "x.json").write_text("{}", encoding="utf-8")
    assert set(imaging.ui_files(tmp_path)) == {"/index.html"}
    with pytest.raises(FileNotFoundError):
        imaging.mount_ui(FastAPI(), tmp_path / "missing")


# ---------------------------------------------------------------------------
# Page structure & assets
# ---------------------------------------------------------------------------


def test_index_references_existing_assets_and_allowed_cdns() -> None:
    client = TestClient(ui_app())
    local, external = [], []
    for tag, attrs in parsed():
        ref = attrs.get("src") if tag == "script" else attrs.get("href") if tag == "link" else None
        if not ref or ref.startswith("data:"):
            continue
        (external if urlsplit(ref).scheme else local).append(ref)
    assert "styles.css" in local and "app.js" in local
    for ref in local:
        assert (WEB / ref).is_file(), ref
        assert client.get("/" + ref).status_code == 200, ref
    for ref in external:
        assert urlsplit(ref).scheme == "https" and urlsplit(ref).hostname in ALLOWED_SCRIPT_HOSTS, ref
    assert any("chart.js" in ref for ref in external)
    # app.js is an ES module; Aladin Lite v3 is loaded from CDS as required.
    assert ("script", {"type": "module", "src": "app.js"}) in parsed()
    assert f"const ALADIN_SRC = '{ALADIN_URL}';" in APP_JS
    remote = set(re.findall(r"https://[a-z0-9.-]+", APP_JS))
    code_hosts = {urlsplit(u).hostname for u in re.findall(r"loadScript\(([^)]+)\)", APP_JS)}
    assert code_hosts == {None}  # loadScript is only ever called with the ALADIN_SRC constant
    assert "https://aladin.cds.unistra.fr" in remote


def test_every_element_id_used_by_app_js_exists() -> None:
    ids = {attrs["id"] for _, attrs in parsed() if attrs.get("id")}
    ids |= set(re.findall(r"\{ id: '([\w-]+)'", APP_JS))  # elements app.js creates itself (chart canvases)
    used = set(re.findall(r"\$\('#([\w-]+)'", APP_JS)) | set(re.findall(r"getElementById\('([\w-]+)'\)", APP_JS))
    assert used, "expected DOM lookups in app.js"
    assert used <= ids, f"missing ids in index.html: {sorted(used - ids)}"
    # A.aladin('#aladin') mounts into an existing container.
    assert "aladin" in ids


def test_accessibility_basics() -> None:
    tags = parsed()
    html_attrs = next(attrs for tag, attrs in tags if tag == "html")
    assert html_attrs.get("lang") == "en" and html_attrs.get("data-theme") == "dark"
    assert any(tag == "meta" and attrs.get("name") == "viewport" for tag, attrs in tags)
    assert any(tag == "main" for tag, _ in tags)
    assert any(tag == "header" for tag, _ in tags) and any(tag == "footer" for tag, _ in tags)
    labels = {attrs.get("for") for tag, attrs in tags if tag == "label"}
    for tag, attrs in tags:
        if tag in {"input", "select", "textarea"} and attrs.get("type") != "checkbox":
            assert attrs.get("id") in labels or attrs.get("aria-label"), f"unlabelled <{tag} id={attrs.get('id')}>"
    for tag, attrs in tags:
        if tag == "button":
            assert attrs.get("type") in {"button", "submit"}, attrs
    icon_buttons = [attrs for tag, attrs in tags if tag == "button" and "icon-btn" in (attrs.get("class") or "")]
    assert icon_buttons and all(attrs.get("aria-label") for attrs in icon_buttons)
    assert any(attrs.get("aria-live") for _, attrs in tags), "progress/status must be announced"
    assert 'class="skip-link"' in INDEX


def test_css_dark_default_responsive_and_reduced_motion() -> None:
    assert re.search(r":root\s*\{[^}]*--bg:\s*#0", CSS), "dark palette is the default"
    assert ':root[data-theme="light"]' in CSS
    assert "@media (max-width: 720px)" in CSS and "@media (max-width: 1100px)" in CSS
    assert "prefers-reduced-motion: reduce" in CSS
    assert ":focus-visible" in CSS
    assert CSS.count("{") == CSS.count("}")


# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_app_js_has_no_syntax_errors(tmp_path: Path) -> None:
    module = tmp_path / "app.mjs"  # force ES-module parsing
    module.write_text(APP_JS, encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(module)], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_app_js_helpers_behave(tmp_path: Path) -> None:
    """Run the exported pure helpers (coordinate parsing, SSE parsing, axis labels) under node."""
    module = tmp_path / "app.mjs"
    module.write_text(APP_JS, encoding="utf-8")
    script = tmp_path / "probe.mjs"
    script.write_text(f"""
const m = await import({json.dumps(module.as_uri())});
const out = {{}};
out.decimal = m.parseCoordinates('187.2779154, +2.0523883');
out.sexagesimal = m.parseCoordinates('12 29 06.70 +02 03 08.6');
out.letters = m.parseCoordinates('12h29m06.70s -02d03m08.6s');
out.names = ['3C 273', 'M 87', 'HD 209458', 'Vega', '400 10', '10 95'].map(m.parseCoordinates);
const sse = 'event: catalog\\ndata: {{"catalog":"gaia_dr3","status":"ok","count":2}}\\n\\n: keep-alive\\n\\n'
  + 'event: done\\r\\ndata: {{"record":\\r\\ndata: {{"x":1}}}}\\r\\n\\r\\n';
out.events = [];
for await (const e of m.sseEvents(new Response(sse))) out.events.push(e);
out.ticks = [m.powerOfTen(1e-12), m.powerOfTen(100), m.powerOfTen(3)];
out.sexa = m.sexagesimal(187.2779154, 2.0523883);
out.nufnu = m.nuFnu({{frequency_hz: 1.4e9, flux_jy: 50}});
out.endpoints = m.ENDPOINTS;
console.log(JSON.stringify(out));
""", encoding="utf-8")
    result = subprocess.run([NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["decimal"] == {"ra": 187.2779154, "dec": 2.0523883}
    assert out["sexagesimal"]["ra"] == pytest.approx(187.2779167, abs=1e-6)
    assert out["sexagesimal"]["dec"] == pytest.approx(2.0523889, abs=1e-6)
    assert out["letters"]["dec"] == pytest.approx(-2.0523889, abs=1e-6)
    assert out["names"] == [None] * 6
    assert out["events"] == [
        {"event": "catalog", "data": {"catalog": "gaia_dr3", "status": "ok", "count": 2}},
        {"event": "done", "data": {"record": {"x": 1}}},
    ]
    assert out["ticks"] == ["10⁻¹²", "10²", ""]
    assert out["sexa"] == "12h29m06.70s +02°03′08.6″"
    # 1 Jy = 1e-23 erg s^-1 cm^-2 Hz^-1: 50 Jy at 1.4 GHz -> 7e-13 erg s^-1 cm^-2.
    assert out["nufnu"] == pytest.approx(7e-13)
    assert {k: (v["method"], v["path"]) for k, v in out["endpoints"].items()} == endpoints_in_app_js()


# ---------------------------------------------------------------------------
# API contract
# ---------------------------------------------------------------------------


def test_every_endpoint_the_ui_calls_is_in_the_api_contract() -> None:
    endpoints = endpoints_in_app_js()
    for name, (method, path) in endpoints.items():
        assert (method, path) in API_CONTRACT, f"{name}: {method} {path} is not part of the API contract"
    # No API path is used outside the ENDPOINTS table...
    block_end = APP_JS.index("});", APP_JS.index("const ENDPOINTS"))
    stray = re.findall(r"['\"`](/(?:api|vo)/[^'\"`]*)['\"`]", APP_JS[block_end:])
    assert stray == [], f"API paths hard-coded outside ENDPOINTS: {stray}"
    # ...and every call site names a declared endpoint.
    called = set(re.findall(r"\bapi\('(\w+)'", APP_JS)) | set(re.findall(r"endpointUrl\('(\w+)'", APP_JS))
    assert called, "expected API call sites"
    assert called <= set(endpoints), f"undeclared endpoints: {called - set(endpoints)}"
    assert "fetch(" not in re.sub(r"(await fetch\((endpointUrl|target)\b)", "", APP_JS).replace("fetchBlob(", ""), \
        "all network access must go through api(), fetchBlob() or the SSE fetch"
    # The UI reaches every feature it is expected to present.
    assert {"search", "searchStream", "sed", "lightcurves", "solarSystem", "cutoutStack", "cutoutSurveys",
            "aiQuery", "aiExplain", "citations"} <= called


def test_imaging_router_implements_its_part_of_the_contract() -> None:
    routes = {(method, route.path) for route in imaging.router.routes for method in route.methods}
    assert {("GET", "/api/v1/cutouts"), ("GET", "/api/v1/cutouts/surveys"), ("GET", "/api/v1/cutouts/stack")} <= routes
    assert routes <= API_CONTRACT


def test_example_chips_do_not_search_high_proper_motion_stars_by_bare_position() -> None:
    """A bare RA/Dec is not propagated for proper motion: Barnard's star (10.4"/yr) goes by name."""
    chips = {attrs.get("data-q"): attrs for tag, attrs in parsed() if tag == "button" and "example" in (attrs.get("class") or "")}
    assert "Barnard's star" in chips
    assert not any(q and q.startswith("269.45") for q in chips), "Barnard's star must not be searched by its J2000 position"
    coordinate_chips = [q for q in chips if q and parse_is_numeric(q)]
    # The coordinate example is the Crab pulsar (pm ~ 12 mas/yr, SIMBAD ICRS 05 34 31.94 +22 00 52.1).
    assert coordinate_chips == ["05 34 31.94 +22 00 52.1"]


def parse_is_numeric(text: str) -> bool:
    return re.fullmatch(r"[\d\s:.+-]+", text) is not None


def test_ui_service_limits_match_the_services() -> None:
    """The UI clamps the SED / light-curve cone to the services' own maximum radius."""
    import sed
    import timedomain

    assert re.search(r"const SED_MAX_RADIUS_ARCSEC = (\d+);", APP_JS).group(1) == f"{sed.MAX_RADIUS_ARCSEC:g}"
    assert re.search(r"const LIGHTCURVE_MAX_RADIUS_ARCSEC = (\d+);", APP_JS).group(1) == \
        f"{timedomain.MAX_LIGHTCURVE_RADIUS_ARCSEC:g}"


# ---------------------------------------------------------------------------
# Behaviour under node (real app.js functions, stubbed fetch, minimal DOM)
# ---------------------------------------------------------------------------

# A minimal DOM stand-in: enough of Node/Element/document for app.js's h(), $(),
# setPanel() and the cutout strip. The module is imported *before* it is installed,
# so app.js's browser-only init() does not run.
FAKE_DOM_JS = r"""
class FakeNode {
  constructor() { this.childNodes = []; this.parentNode = null; }
  append(...items) {
    for (const item of items) {
      const node = item instanceof FakeNode ? item : new FakeText(String(item));
      node.parentNode = this;
      this.childNodes.push(node);
    }
  }
  replaceChildren(...items) { this.childNodes = []; this.append(...items); }
  get textContent() { return this.childNodes.map((c) => c.textContent).join(''); }
  set textContent(v) { this.replaceChildren(String(v)); }
}
class FakeText extends FakeNode {
  constructor(t) { super(); this.data = t; }
  get textContent() { return this.data; }
  set textContent(v) { this.data = String(v); }
}
class FakeElement extends FakeNode {
  constructor(tag) {
    super();
    this.tagName = tag.toUpperCase(); this.attributes = {}; this.dataset = {}; this.style = {};
    this.listeners = {}; this.className = ''; this.value = ''; this.id = '';
    const el = this;
    const names = () => el.className.split(' ').filter(Boolean);
    this.classList = {
      add: (...c) => { el.className = [...new Set([...names(), ...c])].join(' '); },
      remove: (...c) => { el.className = names().filter((n) => !c.includes(n)).join(' '); },
      contains: (c) => names().includes(c),
    };
  }
  setAttribute(k, v) { this.attributes[k] = String(v); if (k === 'id') this.id = String(v); }
  getAttribute(k) { return k in this.attributes ? this.attributes[k] : null; }
  hasAttribute(k) { return k in this.attributes; }
  removeAttribute(k) { delete this.attributes[k]; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  remove() { if (this.parentNode) this.parentNode.childNodes = this.parentNode.childNodes.filter((c) => c !== this); }
  get children() { return this.childNodes.filter((c) => c instanceof FakeElement); }
  *descendants() { for (const c of this.children) { yield c; yield* c.descendants(); } }
  matches(sel) {
    if (sel.startsWith('#')) return this.id === sel.slice(1);
    if (sel.startsWith('.')) return this.classList.contains(sel.slice(1));
    return this.tagName === sel.toUpperCase();
  }
  querySelectorAll(sel) { return [...this.descendants()].filter((e) => e.matches(sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
export function installDom(baseURI) {
  const body = new FakeElement('body');
  globalThis.Node = FakeNode;
  globalThis.document = {
    body, baseURI, readyState: 'complete',
    createElement: (t) => new FakeElement(t),
    createTextNode: (t) => new FakeText(t),
    querySelector: (sel) => body.querySelector(sel),
    querySelectorAll: (sel) => body.querySelectorAll(sel),
    addEventListener() {},
    documentElement: new FakeElement('html'),
  };
  Object.defineProperty(globalThis, 'localStorage', {
    value: { getItem: () => null, setItem() {} }, configurable: true, writable: true,
  });
  return body;
}
export function element(tag, id) {
  const el = globalThis.document.createElement(tag);
  if (id) el.setAttribute('id', id);
  return el;
}
export function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json', ...headers } });
}
export function sse(text, chunks = null) {
  const enc = new TextEncoder();
  const parts = chunks || [text];
  const stream = new ReadableStream({ start(c) { for (const p of parts) c.enqueue(enc.encode(p)); c.close(); } });
  return new Response(stream, { status: 200, headers: { 'content-type': 'text/event-stream; charset=utf-8' } });
}
"""


def run_node(tmp_path: Path, body: str) -> dict:
    """Import app.js as a module plus the DOM stand-in, run ``body`` (which sets ``out``), return ``out``."""
    module = tmp_path / "app.mjs"
    module.write_text(APP_JS, encoding="utf-8")
    dom = tmp_path / "fakedom.mjs"
    dom.write_text(FAKE_DOM_JS, encoding="utf-8")
    script = tmp_path / "probe.mjs"
    script.write_text(
        f"const m = await import({json.dumps(module.as_uri())});\n"
        f"const dom = await import({json.dumps(dom.as_uri())});\n"
        "const out = {};\n" + body + "\nconsole.log(JSON.stringify(out));\n",
        encoding="utf-8",
    )
    result = subprocess.run([NODE, str(script)], capture_output=True, text=True, encoding="utf-8", timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


SEARCH_CASES_JS = r"""
const RECORD = { target: { ra: 187.2779154, dec: 2.0523883 }, catalogs_queried: 3, crossmatch_groups: [] };
async function scenario(streamAnswer, { stream = true } = {}) {
  const calls = [];
  globalThis.fetch = async (url, init = {}) => {
    const u = new URL(String(url));
    calls.push(`${init.method || 'GET'} ${u.pathname}`);
    if (u.pathname.endsWith('/api/v1/search/stream')) return streamAnswer();
    if (u.pathname.endsWith('/api/v1/search') && init.method === 'POST') return dom.json({ ...RECORD, from: 'post' });
    return dom.json({ detail: 'Not Found' }, 404);
  };
  const events = [];
  let fellBack = false;
  try {
    const r = await m.searchRecord({
      input: { name: '3C 273', radius: 10 }, body: { name: '3C 273', radius_arcsec: 10 }, stream,
      signal: new AbortController().signal, onEvent: (e) => events.push(e), onFallback: () => { fellBack = true; },
    });
    return { via: r.via, from: r.record.from || 'stream', calls, events, fellBack };
  } catch (err) {
    return { error: err.name, status: err.status, kind: m.errorKind(err), calls, events };
  }
}
out.missing404 = await scenario(() => dom.json({ detail: 'Not Found' }, 404));
out.notStreaming = await scenario(() => dom.json({ some: 'json' }, 200));
out.networkError = await scenario(() => { throw new TypeError('fetch failed'); });
out.noDone = await scenario(() => dom.sse('event: catalog\ndata: {"catalog":"gaia_dr3","status":"ok","count":1}\n\n'));
out.streamed = await scenario(() => dom.sse('event: catalog\ndata: {"catalog":"gaia_dr3","status":"ok","count":1}\n\n'
  + 'event: done\ndata: {"record": {"target": {"ra": 1, "dec": 2}}}\n\n'));
out.badInput = await scenario(() => dom.json({ detail: 'radius_arcsec: too large' }, 422));
out.rateLimited = await scenario(() => dom.json({ detail: 'Rate limit exceeded' }, 429, { 'Retry-After': '60' }));
out.errorEvent = await scenario(() => dom.sse('event: error\ndata: {"status": 502, "detail": "upstream"}\n\n'));
out.streamOff = await scenario(() => { throw new Error('must not be called'); }, { stream: false });
"""


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_search_falls_back_to_post_when_the_stream_is_missing(tmp_path: Path) -> None:
    out = run_node(tmp_path, "dom.installDom('http://ui.test/');\n" + SEARCH_CASES_JS)
    post = ["GET /api/v1/search/stream", "POST /api/v1/search"]
    for case in ("missing404", "notStreaming", "networkError", "noDone"):
        assert out[case]["via"] == "post" and out[case]["from"] == "post", case
        assert out[case]["calls"] == post and out[case]["fellBack"], case
    assert out["noDone"]["events"] == ["catalog"]  # progress events were still delivered
    assert out["streamed"]["via"] == "stream" and out["streamed"]["calls"] == ["GET /api/v1/search/stream"]
    assert out["streamed"]["events"] == ["catalog"] and not out["streamed"]["fellBack"]
    # Real answers from the stream endpoint are not retried through POST.
    assert out["badInput"] == {"error": "ApiError", "status": 422, "kind": "error",
                               "calls": ["GET /api/v1/search/stream"], "events": []}
    assert out["rateLimited"]["kind"] == "rate-limited" and out["rateLimited"]["calls"] == ["GET /api/v1/search/stream"]
    assert out["errorEvent"]["status"] == 502 and out["errorEvent"]["kind"] == "error"
    assert out["streamOff"]["via"] == "post" and out["streamOff"]["calls"] == ["POST /api/v1/search"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_missing_endpoints_render_as_unavailable(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
dom.installDom('http://ui.test/');
const cases = {
  notFound: new m.ApiError(404, 'Not Found', 'sed'),
  notFoundNoDetail: new m.ApiError(404, '', 'sed'),
  methodNotAllowed: new m.ApiError(405, 'Method Not Allowed', 'sed'),
  notImplemented: new m.ApiError(501, 'Not Implemented', 'sed'),
  network: new m.ApiError(0, 'Network error: fetch failed', 'sed'),
  unresolved: new m.ApiError(404, "Could not resolve 'Nonexistent': not found", 'sed'),
  badInput: new m.ApiError(422, 'radius_arcsec: Input should be less than or equal to 60', 'sed'),
  upstream: new m.ApiError(502, 'hips2fits unavailable', 'sed'),
  unconfigured: new m.ApiError(503, 'ANTHROPIC_API_KEY is not set', 'aiExplain'),
  resolverDown: new m.ApiError(503, 'Name resolver unavailable: timed out', 'search', 30),
  rateLimited: new m.ApiError(429, 'Rate limit exceeded', 'sed', 42),
  plain: new Error('boom'),
  aborted: new DOMException('stop', 'AbortError'),
};
out.kinds = {};
out.panels = {};
for (const [name, err] of Object.entries(cases)) {
  out.kinds[name] = m.errorKind(err);
  const panel = dom.element('div');
  panel.dataset.state = 'loading';
  m.panelError(panel, err, 'The SED service');
  out.panels[name] = { state: panel.dataset.state, text: panel.textContent };
}
""")
    kinds = out["kinds"]
    for name in ("notFound", "notFoundNoDetail", "methodNotAllowed", "notImplemented", "network"):
        assert kinds[name] == "unavailable", name
        assert out["panels"][name]["state"] == "unavailable", name
        assert "The SED service is unavailable on this server." in out["panels"][name]["text"], name
    for name in ("unresolved", "badInput", "upstream", "plain"):
        assert kinds[name] == "error", name
        assert out["panels"][name]["state"] == "error" and "The SED service failed:" in out["panels"][name]["text"], name
    assert "Could not resolve 'Nonexistent'" in out["panels"]["unresolved"]["text"]
    assert kinds["unconfigured"] == "unconfigured" and out["panels"]["unconfigured"]["state"] == "unavailable"
    assert "The SED service is not configured: ANTHROPIC_API_KEY is not set" in out["panels"]["unconfigured"]["text"]
    # 503 + Retry-After: the resolver/archive is down for now (not a configuration problem).
    assert kinds["resolverDown"] == "temporarily-unavailable" and out["panels"]["resolverDown"]["state"] == "rate-limited"
    text = out["panels"]["resolverDown"]["text"]
    assert "The SED service is temporarily unavailable, retry in 30 s" in text and "Name resolver unavailable" in text
    assert "not configured" not in text
    assert kinds["rateLimited"] == "rate-limited" and out["panels"]["rateLimited"]["state"] == "rate-limited"
    assert "rate limited by the server; retry in 42 s" in out["panels"]["rateLimited"]["text"]
    assert kinds["aborted"] == "aborted" and out["panels"]["aborted"]["state"] == "loading"  # left untouched


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_cutout_strip_renders_images_states_and_fits_links(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
const body = dom.installDom('http://ui.test/astro/');
const panel = dom.element('div', 'cutout-panel');
const fov = dom.element('select', 'cutout-fov');
fov.value = '3';
body.append(panel, fov);
const base = '/astro/api/v1/cutouts';
const q = (survey, fmt) => `${base}?ra=187.2779154&dec=2.0523883&fov_arcmin=3.0&survey=${survey}&width=256&height=256&format=${fmt}`;
const mk = (survey, label, extra = {}) => ({ slot: 'x', survey, label, hips_id: `X/P/${survey}`, regime: 'radio', wavelength: '1 GHz',
  wavelength_m: 0.3, in_coverage: true, color: false, url: q(survey, 'png'), fits_survey: survey, fits_label: label,
  fits_url: q(survey, 'fits'), fits_pixel_units: 'Jy/beam', fits_calibrated: true, fits_note: null, epoch: null,
  offset_arcsec: 0, note: null, ...extra });
const stack = { target: { ra: 187.2779154, dec: 2.0523883 }, fov_arcmin: 3, size_px: 256, format: 'png', coverage_checked: true, panels: [
  mk('nvss', 'NVSS 1.4 GHz'),
  mk('2mass', '2MASS J/H/Ks', { color: true, fits_survey: '2mass_k', fits_label: '2MASS Ks', fits_url: q('2mass_k', 'fits'),
    fits_pixel_units: 'DN', fits_calibrated: false, epoch: 1999.25, offset_arcsec: 7.8 }),
  mk('dss2', 'DSS2 colour', { color: true, fits_survey: 'dss2_red', fits_label: 'DSS2 red', fits_url: q('dss2_red', 'fits'),
    fits_pixel_units: 'photographic density', fits_calibrated: false, epoch: 1991.5, offset_arcsec: 88.3,
    note: 'The target moves up to 78" from this centre during the survey\'s 1984.0-1999.0 observations and may lie outside the field.' }),
  mk('vlass', 'VLASS 3 GHz'),
  mk('sdss', 'SDSS DR9', { in_coverage: false, color: true, fits_survey: 'sdss_r', fits_label: 'SDSS DR9 r', fits_url: q('sdss_r', 'fits') }),
  mk('chandra', 'Chandra (CXC)', { color: true, fits_survey: null, fits_label: null, fits_url: null, fits_pixel_units: null,
    fits_calibrated: null, fits_note: 'Chandra (CXC) has no single-band FITS HiPS.' }),
], proper_motion: { pm_ra_masyr: -801.551, pm_dec_masyr: 10362.394, epoch: 2000 } };
const fetched = [];
const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
globalThis.fetch = async (url) => {
  const u = new URL(String(url));
  fetched.push(u.href);
  if (u.pathname === '/astro/api/v1/cutouts/stack') return dom.json(stack);
  if (u.pathname === '/astro/api/v1/cutouts') {
    const survey = u.searchParams.get('survey');
    const headers = { 'content-type': 'image/png', 'X-Cutout-Coverage': survey === 'sdss' || survey === 'vlass' ? '0.0000' : '1.0000' };
    if (survey === 'vlass') headers['X-Cutout-Degraded'] = 'blank image from mirror alaskybis.cds.unistra.fr after alasky: HTTP 500';
    return new Response(png, { status: 200, headers });
  }
  return dom.json({ detail: 'Not Found' }, 404);
};
await m.loadCutouts({ ra: 187.2779154, dec: 2.0523883 }, new AbortController().signal);
out.state = panel.dataset.state;
out.fetched = fetched;
out.figures = panel.querySelectorAll('figure').map((fig) => {
  const img = fig.querySelector('img');
  const buttons = fig.querySelectorAll('button').map((b) => ({ text: b.textContent, disabled: b.hasAttribute('disabled'), title: b.getAttribute('title') }));
  const note = fig.querySelector('.overlay-note');
  const warn = fig.querySelector('.panel-note');
  return { survey: fig.dataset.survey, img: img ? img.getAttribute('src') : null, alt: img ? img.getAttribute('alt') : null,
    cls: fig.className, note: note ? note.textContent : null, warn: warn ? warn.textContent : null, caption: fig.querySelector('figcaption').textContent, buttons };
});
out.statusNoHeaders = m.cutoutStatus(new Headers(), { in_coverage: true });
out.text = panel.textContent;
out.endpoint = m.endpointUrl('sed', { ra: 1, dec: 2 }).href;
""")
    assert out["state"] == "ready"
    assert out["fetched"][0].startswith("http://ui.test/astro/api/v1/cutouts/stack?ra=187.2779154&dec=2.0523883&fov_arcmin=3")
    assert all(u.startswith("http://ui.test/astro/api/v1/cutouts?") for u in out["fetched"][1:])
    assert len(out["fetched"]) == 7  # the stack + one image per panel (no FITS downloads)
    figures = {f["survey"]: f for f in out["figures"]}
    assert list(figures) == ["nvss", "2mass", "dss2", "vlass", "sdss", "chandra"]
    for survey, fig in figures.items():
        assert fig["img"] and fig["img"].startswith("blob:"), survey
    assert figures["nvss"]["cls"] == "cutout" and figures["nvss"]["note"] is None
    # A blank image served after a primary failure is a rendering failure, not the footprint.
    assert "degraded" in figures["vlass"]["cls"] and figures["vlass"]["note"] == "Rendering failed upstream"
    assert "nodata" in figures["sdss"]["cls"] and figures["sdss"]["note"] == "No survey data here"
    fits = {s: f["buttons"][1] for s, f in figures.items()}
    assert fits["nvss"]["text"] == "FITS (Jy/beam)" and not fits["nvss"]["disabled"]
    assert fits["nvss"]["title"] == "Download NVSS 1.4 GHz FITS (single-band survey pixel values: Jy/beam)"
    assert fits["2mass"]["text"] == "FITS: 2MASS Ks (DN)" and "colour composite" in fits["2mass"]["title"]
    assert "DN, not flux-calibrated" in fits["2mass"]["title"]
    # DSS2 red FITS are plate densities: the UI never calls them calibrated.
    assert fits["dss2"]["text"] == "FITS: DSS2 red (photographic density)"
    assert "photographic density, not flux-calibrated" in fits["dss2"]["title"]
    assert not any("calibrated FITS" in f["title"] or f["title"].startswith("Download calibrated") for f in fits.values())
    assert fits["chandra"]["text"] == "No FITS" and fits["chandra"]["disabled"]
    assert fits["chandra"]["title"] == "Chandra (CXC) has no single-band FITS HiPS."
    # Proper motion: panels moved to the survey epoch say so (caption, alt text, strip note).
    assert "Proper motion 10.4″/yr: each panel is centred on the target's position at that survey's mean epoch." in out["text"]
    assert figures["2mass"]["alt"] == ("2MASS J/H/Ks (1 GHz) cutout, 3 arcmin field centred on the target's expected "
                                       "position at epoch 1999.3")
    assert "Epoch 1991.5: 88.3″ from the catalogue position" in figures["dss2"]["caption"]
    assert figures["dss2"]["warn"].endswith("may lie outside the field.")
    assert figures["nvss"]["alt"].endswith("3 arcmin field centred on the target") and figures["nvss"]["warn"] is None
    assert out["statusNoHeaders"] == {"kind": "ok", "message": None}
    # API paths resolve against the document base (sub-path deployments).
    assert out["endpoint"] == "http://ui.test/astro/api/v1/sed?ra=1&dec=2"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_phase_folding_uses_one_reference_epoch_for_all_series(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
const P = 0.5667;
const g = { survey: 'ztf', band: 'g', unit: 'mag', points: [58200.10, 58201.00, 58203.37].map((t) => ({ mjd: t, value: 15, error: 0.02, flag: 0 })) };
const r = { survey: 'ztf', band: 'r', unit: 'mag', points: [58200.35, 58201.00, 58204.91].map((t) => ({ mjd: t, value: 14.6, error: 0.02, flag: 0 })) };
const w1 = { survey: 'neowise', band: 'W1', unit: 'flux', points: [{ mjd: 58150.0, value: 1.2, error: 0.1, flag: 0 }] };
const lc = { series: [g, r, w1], period: { best_period_days: P, series: 'ztf g' } };
const view = m.lightcurveDatasets(lc, { unit: 'mag', folded: true });
out.t0 = view.t0;
out.folded = view.folded;
out.phases = Object.fromEntries(view.datasets.map((d) => [d.label, d.data.filter((p) => p.x < 1).map((p) => [p.mjd, p.x])]));
out.time = m.lightcurveDatasets(lc, { unit: 'mag', folded: false }).datasets[0].data.map((p) => p.x);
out.fold = m.foldPhase(58201.0, P, 58200.1);
out.epochEmpty = m.lightcurveEpoch([]);
""")
    # The shared reference epoch is the earliest epoch of all series (NEOWISE here).
    assert out["t0"] == 58150.0 and out["folded"]
    g = dict(map(tuple, out["phases"]["ztf g"]))
    r = dict(map(tuple, out["phases"]["ztf r"]))
    # The same instant has the same phase in every band.
    assert g[58201.0] == pytest.approx(r[58201.0], abs=1e-12)
    for mjd, phase in list(g.items()) + list(r.items()):
        assert phase == pytest.approx(((mjd - 58150.0) / 0.5667) % 1, abs=1e-9)
    assert out["time"] == [58200.10, 58201.00, 58203.37]
    assert out["fold"] == pytest.approx(((58201.0 - 58200.1) / 0.5667) % 1)
    assert out["epochEmpty"] is None


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_search_input_rejects_out_of_range_coordinates(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
const attempt = (text, radius = 10) => { try { return m.parseSearchInput(text, radius); } catch (err) { return { error: err.message }; } };
for (const t of ['400 10', '10 95', '-5 10', '25 10 00 +10 00 00', '12 29 06.7 +95 00 00', '3C 273', 'M 87', 'Barnard\'s star',
                 '187.2779154 +2.0523883', '12 29 06.70 +02 03 08.6']) out[t] = attempt(t);
out.zeroRadius = attempt('M 87', 0);
out.hugeRadius = attempt('M 87', 5000);
out.empty = attempt('   ');
out.clamp = [m.serviceRadius(120, m.SED_MAX_RADIUS_ARCSEC), m.serviceRadius(10, m.SED_MAX_RADIUS_ARCSEC)];
""")
    assert out["400 10"]["error"].startswith("RA must be in [0, 360) degrees")
    assert out["10 95"]["error"].startswith("Dec must be in [-90, +90] degrees")
    assert out["-5 10"]["error"].startswith("RA must be in [0, 360)")
    assert out["25 10 00 +10 00 00"]["error"].startswith("RA must be 0-23 h")
    assert out["12 29 06.7 +95 00 00"]["error"].startswith("Dec must be at most 90")
    for name in ("3C 273", "M 87", "Barnard's star"):
        assert out[name] == {"text": name, "radius": 10, "name": name}
    assert out["187.2779154 +2.0523883"] == {"text": "187.2779154 +2.0523883", "radius": 10, "ra": 187.2779154, "dec": 2.0523883}
    assert out["12 29 06.70 +02 03 08.6"]["ra"] == pytest.approx(187.2779167, abs=1e-6)
    assert "Radius" in out["zeroRadius"]["error"] and "Radius" in out["hugeRadius"]["error"]
    assert out["empty"]["error"] == "Enter an object name or coordinates."
    assert out["clamp"] == [{"radius": 60, "clamped": True}, {"radius": 10, "clamped": False}]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_sse_parser_keeps_crlf_split_across_chunks(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
async function collect(chunks) {
  const events = [];
  for await (const e of m.sseEvents(dom.sse(null, chunks))) events.push(e);
  return events;
}
out.splitCrlf = await collect(['event: catalog\r', '\ndata: {"a":1}\r\n\r\n']);
out.splitBlank = await collect(['event: done\r\ndata: {"record":1}\r\n\r', '\n']);
out.byteByByte = await collect([...'event: group\r\ndata: 2\r\n\r\n']);
out.crOnly = await collect(['event: catalog\rdata: 3\r\r']);
out.lfOnly = await collect(['event: catalog\ndata: 4\n', '\n']);
out.unterminated = await collect(['event: catalog\ndata: 5\n']);
""")
    assert out["splitCrlf"] == [{"event": "catalog", "data": {"a": 1}}]
    assert out["splitBlank"] == [{"event": "done", "data": {"record": 1}}]
    assert out["byteByByte"] == [{"event": "group", "data": 2}]
    assert out["crOnly"] == [{"event": "catalog", "data": 3}]
    assert out["lfOnly"] == [{"event": "catalog", "data": 4}]
    assert out["unterminated"] == []


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_cutout_strip_ignores_stale_stacks_when_the_fov_changes(tmp_path: Path) -> None:
    """A slow 3' stack answering after a fast 30' one must not overwrite it (FoV selector race)."""
    out = run_node(tmp_path, r"""
const body = dom.installDom('http://ui.test/');
const panel = dom.element('div', 'cutout-panel');
const fov = dom.element('select', 'cutout-fov');
body.append(panel, fov);
const signals = [];
const png = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
globalThis.fetch = async (url, init = {}) => {
  const u = new URL(String(url));
  if (u.pathname === '/api/v1/cutouts/stack') {
    const f = u.searchParams.get('fov_arcmin');
    signals.push({ fov: f, signal: init.signal });
    await sleep(f === '3' ? 300 : 20);  // ignores the abort signal: the worst case
    const p = { slot: 'x', survey: 'dss2', label: 'DSS2', hips_id: 'CDS/P/DSS2/color', regime: 'optical', wavelength: 'x',
      wavelength_m: 5e-7, in_coverage: true, color: true, url: `/api/v1/cutouts?survey=dss2&fov_arcmin=${f}`,
      fits_survey: null, fits_url: null, epoch: null, offset_arcsec: 0, note: null };
    return dom.json({ target: { ra: 1, dec: 2 }, fov_arcmin: Number(f), size_px: 64, format: 'png', coverage_checked: true, panels: [p] });
  }
  await sleep(u.searchParams.get('fov_arcmin') === '3' ? 5 : 1);
  return new Response(png, { status: 200, headers: { 'content-type': 'image/png', 'X-Cutout-Coverage': '1.0000' } });
};
const target = { ra: 1, dec: 2 };
fov.value = '3';
const first = m.loadCutouts(target, new AbortController().signal);
await sleep(5);
fov.value = '30';
const second = m.loadCutouts(target, new AbortController().signal);
await Promise.all([first, second]);
await sleep(50);
out.selectedFov = fov.value;
out.alts = panel.querySelectorAll('img').map((img) => img.getAttribute('alt'));
out.aborted = signals.map((s) => [s.fov, s.signal.aborted]);
out.state = panel.dataset.state;
""")
    assert out["selectedFov"] == "30"
    assert out["alts"] == ["DSS2 (x) cutout, 30 arcmin field centred on the target"]
    assert out["aborted"] == [["3", True], ["30", False]], "the superseded strip's requests are cancelled"
    assert out["state"] == "ready"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_cutout_stack_follows_the_targets_proper_motion(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
out.pm = m.cutoutStackParams({ ra: 269.45207696, dec: 4.69336497, pm_ra_masyr: -801.551, pm_dec_masyr: 10362.394, epoch: 2000, name: "Barnard's star" }, '3');
out.name = m.cutoutStackParams({ ra: 187.2779154, dec: 2.0523883, name: '3C 273' }, '3');
out.bare = m.cutoutStackParams({ ra: 83.63321, dec: 22.01446 }, '10');
out.info = m.targetInfo({ ra: '269.45207696', dec: '4.69336497', pm_ra_masyr: -801.551, pm_dec_masyr: 10362.394, epoch: null, frame: 'icrs' });
""")
    assert out["pm"] == {"ra": 269.45207696, "dec": 4.69336497, "fov_arcmin": "3", "pm_ra_masyr": -801.551,
                         "pm_dec_masyr": 10362.394, "epoch": 2000}
    assert out["name"] == {"name": "3C 273", "fov_arcmin": "3"}  # the server resolves it (with proper motion)
    assert out["bare"] == {"ra": 83.63321, "dec": 22.01446, "fov_arcmin": "10"}
    assert out["info"] == {"ra": 269.45207696, "dec": 4.69336497, "pm_ra_masyr": -801.551, "pm_dec_masyr": 10362.394}


# ---------------------------------------------------------------------------
# Composition with the real API middleware (authentication + quotas)
# ---------------------------------------------------------------------------


def _keyed_app(public: bool = True) -> FastAPI:
    """imaging.router + the UI behind api.py's own auth/quota middleware function (api.app untouched)."""
    import api

    app = FastAPI()
    app.middleware("http")(api.request_metrics)
    app.include_router(imaging.router)
    imaging.mount_ui(app, public=public)
    return app


def test_ui_loads_without_an_api_key_while_the_api_still_requires_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """With API_KEYS set, a browser cannot send X-API-Key on page navigation: the UI shell must load
    (and not count against the quota), while every /api call still needs the key."""
    monkeypatch.setenv("API_KEYS", "secret-key-1")
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "3")
    app = _keyed_app()
    assert app.state.ui_public and set(app.state.ui_routes) == {"/", "/index.html", "/app.js", "/styles.css"}
    client = TestClient(app)
    for _ in range(5):  # well past the 3-per-minute quota
        for path in ("/", "/index.html", "/app.js", "/styles.css"):
            resp = client.get(path)
            assert resp.status_code == 200, (path, resp.text)
            assert resp.headers["cache-control"] == "no-cache"
    assert client.get("/").text == INDEX
    assert client.get("/app.js").headers["content-type"].startswith("text/javascript")
    head = client.head("/")
    assert head.status_code == 200 and head.content == b""
    denied = client.get("/api/v1/cutouts/surveys")
    assert denied.status_code == 401 and denied.json() == {"detail": "Authentication required"}
    assert client.get("/api/v1/cutouts/surveys", headers={"X-API-Key": "secret-key-1"}).status_code == 200
    # Only the exact UI files bypass the middleware.
    assert client.get("/not-a-file.js").status_code == 401
    assert client.post("/").status_code == 401
    assert imaging.is_ui_path(app, "/app.js") and not imaging.is_ui_path(app, "/api/v1/cutouts")


def test_ui_can_be_kept_behind_authentication(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEYS", "secret-key-1")
    client = TestClient(_keyed_app(public=False))
    assert client.get("/").status_code == 401
    assert client.get("/", headers={"X-API-Key": "secret-key-1"}).status_code == 200


def test_ui_middleware_respects_the_root_path(tmp_path: Path) -> None:
    app = FastAPI()
    imaging.mount_ui(app)
    client = TestClient(app, root_path="/astro")
    assert client.get("/astro/").status_code == 200 and "<title>AstroSearch Sky Explorer</title>" in client.get("/astro/").text
    assert client.get("/astro/app.js").status_code == 200


# ---------------------------------------------------------------------------
# Round 3: the first target event carries the searched name to the cutout stack;
# blank-image states from X-Cutout-Blank
# ---------------------------------------------------------------------------


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_first_target_event_sends_the_name_to_the_cutout_stack(tmp_path: Path) -> None:
    """A stream 'start' event whose target has no proper motion (Barnard's star searched by name):
    the very first cutout strip must ask the server to resolve the name (which applies the proper
    motion), not use the bare J2000 position."""
    out = run_node(tmp_path, r"""
const body = dom.installDom('http://ui.test/');
for (const id of ['sed-panel', 'lc-panel', 'cutout-panel', 'sso-panel']) body.append(dom.element('div', id));
const fov = dom.element('select', 'cutout-fov');
fov.value = '3';
body.append(fov);
const fetched = [];
globalThis.fetch = async (url) => { fetched.push(new URL(String(url))); return dom.json({ detail: 'Not Found' }, 404); };
m.state.query = { name: "Barnard's star", radius: 10 };
m.state.target = null;
m.onTargetKnown(m.targetInfo({ ra: 269.45207696, dec: 4.69336497 }), new AbortController().signal);
for (let i = 0; i < 20; i++) await new Promise((r) => setTimeout(r, 0));
const stack = fetched.find((u) => u.pathname === '/api/v1/cutouts/stack');
out.stack = stack ? Object.fromEntries(stack.searchParams) : null;
out.target = m.state.target;
out.known = m.knownTarget({ ra: 1, dec: 2 }, null);
out.keepsOwnName = m.knownTarget({ ra: 1, dec: 2, name: 'M 1' }, { name: 'other' }).name;
""")
    assert out["stack"] == {"name": "Barnard's star", "fov_arcmin": "3"}
    assert out["target"] == {"ra": 269.45207696, "dec": 4.69336497, "name": "Barnard's star"}
    assert out["known"] == {"ra": 1, "dec": 2} and out["keepsOwnName"] == "M 1"


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_cutout_status_distinguishes_confirmed_and_unconfirmed_blanks(tmp_path: Path) -> None:
    out = run_node(tmp_path, r"""
const H = (h) => new Headers(h);
out.noData = m.cutoutStatus(H({ 'X-Cutout-Blank': 'no-data', 'X-Cutout-Coverage': '0.0000' }), { in_coverage: null });
out.unconfirmed = m.cutoutStatus(H({ 'X-Cutout-Blank': 'unconfirmed', 'X-Cutout-Coverage': '0.0000' }), { in_coverage: true });
out.jpegFailure = m.cutoutStatus(H({ 'X-Cutout-Blank': 'rendering-failure', 'X-Cutout-Degraded': 'blank (uniform) JPEG ...' }), { in_coverage: true });
out.emptyNoHeader = m.cutoutStatus(H({ 'X-Cutout-Coverage': '0.0000' }), { in_coverage: true });
out.outsideFootprint = m.cutoutStatus(H({}), { in_coverage: false });
out.data = m.cutoutStatus(H({ 'X-Cutout-Coverage': '1.0000' }), { in_coverage: true });
""")
    assert out["noData"]["kind"] == "nodata" and out["noData"]["message"] == "No survey data here"
    assert out["unconfirmed"]["kind"] == "unconfirmed"
    assert out["unconfirmed"]["message"] == "Blank image; footprint could not be checked"
    assert out["jpegFailure"]["kind"] == "degraded"
    # A blank without a confirmation is never presented as 'no survey data'.
    assert out["emptyNoHeader"]["kind"] == "unconfirmed"
    assert out["outsideFootprint"]["kind"] == "nodata" and out["data"] == {"kind": "ok", "message": None}


def test_unconfirmed_blank_state_is_styled() -> None:
    css = (WEB / "styles.css").read_text(encoding="utf-8")
    assert ".cutout.unconfirmed img" in css

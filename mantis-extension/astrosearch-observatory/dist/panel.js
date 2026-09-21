(function () {
  'use strict';

  const IDS = {
    mega: 'aa065920-3010-4711-8ac7-14bc5fd33a7e',
    semantic: 'd8103fd7-73bd-4a09-8347-6de17f363a8a',
    sky: '69925de6-89f3-4e4d-9f89-42817e4490c2',
    tess: 'e5d48a5a-476c-489e-87e4-61c6c9f6b527'
  };

  const root = document.getElementById('root');
  root.innerHTML = `
    <main class="shell">
      <header class="hero">
        <div class="orb orb-a"></div><div class="orb orb-b"></div>
        <div class="eyebrow"><span class="live-dot"></span> ASTROSEARCH · MIT CSAIL MANTIS</div>
        <h1>Explore the sky<br><span>with evidence attached.</span></h1>
        <p class="lede">One organized cockpit for confirmed systems, Gaia reference stars, telescope signals, and scientifically bounded candidate review.</p>
        <div class="hero-actions">
          <button class="primary" data-map="mega">Launch 50K Mega Atlas <b>↗</b></button>
          <button class="ghost" data-map="tess">Open TESS Signal Lab</button>
        </div>
      </header>

      <section class="metrics" aria-label="Space totals">
        <article><strong>81,322</strong><span>rendered points</span></article>
        <article><strong>50,000</strong><span>Mega Atlas records</span></article>
        <article><strong>34,353</strong><span>Gaia DR3 sources</span></article>
        <article><strong>6,366</strong><span>confirmed planets</span></article>
      </section>

      <section class="section">
        <div class="section-heading"><div><span class="kicker">DATA UNIVERSE</span><h2>Choose a research layer</h2></div><span id="connection" class="status online">4 organized maps</span></div>
        <div class="map-grid">
          <article class="map-card featured">
            <div class="card-top"><span class="icon">✦</span><span class="pill cyan">PRIMARY</span></div>
            <h3>Astronomy Mega Atlas</h3><p>50,000 unified Exoplanet Archive, SIMBAD, and Gaia DR3 records with sky vectors and nearest-host context.</p>
            <div class="mini-stats"><span><b>50K</b> objects</span><span><b>37</b> fields</span><span><b>5</b> queues</span></div>
            <button data-map="mega">Explore Mega Atlas →</button>
          </article>
          <article class="map-card">
            <div class="card-top"><span class="icon violet">⌁</span><span class="pill">SEMANTIC</span></div>
            <h3>Catalog Intelligence</h3><p>Exoplanet systems grouped by descriptive similarity, identity evidence, and discovery context.</p>
            <div class="mini-stats"><span><b>15,647</b> records</span><span><b>1,368</b> clusters</span></div>
            <button data-map="semantic">Open semantic map →</button>
          </article>
          <article class="map-card">
            <div class="card-top"><span class="icon amber">◎</span><span class="pill">SKY</span></div>
            <h3>ICRS Sky Atlas</h3><p>A physical right-ascension and declination view. Spatial layout means sky direction, not language similarity.</p>
            <div class="mini-stats"><span><b>15,647</b> coordinates</span><span><b>3D</b> unit vectors</span></div>
            <button data-map="sky">Open sky atlas →</button>
          </article>
          <article class="map-card">
            <div class="card-top"><span class="icon rose">∿</span><span class="pill rose-pill">SIGNALS</span></div>
            <h3>TESS Signal Lab</h3><p>Real SPOC light curves represented across sectors for repeatability, quality, and shape-comparison review.</p>
            <div class="mini-stats"><span><b>28</b> observations</span><span><b>10</b> hosts</span><span><b>100%</b> synthesis</span></div>
            <button data-map="tess">Open signal lab →</button>
          </article>
        </div>
      </section>

      <section class="section review-section">
        <div class="section-heading"><div><span class="kicker">REVIEW QUEUES</span><h2>Start with signal, not noise</h2></div><span class="safety">Human review required</span></div>
        <div class="review-grid">
          <article class="review-card priority"><div class="review-icon">⌖</div><div><strong>10</strong><h3>Host-neighborhood sources</h3><p>Gaia sources within 60 arcseconds of a catalogued exoplanet host. Proximity is a lead, never an identity claim.</p></div></article>
          <article class="review-card"><div class="review-icon violet-bg">↯</div><div><strong>226</strong><h3>Photometric variables</h3><p>Gaia sources flagged variable for follow-up against cadence, artifacts, and known variability classes.</p></div></article>
          <article class="review-card"><div class="review-icon green-bg">✓</div><div><strong>32,860</strong><h3>Reliable astrometry</h3><p>Gaia reference sources with RUWE ≤ 1.4 for cleaner positional and motion analysis.</p></div></article>
          <article class="review-card"><div class="review-icon amber-bg">◫</div><div><strong>15,647</strong><h3>Known-object baseline</h3><p>The complete confirmed-planet, host-system, and SIMBAD comparison layer.</p></div></article>
        </div>
        <div id="bags" class="bag-strip">
          <span class="bag"><i></i>Known-object baseline <b>15,647</b></span>
          <span class="bag"><i></i>Gaia reference sources <b>34,353</b></span>
          <span class="bag"><i></i>Reliable astrometry <b>32,860</b></span>
          <span class="bag"><i></i>Photometric variables <b>226</b></span>
          <span class="bag"><i></i>Host neighborhoods <b>10</b></span>
        </div>
      </section>

      <section class="section science-section">
        <div><span class="kicker">SCIENTIFIC VIEWS</span><h2>Four ways to interrogate Gaia</h2></div>
        <div class="view-list">
          <div><span>01</span><b>Color–magnitude</b><small>BP−RP × G magnitude</small></div>
          <div><span>02</span><b>Proper motion</b><small>μα* × μδ</small></div>
          <div><span>03</span><b>Distance proxy</b><small>Parallax × apparent magnitude</small></div>
          <div><span>04</span><b>Candidate quality</b><small>Host separation × RUWE</small></div>
        </div>
        <button class="wide" data-map="mega">Open Mega Atlas and choose a scientific page →</button>
      </section>

      <section class="method">
        <div><span class="kicker">INTERPRETATION RULE</span><h2>Interesting is not discovered.</h2></div>
        <p>The workspace separates identity, sky proximity, signal similarity, and data quality. Candidate queues are prompts for repeat-observation and artifact review; no automated result is presented as a new astronomical object.</p>
      </section>

      <footer><span>AstroSearch Observatory v1.0.1</span><span id="active-map">4 research layers · 5 curated queues</span></footer>
      <div id="toast" role="status" aria-live="polite"></div>
    </main>`;

  function show(message, kind) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = kind === 'error' ? 'show error' : 'show';
    window.setTimeout(() => { toast.className = ''; }, 3200);
  }

  async function openMap(key) {
    const id = IDS[key];
    if (!id) return;
    try {
      if (!window.mantis || !window.mantis.maps) throw new Error('Use the Mantis Maps menu to open this layer.');
      await window.mantis.maps.open(id);
      show('Opened ' + ({mega:'Mega Atlas', semantic:'Catalog Intelligence', sky:'ICRS Sky Atlas', tess:'TESS Signal Lab'}[key] || 'map'));
    } catch (error) {
      show(error && error.message ? error.message : String(error), 'error');
    }
  }

  document.querySelectorAll('[data-map]').forEach((button) => {
    button.addEventListener('click', () => openMap(button.dataset.map));
  });

  function within(promise, milliseconds) {
    return Promise.race([
      promise,
      new Promise((_, reject) => window.setTimeout(() => reject(new Error('Mantis bridge timeout')), milliseconds))
    ]);
  }

  async function hydrate() {
    if (!window.mantis || !window.mantis.maps) return;
    try {
      const active = await within(window.mantis.maps.getActive(), 1800);
      if (active) document.getElementById('active-map').textContent = 'Active · ' + (active.name || active.mapName || 'research map');
    } catch (_) {
      // The dashboard remains complete when the host bridge is unavailable.
    }
  }

  hydrate();
})();

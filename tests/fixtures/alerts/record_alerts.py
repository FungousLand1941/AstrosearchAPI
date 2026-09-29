"""Record real ALeRCE / Fink (ZTF, LSST) answers and the crossmatch queries of their alerts.

Live network, polite (a few dozen small requests)::

    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py            # everything
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py validation # class/tag validation only
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py famous     # famous-object enrichments only
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py review     # review-round-2 + random positions
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py round3     # review-round-3 positions
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py backlog    # truncated-window backlog scenarios
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py enrich     # re-record xmatch_* for the stored alerts
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py variables  # Fink CV / RR Lyrae rows + enrichment
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py duplicates [UNTIL_MJD]  # ALeRCE rows per classifier version
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py versions   # add live /probabilities answers to alerce*

Each scenario is stored like the catalog fixtures (see ``tests/fixture_io.py``):
``<scenario>.json`` holds the request/response metadata plus the ``params`` the
scenario was run with, and ``<scenario>.<n>.body`` the raw response bytes. The
offline tests replay them strictly (same URL, query and body) with respx.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
for extra in (ROOT, ROOT / "tests"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import httpx
from fixture_io import redact
from helpers import make_service

from alerts import Alert, AlertEnricher, AlertService, AlertStore, fetch_alerts, now_mjd
from datasets import MetadataStore

# Famous objects (SIMBAD ICRS J2000 positions, queried 2026-09-28).
FAMOUS: dict[str, tuple[float, float]] = {
    # T CrB: recurrent nova / symbiotic star (SIMBAD 'V* T CrB', otype Sy*), Gaia parallax ~1.1 mas.
    "T_CrB": (239.87567594413002, 25.92017038415),
    # AT 2018cow (SIMBAD 'SN 2018cow'); host CGCG 137-068 at z = 0.0141.
    "AT2018cow": (244.00105833333333, 22.268083333333333),
    # M31N 2008-12a, the recurrent nova in M31 (SIMBAD 'RX J0045.4+4154', otype No*).
    "M31N_2008-12a": (11.370375, 41.902806),
    # The nucleus of M82 (a crowded field of X-ray binaries, SNRs and clusters inside M82).
    "M82_nucleus": (148.96969, 69.67938),
    # SN 2014J in M82, 58" from the galaxy's centre (SIMBAD 'SN 2014J').
    "SN2014J": (148.925583, 69.673889),
    # AT 2017gfo, the kilonova of GW170817, in NGC 4993 (SIMBAD 'GrW 170817' is type GWE).
    "AT2017gfo": (197.450375, -23.381481),
    # AT 2019dsg, a TDE; NED lists its host galaxy under the name 'AT 2019dsg'.
    "AT2019dsg": (314.26240, 14.20442),
    # IC 10 X-1 (SIMBAD '[BWF97] IC 10 X-1', HXB): a WR + black-hole binary in IC 10, a PGC 'M'
    # (multiple-system) galaxy.
    "IC10_X-1": (5.120971269999999, 59.28104471),
    # A point 150" N of M86's nucleus, inside M86's D25 ellipse (M86 = NGC 4406 is PGC type 'M').
    "M86_disk": (186.55042, 12.98722),
    # SN 2006gy in NGC 1260 (PGC 12219, type 'M').
    "SN2006gy": (49.36275, 41.40541666666667),
    # Foreground stars projected on nearby galaxies (Gaia DR3 J2016 positions):
    # 381261910408440576 near M31's nucleus (G = 12.2, parallax 75 sigma, RUWE 1.8);
    # 369244286970444416 in M31's disk (parallax 7.5 mas at 60 sigma, RUWE 5.3);
    # 1609299644238883584 in M101's D25 ellipse (G = 17.6, proper motion 16.9 mas/yr).
    "M31_fg_star": (10.6395038, 41.2639317),
    "M31_fg_star2": (10.7444782, 40.9837978),
    "M101_fg_star": (210.9002279, 54.4433872),
    # SN 1987A in the LMC (projected offset ~1.0 kpc at 49.6 kpc).
    "SN1987A": (83.86661833333334, -69.26975372222223),
    # SN 2011dh in M51 (NED names M51 'SN 1994I HOST') and SN 1993J in M81 ('SN 1993J HOST').
    "SN2011dh": (202.521273125, 47.16970075),
    "SN1993J": (148.85322816666667, 69.02047294444446),
    # Catalogued CVs with transient-style names: SIMBAD 'ZTF18aayefwp' (CV*) and
    # 'MASTER OT J073857.06+182648.2' (CV?) = ZTF18aaawtyh (NED lists 'AT 2017abr' there as a galaxy).
    "ZTF18aayefwp": (306.5039620530862, 33.66233011833),
    "ZTF18aaawtyh": (114.73774999999999, 18.44672222222222),
    # Galaxy nuclei whose Gaia DR3 5-parameter solutions have spurious, formally significant proper
    # motions (regressions: they were called Galactic stars and lost their host). M87 (AGN; 7.5 mas/yr
    # at 12 sigma, parallax -3.4 sigma), NGC 4395 (Sy2; 2.8 mas/yr at 17 sigma), NGC 4258 = M106 (Sy2;
    # Gaia source 1.5" from SIMBAD's position), the Seyfert 1 nuclei NGC 3783 (0.24 mas/yr at 15 sigma),
    # NGC 7469, NGC 6814 (RUWE 1.24) and NGC 3516, NGC 4151, and NGC 3115 (an S0 nucleus with
    # extragalactic LMXBs within 2").
    "M87_nucleus": (187.70593076725, 12.391123246083334),
    "NGC4395_nucleus": (186.45359712911997, 33.54686115781999),
    "NGC4258_nucleus": (184.7400833333333, 47.30371944444444),
    "NGC3783_nucleus": (174.7571236746, -37.73861378972),
    "NGC7469_nucleus": (345.8151, 8.8739),
    "NGC6814_nucleus": (295.66910924422996, -10.32364131079),
    "NGC3516_nucleus": (166.697763417, 72.56869399296),
    "NGC4151_nucleus": (182.63573325577997, 39.405850979869996),
    "NGC3115_nucleus": (151.30802937792, -7.718606308970001),
    # ASASSN-14ko, the periodic nuclear transient of ESO 253-G003 (Sy2 host; Gaia nucleus 1.2 mas/yr at 10 sigma).
    "ASASSN-14ko": (81.32554166666667, -46.00563888888889),
    # SN 2020oi in M100 (on a compact cluster, Gaia DSC P(galaxy) = 1) and SN 2006X in M100 (M100 has no CF4
    # distance of its own: Virgo's group distance); SN 2004dj on the cluster Sandage 96 in NGC 2403.
    "SN2020oi": (185.728875, 15.8236),
    "SN2006X": (185.72471, 15.80888),
    "SN2004dj": (114.32101338601001, 65.59939611145),
    # 3" north of M83's nucleus: the host must be named M83, not a 6dFGS fibre entry.
    "M83_near_nucleus": (204.25383, -29.864927777777778),
    # SN 2009ip (SIMBAD type s*b, its LBV progenitor) just outside NGC 7259's D25 ellipse (d_DLR 1.16).
    "SN2009ip": (335.7844166666667, -28.947888888888887),
    # Hosts outside the D25 ellipse: SN 2023bee -> NGC 2708 (d_DLR ~1.5), SN 2018aoz -> NGC 3923 (~1.45).
    "SN2023bee": (134.04841666666667, -3.3255694444444446),
    "SN2018aoz": (177.758, -28.744099999999996),
    # The blazar 3C 273 (SIMBAD BLL): a known variable and AGN.
    "3C273": (187.27791594049, 2.05238823055),
}
# Review round 2 (SIMBAD / Gaia DR3 positions, queried 2026-09-28):
REVIEW: dict[str, tuple[float, float]] = {
    # Galactic stars that Gaia DR3's DSC calls extragalactic or lists as galaxy candidates, with decisive
    # astrometry: the white dwarf WDJ153053.31+690231.98 (parallax 100 sigma, DSC ext 0.985), the dwarf nova
    # UV Per (pm 446 sigma), and two foreground stars on M31 ([LF82] d, 12.9-sigma parallax; M31GC
    # J004237+411422, pm 208 sigma), both Gaia galaxy candidates.
    "WD1647493723250083584": (232.72187939379668, 69.04262554294488),
    "UV_Per": (32.53473412720054, 57.189166068564425),
    "M31_LF82d": (10.675847289712872, 41.26275007396792),
    "M31GC_J004237": (10.653569378979347, 41.2395322366092),
    # Bright SIMBAD stars on M31's disc with huge Gaia excess noise: M_G -13.3 ... -10.5 at M31's distance.
    "M31_WWV2004_LPV": (10.801861658055273, 40.77813034081048),
    "M31_GSC02805-02180": (10.645509790548862, 41.34538678030782),
    "M31_MLV92_187476": (10.862927889768518, 41.081079385159036),
    "M31_GPM11.17+41.19": (11.17444344099562, 41.194919482730555),
    # The eclipsing binary Gaia DR3 6189441739218449664 (RUWE 5.1, pm 11 mas/yr at 31 sigma), 5.9' from
    # NGC 5078, whose 2003 HyperLEDA D25 is 51' (current: 2.6'), and a point 40' from NGC 5078.
    "EB_6189441739218449664": (199.85616010770747, -27.374244054046667),
    "NGC5078_40arcmin": (200.25, -27.95),
    # Background SNe projected on (or beside) nearby giants: PTF 11dws (z = 0.15) on M106, 0.8" from its host;
    # PTF 10hv (z = 0.052) on M101; SN 2008hz (z = 0.0795) and PTF 12gix (z = 0.11) on M31; M31's outskirts:
    # iPTF 15dql (z = 0.13, d_DLR 3.3), PTF 11paw (z = 0.19, 1.7), iPTF 15dhn (z = 0.15, 1.3).
    "PTF11dws": (184.7785, 47.355805555555555),
    "PTF10hv": (210.98408333333336, 54.45863888888889),
    "SN2008hz": (10.827583333333333, 42.170611111111114),
    "PTF12gix": (11.764458333333334, 41.502138888888894),
    "iPTF15dql": (11.512629166666665, 38.96977777777778),
    "PTF11paw": (9.758583333333334, 42.01794444444444),
    "iPTF15dhn": (12.236, 43.059666666666665),
    # SN 2016bam (z = 0.0135) in NGC 2445, with a z = 2.06 QSO 15" away; PTF 12lz (z = 0.07) with a z = 0.336
    # galaxy 48" away and a QSO 53" away, 3 D25 radii from M33.
    "SN2016bam": (116.71966666666667, 39.02272222222222),
    "PTF12lz": (24.32041666666667, 30.120638888888887),
    # SNe on a NED '*' ("star or point source") entry without a Gaia source: SN 2002gn, SN 2018aks.
    "SN2002gn": (29.22733333333333, -1.113611111111111),
    "SN2018aks": (157.46508333333333, 9.01293888888889),
}
# Random blank positions at |b| > 30 deg (the reviewer's sample: random.seed(7), uniform in RA and sin(dec) in
# [-0.5, 1]); rand6 lies 33' from the Ursa Minor dwarf, whose 2003 D25 (logD25 2.54) is 6 times today's.
RANDOM_POSITIONS: dict[str, tuple[float, float]] = {
    "rand0": (192.9175215504081, 2.781850406686493), "rand1": (20.879612918894452, 15.138523680868694),
    "rand2": (13.498437039114556, 8.654079263941124), "rand3": (25.147952486862817, -21.341777431445312),
    "rand4": (152.82690809130503, 47.75511822753619), "rand5": (44.568706013872415, -9.505457671295577),
    "rand6": (225.87596006601214, 67.1557232073773), "rand7": (207.7570615022995, 5.452511913966438),
    "rand8": (351.4518380134512, -25.47555535530242), "rand9": (51.93183000867751, -18.863317720357127),
    "rand10": (230.00884881342625, 3.3592457097746107), "rand11": (197.1880076554408, -23.94230570840837),
    "rand12": (21.45642118784376, -11.014763745280108), "rand13": (244.94399034544293, 8.128198320862905),
    "rand14": (163.14637549347913, -2.8860343815904383), "rand15": (189.0707413721225, 54.36118773443206),
    "rand16": (352.86294509732954, -18.838476241663162), "rand17": (150.5242158426818, 39.47276724689663),
    "rand18": (54.71443247778171, 13.499957978919726), "rand19": (14.114612537077559, 30.153859821429943),
}
# Review round 3 (SIMBAD ICRS positions, queried 2026-09-29):
ROUND3: dict[str, tuple[float, float]] = {
    # Blazars / QSOs whose own AGN entry is at the alert (SIMBAD BLL/Bla with redshifts): 3C 273 (z = 0.158; a
    # z = 0.0053 dwarf lies 10.8" away), BL Lac (z = 0.0655), PKS 2155-304 (z = 0.116, companion 4.4" away),
    # Mrk 421 (z = 0.030, its own D25 galaxy), 3C 279 (z = 0.536) and PG 1553+113 (z = 0.36).
    "3C273": (187.27791594049, 2.05238823055),
    "BL_Lac": (330.68038064033993, 42.27777206022),
    "PKS2155-304": (329.71693843745, -30.225588457719997),
    "Mrk421": (166.11380868146, 38.20883291552),
    "3C279": (194.04652741491665, -5.789312541944445),
    "PG1553+113": (238.92935002022, 11.190101569430002),
    # Galaxy nuclei with a SIMBAD '*' duplicate entry (the Gaia source there is DSC-extragalactic with a large
    # excess noise): LEDA 1798300 on SDSS J125151.03+270706.1 (z = 0.027), 2MASS J09480288+1319111 (z = 0.081),
    # Pul -3 1020035 on LEDA 43358 (z = 0.035), 2MASS J14075302+5336515 (z = 0.078); and MGC 97030, a real
    # foreground star (Gaia pm 9.6 mas/yr at 34 sigma) on a galaxy entry.
    "LEDA1798300": (192.96265666372997, 27.118380852369995),
    "2MASS_J09480288+1319111": (147.01196897129, 13.319751523210002),
    "Pul-3_1020035": (192.382818, 25.996307),
    "2MASS_J14075302+5336515": (211.97093492933, 53.61428059463),
    "MGC97030": (214.53133333333332, 0.08519444444444442),
    # Blank points 50" and 45" from '[RGG2003] new galaxy' (a z = 0.0053 dwarf 9" from 3C 273's PGC 41121).
    "blank_near_dwarf": (187.26112, 2.05167),
    "blank_near_dwarf2": (187.2750, 2.03917),
    # The Sculptor dSph RR Lyrae star EV* SclG V0214, 12' from Sculptor's centre (current D25: 34").
    "SclG_V0214": (14.83046939535, -33.6108759036),
}
# Polite recording: at most this many enrichments at once (each makes ~8 archive requests).
RECORD_CONCURRENCY = 3
LSST_SINCE_MJD = 61200.0  # 2026-06-09: Fink/LSST last processed night so far is 2026-07-14.


class Recorder:
    def __init__(self) -> None:
        self.log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(self, response: httpx.Response) -> None:
        await response.aread()
        self.log.append((response.request, response))

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=120.0, follow_redirects=True, event_hooks={"response": [self.hook]})

    def save(self, name: str, params: dict[str, Any]) -> None:
        for old in HERE.glob(f"{name}.*.body"):
            old.unlink()
        exchanges = []
        for idx, (request, response) in enumerate(self.log):
            (HERE / f"{name}.{idx}.body").write_bytes(redact(response.content))
            exchanges.append({
                "method": request.method,
                "url": str(request.url),
                "request_body": request.content.decode("utf-8", "replace") if request.content else "",
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "match": [],
            })
        meta = {"catalog": name, "params": params, "exchanges": exchanges}
        (HERE / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"{name:<24} {len(exchanges)} exchange(s)")
        self.log.clear()


async def record_broker(name: str, broker: str, since: float, until: float, limit: int, options: dict[str, Any]) -> list[Alert]:
    rec = Recorder()
    async with rec.client() as client:
        result = await fetch_alerts(client, broker, since_mjd=since, until_mjd=until, limit=limit, options=options)
    rec.save(name, {"broker": broker, "since_mjd": since, "until_mjd": until, "limit": limit, "options": options,
                    "n_alerts": len(result.alerts), "warnings": result.warnings, "truncated": result.truncated,
                    "boundary_mjd": result.boundary_mjd})
    return result.alerts


async def record_enrichment(name: str, alerts: list[Alert]) -> None:
    rec = Recorder()
    semaphore = asyncio.Semaphore(RECORD_CONCURRENCY)
    async with rec.client() as client:
        enricher = AlertEnricher(make_service(client))

        async def one(alert: Alert) -> Any:
            async with semaphore:
                return await enricher.enrich(alert)

        results = await asyncio.gather(*(one(a) for a in alerts))
    for alert, res in zip(alerts, results, strict=True):
        print(f"  {alert.alert_id}: status={res.status} new={res.is_new} star={res.known_star} var={res.known_variable} "
              f"host={(res.host or {}).get('name')} ({(res.host or {}).get('method')}) "
              f"failures={[f['catalog'] for f in res.failures]}")
    rec.save(name, {"alerts": [a.as_dict() for a in alerts]})


async def record_validation(name: str, broker: str, since: float, until: float, options: dict[str, Any]) -> None:
    """Record the real answers to an unknown/unsupported class or tag (empty list or plain text, then the class list)."""
    rec = Recorder()
    outcome = "no error"
    async with rec.client() as client:
        try:
            await fetch_alerts(client, broker, since_mjd=since, until_mjd=until, limit=2, options=options)
        except ValueError as exc:
            outcome = f"ValueError: {exc}"[:160]
    print(f"  {name}: {outcome}")
    rec.save(name, {"broker": broker, "since_mjd": since, "until_mjd": until, "limit": 2, "options": options})


async def record_validations(until: float) -> None:
    await record_validation("invalid_alerce", "alerce", round(until - 1.0, 4), until,
                            {"classifier": "stamp_classifier", "class_name": "NotAClass", "mjd_field": "firstmjd"})
    await record_validation("invalid_fink_ztf", "fink", round(until - 1.0, 4), until, {"class_name": "NotAClass"})
    await record_validation("invalid_fink_lsst", "fink_lsst", round(until - 1.0, 4), until, {"class_name": "nope"})
    # Listed by /tags with "API support": false -> HTTP 400 "only available from the Livestream service".
    await record_validation("unsupported_fink_lsst", "fink_lsst", round(until - 1.0, 4), until,
                            {"class_name": "uniform_sample"})


async def record_famous(until: float) -> None:
    famous = [Alert("manual", name, ra, dec, until, None, None, None, None, "", survey="none")
              for name, (ra, dec) in FAMOUS.items()]
    await record_enrichment("xmatch_famous", famous)


async def record_round3(until: float) -> None:
    """Enrichments of the review-round-3 positions."""
    await record_enrichment("xmatch_round3", [Alert("manual", name, ra, dec, until, None, None, None, None, "",
                                                    survey="none") for name, (ra, dec) in ROUND3.items()])


async def record_review(until: float) -> None:
    """Enrichments of the review-round-2 positions and of random blank positions."""
    positions = {**REVIEW, **RANDOM_POSITIONS}
    await record_enrichment("xmatch_review", [Alert("manual", name, ra, dec, until, None, None, None, None, "",
                                                    survey="none") for name, (ra, dec) in positions.items()])


BACKLOG_OVERLAP_DAYS = 0.25
BACKLOG_LIMIT = 2


async def pick_window(broker: str, options: dict[str, Any], until: float, lo: int, hi: int) -> tuple[float, float]:
    """(end, lookback days) of a recent window holding between lo and hi objects (ending just after an alert)."""
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        recent = await fetch_alerts(client, broker, since_mjd=until - 14.0, until_mjd=until, limit=100, options=options)
        if not recent.alerts:
            raise SystemExit(f"no {broker} alerts in the last 14 days")
        end = round(max(a.first_mjd if broker == "alerce" else a.mjd for a in recent.alerts) + 0.001, 4)
        for days in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
            found = await fetch_alerts(client, broker, since_mjd=end - days, until_mjd=end, limit=100, options=options)
            if lo <= len(found.alerts) <= hi and not found.truncated:
                return end, days
    raise SystemExit(f"no {broker} window with {lo}-{hi} objects found")


async def run_backlog(client: httpx.AsyncClient, store: AlertStore, broker: str, options: dict[str, Any], until: float,
                      days: float) -> list[Any]:
    """Default-window polls (fixed clock, limit 2, no crossmatch) until the backlog is gone (shared with the test)."""
    svc = AlertService(store, client, None, clock=lambda: until, lookback_days=days, overlap_days=BACKLOG_OVERLAP_DAYS)
    results = []
    for _ in range(15):
        res = await svc.poll(broker, limit=BACKLOG_LIMIT, crossmatch=False, options=options)
        results.append(res)
        if res.window == "new" and res.backlog is None and len(results) > 1:
            break
    return results


async def record_backlog(name: str, broker: str, options: dict[str, Any], until: float) -> None:
    """A truncated default window ingested completely over successive polls, then the reference (limit 100) fetch."""
    until, days = await pick_window(broker, options, until, 5, 12)
    rec = Recorder()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # MetadataStore keeps its init connection
        store = AlertStore(MetadataStore(f"sqlite:///{Path(tmp, 'rec.sqlite3').as_posix()}"))
        async with rec.client() as client:
            results = await run_backlog(client, store, broker, options, until, days)
            reference = await fetch_alerts(client, broker, since_mjd=results[0].since_mjd, until_mjd=until, limit=100,
                                           options=options)
        stored = sorted(r["id"] for r in store.list(limit=1000))
    ref_ids = sorted(a.alert_id for a in reference.alerts)
    print(f"  {name}: lookback {days} d, reference {len(ref_ids)} objects, stored {len(stored)} after "
          f"{len(results)} polls ({[r.window for r in results]}); complete={set(ref_ids) <= set(stored)}")
    rec.save(name, {"broker": broker, "options": options, "until_mjd": until, "lookback_days": days,
                    "reference_ids": ref_ids, "polls": [r.as_dict() for r in results]})


async def rerecord_enrichments() -> None:
    """Re-record the xmatch_* scenarios for the alerts they already hold (after a query change)."""
    for name in ("xmatch_alerce", "xmatch_fink_ztf", "xmatch_fink_lsst", "xmatch_fink_variables"):
        meta = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
        await record_enrichment(name, [Alert.from_dict(a) for a in meta["params"]["alerts"]])


async def record_variables(until: float) -> None:
    """Real Fink/ZTF rows whose broker cross-match carries SIMBAD variable labels and Gaia parallaxes."""
    rec = Recorder()
    found: list[Alert] = []
    since = round(until - 60.0, 4)
    async with rec.client() as client:
        for cls in ("(TNS) CV", "RRLyrae"):
            result = await fetch_alerts(client, "fink", since_mjd=since, until_mjd=until, limit=2,
                                        options={"class_name": cls})
            found.extend(result.alerts)
    rec.save("fink_variables", {"broker": "fink", "since_mjd": since, "until_mjd": until, "limit": 2,
                                "classes": ["(TNS) CV", "RRLyrae"], "alerts": [a.as_dict() for a in found]})
    await record_enrichment("xmatch_fink_variables", found)


DUPLICATES_OPTIONS = {"classifier": "lc_classifier", "class_name": "AGN", "mjd_field": "lastmjd"}
DUPLICATES_LIMIT = 15


async def record_duplicates(until: float) -> None:
    """ALeRCE lc_classifier AGN rows: one row per classifier version (dedupe + newest-version choice)."""
    since = round(until - 3.0, 4)
    rec = Recorder()
    async with rec.client() as client:
        result = await fetch_alerts(client, "alerce", since_mjd=since, until_mjd=until, limit=DUPLICATES_LIMIT,
                                    options=DUPLICATES_OPTIONS)
    summary = duplicates_params(result)
    print(f"  alerce_duplicates: {len(summary['alert_ids'])} alerts, {len(summary['repeated'])} repeated per version, "
          f"{len(summary['versions'])} resolved, truncated={result.truncated}, requests={result.requests}, "
          f"warnings={result.warnings}")
    rec.save("alerce_duplicates", {"broker": "alerce", "since_mjd": since, "until_mjd": until, "limit": DUPLICATES_LIMIT,
                                   "options": DUPLICATES_OPTIONS, **summary})


def duplicates_params(result: Any) -> dict[str, Any]:
    """The expected results of the duplicates scenario (checked by the offline test)."""
    return {"alert_ids": [a.object_id for a in result.alerts],
            "repeated": [a.object_id for a in result.alerts if len(a.extra.get("classifier_rows") or []) > 1],
            "versions": {a.object_id: a.extra["classifier_version"] for a in result.alerts
                         if a.extra.get("classifier_version")},
            "probabilities": {a.object_id: a.probability for a in result.alerts},
            "truncated": result.truncated, "boundary_mjd": result.boundary_mjd, "requests": result.requests,
            "warnings": result.warnings}


async def add_versions(name: str) -> None:
    """Append to a recorded ALeRCE scenario the ``/objects/{oid}/probabilities`` answers that the fetch now
    requests for every kept object, recorded live; the scenario's other exchanges (including answers already
    recorded) are replayed unchanged. The duplicates scenario's expected results are recomputed."""
    import respx
    from fixture_io import load_exchanges, replay_side_effect

    meta = json.loads((HERE / f"{name}.json").read_text(encoding="utf-8"))
    params = meta["params"]
    replay = replay_side_effect(load_exchanges("alerts", [name]))
    wanted: list[str] = []

    def probabilities(request: httpx.Request) -> httpx.Response:
        # Phase 1 (offline): note every /probabilities request; answer recorded ones, others with a placeholder
        # (which objects are kept, hence requested, does not depend on the answers).
        try:
            return replay(request)
        except Exception:  # noqa: BLE001 - FixtureMismatch: not recorded yet
            if str(request.url) not in wanted:
                wanted.append(str(request.url))
            return httpx.Response(200, json=[])

    async def run(client: httpx.AsyncClient) -> Any:
        if "polls" in params:  # a backlog scenario: the same polls, then the reference fetch
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                store = AlertStore(MetadataStore(f"sqlite:///{Path(tmp, 'rec.sqlite3').as_posix()}"))
                results = await run_backlog(client, store, params["broker"], params["options"], params["until_mjd"],
                                            params["lookback_days"])
                return await fetch_alerts(client, params["broker"], since_mjd=results[0].since_mjd,
                                          until_mjd=params["until_mjd"], limit=100, options=params["options"])
        return await fetch_alerts(client, params["broker"], since_mjd=params["since_mjd"], until_mjd=params["until_mjd"],
                                  limit=params["limit"], options=params["options"])

    with respx.mock(assert_all_called=False) as mock:
        mock.get(url__regex=r".*/objects/[^/]+/probabilities.*").mock(side_effect=probabilities)
        mock.route().mock(side_effect=replay)
        async with httpx.AsyncClient(timeout=120.0) as client:
            await run(client)
    # Phase 2 (live): record the missing answers.
    added: list[tuple[httpx.Request, httpx.Response]] = []
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as live:
        for url in wanted:
            response = await live.get(url)
            added.append((response.request, response))
    start = len(meta["exchanges"])
    for idx, (request, response) in enumerate(added, start=start):
        (HERE / f"{name}.{idx}.body").write_bytes(redact(response.content))
        meta["exchanges"].append({"method": request.method, "url": str(request.url), "request_body": "",
                                  "status_code": response.status_code,
                                  "content_type": response.headers.get("content-type", ""), "match": []})
    (HERE / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"{name:<24} +{len(added)} /probabilities exchange(s)")
    if name == "alerce_duplicates":
        with respx.mock(assert_all_called=False) as mock:
            mock.route().mock(side_effect=replay_side_effect(load_exchanges("alerts", [name])))
            async with httpx.AsyncClient(timeout=120.0) as client:
                result = await run(client)
        params.update(duplicates_params(result))
        (HERE / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        print(f"  alerce_duplicates: {len(result.alerts)} alerts, warnings={result.warnings}")


async def main() -> None:
    os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    # An explicit window end (UTC MJD) re-records a scenario over the same window, e.g.
    # ``record_alerts.py duplicates 61311.5221``.
    until = float(sys.argv[2]) if len(sys.argv) > 2 else round(now_mjd(), 4)
    if mode in {"validation", "all"}:
        await record_validations(until)
    if mode in {"famous", "all"}:
        await record_famous(until)
    if mode in {"review", "all"}:
        await record_review(until)
    if mode in {"round3", "all"}:
        await record_round3(until)
    if mode == "enrich":
        await rerecord_enrichments()
    if mode == "versions":
        for name in ("alerce", "alerce_backlog", "alerce_duplicates"):
            await add_versions(name)
    if mode in {"duplicates", "all"}:
        await record_duplicates(until)
    if mode in {"variables", "all"}:
        await record_variables(until)
    if mode in {"backlog", "all"}:
        await record_backlog("fink_backlog", "fink", {"class_name": "SN candidate"}, until)
        await record_backlog("alerce_backlog", "alerce",
                             {"classifier": "stamp_classifier", "class_name": "SN", "mjd_field": "firstmjd"}, until)
    if mode == "all":
        alerce = await record_broker("alerce", "alerce", round(until - 3.0, 4), until, 3,
                                     {"classifier": "stamp_classifier", "class_name": "SN", "mjd_field": "firstmjd"})
        fink = await record_broker("fink_ztf", "fink", round(until - 10.0, 4), until, 3, {"class_name": "SN candidate"})
        lsst = await record_broker("fink_lsst", "fink_lsst", LSST_SINCE_MJD, until, 3,
                                   {"class_name": "extragalactic_new_candidate"})
        await record_enrichment("xmatch_alerce", alerce)
        await record_enrichment("xmatch_fink_ztf", fink)
        await record_enrichment("xmatch_fink_lsst", lsst)


if __name__ == "__main__":
    asyncio.run(main())

"""Record real ALeRCE / Fink (ZTF, LSST) answers and the crossmatch queries of their alerts.

Live network, polite (a few dozen small requests)::

    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py            # everything
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py validation # class/tag validation only
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py famous     # famous-object enrichments only
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py backlog    # truncated-window backlog scenarios
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py enrich     # re-record xmatch_* for the stored alerts
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py variables  # Fink CV / RR Lyrae rows + enrichment
    .venv/Scripts/python.exe tests/fixtures/alerts/record_alerts.py duplicates # ALeRCE rows repeated per classifier version

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
    ids = [a.object_id for a in result.alerts]
    repeated = [a.object_id for a in result.alerts if a.extra.get("classifier_choice")]
    print(f"  alerce_duplicates: {len(ids)} alerts ({len(set(ids))} unique), {len(repeated)} repeated per version, "
          f"truncated={result.truncated}, requests={result.requests}, warnings={result.warnings}")
    rec.save("alerce_duplicates", {"broker": "alerce", "since_mjd": since, "until_mjd": until, "limit": DUPLICATES_LIMIT,
                                   "options": DUPLICATES_OPTIONS, "alert_ids": ids, "repeated": repeated,
                                   "truncated": result.truncated, "boundary_mjd": result.boundary_mjd,
                                   "requests": result.requests})


async def main() -> None:
    os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
    until = round(now_mjd(), 4)
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in {"validation", "all"}:
        await record_validations(until)
    if mode in {"famous", "all"}:
        await record_famous(until)
    if mode == "enrich":
        await rerecord_enrichments()
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

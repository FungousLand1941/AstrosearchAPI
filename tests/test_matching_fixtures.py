"""Recorded real-archive cones for the Bayesian matching engine.

Recording (live network, one cone query per catalogue and target)::

    .venv/Scripts/python.exe tests/test_matching_fixtures.py record            # every target
    .venv/Scripts/python.exe tests/test_matching_fixtures.py record kruger60a  # some targets
    .venv/Scripts/python.exe tests/test_matching_fixtures.py record-search m13_core_3arcsec

``record-search`` records whole searches of RECORDED_SEARCHES end to end through the
service -- every request it sends (density probes included) and, for names, the Sesame
answer -- and ``replay_search`` replays them strictly.

stores ``tests/fixtures/matching/<key>/<catalog>.json`` + bodies with the helpers of
``fixture_io`` (same format, strict request replay), and the Sesame answers used by
name searches in ``tests/fixtures/matching/sesame/<file>.xml``. The offline tests below
(crowded fields) and in ``test_matching_regressions.py`` replay them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import respx

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import fixture_io
from fixture_io import load_exchanges, replay_side_effect
from helpers import make_service, offline_client

MATCHING = fixture_io.FIXTURES / "matching"
SESAME_DIR = MATCHING / "sesame"

# A crowded bulge field near Baade's window (l = 4.16, b = -3.30): Gaia DR3 holds 268 sources
# within 30" (~95 per arcmin^2), so a 20" cone holds ~120 stars seen by several surveys.
CROWDED_KEY = "bulge_field"
CROWDED = {"ra": 272.0, "dec": -27.0, "radius_arcsec": 20.0}
CROWDED_CATALOGS = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2"]
FIXTURE_DIR = f"matching/{CROWDED_KEY}"

# Sesame answers recorded for name searches: file stem -> name.
SESAME_NAMES = {"gj860a": "GJ 860 A", "barnards_star": "Barnard's star"}
BARNARD_PM = (-801.551, 10362.394)

# Every recorded target: position (and epoch / proper motion, which shape the cones), cone
# radius and catalogues. Positions are copied exactly into the tests that replay them.
MATCHING_TARGETS: dict[str, dict[str, Any]] = {
    CROWDED_KEY: {**CROWDED, "catalogs": CROWDED_CATALOGS},
    # Baade's window at the default 3" radius: Gaia DR3 counts 1.09e6 per deg^2 in 0.1 deg
    # (34,201 sources in a 0.1-deg cone, VizieR I/355 COUNT(*), 2026-09-28).
    "baade_3arcsec": {"ra": 272.0, "dec": -27.0, "radius_arcsec": 3.0, "catalogs": ["gaia_dr3", "twomass_psc"]},
    # High proper-motion stars at their SIMBAD J2000 coordinates, searched without an epoch.
    "barnard_undated": {"ra": 269.45207696, "dec": 4.69336497, "radius_arcsec": 10.0,
                        "catalogs": ["simbad", "gaia_dr3", "twomass_psc"]},
    "tau_ceti_undated": {"ra": 26.01701307, "dec": -15.93747989, "radius_arcsec": 10.0,
                         "catalogs": ["simbad", "gaia_dr3", "twomass_psc"]},
    "61cyga_undated": {"ra": 316.72474829, "dec": 38.74941732, "radius_arcsec": 10.0,
                       "catalogs": ["simbad", "gaia_dr3", "twomass_psc"]},
    # 3C 273 typed with three decimals (1.43" from the quasar).
    "3c273_rounded": {"ra": 187.278, "dec": 2.052, "radius_arcsec": 10.0,
                      "catalogs": ["simbad", "gaia_dr3", "twomass_psc", "panstarrs_dr2", "sdss", "nvss"]},
    # Gaia DR3 field stars (parallax SNR > 10) near unrelated NVSS sources, at J2016.
    "nvss_star_629415710493682432": {"ra": 150.8841469495608, "dec": 21.170397587798014, "epoch": 2016.0,
                                     "radius_arcsec": 35.0, "catalogs": ["gaia_dr3", "nvss"]},
    "nvss_star_629415706201972480": {"ra": 150.88452657706515, "dec": 21.170814829645522, "epoch": 2016.0,
                                     "radius_arcsec": 35.0, "catalogs": ["gaia_dr3", "nvss"]},
    "nvss_star_629559574718166784": {"ra": 151.3953787449629, "dec": 21.85844960158126, "epoch": 2016.0,
                                     "radius_arcsec": 35.0, "catalogs": ["gaia_dr3", "nvss"]},
    "nvss_star_628572522514194048": {"ra": 151.56290524316677, "dec": 20.591059342679564, "epoch": 2016.0,
                                     "radius_arcsec": 35.0, "catalogs": ["gaia_dr3", "nvss"]},
    # Gaia DR3 2767031379074805376 (G = 15.6) 5.5" from SIMBAD's 'HVC 104.2-48-168' (E quality).
    "hvc_star": {"ra": 8.88021305510004e-05, "dec": 12.998332944400289, "epoch": 2016.0, "radius_arcsec": 8.0,
                 "catalogs": ["simbad", "gaia_dr3"]},
    # Gaia DR3 2846803146692579712 at RA 359.99998: Pan-STARRS lists it twice (8.8 mas apart).
    "ps1_duplicate": {"ra": 359.9999806544865, "dec": 21.434972343632282, "epoch": 2016.0, "radius_arcsec": 3.0,
                      "catalogs": ["panstarrs_dr2", "gaia_dr3"]},
    # Kruger 60 A as resolved by name ('GJ 860 A' -> HD 239960A, SIMBAD J2000 + motion).
    "kruger60a": {"ra": 336.99815645, "dec": 57.69502238, "epoch": 2000.0, "pm_ra_masyr": -725.227,
                  "pm_dec_masyr": -223.461, "radius_arcsec": 16.0,
                  "catalogs": ["simbad", "gaia_dr3", "twomass_psc", "allwise"]},
    # Quasars whose AllWISE detections are 5-15 sigma off in the catalogue errors.
    "3c279": {"ra": 194.04652741, "dec": -5.78931254, "radius_arcsec": 5.0,
              "catalogs": ["allwise", "gaia_dr3", "simbad"]},
    "cygnus_a": {"ra": 299.86815237, "dec": 40.7339159, "radius_arcsec": 5.0,
                 "catalogs": ["allwise", "gaia_dr3", "simbad"]},
    "crf3_5763535293039121152": {"ra": 133.77828766791566, "dec": -3.0602251925444897, "radius_arcsec": 3.0,
                                 "catalogs": ["allwise", "gaia_dr3"]},
    "crf3_3975233510826944640": {"ra": 178.82623185784482, "dec": 19.66173009584727, "radius_arcsec": 3.0,
                                 "catalogs": ["allwise", "gaia_dr3"]},
    "crf3_903276775341338112": {"ra": 125.54192477481958, "dec": 33.90073764428269, "radius_arcsec": 3.0,
                                "catalogs": ["allwise", "gaia_dr3"]},
}


def barnard_2016_target() -> dict[str, Any]:
    """Barnard's star by name moved to J2016.0 with the resolver motion (what a name search
    with epoch=2016 queries)."""
    from crossmatch import resolved_search_target
    from providers import SesameResolver

    xml = (SESAME_DIR / "barnards_star.xml").read_text(encoding="utf-8")
    spec = resolved_search_target(SesameResolver.parse_response("Barnard's star", xml), epoch=2016.0)
    return {"ra": spec["ra"], "dec": spec["dec"], "epoch": 2016.0, "pm_ra_masyr": spec["pm_ra_masyr"],
            "pm_dec_masyr": spec["pm_dec_masyr"], "radius_arcsec": 10.0,
            "catalogs": ["gaia_dr3", "simbad", "twomass_psc"]}


def _register() -> None:
    # fixture_io records targets listed in its dictionaries: add ours at run time.
    for key, spec in MATCHING_TARGETS.items():
        fixture_io.EPOCH_TARGETS.setdefault(key, {k: v for k, v in spec.items() if k != "catalogs"})
    if (SESAME_DIR / "barnards_star.xml").exists():
        spec = barnard_2016_target()
        MATCHING_TARGETS.setdefault("barnard_name_2016", spec)
        fixture_io.EPOCH_TARGETS.setdefault("barnard_name_2016", {k: v for k, v in spec.items() if k != "catalogs"})


async def record_sesame() -> None:
    import httpx

    from providers import SesameResolver

    SESAME_DIR.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        for stem, name in SESAME_NAMES.items():
            resolver = SesameResolver(client)
            from urllib.parse import quote

            response = await client.get(f"{resolver.endpoint}?{quote(name, safe='')}",
                                        headers={"Accept": "application/xml, text/xml"})
            response.raise_for_status()
            (SESAME_DIR / f"{stem}.xml").write_text(response.text, encoding="utf-8")
            print(f"sesame     {name:<24} {len(response.text)} bytes")


async def record(keys: list[str] | None = None) -> None:
    await record_sesame()
    _register()
    # Default: every target not recorded yet.
    for key in keys or [k for k in MATCHING_TARGETS if not recorded(k)]:
        spec = MATCHING_TARGETS[key]
        for name in spec["catalogs"]:
            catalog, records, outcome = await fixture_io._record_catalog(name, key, spec["ra"], spec["dec"])
            if records:
                fixture_io._save(f"matching/{key}", catalog, records, coords_key=key)
            print(f"{key:<30} {catalog:<16} {outcome}")


def recorded(key: str) -> bool:
    return (MATCHING / key).exists()


def fixture_mismatches(record: Any) -> list[str]:
    """Every strict-replay mismatch hidden in a record (``UnifiedRecord`` or its dict): a
    ``FixtureMismatch`` is an AssertionError, but the executor records a catalogue's
    exception as an ordinary failure and a density probe records its own; both are found here."""
    data = record if isinstance(record, dict) else record.as_dict()
    found = [f"{f.get('catalog')}: {f.get('message')}" for f in data.get("failures") or []
             if f.get("error_type") == fixture_io.FixtureMismatch.__name__]
    densities = ((data.get("provenance") or {}).get("association") or {}).get("densities") or {}
    for name, info in densities.items():
        probe = (info or {}).get("probe") or {}
        if str(probe.get("error") or "").startswith(fixture_io.FixtureMismatch.__name__):
            found.append(f"{name} density probe: {probe.get('error')}")
    return found


def assert_replayed_exactly(record: Any) -> None:
    """Fail when any request of a replay did not match its recording (a query regression)."""
    mismatches = fixture_mismatches(record)
    if mismatches:
        raise fixture_io.FixtureMismatch("replayed requests differ from the recordings: " + " | ".join(mismatches))


async def replay_crossmatch(key: str, *, registry=None, allow_mismatch: bool = False, **kwargs):
    """``service.crossmatch`` on the recorded cones of ``key`` (position, epoch and motion
    of MATCHING_TARGETS; keyword arguments override / extend them). A request that does not
    match its recording fails the replay (``allow_mismatch=True`` to inspect it instead)."""
    _register()
    spec = MATCHING_TARGETS[key]
    exchanges = load_exchanges(f"matching/{key}")
    params: dict[str, Any] = {"radius_arcsec": spec["radius_arcsec"], "catalogs": spec["catalogs"],
                              "epoch": spec.get("epoch"), "pm_ra_masyr": spec.get("pm_ra_masyr"),
                              "pm_dec_masyr": spec.get("pm_dec_masyr")}
    params.update(kwargs)
    ra, dec = params.pop("ra", spec["ra"]), params.pop("dec", spec["dec"])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client, registry)
            record = await service.crossmatch(ra, dec, **params)
    if not allow_mismatch:
        assert_replayed_exactly(record)
    return record


# Searches recorded end to end (``record-search``): a name (stream path, with its Sesame
# answer in SESAME_DIR/<sesame>.xml) or coordinates, with the radius and catalogues.
RECORDED_SEARCHES: dict[str, dict[str, Any]] = {
    # Named extragalactic target whose SIMBAD entry lists a (Gaia-noise) parallax.
    "named_3c273": {"name": "3C 273", "sesame": "3c273_sesame", "radius_arcsec": 5.0,
                    "catalogs": ["simbad", "gaia_dr3", "first", "nvss", "rosat", "chandra", "xmm", "vlass"]},
    # Named radio / X-ray emitting stars with SIMBAD parallaxes: a pulsar, an X-ray binary,
    # a colliding-wind Wolf-Rayet binary.
    "named_psr_b0950": {"name": "PSR B0950+08", "sesame": "psr_b0950", "radius_arcsec": 5.0,
                        "catalogs": ["simbad", "gaia_dr3", "nvss", "first", "vlass"]},
    "named_ss433": {"name": "SS 433", "sesame": "ss433", "radius_arcsec": 5.0,
                    "catalogs": ["simbad", "gaia_dr3", "nvss", "vlass", "rosat", "xmm"]},
    "named_wr147": {"name": "WR 147", "sesame": "wr147", "radius_arcsec": 3.0,
                    "catalogs": ["simbad", "gaia_dr3", "vlass", "chandra", "xmm"]},
    # Named extended objects: a supernova remnant, an HII region, a globular cluster.
    "named_crab": {"name": "Crab Nebula", "sesame": "crab", "radius_arcsec": 10.0,
                   "catalogs": ["simbad", "ned", "gaia_dr3", "twomass_psc"]},
    "named_m42": {"name": "M 42", "sesame": "m42", "radius_arcsec": 10.0,
                  "catalogs": ["simbad", "gaia_dr3", "twomass_psc", "chandra"]},
    "named_m13": {"name": "M 13", "sesame": "m13", "radius_arcsec": 3.0, "catalogs": ["simbad", "ned", "gaia_dr3"]},
    # A saturated AllWISE star whose row has no w1mjdmean.
    "named_aldebaran": {"name": "Aldebaran", "sesame": "aldebaran", "radius_arcsec": 5.0,
                        "catalogs": ["simbad", "gaia_dr3", "allwise"]},
    # Sco X-1 by undated coordinates: SIMBAD 'LXB', NED 'V*'.
    "scox1_coordinates": {"ra": 244.97945528, "dec": -15.64028269, "radius_arcsec": 5.0,
                          "catalogs": ["simbad", "ned", "nvss", "vlass"]},
    # An arbitrary point in M 13's core (Gaia DR3: 2.9e6 per deg^2 within 30"), at 3" (the
    # density probe is recorded with it) and at 30".
    "m13_core_3arcsec": {"ra": 250.4237, "dec": 36.4614, "radius_arcsec": 3.0, "catalogs": ["gaia_dr3"]},
    "m13_core_30arcsec": {"ra": 250.4237, "dec": 36.4614, "radius_arcsec": 30.0, "catalogs": ["gaia_dr3"]},
    # Undated SIMBAD J2000 coordinates of fast movers at small radii (Gaia's J2016 rows lie
    # outside the cone): eps Eri (0.98"/yr) and GJ 15 A (2.9"/yr, 2MASS row 3.4" away).
    "epseri_undated": {"ra": 53.232685, "dec": -9.458261, "radius_arcsec": 3.0,
                       "catalogs": ["simbad", "gaia_dr3", "twomass_psc"]},
    "gj15a_undated": {"ra": 4.59535873, "dec": 44.02295948, "radius_arcsec": 5.0,
                      "catalogs": ["simbad", "gaia_dr3", "twomass_psc"]},
    # 3C 273 typed '12 29 07 +02 03 09' and parsed to floats by the web UI.
    "3c273_sexagesimal": {"ra": 15 * (12 + 29 / 60 + 7 / 3600), "dec": 2 + 3 / 60 + 9 / 3600, "radius_arcsec": 10.0,
                          "catalogs": ["simbad", "gaia_dr3", "panstarrs_dr2"]},
    # Round 3. Named extended objects whose centre lies on a point source: the Coma cluster
    # (SIMBAD 'ACO 1656', no Sesame error, 0.03" from the galaxy NGC 4876), Cas A (SIMBAD and
    # NED centres 15" apart, NED listing it twice), M 42 with the X-ray catalogues (the
    # Trapezium's stars) and NED, the Perseus and Virgo clusters (NGC 1275's X-ray nucleus,
    # M87's globular clusters).
    "named_coma": {"name": "Coma Cluster", "sesame": "coma", "radius_arcsec": 20.0,
                   "catalogs": ["simbad", "ned", "gaia_dr3", "twomass_psc", "allwise", "chandra", "xmm"]},
    "named_casa": {"name": "Cas A", "sesame": "casa", "radius_arcsec": 20.0,
                   "catalogs": ["simbad", "ned", "gaia_dr3", "chandra", "xmm", "nvss"]},
    "named_m42_xray": {"name": "M 42", "sesame": "m42_xray", "radius_arcsec": 20.0,
                       "catalogs": ["simbad", "ned", "gaia_dr3", "chandra", "xmm"]},
    "named_perseus": {"name": "Perseus Cluster", "sesame": "perseus", "radius_arcsec": 10.0,
                      "catalogs": ["simbad", "ned", "chandra", "xmm"]},
    "named_virgo": {"name": "Virgo Cluster", "sesame": "virgo", "radius_arcsec": 10.0, "catalogs": ["simbad", "ned"]},
    # Fast movers that 5XMM lists once per epoch along their track (Proxima Cen: 3.8"/yr, three
    # rows 2001-2017; 61 Cyg A: 5.2"/yr), and a compilation listing one object twice (Cyg X-3:
    # SIMBAD 'V* V1521 Cyg' and 'NVSS J203225+405728' 0.1" apart).
    "named_proxima": {"name": "Proxima Centauri", "sesame": "proxima", "radius_arcsec": 10.0,
                      "catalogs": ["simbad", "gaia_dr3", "xmm", "chandra"]},
    "named_61cyga": {"name": "61 Cyg A", "sesame": "61cyga", "radius_arcsec": 10.0,
                     "catalogs": ["simbad", "gaia_dr3", "xmm"]},
    "named_cygx3": {"name": "Cyg X-3", "sesame": "cygx3", "radius_arcsec": 5.0, "catalogs": ["simbad", "ned", "gaia_dr3"]},
    # 47 Tuc's core at 3" (one Gaia row: no Poisson excess, a crowded region) and at 30".
    "tuc47_core_3arcsec": {"ra": 6.0236, "dec": -72.0813, "radius_arcsec": 3.0, "catalogs": ["gaia_dr3", "twomass_psc"]},
    "tuc47_core_30arcsec": {"ra": 6.0236, "dec": -72.0813, "radius_arcsec": 30.0,
                            "catalogs": ["gaia_dr3", "twomass_psc"]},
    # Final-review round. M82 by name: SIMBAD lists 'M 82' at the resolved position and dozens of
    # radio / X-ray sources of its starburst within 5" (the resolved row must stay the target).
    "named_m82": {"name": "M82", "sesame": "m82", "radius_arcsec": 5.0,
                  "catalogs": ["simbad", "ned", "gaia_dr3", "chandra", "vlass"]},
    # Named galaxies whose NED record is the target itself (NED's 'NGC 4565' 1.2", 'Messier 101'
    # 0.8", 'NGC 7318a' 0.7" from the SIMBAD-resolved centre), and RR Lyr (NED 'RR Lyr').
    "named_ngc4565": {"name": "NGC 4565", "sesame": "ngc4565", "radius_arcsec": 10.0,
                      "catalogs": ["simbad", "ned", "twomass_psc", "allwise", "gaia_dr3"]},
    "named_m101": {"name": "M101", "sesame": "m101", "radius_arcsec": 10.0, "catalogs": ["simbad", "ned"]},
    "named_ngc7318a": {"name": "NGC 7318A", "sesame": "ngc7318a", "radius_arcsec": 5.0,
                       "catalogs": ["simbad", "ned", "gaia_dr3"]},
    "named_rrlyr": {"name": "RR Lyr", "sesame": "rrlyr", "radius_arcsec": 5.0, "catalogs": ["simbad", "ned", "gaia_dr3"]},
    # Coordinates near extragalactic star clusters: two globular clusters of M87 1.9" apart, and
    # five young star clusters of Stephan's Quintet (NED [FGD2015]) within 4".
    "m87_halo_gcs": {"ra": 187.718824, "dec": 12.378813, "radius_arcsec": 3.0, "catalogs": ["gaia_dr3", "simbad"]},
    "stephan_yscs": {"ra": 339.0063, "dec": 33.9656, "radius_arcsec": 10.0, "catalogs": ["simbad", "ned"]},
}


class FixedResolver:
    """A resolver answering with a recorded Sesame response."""

    def __init__(self, obj: Any) -> None:
        self.obj = obj

    async def resolve(self, _query: str) -> Any:
        return self.obj


def recorded_resolver(key: str) -> FixedResolver:
    from providers import SesameResolver

    spec = RECORDED_SEARCHES[key]
    xml = (SESAME_DIR / f"{spec['sesame']}.xml").read_text(encoding="utf-8")
    return FixedResolver(SesameResolver.parse_response(spec["name"], xml))


async def _run_search(service: Any, key: str, **kwargs: Any) -> dict[str, Any]:
    """Run a RECORDED_SEARCHES entry; returns {"record": dict, "start": dict | None}. A name
    goes through the stream (with its recorded Sesame answer) unless ``ra``/``dec`` are
    given, which run ``crossmatch`` (the POST /search path) on the same recorded cones."""
    spec = RECORDED_SEARCHES[key]
    params = {"radius_arcsec": spec["radius_arcsec"], "catalogs": spec["catalogs"], **kwargs}
    if "name" in spec and "ra" not in kwargs:
        resolver = kwargs.pop("resolver", None) or recorded_resolver(key)
        params.pop("resolver", None)
        events = [e async for e in service.crossmatch_stream(name=spec["name"], resolver=resolver, **params)]
        return {"start": events[0]["data"], "record": events[-1]["data"]["record"], "events": events}
    ra = params.pop("ra") if "ra" in params else spec["ra"]
    dec = params.pop("dec") if "dec" in params else spec["dec"]
    record = await service.crossmatch(ra, dec, **params)
    return {"start": None, "record": record.as_dict(), "unified": record}


async def replay_search(key: str, *, registry=None, allow_mismatch: bool = False, **kwargs: Any) -> dict[str, Any]:
    """Replay a RECORDED_SEARCHES entry offline (strict: a request that differs from its
    recording fails, unless ``allow_mismatch``). Keyword arguments go to the search."""
    exchanges = load_exchanges(f"matching/{key}")
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            result = await _run_search(make_service(client, registry), key, **kwargs)
    if not allow_mismatch:
        assert_replayed_exactly(result["record"])
    return result


def search_recorded(key: str) -> bool:
    spec = RECORDED_SEARCHES[key]
    return (MATCHING / key).exists() and ("sesame" not in spec or (SESAME_DIR / f"{spec['sesame']}.xml").exists())


def _catalog_of(request: Any, names: list[str]) -> str:
    """The catalogue of ``names`` a recorded request belongs to (its table / catalog token,
    else its endpoint)."""
    from models import CatalogRegistry

    registry = CatalogRegistry()
    text = fixture_io._request_text(request)
    hits = [n for n in names if any(tok in text for tok in fixture_io.match_tokens(registry.get(n)))]
    if len(hits) == 1:
        return hits[0]
    base = f"{request.url.scheme}://{request.url.host}{request.url.path}"
    by_endpoint = [n for n in names if str(registry.get(n).endpoint or "").split("?")[0] == base]
    if len(by_endpoint) == 1:
        return by_endpoint[0]
    raise RuntimeError(f"cannot tell which catalogue {base} belongs to: {hits or by_endpoint}")


async def record_search(key: str) -> None:
    import json
    import os
    from urllib.parse import quote

    import httpx

    from crossmatch import CrossmatchService
    from models import CatalogRegistry
    from providers import CacheManager, SesameResolver, provider_map

    os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
    spec = RECORDED_SEARCHES[key]
    if "name" in spec:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as plain:
            url = f"{SesameResolver(plain).endpoint}?{quote(spec['name'], safe='')}"
            response = await plain.get(url, headers={"Accept": "application/xml, text/xml"})
            response.raise_for_status()
            SESAME_DIR.mkdir(parents=True, exist_ok=True)
            (SESAME_DIR / f"{spec['sesame']}.xml").write_text(response.text, encoding="utf-8")
    log: list[tuple[Any, Any]] = []

    async def hook(response: Any) -> None:
        await response.aread()
        log.append((response.request, response))

    async with httpx.AsyncClient(timeout=180.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        service = CrossmatchService(CatalogRegistry(), provider_map(client, timeout=180.0, cache=CacheManager(None)),
                                    timeout=170.0)
        result = await _run_search(service, key)
    failures = result["record"].get("failures") or []
    if failures:
        raise RuntimeError(f"{key}: not recorded, failures {failures}")
    per_catalog: dict[str, list[dict[str, Any]]] = {}
    for request, response in log:
        name = _catalog_of(request, spec["catalogs"])
        per_catalog.setdefault(name, []).append({
            "method": request.method, "url": str(request.url),
            "request_body": request.content.decode("utf-8", "replace") if request.content else "",
            "status_code": response.status_code, "content_type": response.headers.get("content-type", ""),
            "match": [], "_content": response.content})
    folder = MATCHING / key
    folder.mkdir(parents=True, exist_ok=True)
    for name, records in per_catalog.items():
        for old in folder.glob(f"{name}.*.body"):
            old.unlink()
        exchanges = []
        for idx, rec in enumerate(records):
            (folder / f"{name}.{idx}.body").write_bytes(fixture_io.redact(rec.pop("_content")))
            exchanges.append(rec)
        meta = {"catalog": name, "target": key, "search": {k: v for k, v in spec.items() if k != "catalogs"},
                "exchanges": exchanges}
        (folder / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{key:<24} {name:<16} {len(exchanges)} exchange(s)")


async def crowded_record(**kwargs):
    return await replay_crossmatch(CROWDED_KEY, **kwargs)


def independent_members(group: dict) -> list[dict]:
    return [m for m in group["members"] if m["coincident_with"] is None]


async def test_crowded_field_replay_gives_distinct_objects_one_row_per_catalogue() -> None:
    record = await crowded_record()
    assert record.failures == []
    counts = {name: res["row_count"] for name, res in record.catalog_results.items()}
    assert counts["gaia_dr3"] >= 20, counts  # it is crowded
    groups = record.crossmatch_groups
    multi = [g for g in groups if len(independent_members(g)) >= 2]
    assert len(multi) >= 5, [g["catalogs"] for g in groups]
    for group in groups:
        members = independent_members(group)
        catalogs = [m["catalog"] for m in members]
        assert len(catalogs) == len(set(catalogs)), group  # never two rows of one catalogue
    # Every in-radius row is in exactly one group.
    ids = [(m["catalog"], m["source_id"]) for g in groups for m in g["members"]]
    assert len(ids) == len(set(ids)) == sum(counts.values())
    # Multi-catalogue objects are secure: members agree within their errors.
    secure = [g for g in multi if g["match_probability"] is not None and g["match_probability"] > 0.99]
    assert len(secure) >= 0.6 * len(multi)
    for group in secure:
        gaia = [m for m in group["members"] if m["catalog"] == "gaia_dr3"]
        others = [m for m in group["members"] if m["catalog"] != "gaia_dr3"]
        for m in others:
            if gaia:
                from models import haversine_arcsec

                # 2MASS / AllWISE / PS1 counterparts of a Gaia star lie within ~1" of it.
                assert haversine_arcsec(m["ra"], m["dec"], gaia[0]["ra"], gaia[0]["dec"]) < 1.5


async def test_crowded_field_densities_are_local() -> None:
    record = await crowded_record()
    densities = record.provenance["association"]["densities"]
    # The Galactic plane is far denser in Gaia than the all-sky mean (43,900 per deg^2):
    # the density used by the prior comes from the density map and the cone, not the
    # catalogue average. Baade's window holds ~1.09e6 Gaia sources per deg^2.
    gaia = densities["gaia_dr3"]
    assert gaia["prior_source"] == "density_map" and gaia["method"] == "gamma_poisson"
    assert 0.5 * 1.09e6 < gaia["density_per_deg2"] < 2.0 * 1.09e6
    assert gaia["density_per_deg2"] > 10 * gaia["sky_mean_per_deg2"]
    # 2MASS and Pan-STARRS follow the stars (scaled from the Gaia map), AllWISE is
    # confusion-limited at ~2.5e4 per deg^2 everywhere (0.1-deg COUNT(*) here: 778 sources,
    # 2.48e4 per deg^2): its prior mean stays there, and the estimate from the 6 AllWISE rows
    # of this 20" cone (all field rows: nothing sits at the searched point, which is exact)
    # is within its Poisson scatter (6 rows: 90% interval 2.6-11.8 rows, 2.7e4-1.2e5 per deg^2).
    assert densities["twomass_psc"]["density_per_deg2"] > 5 * densities["twomass_psc"]["sky_mean_per_deg2"]
    assert densities["panstarrs_dr2"]["density_per_deg2"] > 5 * densities["panstarrs_dr2"]["sky_mean_per_deg2"]
    allwise = densities["allwise"]
    assert 1.0e4 < allwise["prior_mean_per_deg2"] < 5.0e4
    assert allwise["rows"] == allwise["field_rows"] == 6
    assert 2.7e4 < allwise["density_per_deg2"] < 1.2e5


async def _record_searches(keys: list[str]) -> None:
    for key in keys:
        await record_search(key)


if __name__ == "__main__":
    if sys.argv[1:2] == ["record"]:
        asyncio.run(record(sys.argv[2:] or None))
    elif sys.argv[1:2] == ["record-search"]:
        asyncio.run(_record_searches(sys.argv[2:] or [k for k in RECORDED_SEARCHES if not search_recorded(k)]))
    else:
        print(__doc__)

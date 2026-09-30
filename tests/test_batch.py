"""Offline tests for batch.py: target parsing, query building, replay of recorded TAP-upload / CDS XMatch answers,
agreement with the recorded per-target cone searches, retries, chunk splitting, cone fallback, router and CLI.

Batch recordings live in tests/fixtures/batch/<case>/ (re-record with
``.venv/Scripts/python.exe tests/test_batch_fixtures.py record``); the per-target cone recordings used for the
agreement checks are the canary fixtures in tests/fixtures/<target>/.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import re
from typing import Any

import httpx
import pytest
import respx
from fixture_io import TARGETS, load_exchanges
from helpers import offline_client
from test_batch_fixtures import (
    CANARY_RADIUS,
    CANARY_TARGETS,
    GENERIC_CATALOGS,
    GENERIC_TARGETS,
    UPLOAD_CATALOGS,
    VEGA,
    VEGA_ADJACENT,
    VEGA_CATALOGS,
    batch_replay_side_effect,
    load_batch_exchanges,
    multipart_fields,
    request_body,
)

import batch
from batch import (
    BatchCrossmatcher,
    BatchError,
    BatchResult,
    build_upload_adql,
    parse_targets,
    read_targets_csv,
    read_targets_json,
    upload_csv,
    upload_votable,
)
from models import CatalogRegistry

CONE_ONLY = ["ned", "exoplanet_archive", "panstarrs_dr2", "sdss"]


def canary_cone_exchanges(catalogs: list[str]) -> list[Any]:
    out: list[Any] = []
    for key in TARGETS:
        out.extend(load_exchanges(key, catalogs))
    return out


async def replay_batch(case: str, targets, catalogs, radius, *, strategies=None, cone_catalogs=None,
                       engine: BatchCrossmatcher | None = None, **kwargs) -> BatchResult:
    exchanges = load_batch_exchanges(case) if case else []
    cone = canary_cone_exchanges(cone_catalogs) if cone_catalogs else None
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=batch_replay_side_effect(exchanges, cone))
        async with offline_client() as client:
            eng = engine or BatchCrossmatcher(client=client, **kwargs)
            eng.client = client
            # The recordings hold VLASS/LoTSS answers of TAPVizieR table uploads (still an explicit strategy);
            # their default CDS XMatch path is replayed by test_batch_final from its own recordings.
            names = list(catalogs) if catalogs is not None else eng.default_catalogs()
            recorded = {name: "upload" for name in RECORDED_TAPVIZIER_UPLOADS if name in names}
            return await eng.run(targets, catalogs, radius_arcsec=radius, strategies={**recorded, **(strategies or {})})


RECORDED_TAPVIZIER_UPLOADS = ("vlass", "lotss")


@pytest.fixture(scope="module")
def canary() -> BatchResult:
    return asyncio.run(replay_batch("canary", CANARY_TARGETS, UPLOAD_CATALOGS, CANARY_RADIUS))


def ids(result: BatchResult, target: str, catalog: str) -> list[str]:
    return [m["source_id"] for m in result.target_matches(target).get(catalog, [])]


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def test_parse_targets_aliases_and_normalisation():
    targets = parse_targets([
        {"Name": " 3C  273 ", "RAJ2000": "187.2779154", "DEJ2000": 2.0523883},
        {"ra": 10, "dec": -5, "epoch": 2000, "pmra": 5.0, "pmdec": -3.0, "plx": 12.0, "radius": 4},
        {"ra": 359.9999999, "dec": 90},
    ])
    assert [t.id for t in targets] == ["3C 273", "2", "3"]
    assert targets[0].target.ra == pytest.approx(187.2779154, abs=1e-9)
    assert targets[1].target.epoch == 2000.0 and targets[1].target.proper_motion == (5.0, -3.0)
    assert targets[1].target.parallax_mas == 12.0 and targets[1].radius_arcsec == 4.0
    assert targets[2].target.dec == 90.0


@pytest.mark.parametrize("items, message", [
    ([{"id": "a", "ra": 1, "dec": 1}, {"id": "a", "ra": 2, "dec": 2}], "duplicate id"),
    ([{"id": "a", "ra": 1}], "ra and dec"),
    ([{"ra": 1, "dec": 95}], "DEC must be within"),
    ([{"ra": 1, "dec": 1, "pmra": 3}], "given together"),
    ([{"ra": 1, "dec": 1, "radius_arcsec": 500}], "radius_arcsec must be in"),
    ([{"ra": "abc", "dec": 1}], "numeric"),
    ([{"ra": 1, "dec": 1, "epoch": 1500}], "epoch must be"),
    ([], "no targets"),
    (["x"], "expected an object"),
])
def test_parse_targets_rejects_invalid(items, message):
    with pytest.raises(BatchError, match=message):
        parse_targets(items)


def test_parse_targets_limit():
    with pytest.raises(BatchError, match="at most 2"):
        parse_targets([{"ra": i, "dec": 0} for i in range(3)], max_targets=2)


def test_read_targets_csv_comments_bom_blanks():
    text = "﻿# my targets\nid,ra,dec,epoch,pmra,pmdec\nA,187.2779154,2.0523883,,,\nB,279.23473479,38.78368896,2000,200.94,286.23\n"
    targets = read_targets_csv(text.encode("utf-8"))
    assert [t.id for t in targets] == ["A", "B"]
    assert targets[0].target.epoch is None and targets[0].target.proper_motion is None
    assert targets[1].target.proper_motion == (200.94, 286.23)
    with pytest.raises(BatchError, match="ra and dec columns"):
        read_targets_csv("id,x,y\n1,2,3\n")


def test_read_targets_json_shapes(tmp_path):
    assert len(read_targets_json('[{"ra": 1, "dec": 2}]')) == 1
    assert read_targets_json({"targets": [{"id": 7, "ra": 1, "dec": 2}]})[0].id == "7"
    with pytest.raises(BatchError):
        read_targets_json('{"foo": 1}')
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"targets": CANARY_TARGETS}), encoding="utf-8")
    assert [t.id for t in batch.load_targets_file(path)] == list(TARGETS)
    csv_path = tmp_path / "t.csv"
    csv_path.write_text("ra,dec\n1,2\n", encoding="utf-8")
    assert batch.load_targets_file(csv_path)[0].id == "1"


# ---------------------------------------------------------------------------
# Upload payloads & ADQL
# ---------------------------------------------------------------------------


def test_upload_votable_roundtrip_is_exact():
    from astropy.io.votable import parse

    rows = [(0, 187.2779154, 2.0523883, 10.0 / 3600.0), (41, 0.1, -89.99999, 1e-4)]
    payload = upload_votable(rows)
    assert payload == upload_votable(rows)  # deterministic (replayable)
    table = parse(io.BytesIO(payload)).get_first_table()
    assert [f.name for f in table.fields] == ["t_idx", "t_ra", "t_dec", "t_rad"]
    assert table.fields[0].datatype == "int"  # 32-bit: Xamin 64-bit nulls overflow C long on Windows
    for got, want in zip(table.array.tolist(), rows):
        assert tuple(got) == want
    assert payload.isascii()
    assert upload_csv([(3, 1.5, -2.25)]) == b"t_idx,t_ra,t_dec\n3,1.5,-2.25\n"


def test_build_upload_adql_per_service():
    reg = CatalogRegistry()
    simbad = build_upload_adql(reg.get("simbad"), batch.UPLOAD_SERVICES[reg.get("simbad").endpoint])
    assert simbad.startswith("SELECT tu.t_idx, b.main_id, b.ra, b.dec")
    assert ("FROM basic AS b LEFT OUTER JOIN allfluxes AS f ON f.oidref = b.oid, TAP_UPLOAD.targets AS tu "
            "WHERE 1 = CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', tu.t_ra, tu.t_dec, tu.t_rad))") in simbad
    irsa = build_upload_adql(reg.get("twomass_psc"), batch.UPLOAD_SERVICES[reg.get("twomass_psc").endpoint])
    assert " JOIN " not in irsa and "FROM fp_psc, TAP_UPLOAD.targets AS tu WHERE" in irsa  # IRSA: comma form only
    rosat = build_upload_adql(reg.get("rosat"), batch.UPLOAD_SERVICES[reg.get("rosat").endpoint])
    assert rosat.startswith("SELECT tu.t_idx AS t_idx, name AS name, ra AS ra")  # Xamin prefixes unaliased columns
    assert '"time" AS "time"' in rosat
    vlass = build_upload_adql(reg.get("vlass"), batch.UPLOAD_SERVICES[reg.get("vlass").endpoint])
    assert 'FROM "J/ApJ/914/42/table5", TAP_UPLOAD.targets AS tu' in vlass
    assert batch.UPLOAD_SERVICES[reg.get("vlass").endpoint].upload_endpoint.startswith("http://tapvizier")


def test_strategy_table_and_defaults():
    engine = BatchCrossmatcher()
    table = engine.strategy_table()
    assert table["gaia_dr3"]["strategy"] == "xmatch" and table["gaia_dr3"]["vizier_table"] == "vizier:I/355/gaiadr3"
    for name in ("simbad", "twomass_psc", "allwise", "first", "nvss", "rosat", "chandra", "xmm"):
        assert table[name]["strategy"] == "upload", name
    # VizieR-hosted registry catalogs: CDS XMatch (TAPVizieR uploads stalled for minutes, live).
    for name, table_id in (("vlass", "vizier:J/ApJ/914/42/table5"), ("lotss", "vizier:J/A+A/707/A198/lotssdr3")):
        assert table[name]["strategy"] == "xmatch" and table[name]["vizier_table"] == table_id, name
    for name in CONE_ONLY:
        assert table[name]["strategy"] == "cone", name
    defaults = engine.default_catalogs()
    assert set(defaults) == set(UPLOAD_CATALOGS)  # enabled and upload/xmatch capable


@pytest.mark.parametrize("catalogs, strategies, message", [
    (["nope"], None, "unknown catalog"),
    (["ned"], {"ned": "upload"}, "does not support TAP uploads"),
    (["simbad"], {"simbad": "xmatch"}, "no CDS XMatch"),
    (["vizier:II/246/out"], {"vizier:II/246/out": "cone"}, "xmatch strategy only"),
    (["simbad"], {"first": "cone"}, "not requested"),
    (["simbad"], {"simbad": "magic"}, "unknown strategy"),
])
def test_resolve_rejects(catalogs, strategies, message):
    with pytest.raises(BatchError, match=message):
        asyncio.run(BatchCrossmatcher().run(CANARY_TARGETS, catalogs, radius_arcsec=5, strategies=strategies))


@pytest.mark.parametrize("radius", [0.0, -1.0, 181.0, float("nan")])
def test_radius_bounds(radius):
    with pytest.raises(BatchError, match="radius_arcsec"):
        asyncio.run(BatchCrossmatcher().run(CANARY_TARGETS, ["simbad"], radius_arcsec=radius))


# ---------------------------------------------------------------------------
# Replay of recorded uploads / XMatch
# ---------------------------------------------------------------------------


def test_canary_one_request_per_catalog(canary):
    assert canary.request_count == len(UPLOAD_CATALOGS)
    for name, run in canary.runs.items():
        assert run.requests == 1 and run.chunks == 1 and not run.errors, name
        assert run.strategy == ("xmatch" if name == "gaia_dr3" else "upload")
        assert run.citation
    assert canary.summary()["strategies"]["gaia_dr3"] == "xmatch"


@pytest.mark.parametrize("target, catalog, expected", [
    ("3c273", "gaia_dr3", "3700386905605055360"),
    ("3c273", "simbad", "3C 273"),
    ("3c273", "twomass_psc", "12290669+0203085"),
    ("3c273", "allwise", "J122906.69+020308.6"),
    ("3c273", "first", "FIRST J122906.7+020308"),
    ("3c273", "nvss", "NVSS J122906+020305"),
    ("m87", "simbad", "M 87"),
    ("m87", "gaia_dr3", "3907709439453756032"),
    ("m87", "allwise", "J123049.43+122328.0"),
    ("hd209458", "gaia_dr3", "1779546757669063552"),
    ("hd209458", "simbad", "HD 209458"),
    ("hd209458", "twomass_psc", "22031077+1853036"),
])
def test_canary_known_identities(canary, target, catalog, expected):
    found = ids(canary, target, catalog)
    assert found and found[0] == expected


def test_canary_canonical_fields(canary):
    m = canary.target_matches("3c273")
    gaia = m["gaia_dr3"][0]
    assert gaia["epoch"] == 2016.0 and gaia["strategy"] == "xmatch"
    assert gaia["data"]["source_id"] == 3700386905605055360 and "ra_error" in gaia["data"]
    # Gaia DR3 ra_error/dec_error of 3C 273: 0.0181 / 0.0129 mas -> RMS sigma in arcsec
    assert gaia["positional_error_arcsec"] == pytest.approx(math.sqrt((0.0181**2 + 0.0129**2) / 2) / 1000, rel=1e-3)
    assert gaia["separation_arcsec"] < 0.01
    tm = m["twomass_psc"][0]
    assert 1997.4 <= tm["epoch"] <= 2001.2 and tm["positional_error_arcsec"] > 0
    nvss = m["nvss"][0]
    assert nvss["epoch"] is None and nvss["epoch_range"] == [1993.7, 1997.3]
    simbad = m["simbad"][0]
    assert simbad["epoch"] == 2000.0 and simbad["physical"]["object_type"] == "BLL"
    for name, matches in m.items():
        for rank, match in enumerate(matches, start=1):
            assert match["rank"] == rank and match["catalog"] == name
            assert match["separation_arcsec"] <= CANARY_RADIUS + 1e-6
            assert 0.0 <= match["confidence"] <= 1.0
        seps = [x["separation_arcsec"] for x in matches]
        assert seps == sorted(seps)


def test_canary_upload_columns_are_removed(canary):
    for matches in canary.target_matches("m87").values():
        for match in matches:
            assert not {"t_idx", "t_ra", "t_dec", "t_rad", "angDist"} & set(match["data"])


@pytest.mark.parametrize("catalog", UPLOAD_CATALOGS)
def test_batch_agrees_with_recorded_cone_searches(canary, catalog):
    """Same rows (ids) and separations (< 0.01") as the recorded per-target cone searches (radius 10")."""
    cone = asyncio.run(replay_batch("", CANARY_TARGETS, [catalog], CANARY_RADIUS, strategies={catalog: "cone"},
                                    cone_catalogs=[catalog]))
    assert cone.runs[catalog].strategy == "cone" and cone.runs[catalog].requests == len(CANARY_TARGETS)
    for key in TARGETS:
        up = {m["source_id"]: m["separation_arcsec"] for m in canary.target_matches(key).get(catalog, [])}
        cs = {m["source_id"]: m["separation_arcsec"] for m in cone.target_matches(key).get(catalog, [])}
        assert set(up) == set(cs), (key, catalog)
        for sid in up:
            assert abs(up[sid] - cs[sid]) < 0.01, (key, catalog, sid)


def test_vega_epoch_and_proper_motion():
    result = asyncio.run(replay_batch("vega", [VEGA, VEGA_ADJACENT], VEGA_CATALOGS, 5.0))
    vega = result.target_matches("Vega")
    assert ids(result, "Vega", "simbad") == ["* alf Lyr"]
    assert ids(result, "Vega", "twomass_psc") == ["18365633+3847012"]
    tm = vega["twomass_psc"][0]
    # As in a single-object crossmatch, Vega's parallax is adopted from its SIMBAD row (130.23 mas, the Hipparcos
    # value of van Leeuwen 2007 quoted by SIMBAD) and 2MASS (no proper motions, single-epoch positions) is moved with Vega's motion
    # AND corrected for the annual parallax.
    assert tm["epoch_propagation"] == "target_pm_parallax"
    parallax = result.target_association("Vega")["target_parallax"]
    assert parallax["parallax_mas"] == pytest.approx(130.23, abs=0.5)
    assert result.target_association("Vega")["target_proper_motion"]["source"] == "input"
    # 2MASS observed Vega at ~1999.3: moving it with Vega's 0.349"/yr to J2000 changes its separation by at most
    # pm x |dt| + parallax (triangle inequality); the saturated 2MASS position agrees to < 2 sigma (err_maj 0.29").
    assert 1997.4 <= tm["epoch"] <= 2001.2
    shift = math.hypot(VEGA["pm_ra_masyr"], VEGA["pm_dec_masyr"]) / 1000.0 * abs(2000.0 - tm["epoch"]) + 0.13023
    assert abs(tm["separation_arcsec"] - tm["query_separation_arcsec"]) <= shift + 1e-6
    assert tm["separation_arcsec"] < 2.0 * tm["positional_error_arcsec"]
    assert vega.get("gaia_dr3", []) == []  # Vega (G ~ 0) has no Gaia DR3 entry
    assert not any(result.target_matches("Vega+60N").get(c) for c in VEGA_CATALOGS)
    # Vega's proper-motion cone (8.6") and the plain 5" cone fall in one XMatch radius bucket: one request.
    assert result.runs["gaia_dr3"].requests == 1


def test_generic_vizier_tables_via_xmatch(canary):
    """The spec's VizieR tables carry every canonical field (ids, 1-sigma positional errors, epochs)."""
    result = asyncio.run(replay_batch("generic", GENERIC_TARGETS, GENERIC_CATALOGS, 5.0))
    for name in GENERIC_CATALOGS:
        run = result.runs[name]
        assert run.strategy == "xmatch" and run.requests == 1 and not run.errors, (name, run.errors)
    m = result.target_matches("3c273")
    irsa_2mass = canary.target_matches("3c273")["twomass_psc"][0]
    tm = m["vizier:II/246/out"][0]
    assert tm["source_id"] == irsa_2mass["source_id"] == "12290669+0203085"
    assert abs(tm["separation_arcsec"] - irsa_2mass["separation_arcsec"]) < 0.01
    # XMatch errHalfMaj/errHalfMin (0.17/0.08") = 2MASS err_maj/err_min (1-sigma): RMS 0.1329"
    assert tm["positional_error_arcsec"] == pytest.approx(math.sqrt((0.17**2 + 0.08**2) / 2), rel=1e-4)
    assert tm["positional_error_arcsec"] == pytest.approx(irsa_2mass["positional_error_arcsec"], rel=1e-6)
    assert tm["epoch"] == pytest.approx(irsa_2mass["epoch"], abs=1e-6)  # MeasureJD = IRSA jdate
    aw = m["vizier:II/328/allwise"][0]
    irsa_aw = canary.target_matches("3c273")["allwise"][0]
    assert aw["source_id"] == irsa_aw["source_id"] == "J122906.69+020308.6"  # designation, not the numeric ID
    assert aw["positional_error_arcsec"] == pytest.approx(irsa_aw["positional_error_arcsec"], rel=2e-3)
    assert aw["epoch"] is None and aw["epoch_range"] == [2010.0, 2011.2]  # no per-source epoch in VizieR II/328
    assert "pmRA" not in aw["data"]
    ps1 = m["vizier:II/349/ps1"][0]
    assert ps1["source_id"] == "110461872779253351"
    # errHalfMaj/errHalfMin 0.039/0.024" (= e_DEJ2000/e_RAJ2000) with the 15 mas systematic in quadrature
    assert ps1["positional_error_arcsec"] == pytest.approx(math.hypot(math.sqrt((0.039**2 + 0.024**2) / 2), 0.015), rel=1e-3)
    assert ps1["epoch"] == pytest.approx(2011.5451, abs=1e-3)  # Epoch MJD 55761.365
    sdss = m["vizier:V/154/sdss16"]
    assert [s["source_id"] for s in sdss] == ["1237651735760142397"]  # the PRIMARY (mode 1) photometric object
    assert sdss[0]["epoch"] == pytest.approx(2000.3389, abs=1e-3) and sdss[0]["positional_error_arcsec"] > 0.04
    gaia = m["vizier:I/355/gaiadr3"][0]
    irsa_gaia = canary.target_matches("3c273")["gaia_dr3"][0]
    assert gaia["source_id"] == irsa_gaia["source_id"] == "3700386905605055360" and gaia["epoch"] == 2016.0
    assert gaia["positional_error_arcsec"] == pytest.approx(irsa_gaia["positional_error_arcsec"], rel=1e-9)
    assert not any("epoch unknown" in w for w in result.runs["vizier:I/355/gaiadr3"].warnings)
    for key in ("3c273", "hd209458"):
        for name, matches in result.target_matches(key).items():
            for match in matches:
                assert match["positional_error_arcsec"] is not None, (key, name)
                assert match["epoch"] is not None or match["epoch_range"], (key, name)


@pytest.mark.parametrize("catalog", CONE_ONLY)
def test_cone_only_catalogs_use_bounded_cone_searches(catalog):
    result = asyncio.run(replay_batch("", CANARY_TARGETS, [catalog], CANARY_RADIUS, cone_catalogs=[catalog]))
    run = result.runs[catalog]
    assert run.strategy == "cone" and run.requests == len(CANARY_TARGETS) and not run.errors
    if catalog == "ned":
        assert ids(result, "3c273", "ned")[0] == "3C 273"
    if catalog == "exoplanet_archive":
        assert set(ids(result, "hd209458", "exoplanet_archive")) == {"HD 209458 b"}
    for key in TARGETS:
        for m in result.target_matches(key).get(catalog, []):
            assert m["strategy"] == "cone"


# ---------------------------------------------------------------------------
# Resilience: retries, chunk splitting, fallback, wide cones
# ---------------------------------------------------------------------------


def test_retry_after_connection_reset():
    exchanges = load_batch_exchanges("canary", ["gaia_dr3"])
    handler = batch_replay_side_effect(exchanges)
    calls = {"n": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ReadError("[WinError 10054] An existing connection was forcibly closed by the remote host")
        return handler(request)

    async def go() -> BatchResult:
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=flaky)
            async with offline_client() as client:
                engine = BatchCrossmatcher(client=client)
                engine.retry_backoff_seconds = 0.0
                return await engine.run(CANARY_TARGETS, ["gaia_dr3"], radius_arcsec=CANARY_RADIUS)

    result = asyncio.run(go())
    run = result.runs["gaia_dr3"]
    assert run.requests == 2 and run.retries == 1 and not run.errors
    assert ids(result, "3c273", "gaia_dr3")[0] == "3700386905605055360"


def _tap_json_rows(targets_votable: bytes, per_target: int) -> list[list[Any]]:
    rows = re.findall(rb"<TR><TD>(\d+)</TD><TD>([^<]+)</TD><TD>([^<]+)</TD>", targets_votable)
    return [[int(i), f"obj-{int(i)}-{k}", float(ra), float(dec)] for i, ra, dec in rows for k in range(per_target)]


def test_overflowing_chunks_are_split(monkeypatch):
    """An answer with MAXREC rows means 'truncated': the chunk is halved until every answer fits."""
    service = batch.UPLOAD_SERVICES[batch.SIMBAD_TAP]
    monkeypatch.setitem(batch.UPLOAD_SERVICES, batch.SIMBAD_TAP, batch.UploadService(
        service.key, service.upload_endpoint, chunk_size=4, max_rows=2, upload_row_limit=service.upload_row_limit))
    meta = [{"name": "t_idx", "datatype": "INTEGER"},
            {"name": "main_id", "datatype": "CHAR", "ucd": "meta.id;meta.main"},
            {"name": "ra", "datatype": "DOUBLE", "unit": "deg", "ucd": "pos.eq.ra;meta.main"},
            {"name": "dec", "datatype": "DOUBLE", "unit": "deg", "ucd": "pos.eq.dec;meta.main"}]
    sizes: list[int] = []

    def fake_simbad(request: httpx.Request) -> httpx.Response:
        fields = multipart_fields(request.headers["content-type"], request_body(request))
        assert fields["MAXREC"] == "2" and fields["UPLOAD"] == "targets,param:targets"
        rows = _tap_json_rows(fields["targets"].encode(), per_target=1)
        sizes.append(len(rows))
        return httpx.Response(200, json={"metadata": meta, "data": rows[:2]})

    async def go() -> BatchResult:
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=fake_simbad)
            async with offline_client() as client:
                return await BatchCrossmatcher(client=client).run(CANARY_TARGETS, ["simbad"], radius_arcsec=5)

    result = asyncio.run(go())
    run = result.runs["simbad"]
    assert sizes == [4, 2, 2, 1, 1, 1, 1] and run.requests == 7 and run.split_chunks == 3
    assert run.matched_targets == 4 and not run.errors
    for key in TARGETS:
        match = result.target_matches(key)["simbad"][0]
        assert match["source_id"].startswith("obj-") and match["separation_arcsec"] == pytest.approx(0.0, abs=1e-6)


def test_failed_upload_falls_back_to_cone_search():
    """SIMBAD upload answers 503 every time: after the retries the chunk is cone-searched (recorded cones)."""
    cone_handler = batch_replay_side_effect([], canary_cone_exchanges(["simbad"]))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("content-type", "").startswith("multipart/form-data"):
            return httpx.Response(503, text="Service Unavailable")
        return cone_handler(request)

    async def go() -> BatchResult:
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=handler)
            async with offline_client() as client:
                engine = BatchCrossmatcher(client=client)
                engine.retry_backoff_seconds = 0.0
                return await engine.run(CANARY_TARGETS, ["simbad"], radius_arcsec=CANARY_RADIUS)

    result = asyncio.run(go())
    run = result.runs["simbad"]
    assert run.fallback_targets == 4 and run.retries == 4
    assert run.requests == 5 + 4  # five upload attempts, then one cone per target
    assert "HTTP 503" in run.errors[0]
    assert ids(result, "3c273", "simbad")[0] == "3C 273"
    assert all(m["strategy"] == "cone" for m in result.target_matches("3c273")["simbad"])


def test_upload_failure_without_fallback_is_reported():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="<VOTABLE><RESOURCE><INFO name=\"QUERY_STATUS\" value=\"ERROR\">bad</INFO></RESOURCE></VOTABLE>",
                              headers={"content-type": "text/xml"})

    async def go() -> BatchResult:
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=handler)
            async with offline_client() as client:
                return await BatchCrossmatcher(client=client, fallback_to_cone=False).run(
                    CANARY_TARGETS, ["first"], radius_arcsec=5)

    result = asyncio.run(go())
    run = result.runs["first"]
    assert run.requests == 1 and run.failed_targets == 4 and "CatalogQueryError" in run.errors[0]
    assert set(result.failures[0]) == {"first"}


def test_xmatch_cone_wider_than_180_arcsec_is_capped_without_fallback():
    """Epoch 1950 without a proper motion: the Gaia cone is widened to ~305" (> XMatch's 180" limit). Without a
    cone fallback it is capped at 180" -- the trade-off models.plan_cone makes at EPOCH_PAD_MAX_ARCSEC -- and the
    warning says so (it used to claim a cone search that never happened, and the target failed)."""
    sent: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(multipart_fields(request.headers["content-type"], request_body(request)))
        return httpx.Response(200, content=b'<?xml version="1.0"?><VOTABLE version="1.3"><RESOURCE type="results">'
                              b'<INFO name="QUERY_STATUS" value="OK"/><TABLE><FIELD name="angDist" datatype="double"/>'
                              b'<FIELD name="t_idx" datatype="int"/><DATA><TABLEDATA></TABLEDATA></DATA></TABLE>'
                              b"</RESOURCE></VOTABLE>", headers={"content-type": "text/xml"})

    async def go() -> BatchResult:
        with respx.mock(assert_all_mocked=True) as router:
            router.route().mock(side_effect=handler)
            async with offline_client() as client:
                return await BatchCrossmatcher(client=client, fallback_to_cone=False).run(
                    [{"id": "old", "ra": 10.0, "dec": 10.0, "epoch": 1950.0}], ["gaia_dr3"], radius_arcsec=5)

    result = asyncio.run(go())
    run = result.runs["gaia_dr3"]
    assert run.requests == 1 and run.failed_targets == 0 and run.refused_targets == 0
    assert [f["distMaxArcsec"] for f in sent] == ["180.000"]
    assert sent[0]["cat1"].splitlines()[1] == "0,10.0,10.0"
    assert not any("cone search" in w for w in run.warnings)
    assert any("capped at the XMatch limit of 180 arcsec (the cone fallback is disabled)" in w for w in run.warnings)
    assert any("epoch gap 66.0 yr needs a 698 arcsec cone" in w and "capped at the CDS XMatch limit" in w
               for w in run.warnings), run.warnings


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_rows_parquet_csv_json(canary, tmp_path):
    import pyarrow.parquet as pq

    rows = canary.rows()
    assert len(rows) == sum(run.total_matches for run in canary.runs.values())
    first = next(r for r in rows if r["target_id"] == "3c273" and r["catalog"] == "simbad")
    assert first["source_id"] == "3C 273" and first["strategy"] == "upload" and json.loads(first["data_json"])["main_id"] == "3C 273"
    path = canary.write(tmp_path / "out.parquet")
    table = pq.read_table(path)
    assert table.num_rows == len(rows)
    summary = json.loads(table.schema.metadata[b"astrosearch.batch.summary"])
    assert summary["request_count"] == len(UPLOAD_CATALOGS) and summary["strategies"]["simbad"] == "upload"
    assert table.column("target_id").to_pylist().count("m87") == sum(1 for r in rows if r["target_id"] == "m87")
    csv_text = canary.write(tmp_path / "out.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("target_id,target_ra,target_dec")
    data = json.loads(canary.write(tmp_path / "out.json").read_text(encoding="utf-8"))
    assert [t["id"] for t in data["targets"]] == list(TARGETS)
    assert data["catalogs"]["gaia_dr3"]["strategy"] == "xmatch"
    no_data = canary.as_dict(include_data=False)
    assert "data" not in no_data["targets"][0]["matches"]["simbad"][0]
    with pytest.raises(BatchError):
        canary.write(tmp_path / "out.xlsx")


def test_nearest_only(canary):
    async def go():
        exchanges = load_batch_exchanges("canary", ["simbad"])
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=batch_replay_side_effect(exchanges))
            async with offline_client() as client:
                return await BatchCrossmatcher(client=client).run(CANARY_TARGETS, ["simbad"], radius_arcsec=CANARY_RADIUS,
                                                                  nearest_only=True)

    result = asyncio.run(go())
    assert ids(result, "m87", "simbad") == ["M 87"]
    assert len(ids(canary, "m87", "simbad")) > 1


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def make_app():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(batch.router)
    return app


def _client_call(method: str, path: str, *, catalogs: list[str], **kwargs):
    from fastapi.testclient import TestClient

    exchanges = load_batch_exchanges("canary", catalogs)
    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=batch_replay_side_effect(exchanges))
        with TestClient(make_app()) as client:
            return client.request(method, path, **kwargs)


def test_router_json_body():
    body = {"targets": CANARY_TARGETS, "catalogs": ["simbad", "first"], "radius_arcsec": CANARY_RADIUS,
            "include_data": False}
    response = _client_call("POST", "/api/v1/batch/crossmatch", catalogs=["simbad", "first"], json=body)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"]["request_count"] == 2 and response.headers["x-batch-request-count"] == "2"
    by_id = {t["id"]: t for t in data["targets"]}
    assert by_id["3c273"]["matches"]["simbad"][0]["source_id"] == "3C 273"
    assert by_id["3c273"]["matches"]["first"][0]["source_id"] == "FIRST J122906.7+020308"
    assert "data" not in by_id["3c273"]["matches"]["simbad"][0]
    assert data["catalogs"]["simbad"]["strategy"] == "upload"


def test_router_csv_body_rows_format():
    csv_text = "id,ra,dec\n" + "".join(f"{t['id']},{t['ra']!r},{t['dec']!r}\n" for t in CANARY_TARGETS)
    response = _client_call("POST", "/api/v1/batch/crossmatch?catalogs=nvss&radius_arcsec=10&format=rows",
                            catalogs=["nvss"], content=csv_text, headers={"content-type": "text/csv"})
    assert response.status_code == 200, response.text
    rows = response.json()["rows"]
    assert {r["source_id"] for r in rows} == {"NVSS J122906+020305", "NVSS J123049+122321"}
    assert all(r["strategy"] == "upload" and r["catalog"] == "nvss" for r in rows)


def test_router_multipart_parquet():
    import pyarrow.parquet as pq

    csv_text = "name,RAJ2000,DEJ2000\n" + "".join(f"{t['id']},{t['ra']!r},{t['dec']!r}\n" for t in CANARY_TARGETS)
    response = _client_call(
        "POST", "/api/v1/batch/crossmatch", catalogs=["allwise"],
        files={"file": ("targets.csv", csv_text.encode(), "text/csv")},
        data={"catalogs": "allwise", "radius_arcsec": "10", "format": "parquet"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/vnd.apache.parquet"
    table = pq.read_table(io.BytesIO(response.content))
    assert set(table.column("source_id").to_pylist()) >= {"J122906.69+020308.6", "J123049.43+122328.0",
                                                          "J220310.79+185303.3"}


@pytest.mark.parametrize("kwargs, status, fragment", [
    ({"json": {"targets": [{"ra": 1, "dec": 95}], "catalogs": ["simbad"]}}, 422, ""),
    ({"json": {"targets": [], "catalogs": ["simbad"]}}, 422, ""),
    ({"json": {"targets": CANARY_TARGETS, "catalogs": ["nope"]}}, 422, "unknown catalog"),
    ({"json": {"targets": CANARY_TARGETS, "catalogs": ["simbad"], "radius_arcsec": 500}}, 422, ""),
    ({"json": {"targets": CANARY_TARGETS, "catalogs": ["simbad"], "format": "xlsx"}}, 422, ""),
    ({"content": "id,x\n1,2\n", "headers": {"content-type": "text/csv"}}, 422, "ra and dec"),
    ({"content": b"{not json", "headers": {"content-type": "application/json"}}, 422, "invalid JSON"),
    ({"json": {"targets": [{"id": "a", "ra": 1, "dec": 1}, {"id": "a", "ra": 2, "dec": 2}], "catalogs": ["simbad"]}},
     422, "duplicate id"),
])
def test_router_rejects_bad_requests(kwargs, status, fragment):
    response = _client_call("POST", "/api/v1/batch/crossmatch", catalogs=[], **kwargs)
    assert response.status_code == status, response.text
    assert fragment in response.text


def test_router_strategies():
    response = _client_call("GET", "/api/v1/batch/strategies", catalogs=[])
    assert response.status_code == 200
    data = response.json()
    assert data["catalogs"]["gaia_dr3"]["strategy"] == "xmatch" and data["catalogs"]["ned"]["strategy"] == "cone"
    assert data["max_radius_arcsec"] == 180.0 and "simbad" in data["default_catalogs"]


def test_router_all_catalogs_failed_is_502():
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(return_value=httpx.Response(400, text="bad"))
        with TestClient(make_app()) as client:
            response = client.post("/api/v1/batch/crossmatch?strategies=first=upload",
                                   json={"targets": CANARY_TARGETS[:1], "catalogs": ["first"]})
    # A deterministic HTTP 400 on the upload is final (no cone fallback repeating it): every target failed.
    assert response.status_code == 502, response.text
    assert response.json()["detail"]["catalogs"]["first"]["failed_targets"] == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="astrosearch")
    batch.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def test_cli_batch_parquet(tmp_path, capsys):
    import pyarrow.parquet as pq

    targets = tmp_path / "targets.csv"
    targets.write_text("id,ra,dec\n" + "".join(f"{t['id']},{t['ra']!r},{t['dec']!r}\n" for t in CANARY_TARGETS),
                       encoding="utf-8")
    out = tmp_path / "matches.parquet"
    args = _parse_cli(["batch", "--targets", str(targets), "--catalogs", "simbad,gaia_dr3", "--radius", "10",
                       "--out", str(out)])
    assert args.handler is batch.run_cli
    exchanges = load_batch_exchanges("canary", ["simbad", "gaia_dr3"])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=batch_replay_side_effect(exchanges))
        code = args.handler(args)
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "simbad                upload" in printed and "gaia_dr3              xmatch" in printed
    assert "-> 2 requests" in printed
    table = pq.read_table(out)
    assert "3700386905605055360" in table.column("source_id").to_pylist()


def test_cli_errors(tmp_path, capsys):
    args = _parse_cli(["batch", "--targets", str(tmp_path / "missing.csv")])
    assert args.handler(args) == 2
    bad = tmp_path / "bad.csv"
    bad.write_text("id,ra,dec\n1,10,95\n", encoding="utf-8")
    args = _parse_cli(["batch", "--targets", str(bad)])
    assert args.handler(args) == 2
    assert "DEC must be within" in capsys.readouterr().out
    good = tmp_path / "good.csv"
    good.write_text("ra,dec\n1,1\n", encoding="utf-8")
    args = _parse_cli(["batch", "--targets", str(good), "--format", "json", "--catalogs", "nope"])
    assert args.handler(args) == 2

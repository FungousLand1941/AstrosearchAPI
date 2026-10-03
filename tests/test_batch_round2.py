"""Offline regression tests for the round-2 review of batch.py.

Covers: generic ``vizier:<table>`` XMatch rows read from the positions XMatch matched on (Hipparcos, Tycho-2) with
correct epochs and identifiers; undated SIMBAD confidences; VizieR views sharing their survey's association
priors; epoch-widened cones beyond the XMatch limit (capped / refused / accurate reasons); O(1) target lookup;
event-loop responsiveness (cone planning, parallel conversions, the REST response); malformed input (422 / exit
code 2); failures in flat outputs; client-side limits as 422; connect/pool timeouts; CLI output errors; client
disconnects; the catalog cap; area-scaled upload chunks and OVERFLOW detection before parsing.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import re
import time
from typing import Any

import httpx
import pytest
import respx
from test_batch import make_app, replay_batch
from test_batch_fixtures import (
    BRIGHT_STARS,
    CANARY_TARGETS,
    DENSITY_RADIUS,
    DENSITY_TARGETS,
    HIPPARCOS_TARGETS,
    TYCHO_FIELD_TARGETS,
    WIDE_EPOCH_TARGETS,
    batch_replay_side_effect,
    load_batch_exchanges,
    multipart_fields,
    request_body,
)
from test_batch_regressions import run_with, simbad_json, simbad_rows_for, uploaded_targets, xmatch_votable

import astrometry
import batch
from batch import BatchCrossmatcher, BatchError, BatchResult, BatchTarget, CatalogRun, parse_targets
from models import CatalogRegistry, ColumnMeta

HIP_IDS = {"Barnard": "87937", "61 Cyg A": "104214", "Arcturus": "69673", "Sirius": "32349", "Procyon": "37279",
           "Altair": "97649", "HD 209458": "108859"}
GENERIC_HIP = ["vizier:I/239/hip_main", "vizier:I/311/hip2"]


def _replay(case: str, targets, catalogs, radius, **kwargs) -> BatchResult:
    return asyncio.run(replay_batch(case, targets, catalogs, radius, **kwargs))


def _cli(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="astrosearch")
    batch.register_cli(parser.add_subparsers(dest="command"))
    return parser.parse_args(argv)


def _router_call(handler, method: str, path: str, **kwargs) -> httpx.Response:
    from fastapi.testclient import TestClient

    with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=handler)
        with TestClient(make_app()) as client:
            return client.request(method, path, **kwargs)


def _no_upstream(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not be called
    raise AssertionError(f"no upstream request expected: {request.url}")


# ---------------------------------------------------------------------------
# 1. Generic vizier:<table> rows: XMatch positions, epochs (live recordings)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hipparcos() -> BatchResult:
    return _replay("hipparcos", HIPPARCOS_TARGETS, [*GENERIC_HIP, "vizier:I/259/tyc2"], 5.0)


@pytest.mark.parametrize("catalog", GENERIC_HIP)
def test_hipparcos_rows_are_read_at_the_positions_xmatch_matched(hipparcos, catalog):
    """XMatch returned one Hipparcos row per star (J2000 _RAJ2000 positions); every star is kept, at the J2000
    separation. The J1991.25 meta.main positions (RAICRS / RArad) had dropped 6 of 7 stars and put HD 209458
    at 0.29"."""
    run = hipparcos.runs[catalog]
    assert run.rows_returned == 14 and run.matched_targets == 14 and run.failed_targets == 0 and not run.errors
    for name in BRIGHT_STARS:
        for target_id in (name, f"{name}@pm"):
            matches = hipparcos.target_matches(target_id)[catalog]
            assert [m["source_id"] for m in matches] == [HIP_IDS[name]], (target_id, catalog)
            match = matches[0]
            assert match["epoch"] == 2000.0, (target_id, match["epoch"])  # VizieR propagated with the row's pm
            assert match["separation_arcsec"] < 0.2, (target_id, match["separation_arcsec"])
            assert match["pm_ra_masyr"] is not None and match["pm_dec_masyr"] is not None
            assert "_RAJ2000" in match["data"] and "angDist" not in match["data"]
    hd = hipparcos.target_matches("HD 209458")[catalog][0]
    assert hd["separation_arcsec"] < 0.02  # 0.006" / 0.012" (was 0.29" from the J1991.25 position)
    barnard = hipparcos.target_matches("Barnard@pm")[catalog][0]
    assert barnard["epoch_propagation"] == "source_pm" and barnard["confidence"] > 0.9


def test_tycho2_rows_use_j2000_positions_epochs_and_composite_ids(hipparcos):
    """Tycho-2: the time.epoch column EpRAm (1988.86) is the mean epoch of the measurements, not the epoch of
    _RAJ2000; re-dating the row moved Barnard's match out of the cone (target at epoch 2000 with pm) and put HD
    209458 at 0.318" instead of 0.016". Identifiers are TYC1-TYC2-TYC3, never the observation count Num."""
    run = hipparcos.runs["vizier:I/259/tyc2"]
    assert run.rows_returned == 4 and run.matched_targets == 4 and not run.errors
    for target_id, tyc in (("Barnard", "425-2502-1"), ("Barnard@pm", "425-2502-1"), ("HD 209458", "1688-1821-1"),
                           ("HD 209458@pm", "1688-1821-1")):
        [match] = hipparcos.target_matches(target_id)["vizier:I/259/tyc2"]
        assert match["source_id"] == tyc and match["epoch"] == 2000.0, (target_id, match)
        assert match["data"]["Num"] in (5, 15)  # the former (colliding) 'identifier'
    for target_id in ("HD 209458", "HD 209458@pm"):
        assert hipparcos.target_matches(target_id)["vizier:I/259/tyc2"][0]["separation_arcsec"] < 0.03
    # VizieR declares Tycho-2 pmRA in 'ma/yr' (sic): read as mas/yr, not dropped.
    barnard = hipparcos.target_matches("Barnard@pm")["vizier:I/259/tyc2"][0]
    assert barnard["pm_ra_masyr"] == pytest.approx(-798.8) and barnard["pm_dec_masyr"] == pytest.approx(10277.3)
    assert barnard["epoch_propagation"] == "source_pm"


def _generic_columns() -> list[ColumnMeta]:
    return [ColumnMeta("_RAJ2000", "deg", "pos.eq.ra"), ColumnMeta("_DEJ2000", "deg", "pos.eq.dec"),
            ColumnMeta("TYC1", None, "meta.id.part;meta.main"), ColumnMeta("TYC2", None, "meta.id.part;meta.main"),
            ColumnMeta("TYC3", None, "meta.id.part;meta.main"), ColumnMeta("RAm", "deg", "pos.eq.ra;meta.main"),
            ColumnMeta("DEm", "deg", "pos.eq.dec;meta.main"), ColumnMeta("pmRA", "ma / yr", "pos.pm;pos.eq.ra"),
            ColumnMeta("pmDE", "mas/yr", "pos.pm;pos.eq.dec"), ColumnMeta("EpRAm", "yr", "time.epoch"),
            ColumnMeta("Num", None, "meta.id")]


def test_describe_generic_answer_positions_epochs_ids():
    catalog, _ = batch.xmatch_catalog_definition("vizier:I/999/test", CatalogRegistry())
    moving = {"_RAJ2000": 10.0, "_DEJ2000": 5.0, "TYC1": 1, "TYC2": 22, "TYC3": 1, "RAm": 10.1, "DEm": 5.1,
              "pmRA": 12.0, "pmDE": -3.0, "EpRAm": 1990.5, "Num": 7}
    still = {**moving, "TYC2": 23, "pmRA": None, "pmDE": None, "EpRAm": 1991.25}
    unnamed = {**moving, "TYC3": None, "pmRA": None, "pmDE": None, "EpRAm": None}
    grouped = {0: [moving, still], 1: [unnamed]}
    definition, columns = batch.describe_generic_answer(catalog, _generic_columns(), grouped)
    assert definition.parameters["ra_field"] == "_RAJ2000" and definition.parameters["dec_field"] == "_DEJ2000"
    assert definition.parameters["field_map"] == {"pmra": "pmRA", "pmdec": "pmDE"}
    assert next(c for c in columns if c.name == "pmRA").unit == "mas/yr"
    assert [moving["_source_id"], still["_source_id"]] == ["1-22-1", "1-23-1"]
    assert unnamed["_source_id"] == "J010.000000+05.000000"  # incomplete TYC parts: position designation
    assert (moving["_epoch"], still["_epoch"], unnamed["_epoch"]) == (2000.0, 1991.25, None)
    converter = batch._RowConverter(None, "cds_xmatch")
    result = converter._sources(definition, [moving, still, unnamed], 5.0, None, {}, columns=columns)
    got = {s.source_id: (s.ra, s.dec, s.epoch, s.proper_motion_ra_masyr) for s in result}
    assert got == {"1-22-1": (10.0, 5.0, 2000.0, 12.0), "1-23-1": (10.0, 5.0, 1991.25, None),
                   "J010.000000+05.000000": (10.0, 5.0, None, None)}


def test_describe_generic_answer_prefers_meta_id_main_and_keeps_tables_without_j2000():
    catalog, _ = batch.xmatch_catalog_definition("vizier:I/999/test", CatalogRegistry())
    columns = [ColumnMeta("HIP", None, "meta.id;meta.main"), ColumnMeta("RAdeg", "deg", "pos.eq.ra;meta.main"),
               ColumnMeta("DEdeg", "deg", "pos.eq.dec;meta.main"), ColumnMeta("Obs", "d", "time.epoch")]
    row = {"HIP": 12, "RAdeg": 1.0, "DEdeg": 2.0, "Obs": 51544.5}
    definition, out = batch.describe_generic_answer(catalog, columns, {0: [row]})
    assert definition.parameters["id_field"] == "HIP" and "ra_field" not in definition.parameters
    assert definition.epoch is None and [c.name for c in out] == ["HIP", "RAdeg", "DEdeg", "Obs"]
    [source] = batch._RowConverter(None, "cds_xmatch")._sources(definition, [row], 5.0, None, {}, columns=out)
    assert (source.source_id, source.epoch) == ("12", 2000.0)  # its own time.epoch column (MJD 51544.5)


def test_generic_rows_are_cut_back_with_xmatch_angdist():
    """The per-target cut-back uses XMatch's own angDist (to the positions it matched on), whatever the
    table's meta.main columns say."""
    targets = [{"id": "A", "ra": 150.0, "dec": 2.0, "radius_arcsec": 2.0},
               {"id": "B", "ra": 151.0, "dec": 2.0, "radius_arcsec": 3.5}]
    fields = ("angDist", "t_idx", "t_ra", "t_dec", "_RAJ2000", "_DEJ2000", "HIP", "RAICRS", "DEICRS")
    ints = {"t_idx", "HIP"}
    ucds = {"_RAJ2000": "pos.eq.ra", "_DEJ2000": "pos.eq.dec", "HIP": "meta.id;meta.main",
            "RAICRS": "pos.eq.ra;meta.main", "DEICRS": "pos.eq.dec;meta.main"}

    def handler(request: httpx.Request) -> httpx.Response:
        rows = []
        for idx, ra, dec in uploaded_targets(request):
            for k, off in enumerate((1.0, 3.0)):
                # meta.main positions 90" away (another epoch), _RAJ2000 at the matched offset
                rows.append((off, idx, ra, dec, ra, dec + off / 3600.0, 10 * idx + k, ra, dec + 0.025))
        head = "".join(f'<FIELD name="{n}" datatype="{"int" if n in ints else "double"}"'
                       + (f' ucd="{ucds[n]}"' if n in ucds else "") + "/>" for n in fields)
        body = "".join("<TR>" + "".join(f"<TD>{v}</TD>" for v in row) + "</TR>" for row in rows)
        return httpx.Response(200, headers={"content-type": "text/xml"}, content=(
            '<?xml version="1.0"?><VOTABLE version="1.3"><RESOURCE type="results"><INFO name="QUERY_STATUS" '
            f'value="OK"/><TABLE>{head}<DATA><TABLEDATA>{body}</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>'
        ).encode())

    result = asyncio.run(run_with(handler, targets, ["vizier:I/999/test"], 5.0))
    assert [m["source_id"] for m in result.target_matches("A")["vizier:I/999/test"]] == ["0"]
    assert [m["source_id"] for m in result.target_matches("B")["vizier:I/999/test"]] == ["10", "11"]
    assert result.target_matches("B")["vizier:I/999/test"][1]["separation_arcsec"] == pytest.approx(3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. Undated targets: exact SIMBAD matches of moving stars
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("catalogs", [["simbad"], ["simbad", "twomass_psc"]])
def test_undated_moving_stars_keep_their_simbad_posterior(catalogs):
    """Sirius, Arcturus and HD 209458 without an epoch (the default batch input), at their SIMBAD positions:
    the exact SIMBAD rows had confidence 0.0 / 0.0 / 0.066 (0.0007 with 2MASS added) when every row with a
    proper motion was moved to J2016 but the target was not."""
    result = _replay("hipparcos", HIPPARCOS_TARGETS, catalogs, 5.0)  # the recorded upload holds both variants
    for target_id, main_id in (("Sirius", "* alf CMa"), ("HD 209458", "HD 209458"), ("Altair", "* alf Aql"),
                               ("Barnard", "NAME Barnard's star")):
        match = result.target_matches(target_id)["simbad"][0]
        assert match["source_id"] == main_id and match["separation_arcsec"] < 0.01
        assert match["confidence"] > 0.9, (target_id, catalogs, match["confidence"])
    # Arcturus: SIMBAD also lists '* alf Boo B' 0.24" away, which shares the posterior; the exact row is the
    # most probable counterpart and the two together nearly certain (0.0 before).
    a, b = result.target_matches("Arcturus")["simbad"][:2]
    assert (a["source_id"], b["source_id"]) == ("* alf Boo", "* alf Boo B")
    assert a["confidence"] > b["confidence"] and a["confidence"] + b["confidence"] > 0.9, (a, b)


def test_undated_confidence_through_the_router():
    exchanges = load_batch_exchanges("hipparcos", ["simbad"])
    response = _router_call(batch_replay_side_effect(exchanges), "POST",
                            "/api/v1/batch/crossmatch?format=rows&include_data=false",
                            json={"targets": HIPPARCOS_TARGETS, "catalogs": ["simbad"], "radius_arcsec": 5.0})
    assert response.status_code == 200, response.text
    rows = {(r["target_id"], r["source_id"]): r for r in response.json()["rows"]}
    assert rows[("Sirius", "* alf CMa")]["confidence"] > 0.9
    assert rows[("HD 209458", "HD 209458")]["confidence"] > 0.9


# ---------------------------------------------------------------------------
# 3. VizieR views use their survey's association priors
# ---------------------------------------------------------------------------


def _confidences(result: BatchResult, catalog: str) -> dict[tuple[str, str], float]:
    return {(t.id, m["source_id"]): m["confidence"] for t in result.targets
            for m in result.target_matches(t.id).get(catalog, [])}


@pytest.mark.parametrize("survey, view", [("twomass_psc", "vizier:II/246/out"), ("gaia_dr3", "vizier:I/355/gaiadr3")])
def test_vizier_view_gets_the_same_posterior_as_its_survey(survey, view):
    registry_run = _replay("density", DENSITY_TARGETS, [survey], DENSITY_RADIUS)
    view_run = _replay("density", DENSITY_TARGETS, [view], DENSITY_RADIUS)
    a, b = _confidences(registry_run, survey), _confidences(view_run, view)
    assert a and set(a) == set(b)
    for key, value in a.items():
        assert b[key] == value, (key, value, b[key])
    if survey == "twomass_psc":
        assert a[("M13 core", "16414162+3627415")] > 0.8  # a crowded field: was 0.108 via the view


def test_view_priors_are_what_makes_them_equal(monkeypatch):
    """Without the registered aliases the view gets the generic prior instead of 2MASS's density at the
    position (the M13 core is far denser than the generic 1000 per deg^2): a different posterior for the same
    row, so the equality above is not a coincidence."""
    for table in (astrometry.CATALOG_SKY_DENSITY, astrometry.DENSITY_MAP_SCALING, astrometry.CATALOG_RESOLUTION_ARCSEC):
        assert table["vizier:II/246/out"] == table["twomass_psc"]
        monkeypatch.delitem(table, "vizier:II/246/out")
    monkeypatch.setattr(batch, "_view_aliases", dict)
    survey = _confidences(_replay("density", DENSITY_TARGETS, ["twomass_psc"], DENSITY_RADIUS), "twomass_psc")
    view = _confidences(_replay("density", DENSITY_TARGETS, ["vizier:II/246/out"], DENSITY_RADIUS), "vizier:II/246/out")
    key = ("M13 core", "16414162+3627415")
    assert abs(view[key] - survey[key]) > 0.05, (view[key], survey[key])


# ---------------------------------------------------------------------------
# 4, 11, 12. Cones wider than the XMatch limit
# ---------------------------------------------------------------------------


def test_epoch_widened_view_cone_is_capped_and_matched():
    """3C 286 at epoch 2024 without a proper motion against 2MASS (VizieR II/246): the 284" epoch-widened cone
    used to fail as 'request failed' with a warning claiming a cone search. It is capped at 180" (recorded live)
    and finds 13310829+3030331 at 0.147", as twomass_psc does with its 284" upload cone."""
    exchanges = load_batch_exchanges("wide_epoch", ["vizier:II/246/out"])
    assert [e.fields["distMaxArcsec"] for e in exchanges] == ["180.000"]
    result = _replay("wide_epoch", WIDE_EPOCH_TARGETS, ["vizier:II/246/out", "twomass_psc"], 5.0)
    for catalog in ("vizier:II/246/out", "twomass_psc"):
        [match] = result.target_matches("3C 286@2024")[catalog]
        assert match["source_id"] == "13310829+3030331" and match["separation_arcsec"] == pytest.approx(0.147, abs=0.002)
    run = result.runs["vizier:II/246/out"]
    assert run.failed_targets == 0 and run.refused_targets == 0 and not run.errors
    assert not any("cone search" in w for w in run.warnings)
    assert any("capped at the XMatch limit of 180 arcsec (VizieR tables have no cone fallback)" in w
               for w in run.warnings)
    assert any("epoch gap 26.6 yr needs a 284 arcsec cone" in w for w in run.warnings)


# Epoch 1900 with 3"/yr against Gaia DR3 (2016.0): the cone must enclose the track position at 2016 (348" away)
# and the field at 1900 -- a 185" cone that cannot be capped without losing the target.
FAST_OLD = {"id": "fast1900", "ra": 40.0, "dec": 10.0, "epoch": 1900.0, "pmra": 3000.0, "pmdec": 0.0}


@pytest.mark.parametrize("catalog, fallback, why", [
    ("vizier:I/355/gaiadr3", True, "VizieR tables cannot be cone-searched in a batch"),
    ("gaia_dr3", False, "the cone fallback is disabled"),
])
def test_proper_motion_cone_beyond_the_limit_is_refused_with_its_reason(catalog, fallback, why):
    result = asyncio.run(run_with(_no_upstream, [FAST_OLD], [catalog], 5.0, fallback_to_cone=fallback))
    run = result.runs[catalog]
    assert run.requests == 0 and run.failed_targets == 1 and run.refused_targets == 1
    message = result.failures[0][catalog]
    assert re.search(r"cone of 1\d\d\.\d arcsec \(mode target_pm\) exceeds the CDS XMatch limit of 180 arcsec", message)
    assert why in message and "request failed" not in message
    assert not any("cone search" in w for w in run.warnings)


def test_proper_motion_cone_beyond_the_limit_falls_back_to_a_cone_search():
    cones: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        cones.append(str(request.url))
        return httpx.Response(503, text="down")

    result = asyncio.run(run_with(handler, [FAST_OLD], ["gaia_dr3"], 5.0))
    run = result.runs["gaia_dr3"]
    assert run.fallback_targets == 1 and run.refused_targets == 0 and cones
    assert not any("cdsxmatch" in url for url in cones)
    assert any("sent to per-target cone searches" in w for w in run.warnings)


def test_mixed_run_reports_each_targets_own_reason():
    """A refused too-wide target keeps its own reason; the others report their chunk's HTTP 503 (a too-wide
    target used to inherit whatever error the last chunk had)."""
    targets = [FAST_OLD, {"id": "plain", "ra": 50.0, "dec": 10.0}, {"id": "plain2", "ra": 51.0, "dec": 10.0}]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="Service Unavailable")

    result = asyncio.run(run_with(handler, targets, ["vizier:I/355/gaiadr3"], 5.0))
    failures = {t.id: result.failures[i]["vizier:I/355/gaiadr3"] for i, t in enumerate(result.targets)}
    assert "exceeds the CDS XMatch limit" in failures["fast1900"] and "503" not in failures["fast1900"]
    for target_id in ("plain", "plain2"):
        assert "HTTP 503" in failures[target_id] and "XMatch limit" not in failures[target_id]
    run = result.runs["vizier:I/355/gaiadr3"]
    assert run.refused_targets == 1 and run.failed_targets == 3


def test_refused_everything_is_422_and_cli_exit_2(tmp_path, capsys):
    response = _router_call(_no_upstream, "POST", "/api/v1/batch/crossmatch",
                            json={"targets": [FAST_OLD], "catalogs": ["vizier:I/355/gaiadr3"], "radius_arcsec": 5})
    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert detail["message"] == "no catalog could be queried for any target"
    assert detail["catalogs"]["vizier:I/355/gaiadr3"]["requests"] == 0
    path = tmp_path / "t.json"
    path.write_text(json.dumps([FAST_OLD]), encoding="utf-8")
    args = _cli(["batch", "--targets", str(path), "--catalogs", "vizier:I/355/gaiadr3", "--radius", "5"])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_no_upstream)
        assert args.handler(args) == 2
    assert "exceeds the CDS XMatch limit" in capsys.readouterr().out


def test_upstream_failure_of_every_catalog_is_still_502(monkeypatch):
    monkeypatch.setattr(BatchCrossmatcher, "retry_backoff_seconds", 0.0)
    response = _router_call(lambda request: httpx.Response(503, text="down"), "POST", "/api/v1/batch/crossmatch",
                            json={"targets": CANARY_TARGETS[:1], "catalogs": ["vizier:I/355/gaiadr3"]})
    assert response.status_code == 502, response.text


def test_cone_target_cap_is_a_422_before_any_request(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BATCH_MAX_CONE_TARGETS", "3")
    targets = [{"id": f"t{i}", "ra": 10.0 + i, "dec": 5.0} for i in range(4)]
    response = _router_call(_no_upstream, "POST", "/api/v1/batch/crossmatch",
                            json={"targets": targets, "catalogs": ["ned"]})
    assert response.status_code == 422 and "BATCH_MAX_CONE_TARGETS" in response.text
    path = tmp_path / "t.csv"
    path.write_text("id,ra,dec\n" + "".join(f"{t['id']},{t['ra']},{t['dec']}\n" for t in targets), encoding="utf-8")
    args = _cli(["batch", "--targets", str(path), "--catalogs", "ned"])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_no_upstream)
        assert args.handler(args) == 2
    assert "BATCH_MAX_CONE_TARGETS" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 5. Tycho-2 identifiers in a crowded field (live recording)
# ---------------------------------------------------------------------------


def test_tycho2_identifiers_are_unique_in_a_crowded_field():
    result = _replay("tycho_field", TYCHO_FIELD_TARGETS, ["vizier:I/259/tyc2"], 170.0)
    matches = result.target_matches("h Per")["vizier:I/259/tyc2"]
    assert len(matches) >= 4
    ids = [m["source_id"] for m in matches]
    assert len(set(ids)) == len(ids)
    for match in matches:
        data = match["data"]
        assert match["source_id"] == f"{data['TYC1']}-{data['TYC2']}-{data['TYC3']}"
    nums = [m["data"]["Num"] for m in matches]
    assert len(set(nums)) < len(nums)  # the observation counts that were used as ids do collide here


# ---------------------------------------------------------------------------
# 6. O(1) target lookup
# ---------------------------------------------------------------------------


def test_target_lookup_is_constant_time():
    n = 50_000
    targets = [BatchTarget(f"t{i}", parse_targets([{"ra": 1.0, "dec": 1.0}])[0].target) for i in range(n)]
    result = BatchResult(targets, ["simbad"], 3.0, {"simbad": CatalogRun("simbad", "upload", None)}, {}, {}, 0.0)
    started = time.perf_counter()
    for item in targets:
        assert result.target_matches(item.id) == {} and result.target_association(item.id) == {}
    assert time.perf_counter() - started < 2.0  # 20k targets took 22 s with a scan per call
    with pytest.raises(KeyError):
        result.target_matches("missing")


# ---------------------------------------------------------------------------
# 7, 8. Event loop responsiveness
# ---------------------------------------------------------------------------

_EMPTY_VOTABLE = (b'<?xml version="1.0"?><VOTABLE version="1.3"><RESOURCE type="results"><INFO name="QUERY_STATUS" '
                  b'value="OK"/><TABLE><FIELD name="t_idx" datatype="int"/><FIELD name="name" datatype="char" '
                  b'arraysize="*"/><FIELD name="ra" datatype="double"/><FIELD name="dec" datatype="double"/><DATA>'
                  b"<TABLEDATA></TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>")


def _empty_answer(request: httpx.Request) -> httpx.Response:
    host = request.url.host or ""
    if "cdsxmatch" in host:
        return httpx.Response(200, content=xmatch_votable([]), headers={"content-type": "text/xml"})
    if "simbad" in host:
        return httpx.Response(200, json=simbad_json([]))
    return httpx.Response(200, content=_EMPTY_VOTABLE, headers={"content-type": "text/xml"})


async def _with_ticker(work) -> tuple[Any, float]:
    gaps: list[float] = []
    done = asyncio.Event()

    async def ticker() -> None:
        last = time.perf_counter()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    try:
        result = await work
    finally:
        done.set()
        await tick
    return result, max(gaps)


def test_six_catalogs_of_moving_targets_do_not_block_the_event_loop():
    """8000 targets with an epoch and a proper motion x 6 catalogs: cone planning (~60 us per target and
    catalog) ran on the loop for every catalog in one iteration and six conversion threads starved the loop of
    the GIL -- a 4.2 s worst stall measured before the fix (0.24 s after)."""
    n = 8000
    targets = [{"id": str(i), "ra": (i * 0.113) % 360, "dec": -60 + (i * 0.037) % 120, "epoch": 2010.0,
                "pmra": 50.0, "pmdec": -30.0} for i in range(n)]
    catalogs = ["simbad", "twomass_psc", "allwise", "first", "nvss", "gaia_dr3"]

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(_empty_answer)) as client:
            engine = BatchCrossmatcher(client=client, chunk_sizes={"simbad": n, "irsa": n, "heasarc": n, "xmatch": n})
            return await _with_ticker(engine.run(targets, catalogs, radius_arcsec=3.0))

    result, worst = asyncio.run(scenario())
    assert all(run.requests == 1 and run.failed_targets == 0 for run in result.runs.values())
    assert worst < 1.0, worst


def test_router_json_response_is_serialised_off_the_event_loop(monkeypatch):
    """12000 targets x 2 SIMBAD rows through the REST router (format=json): the response is serialised once, to
    bytes, in a worker thread (it was json.dumps'ed twice, the second time by JSONResponse.render on the loop:
    1.04 s for 20k targets x 2 matches). The stall bound leaves room for the garbage collector, whose full
    collections of a large batch's heap also hold the GIL."""
    import threading

    import fastapi.responses

    n = 12_000
    threads: list[str] = []
    original_bytes = batch._json_bytes

    def spy(data):
        threads.append(threading.current_thread().name)
        return original_bytes(data)

    monkeypatch.setattr(batch, "_json_bytes", spy)
    targets = [{"id": str(i), "ra": (i * 0.113) % 360, "dec": -60 + (i * 0.037) % 120} for i in range(n)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=simbad_json(simbad_rows_for(request, per_target=2)))

    async def scenario():
        app = make_app()
        with respx.mock(assert_all_called=False, assert_all_mocked=False) as router:
            router.route(host="testserver").pass_through()
            router.route().mock(side_effect=handler)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=600) as client:
                return await _with_ticker(client.post("/api/v1/batch/crossmatch?catalogs=simbad&radius_arcsec=5",
                                                      json={"targets": targets}))

    original = fastapi.responses.JSONResponse.render

    def forbidden(self, content):  # pragma: no cover - must not be called
        raise AssertionError("the batch response must not be rendered by JSONResponse on the event loop")

    fastapi.responses.JSONResponse.render = forbidden
    try:
        response, worst = asyncio.run(scenario())
    finally:
        fastapi.responses.JSONResponse.render = original
    assert response.status_code == 200 and response.headers["content-type"] == "application/json"
    data = response.json()
    assert len(data["targets"]) == n and data["summary"]["total_matches"] == 2 * n
    assert len(threads) == 1 and threads[0] != threading.main_thread().name
    assert worst < 1.0, worst


def test_non_finite_values_become_null_in_json():
    assert json.loads(batch._json_bytes({"a": [1.5, float("nan"), {"b": float("inf")}], "c": "x"})) == {
        "a": [1.5, None, {"b": None}], "c": "x"}


# ---------------------------------------------------------------------------
# 9. Malformed input: 422 / exit code 2
# ---------------------------------------------------------------------------

STRAY_QUOTE_CSV = 'id,ra,dec\n"Barnard\'s star,269.452,4.693\n' + "".join(f"t{i},{i * 0.01:.4f},1.0\n" for i in range(10000))
DEEP_JSON = "[" * 100_000


def test_stray_quote_csv_is_a_batch_error():
    assert len(STRAY_QUOTE_CSV) > 131072
    with pytest.raises(BatchError, match="malformed CSV"):
        batch.read_targets_csv(STRAY_QUOTE_CSV)


@pytest.mark.parametrize("payload", [DEEP_JSON, '{"targets": ' + "[" * 100_000 + "}"], ids=["array", "object"])
def test_deeply_nested_json_is_a_batch_error(payload):
    with pytest.raises(BatchError, match="nested too deeply"):
        batch.read_targets_json(payload)


@pytest.mark.parametrize("field", ["ra", "dec", "epoch", "radius_arcsec", "pmra", "parallax"])
def test_huge_integers_are_a_batch_error(field):
    item: dict[str, Any] = {"ra": 1, "dec": 1, field: 10**400}
    if field == "pmra":
        item["pmdec"] = 1
    with pytest.raises(BatchError, match="finite"):
        parse_targets([item])


@pytest.mark.parametrize("kwargs", [
    {"content": STRAY_QUOTE_CSV.encode(), "headers": {"content-type": "text/csv"}},
    {"files": {"file": ("t.csv", STRAY_QUOTE_CSV.encode(), "text/csv")}, "data": {"catalogs": "simbad"}},
    {"content": DEEP_JSON.encode(), "headers": {"content-type": "application/json"}},
    {"files": {"file": ("t.json", DEEP_JSON.encode(), "application/json")}, "data": {"catalogs": "simbad"}},
    {"content": b'{"targets": [{"ra": 1' + b"0" * 400 + b', "dec": 1}]}', "headers": {"content-type": "application/json"}},
    {"content": b'{"targets": [{"ra": 1, "dec": 1, "epoch": 1' + b"0" * 400 + b"}]}",
     "headers": {"content-type": "application/json"}},
    {"content": b'{"targets": [{"ra": 1, "dec": 1, "radius_arcsec": 1' + b"0" * 400 + b"}]}",
     "headers": {"content-type": "application/json"}},
], ids=["csv-stray-quote", "multipart-csv-stray-quote", "json-deep", "multipart-json-deep", "json-huge-ra",
        "json-huge-epoch", "json-huge-radius"])
def test_router_malformed_input_is_422(kwargs):
    response = _router_call(_no_upstream, "POST", "/api/v1/batch/crossmatch?catalogs=simbad", **kwargs)
    assert response.status_code == 422, (response.status_code, response.text[:300])


@pytest.mark.parametrize("name, content", [
    ("stray.csv", STRAY_QUOTE_CSV), ("deep.json", DEEP_JSON), ("huge.json", '[{"ra": 1' + "0" * 400 + ', "dec": 1}]'),
], ids=["stray-quote", "deep", "huge"])
def test_cli_malformed_input_is_exit_2(tmp_path, capsys, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    args = _cli(["batch", "--targets", str(path), "--catalogs", "simbad"])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_no_upstream)
        assert args.handler(args) == 2
    assert capsys.readouterr().out.startswith("Error:")


# ---------------------------------------------------------------------------
# 10. Failures in the flat outputs
# ---------------------------------------------------------------------------


def _simbad_400_first_ok(request: httpx.Request) -> httpx.Response:
    if "simbad" in (request.url.host or ""):
        return httpx.Response(400, text='<VOTABLE><RESOURCE><INFO name="QUERY_STATUS" value="ERROR">bad</INFO>'
                                        "</RESOURCE></VOTABLE>", headers={"content-type": "text/xml"})
    return batch_replay_side_effect(load_batch_exchanges("canary", ["first"]))(request)


def _failing_simbad_result() -> BatchResult:
    return asyncio.run(run_with(_simbad_400_first_ok, CANARY_TARGETS, ["simbad", "first"], 10.0))


def test_flat_rows_mark_failed_queries():
    result = _failing_simbad_result()
    rows = result.rows(include_data=False)
    failed = [r for r in rows if r["status"] == "failed"]
    assert {r["target_id"] for r in failed} == {t["id"] for t in CANARY_TARGETS}
    assert all(r["catalog"] == "simbad" and "CatalogQueryError" in r["error"] and r["source_id"] is None
               and r["strategy"] == "upload" for r in failed)
    matched = [r for r in rows if r["status"] == "match"]
    assert matched and all(r["catalog"] == "first" and r["error"] is None for r in matched)
    assert len(matched) == result.runs["first"].total_matches
    assert result.failed_target_count == 4
    table = result.to_arrow(include_data=False)
    assert table.column("status").to_pylist().count("failed") == 4
    failures = json.loads(table.schema.metadata[b"astrosearch.batch.failures"])
    assert set(failures) == {t["id"] for t in CANARY_TARGETS} and all("simbad" in v for v in failures.values())
    text = result.to_csv_text(include_data=False)
    header = text.splitlines()[0].split(",")
    assert "status" in header and "error" in header
    assert sum(1 for line in text.splitlines()[1:] if ",simbad,upload,failed," in line) == 4


@pytest.mark.parametrize("fmt", ["csv", "rows", "parquet"])
def test_router_flat_formats_report_failures(fmt):
    import pyarrow.parquet as pq

    response = _router_call(_simbad_400_first_ok, "POST", f"/api/v1/batch/crossmatch?format={fmt}",
                            json={"targets": CANARY_TARGETS, "catalogs": ["simbad", "first"], "radius_arcsec": 10})
    assert response.status_code == 200, response.text
    assert response.headers["x-batch-failed-targets"] == "4"
    if fmt == "csv":
        lines = response.text.splitlines()
        assert sum(1 for line in lines if ",simbad,upload,failed," in line) == 4
    elif fmt == "rows":
        data = response.json()
        assert sum(1 for r in data["rows"] if r["status"] == "failed") == 4
        assert set(data["failures"]) == {t["id"] for t in CANARY_TARGETS}
    else:
        table = pq.read_table(io.BytesIO(response.content))
        assert table.column("status").to_pylist().count("failed") == 4
        assert b"astrosearch.batch.failures" in table.schema.metadata


# ---------------------------------------------------------------------------
# 13. Connect / pool timeouts
# ---------------------------------------------------------------------------


def test_connect_timeouts_are_retried_not_split_and_open_the_circuit():
    """A 64-target chunk whose every request cannot connect: 5 attempts (the breaker opens at the 5th), no split
    (it made 132 requests and 63 splits before)."""
    guards: dict[str, Any] = {}
    targets = [{"id": f"t{i}", "ra": 10.0 + i * 0.01, "dec": 5.0} for i in range(64)]

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timed out", request=request)

    result = asyncio.run(run_with(handler, targets, ["simbad"], 5.0, guards=guards, fallback_to_cone=False,
                                  chunk_sizes={"simbad": 64}))
    run = result.runs["simbad"]
    assert run.requests == 5 and run.split_chunks == 0 and run.failed_targets == 64
    assert guards[f"batch-upload:{batch.SIMBAD_TAP}"].state == "open"
    assert all("ConnectTimeout" in result.failures[i]["simbad"] for i in range(64))


def test_pool_timeouts_are_retried_without_a_verdict_on_the_endpoint():
    guards: dict[str, Any] = {}
    targets = [{"id": f"t{i}", "ra": 10.0 + i * 0.01, "dec": 5.0} for i in range(8)]

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("no free connection", request=request)

    result = asyncio.run(run_with(handler, targets, ["simbad"], 5.0, guards=guards, fallback_to_cone=False))
    run = result.runs["simbad"]
    assert run.requests == 5 and run.split_chunks == 0 and run.failed_targets == 8
    assert guards[f"batch-upload:{batch.SIMBAD_TAP}"].state == "closed"


# ---------------------------------------------------------------------------
# 14. CLI output errors
# ---------------------------------------------------------------------------


def test_cli_output_directory_is_rejected_before_running(tmp_path, capsys):
    targets = tmp_path / "t.csv"
    targets.write_text("ra,dec\n1,1\n", encoding="utf-8")
    args = _cli(["batch", "--targets", str(targets), "--catalogs", "simbad", "--out", str(tmp_path)])
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=_no_upstream)
        assert args.handler(args) == 2
    assert "is a directory" in capsys.readouterr().out


def test_cli_write_failure_after_the_run_is_exit_2(tmp_path, capsys, monkeypatch):
    targets = tmp_path / "t.csv"
    targets.write_text("id,ra,dec\n" + "".join(f"{t['id']},{t['ra']!r},{t['dec']!r}\n" for t in CANARY_TARGETS),
                       encoding="utf-8")
    out = tmp_path / "out.parquet"
    args = _cli(["batch", "--targets", str(targets), "--catalogs", "simbad", "--radius", "10", "--out", str(out)])

    def fail(self, path, fmt=None, *, include_data=True):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(BatchResult, "write", fail)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=batch_replay_side_effect(load_batch_exchanges("canary", ["simbad"])))
        assert args.handler(args) == 2
    printed = capsys.readouterr().out
    assert "simbad" in printed and "Error: could not write" in printed  # the report was printed first


# ---------------------------------------------------------------------------
# 15. Client disconnect cancels the batch
# ---------------------------------------------------------------------------


def test_client_disconnect_cancels_the_batch():
    """The client disconnects once the first cone is in flight: no further cone may start, the in-flight
    cones are cancelled (not left running in the background) and the route answers 499 promptly.

    Synchronised on events, not on wall-clock sleeps: the time is measured from the disconnect, so a
    loaded machine that is slow to set the batch up (building an HTTP client, parsing the body) cannot
    make it fail, while an uncancelled batch still would (each cone takes CONE_SECONDS)."""
    CONE_SECONDS = 5.0  # uncancelled: the in-flight wave alone takes 5 s, the whole batch 40 / 8 x 5 s = 25 s
    calls = {"before": 0, "after": 0, "active": 0, "finished": 0}
    gone = asyncio.Event()
    cone_started = asyncio.Event()

    async def slow_cone(request: httpx.Request) -> httpx.Response:
        calls["after" if gone.is_set() else "before"] += 1
        calls["active"] += 1
        cone_started.set()
        try:
            await asyncio.sleep(CONE_SECONDS)
            calls["finished"] += 1
            return httpx.Response(200, json=simbad_json([]))
        finally:
            calls["active"] -= 1

    body = json.dumps({"targets": [{"id": f"t{i}", "ra": 10.0 + i, "dec": 5.0} for i in range(40)],
                       "catalogs": ["simbad"], "strategies": {"simbad": "cone"}}).encode()

    async def scenario() -> tuple[list[dict[str, Any]], float]:
        app = make_app()
        sent: list[dict[str, Any]] = []
        body_sent = False
        timeline: dict[str, float] = {}

        async def receive() -> dict[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await cone_started.wait()  # the batch is querying the archive
            await asyncio.sleep(0.05)  # let the first wave of cones start
            timeline["disconnect"] = time.perf_counter()
            gone.set()
            return {"type": "http.disconnect"}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                 "scheme": "http", "path": "/api/v1/batch/crossmatch", "raw_path": b"/api/v1/batch/crossmatch",
                 "query_string": b"", "root_path": "", "server": ("testserver", 80), "client": ("127.0.0.1", 1),
                 "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]}
        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=slow_cone)
            # (a generous cap only so that a broken watcher fails the test instead of hanging it)
            await asyncio.wait_for(app(scope, receive, send), timeout=120.0)
            elapsed = time.perf_counter() - timeline["disconnect"]
            # Nothing may still be running in the background: cancelled cones have left slow_cone.
            for _ in range(20):
                if calls["active"] == 0:
                    break
                await asyncio.sleep(0.05)
        return sent, elapsed

    sent, elapsed = asyncio.run(scenario())
    assert gone.is_set(), "the route never read the disconnect"
    assert calls["before"] > 0 and calls["after"] == 0, calls
    assert calls["active"] == 0 and calls["finished"] == 0, calls  # in-flight cones cancelled, none completed
    assert elapsed < CONE_SECONDS / 2, elapsed  # cancelled at once; uncancelled >= CONE_SECONDS after it
    assert sent and sent[0]["status"] == batch.CLIENT_CLOSED_REQUEST


# ---------------------------------------------------------------------------
# 16. Catalog cap
# ---------------------------------------------------------------------------


def test_catalog_count_is_capped_before_any_request(monkeypatch):
    names = [f"vizier:J/X/{i}/t" for i in range(21)]
    response = _router_call(_no_upstream, "POST", "/api/v1/batch/crossmatch",
                            json={"targets": CANARY_TARGETS[:1], "catalogs": names})
    assert response.status_code == 422 and "BATCH_MAX_CATALOGS" in response.text
    monkeypatch.setenv("BATCH_MAX_CATALOGS", "2")
    with pytest.raises(BatchError, match="at most 2"):
        asyncio.run(BatchCrossmatcher().run(CANARY_TARGETS, ["simbad", "first", "nvss"], radius_arcsec=5))


# ---------------------------------------------------------------------------
# 17. Upload chunks scaled by cone area; OVERFLOW detected before parsing
# ---------------------------------------------------------------------------


def test_upload_chunks_shrink_for_wide_cones():
    """2000 targets with 60" cones: SIMBAD chunks hold at most 5000 / 36 targets (the expected rows per request
    stay those of 10" cones); 10" cones keep the full 5000-target chunks."""
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        rows = simbad_rows_for(request)
        sizes.append(len(rows))
        return httpx.Response(200, json=simbad_json(rows))

    wide = [{"id": str(i), "ra": (i * 0.113) % 360, "dec": 10.0, "radius_arcsec": 60.0} for i in range(2000)]
    result = asyncio.run(run_with(handler, wide, ["simbad"], 5.0))
    assert max(sizes) <= 5000 // 36 and sum(sizes) == 2000 and result.runs["simbad"].matched_targets == 2000
    sizes.clear()
    narrow = [{**t, "radius_arcsec": 10.0} for t in wide]
    asyncio.run(run_with(handler, narrow, ["simbad"], 5.0))
    assert sizes == [2000]


def test_overflow_status_splits_without_parsing(monkeypatch):
    parsed: list[int] = []
    original = batch.BatchCrossmatcher._parse

    def counting(response, label, expected):
        parsed.append(1)
        return original(response, label, expected)

    monkeypatch.setattr(batch.BatchCrossmatcher, "_parse", staticmethod(counting))
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fields = multipart_fields(request.headers["content-type"], request_body(request))
        n = len(re.findall(r"<TR>", fields["targets"]))
        sizes.append(n)
        status = "OVERFLOW" if n > 1 else "OK"
        return httpx.Response(200, headers={"content-type": "text/xml"}, content=(
            '<?xml version="1.0"?><VOTABLE version="1.3"><RESOURCE type="results"><INFO name="QUERY_STATUS" '
            'value="OK"/><TABLE><FIELD name="t_idx" datatype="int"/><FIELD name="name" datatype="char" '
            'arraysize="*"/><FIELD name="ra" datatype="double"/><FIELD name="dec" datatype="double"/><DATA>'
            f'<TABLEDATA></TABLEDATA></DATA></TABLE><INFO name="QUERY_STATUS" value="{status}"/></RESOURCE></VOTABLE>'
        ).encode())

    result = asyncio.run(run_with(handler, CANARY_TARGETS, ["first"], 10.0))
    assert sizes == [4, 2, 2, 1, 1, 1, 1] and result.runs["first"].split_chunks == 3
    assert len(parsed) == 4  # only the four single-target answers were parsed
    assert batch._declares_overflow(b'<INFO value="OVERFLOW" name="QUERY_STATUS"/>')
    assert not batch._declares_overflow(b'<INFO name="QUERY_STATUS" value="OK"/> OVERFLOW in a description')


# ---------------------------------------------------------------------------
# Target position error from the precision of the coordinates as typed (as CrossmatchService.prepare)
# ---------------------------------------------------------------------------


def test_target_sigma_follows_the_coordinate_precision_like_a_single_object_search():
    targets = parse_targets([{"id": "coarse", "ra": "187.278", "dec": "2.052"},
                             {"id": "fine", "ra": "187.27791542", "dec": "2.05238833"}])
    coarse, fine = targets
    assert coarse.coordinate_sigma_arcsec == pytest.approx(1.04, abs=0.01) and fine.coordinate_sigma_arcsec < 1e-3

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=simbad_json(simbad_rows_for(request)))

    result = asyncio.run(run_with(handler, targets, ["simbad"], 5.0))
    floor = batch.AssociationConfig().target_sigma_arcsec
    assert result.target_association("coarse")["target_sigma_arcsec"] == pytest.approx(
        math.hypot(floor, coarse.coordinate_sigma_arcsec))
    assert result.target_association("fine")["target_sigma_arcsec"] == floor

"""Tests for vo_server: Simple Cone Search, TAP/ADQL, VOSI, UWS and the ``vo`` CLI.

Offline tests serve rows recorded from the real archives for 3C 273
(tests/fixtures/vo_server/3c273_sources.json, derived from the raw responses in
tests/fixtures/3c273 by the real provider code) through a stand-in service that runs
the real matching and grouping code (crossmatch.match_target / _group_matches).
VOTables are validated with astropy.io.votable; pyvo talks to a real uvicorn server
started on a free port. Live tests (``-m live``) query the real archives for 3C 273.

Regenerate the recorded rows (offline, from the raw fixtures)::

    .venv/Scripts/python.exe tests/test_vo_server.py record
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import math
import re
import socket
import sys
import threading
import time
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import vo_server
from crossmatch import _group_matches, match_target
from models import CatalogFailure, CatalogRegistry, CatalogSource, UnifiedRecord, validate_target

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "vo_server" / "3c273_sources.json"
# 3C 273 (SIMBAD, ICRS J2000) -- the position the raw archive fixtures were recorded at.
RA_3C273, DEC_3C273 = 187.2779154, 2.0523883
TEN_ARCSEC_DEG = 10.0 / 3600.0
GAIA_3C273 = "3700386905605055360"


# ---------------------------------------------------------------------------
# Recorded-row stand-in for CrossmatchService
# ---------------------------------------------------------------------------


def load_recorded_sources() -> tuple[list[CatalogSource], dict[str, str], list[str]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    sources = [CatalogSource(**item) for item in payload["sources"]]
    return sources, payload["citations"], payload["catalogs"]


class RecordedCrossmatchService:
    """Offline stand-in for crossmatch.CrossmatchService.

    Serves the rows recorded for 3C 273 and runs the real matching/grouping code on
    them; ``failing`` catalogs are reported as upstream failures (HTTP 503) and
    ``truncate`` ({catalog: n}) keeps only the n nearest rows of a catalog and reports
    it as truncated, exactly like CrossmatchService does when an archive holds more
    than max_rows rows in the cone (``truncated`` + ``excess_row_count`` + a warning).
    """

    def __init__(self, *, failing: frozenset[str] = frozenset(), delay_s: float = 0.0,
                 truncate: dict[str, int] | None = None) -> None:
        self.registry = CatalogRegistry()
        self.sources, self.citations, self.catalogs = load_recorded_sources()
        self.failing = failing
        self.delay_s = delay_s
        self.truncate = dict(truncate or {})
        self.calls: list[dict[str, Any]] = []
        self.completed = 0

    async def crossmatch(self, ra: float, dec: float, *, radius_arcsec: float | None = None, query: Any = None,
                         **_: Any) -> UnifiedRecord:
        self.calls.append({"ra": ra, "dec": dec, "radius_arcsec": radius_arcsec,
                           "catalogs": list(query.catalogs) if query is not None and query.catalogs else None})
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        target = validate_target(ra, dec)
        radius = float(query.radius_arcsec if query is not None else radius_arcsec)
        names = [n for n in self.catalogs if query is None or not query.catalogs or n in query.catalogs]
        ok = [n for n in names if n not in self.failing]
        kept, counts, excess = [], {}, {}
        for match in match_target(target, [s for s in self.sources if s.catalog in ok], radius):  # nearest first
            limit = self.truncate.get(match.catalog)
            if limit is not None and counts.get(match.catalog, 0) >= limit:
                excess[match.catalog] = excess.get(match.catalog, 0) + 1
                continue
            counts[match.catalog] = counts.get(match.catalog, 0) + 1
            kept.append(match)
        groups = _group_matches(kept, target, radius)
        failures = [CatalogFailure(n, error_type="CatalogUnavailableError", message="HTTP 503 (simulated outage)").as_dict()
                    for n in names if n in self.failing]
        counterparts: dict[str, list[dict[str, Any]]] = {}
        for group in groups:
            for member in group["members"]:
                counterparts.setdefault(member["metadata"]["wavelength"], []).append(member)
        self.completed += 1
        return UnifiedRecord(
            target=target.as_dict(),
            catalogs_queried=len(names),
            catalog_results={n: {"status": "success", "row_count": counts.get(n, 0), "truncated": n in excess,
                                 "excess_row_count": excess.get(n, 0)} for n in ok},
            counterparts=counterparts,
            failures=failures,
            provenance={"citations": {n: self.citations[n] for n in ok if n in self.citations},
                        "warnings": [f"{n}: truncated at the {counts.get(n, 0)} nearest rows (simulated)" for n in excess]},
            crossmatch_groups=groups,
        )


def make_app(service: Any):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(vo_server.router)
    app.state.service = service
    app.state.registry = service.registry
    return app


@pytest.fixture
def stub() -> RecordedCrossmatchService:
    return RecordedCrossmatchService()


@pytest.fixture
def client(stub: RecordedCrossmatchService) -> Iterator[Any]:
    from fastapi.testclient import TestClient

    with TestClient(make_app(stub)) as test_client:
        yield test_client


def votable(body: bytes):
    from astropy.io.votable import parse

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any VOTable spec warning fails the test
        return parse(io.BytesIO(body), verify="exception")


def assert_valid_votable(body: bytes) -> None:
    from astropy.io.votable import validate

    report = io.StringIO()
    assert validate(io.BytesIO(body), output=report), report.getvalue()


# Simple Cone Search 1.03 section 2.2 mandates the UCD1 names ID_MAIN, POS_EQ_RA_MAIN and
# POS_EQ_DEC_MAIN (VizieR and HEASARC cone searches serve them in VOTable 1.4 too).
# astropy checks VOTable >= 1.2 UCDs against the UCD1+ vocabulary and reports exactly
# these as W06; every other check must still pass.
SCS_UCD1_NAMES = ("ID_MAIN", "POS_EQ_RA_MAIN", "POS_EQ_DEC_MAIN")


def _is_scs_ucd1_warning(message: str) -> bool:
    return "W06" in message and any(f"Invalid UCD '{name}'" in message for name in SCS_UCD1_NAMES)


def scs_votable(body: bytes):
    """Parse an SCS response: like :func:`votable`, tolerating only W06 on the SCS 1.03 UCD1 names."""
    from astropy.io.votable import parse

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed = parse(io.BytesIO(body), verify="warn")
    unexpected = [str(w.message) for w in caught if not _is_scs_ucd1_warning(str(w.message))]
    assert not unexpected, unexpected
    return parsed


def assert_valid_scs_votable(body: bytes) -> None:
    from astropy.io.votable import validate

    report = io.StringIO()
    if validate(io.BytesIO(body), output=report):
        return
    violations = [line for line in report.getvalue().splitlines() if re.match(r"^\d+: [EW]\d\d:", line)]
    assert violations and all(_is_scs_ucd1_warning(v) for v in violations), report.getvalue()
    assert len(violations) <= len(SCS_UCD1_NAMES), report.getvalue()


def cone_adql(radius_deg: float = TEN_ARCSEC_DEG, extra: str = "", select: str = "*") -> str:
    return (f"SELECT {select} FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {RA_3C273}, {DEC_3C273}, {radius_deg!r})){extra}")


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Recorded fixture sanity (real 3C 273 truth carried by the recorded rows)
# ---------------------------------------------------------------------------


def test_recorded_rows_are_real_3c273_counterparts() -> None:
    sources, citations, catalogs = load_recorded_sources()
    by_catalog = {(s.catalog, s.source_id): s for s in sources}
    assert ("gaia_dr3", GAIA_3C273) in by_catalog
    assert ("simbad", "3C 273") in by_catalog and ("ned", "3C 273") in by_catalog
    assert ("first", "FIRST J122906.7+020308") in by_catalog and ("chandra", "2CXO J122906.6+020308") in by_catalog
    assert by_catalog[("ned", "3C 273")].metadata["physical"]["redshift"] == pytest.approx(0.158339, abs=1e-5)
    assert set(citations) <= set(catalogs) and "gaia_dr3" in citations
    for s in sources:  # every recorded row lies inside the 10" recording cone
        assert math.hypot((s.ra - RA_3C273) * math.cos(math.radians(DEC_3C273)), s.dec - DEC_3C273) * 3600 <= 10.05


# ---------------------------------------------------------------------------
# Column metadata: UCD1+ and VOUnit validity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", vo_server.TABLES, ids=lambda t: t.name)
def test_column_ucds_units_and_types_are_valid(table: vo_server.TableDef) -> None:
    from astropy import units as u
    from astropy.io.votable.ucd import check_ucd

    names = [c.name for c in table.columns]
    assert len(names) == len(set(names))
    for col in table.columns:
        if col.ucd:
            assert check_ucd(col.ucd, check_controlled_vocabulary=True), (table.name, col.name, col.ucd)
        if col.unit:
            u.Unit(col.unit, format="vounit")
        assert col.datatype in {"char", "unicodeChar", "short", "int", "long", "float", "double"}
        assert (col.arraysize == "*") == col.is_string


def test_matches_table_has_the_cone_search_ucds() -> None:
    ucds = {c.ucd: c.name for c in vo_server.MATCHES_COLUMNS}
    assert ucds["meta.id;meta.main"] == "match_id"
    assert ucds["pos.eq.ra;meta.main"] == "ra" and ucds["pos.eq.dec;meta.main"] == "dec"
    assert ucds["pos.angDistance"] == "separation" and ucds["stat.likelihood"] == "match_confidence"
    assert ucds["instr.bandpass"] == "wavelength"
    assert [c.name for c in vo_server.MATCHES_COLUMNS if c.verb == 1] == ["match_id", "ra", "dec"]
    columns = {c.name: c for c in vo_server.MATCHES_COLUMNS}
    # The metadata must not promise what is not done: no proper-motion propagation, and
    # the group is a positional candidate set, not a confirmed physical association.
    assert "no proper-motion propagation" in columns["separation"].description
    assert "propagated with proper motion" not in columns["separation"].description
    assert "NOT a confirmed physical association" in columns["group_id"].description
    assert "at most one detection per catalog" in columns["group_id"].description
    assert columns["group_id"].ucd == "meta.id"  # not meta.id.assoc (associated counterpart)
    assert "physical-object" not in vo_server.TABLES_BY_NAME["astrosearch.matches"].description
    # The documented k is the one used.
    assert "k = 3.72" in columns["group_id"].description and vo_server.GROUP_K == pytest.approx(3.7169, abs=1e-4)
    # Cone Search publishes the SCS 1.03 UCD1 names for the three required FIELDs.
    assert vo_server.SCS_UCD1 == {"match_id": "ID_MAIN", "ra": "POS_EQ_RA_MAIN", "dec": "POS_EQ_DEC_MAIN"}


def test_standards_are_cited_with_their_authors() -> None:
    # IVOA REC ConeSearch-20080222: Williams, Hanisch, Szalay, Plante (ed. Plante).
    assert "Williams, Hanisch, Szalay & Plante 2008" in vo_server.__doc__ and "Graham" not in vo_server.__doc__.split(
        "Table Access Protocol")[0]


def test_wavelength_descriptions_name_every_registry_value() -> None:
    """A user filtering wavelength = 'radio' must be told that SIMBAD/NED rows are 'multi'/'extragalactic'."""
    values = {c.wavelength for c in CatalogRegistry().catalogs.values()}
    assert {"multi", "extragalactic", "exoplanet", "radio", "optical", "infrared", "xray"} <= values
    for table in ("astrosearch.matches", "astrosearch.catalogs"):
        description = vo_server.TABLES_BY_NAME[table].column("wavelength").description
        for value in values:
            assert f"'{value}'" in description, (table, value)


# ---------------------------------------------------------------------------
# ADQL parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "SELECT * FROM astrosearch.catalogs",
    "select distinct top 5 name as n, wavelength from astrosearch.catalogs c where c.enabled = 1 order by 1 desc offset 1;",
    'SELECT "name" FROM "astrosearch"."catalogs" WHERE NOT (max_rows < 10 OR max_rows IS NULL)',
    "SELECT t.* FROM TAP_SCHEMA.tables AS t -- trailing comment",
    "SELECT COUNT(*) AS n FROM astrosearch.catalogs WHERE name LIKE 'gaia%' AND wavelength NOT IN ('radio', 'xray')",
    cone_adql(extra=" AND separation BETWEEN 0 AND 5 AND catalog = 'gaia_dr3' ORDER BY separation ASC, catalog"),
    ("SELECT DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', 187.0, 2.0)) AS d FROM astrosearch.matches "
     "WHERE DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', 187.0, 2.0)) < 0.01"),
])
def test_parser_accepts_supported_subset(query: str) -> None:
    parsed = vo_server.parse_adql(query)
    assert parsed.items and parsed.table.parts


def test_parser_structure() -> None:
    q = vo_server.parse_adql("SELECT DISTINCT TOP 3 a AS x, b.* FROM s.t AS b WHERE a > -1 AND (c = 'it''s' OR d IS NOT NULL) "
                             "ORDER BY x DESC OFFSET 2")
    assert q.distinct and q.top == 3 and q.offset == 2
    assert q.items[0].alias == "x" and q.items[1].star and q.items[1].star_qualifier == ("b",)
    assert q.table.parts == ("s", "t") and q.table.alias == "b"
    assert isinstance(q.where, vo_server.And)
    first = q.where.items[0]
    assert first.op == ">" and first.left.parts == ("a",) and first.right.value == -1
    inner = q.where.items[1]
    assert isinstance(inner, vo_server.Or) and inner.items[0].right.value == "it's"
    assert q.order_by[0].descending


@pytest.mark.parametrize(("query", "message"), [
    ("", "Empty QUERY"),
    ("SELECT * FROM astrosearch.catalogs a JOIN astrosearch.matches b ON a.name = b.catalog", "Joins are not supported"),
    ("SELECT * FROM astrosearch.catalogs, astrosearch.matches", "Joins are not supported"),
    ("SELECT wavelength FROM astrosearch.catalogs GROUP BY wavelength", "GROUP BY / HAVING are not supported"),
    ("SELECT * FROM (SELECT * FROM astrosearch.catalogs) AS s", "Subqueries are not supported"),
    ("SELECT * FROM astrosearch.catalogs WHERE name IN (SELECT name FROM astrosearch.catalogs)", "IN subqueries"),
    ("SELECT * FROM astrosearch.catalogs UNION SELECT * FROM astrosearch.catalogs", "Set operations"),
    ("WITH x AS (SELECT 1) SELECT * FROM x", "Common table expressions"),
    ("SELECT CASE WHEN enabled = 1 THEN 'y' END FROM astrosearch.catalogs", "CASE expressions are not supported"),
    ("SELECT * FROM astrosearch.catalogs WHERE name = 'unterminated", "Unterminated string literal"),
    ("SELECT * FROM astrosearch.catalogs WHERE name ~ 'x'", "Unexpected character"),
    ("SELECT * FROM astrosearch.catalogs WHERE", "Expected a value expression at end of query"),
    ("SELECT * astrosearch.catalogs", "Expected FROM"),
    ("SELECT TOP x * FROM astrosearch.catalogs", "TOP must be a non-negative integer"),
    ("SELECT * FROM astrosearch.catalogs WHERE name", "Expected a comparison"),
    ("SELECT * FROM astrosearch.catalogs extra junk", "Unexpected token"),
])
def test_parser_rejects_unsupported_constructs_clearly(query: str, message: str) -> None:
    with pytest.raises(vo_server.ADQLError, match=message.replace("(", r"\(").replace(")", r"\)")):
        vo_server.parse_adql(query)


def test_parser_limits_hostile_input() -> None:
    nested = "SELECT name FROM astrosearch.catalogs WHERE " + "(" * 5000 + "max_rows > 1" + ")" * 5000
    with pytest.raises(vo_server.ADQLError, match="too deeply nested"):
        vo_server.parse_adql(nested)
    with pytest.raises(vo_server.ADQLError, match="longer than"):
        vo_server.parse_adql("SELECT name FROM astrosearch.catalogs WHERE name = '" + "x" * 200_000 + "'")


# ---------------------------------------------------------------------------
# ADQL semantics on registry / TAP_SCHEMA tables (no upstream access)
# ---------------------------------------------------------------------------


def adql(query: str, **kw: Any) -> vo_server.ResultTable:
    return run(vo_server.execute_adql(query, service=None, registry=CatalogRegistry(), **kw))


def test_catalogs_table_mirrors_registry() -> None:
    registry = CatalogRegistry()
    result = adql("SELECT * FROM astrosearch.catalogs")
    rows = {r["name"]: r for r in result.as_dicts()}
    assert set(rows) == set(registry.catalogs)
    gaia = registry.get("gaia_dr3")
    assert rows["gaia_dr3"]["endpoint"] == gaia.endpoint and rows["gaia_dr3"]["citation"] == gaia.citation
    assert rows["gaia_dr3"]["enabled"] == 1 and rows["gaia_dr3"]["wavelength"] == "optical"
    enabled = adql("SELECT COUNT(*) AS n FROM astrosearch.catalogs WHERE enabled = 1")
    assert enabled.rows == [(len(registry.enabled_catalogs()),)]
    assert enabled.columns[0].name == "n" and enabled.columns[0].datatype == "long"


def test_tap_schema_describes_every_published_column() -> None:
    columns = adql("SELECT table_name, column_name, ucd, unit, datatype, principal, std, column_index FROM TAP_SCHEMA.columns",
                   maxrec=10000)
    expected = {(t.name, c.name) for t in vo_server.TABLES for c in t.columns}
    assert {(r["table_name"], r["column_name"]) for r in columns.as_dicts()} == expected
    ra = next(r for r in columns.as_dicts() if r["table_name"] == "astrosearch.matches" and r["column_name"] == "ra")
    assert ra["ucd"] == "pos.eq.ra;meta.main" and ra["unit"] == "deg" and ra["datatype"] == "double" and ra["principal"] == 1
    tables = adql("SELECT table_name, table_type FROM TAP_SCHEMA.tables WHERE schema_name = 'astrosearch' ORDER BY table_index")
    assert tables.rows == [("astrosearch.matches", "view"), ("astrosearch.catalogs", "table")]
    size = adql('SELECT "size" FROM TAP_SCHEMA.columns WHERE column_name = \'ra\'')
    assert size.rows and all(r == (None,) for r in size.rows)
    assert adql("SELECT * FROM tap_schema.keys").rows == []
    assert adql("SELECT schema_name FROM TAP_SCHEMA.schemas ORDER BY schema_index").rows == [("astrosearch",), ("TAP_SCHEMA",)]


def test_sql_semantics_null_like_between_in_integer_division() -> None:
    q = "SELECT name, epoch, 7/2 AS i, 7/2.0 AS f, -7/2 AS neg, ROUND(2.5) AS r, ROUND(-2.5) AS rn, " \
        "UPPER(name) || '!' AS up, POWER(2, 10) AS p, SQRT(-1) AS bad FROM astrosearch.catalogs WHERE name = 'gaia_dr3'"
    result = adql(q)
    row = result.as_dicts()[0]
    assert row["i"] == 3 and row["f"] == 3.5 and row["neg"] == -3  # integer division truncates toward zero
    types = {c.name: c.datatype for c in result.columns}
    assert types["i"] == "long" and types["f"] == "double" and types["up"] == "char" and types["epoch"] == "double"
    assert row["r"] == 3.0 and row["rn"] == -3.0  # SQL rounds half away from zero
    assert row["up"] == "GAIA_DR3!" and row["p"] == 1024.0 and row["bad"] is None
    # Gaia DR3's per-row ref_epoch column is J2016.0 for every source (Gaia DR3 data model).
    assert row["epoch"] == 2016.0
    # NULL never compares true or false: NVSS publishes no epoch.
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name = 'nvss' AND epoch > 0").rows == []
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name = 'nvss' AND NOT epoch > 0").rows == []
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name = 'nvss' AND epoch IS NULL").rows == [("nvss",)]
    like = adql("SELECT name FROM astrosearch.catalogs WHERE name LIKE 'rosat%' ORDER BY name")
    assert like.rows == [("rosat",), ("rosat_bsc",)]
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name LIKE 'ROSAT_'").rows == []
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name ILIKE 'ROSAT_BS_'").rows == [("rosat_bsc",)]
    registry = CatalogRegistry()
    assert [registry.get(n).timeout_seconds for n in ("first", "nvss", "gaia_dr3")] == [60.0, 60.0, 90.0]
    between = adql("SELECT name FROM astrosearch.catalogs WHERE timeout_seconds BETWEEN 60 AND 60 "
                   "AND name IN ('first', 'nvss', 'gaia_dr3') ORDER BY name")
    assert between.rows == [("first",), ("nvss",)]
    outside = adql("SELECT name FROM astrosearch.catalogs WHERE timeout_seconds NOT BETWEEN 60 AND 60 "
                   "AND name IN ('first', 'nvss', 'gaia_dr3')")
    assert outside.rows == [("gaia_dr3",)]
    wide = adql("SELECT name FROM astrosearch.catalogs WHERE timeout_seconds BETWEEN 59.5 AND 90 "
                "AND name IN ('first', 'gaia_dr3') ORDER BY name")
    assert wide.rows == [("first",), ("gaia_dr3",)]  # both bounds inclusive
    # BETWEEN with a NULL operand is unknown, and so is its negation.
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name = 'nvss' AND epoch BETWEEN 0 AND 3000").rows == []
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name = 'nvss' AND epoch NOT BETWEEN 0 AND 3000").rows == []
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name = 'gaia_dr3' AND epoch BETWEEN 2016 AND 2016"
                ).rows == [("gaia_dr3",)]
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name IN ('first', NULL)").rows == [("first",)]
    assert adql("SELECT name FROM astrosearch.catalogs WHERE name NOT IN ('first', NULL)").rows == []


def test_distance_and_contains_are_great_circle_degrees() -> None:
    row = adql("SELECT DISTANCE(POINT('ICRS', 0, 0), POINT('ICRS', 0, 1)) AS a, DISTANCE(10, 89, 190, 89) AS b, "
               "DISTANCE(POINT('ICRS', 359.5, 0), POINT('ICRS', 0.5, 0)) AS c, "
               "CONTAINS(POINT('ICRS', 0.5, 0), CIRCLE('ICRS', 359.9, 0, 0.61)) AS inside, "
               "CONTAINS(POINT(0.5, 0), CIRCLE(359.9, 0, 0.59)) AS outside FROM TAP_SCHEMA.schemas "
               "WHERE schema_name = 'TAP_SCHEMA'").as_dicts()[0]
    assert row["a"] == pytest.approx(1.0, abs=1e-12)
    assert row["b"] == pytest.approx(2.0, abs=1e-9)  # over the pole
    assert row["c"] == pytest.approx(1.0, abs=1e-12)  # across RA = 0
    assert row["inside"] == 1 and row["outside"] == 0
    dist_col = adql("SELECT DISTANCE(0, 0, 1, 0) AS d FROM TAP_SCHEMA.schemas").columns[0]
    assert dist_col.unit == "deg" and dist_col.ucd == "pos.angDistance"


def test_order_distinct_offset_top_maxrec_overflow() -> None:
    names = sorted(CatalogRegistry().catalogs)
    assert [r[0] for r in adql("SELECT name FROM astrosearch.catalogs ORDER BY name").rows] == names
    assert [r[0] for r in adql("SELECT name AS n FROM astrosearch.catalogs ORDER BY n DESC").rows] == names[::-1]
    assert [r[0] for r in adql("SELECT TOP 2 name FROM astrosearch.catalogs ORDER BY 1 OFFSET 1").rows] == names[1:3]
    distinct = adql("SELECT DISTINCT provider FROM astrosearch.catalogs")
    assert len(distinct.rows) == len({r[0] for r in distinct.rows}) >= 3
    # NULLs sort last ascending and first descending (PostgreSQL semantics).
    epochs = [r[0] for r in adql("SELECT epoch FROM astrosearch.catalogs ORDER BY epoch").rows]
    assert epochs[-1] is None and epochs[0] is not None
    assert adql("SELECT epoch FROM astrosearch.catalogs ORDER BY epoch DESC").rows[0] == (None,)
    capped = adql("SELECT name FROM astrosearch.catalogs", maxrec=3)
    assert capped.status == "OVERFLOW" and len(capped.rows) == 3
    topped = adql("SELECT TOP 3 name FROM astrosearch.catalogs", maxrec=3)
    assert topped.status == "OK" and len(topped.rows) == 3  # TOP, not MAXREC, limited the rows
    empty = adql("SELECT name FROM astrosearch.catalogs", maxrec=0)
    assert empty.rows == [] and [c.name for c in empty.columns] == ["name"]
    # DALI 1.1: MAXREC=0 returns "metadata, no results, and an overflow indicator" --
    # whatever the table and even when no row would match; TOP 0 is not MAXREC (status OK).
    assert empty.status == "OVERFLOW"
    assert adql("SELECT name FROM astrosearch.catalogs WHERE 1 = 0", maxrec=0).status == "OVERFLOW"
    assert adql("SELECT * FROM TAP_SCHEMA.tables", maxrec=0).status == "OVERFLOW"
    top0 = adql("SELECT TOP 0 name FROM astrosearch.catalogs")
    assert top0.rows == [] and top0.status == "OK"
    # The service's hard limit caps any MAXREC (and the default).
    hard = vo_server.VOConfig(default_maxrec=3, hard_maxrec=5)
    capped_hard = adql("SELECT name FROM astrosearch.catalogs", maxrec=100, config=hard)
    assert len(capped_hard.rows) == 5 and capped_hard.status == "OVERFLOW"
    default = adql("SELECT name FROM astrosearch.catalogs", config=hard)
    assert len(default.rows) == 3 and default.status == "OVERFLOW"
    _query, maxrec, _fmt, _media = vo_server.validate_tap_parameters(
        {"LANG": "ADQL", "QUERY": "SELECT 1 FROM TAP_SCHEMA.schemas", "MAXREC": "1000000"}, hard)
    assert maxrec == 5


def test_aggregates() -> None:
    registry = CatalogRegistry()
    providers = [c.provider for c in registry.catalogs.values()]
    row = adql("SELECT COUNT(*) AS n, COUNT(DISTINCT provider) AS p, MIN(name) AS lo, MAX(max_rows) AS hi, "
               "AVG(max_rows) AS mean, SUM(enabled) AS on_count FROM astrosearch.catalogs").as_dicts()[0]
    assert row["n"] == len(registry.catalogs) and row["p"] == len(set(providers))
    assert row["lo"] == min(registry.catalogs) and row["on_count"] == len(registry.enabled_catalogs())
    limit = vo_server.VOConfig.from_env().catalog_row_limit
    vo_rows = [max(c.max_rows, limit) for c in registry.catalogs.values()]
    assert row["mean"] == pytest.approx(sum(vo_rows) / len(vo_rows)) and row["hi"] == max(vo_rows)


@pytest.mark.parametrize(("query", "message"), [
    ("SELECT ra FROM astrosearch.catalogs", "Unknown column 'ra' in table astrosearch.catalogs"),
    ("SELECT * FROM astrosearch.nothing", "Unknown table 'astrosearch.nothing'"),
    ("SELECT x.name FROM astrosearch.catalogs AS c", "Unknown table qualifier 'x'"),
    ("SELECT name FROM astrosearch.catalogs WHERE name = 3", "Cannot compare a string with a number"),
    ("SELECT name FROM astrosearch.catalogs WHERE max_rows LIKE '2%'", "LIKE requires string operands"),
    ("SELECT name, COUNT(*) FROM astrosearch.catalogs", "Aggregate queries may only select aggregates"),
    ("SELECT name FROM astrosearch.catalogs WHERE COUNT(*) > 1", "Aggregate functions are not allowed in WHERE"),
    ("SELECT BOX('ICRS', 1, 2, 3, 4) FROM astrosearch.catalogs", "Geometry function BOX is not supported"),
    ("SELECT FOO(name) FROM astrosearch.catalogs", "Unknown or unsupported function FOO"),
    ("SELECT POINT('ICRS', 1, 2) FROM astrosearch.catalogs", "Geometry values .* cannot be selected"),
    ("SELECT SQRT(1, 2) FROM astrosearch.catalogs", "SQRT takes 1 argument"),
    ("SELECT * FROM astrosearch.matches WHERE catalog = 'gaia_dr3'", "must constrain position with a top-level"),
    (("SELECT * FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 1, 2, 3)) "
      "OR catalog = 'x'"), "must constrain position"),
    (cone_adql(radius_deg=1.0), "exceeds this service's maximum"),
    ("SELECT * FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 10, 95, 0.01))",
     "DEC must be within"),
    ("SELECT * FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 10, 10, 10/3600))",
     r"radius must be positive.*integer division"),
    ("SELECT * FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('GALACTIC', ra, dec), CIRCLE('ICRS', 10, 10, 0.01))",
     "only ICRS is supported"),
    ("SELECT * FROM astrosearch.matches WHERE 1 = CONTAINS(CIRCLE('ICRS', 10, 10, 0.01), POINT('ICRS', ra, dec))",
     r"Only CONTAINS\(POINT"),
    ("SELECT * FROM astrosearch.matches WHERE POINT('ICRS', ra, dec) = POINT('ICRS', 1, 1)", "Geometry values cannot be compared"),
    ("SELECT name FROM astrosearch.catalogs ORDER BY 5", "ORDER BY position 5 is out of range"),
])
def test_semantic_errors_are_reported(query: str, message: str) -> None:
    with pytest.raises(vo_server.ADQLError, match=message):
        adql(query)


def _regex_like(value: str, pattern: str) -> bool:
    import re

    return re.fullmatch("".join("." if c == "_" else ".*" if c == "%" else re.escape(c) for c in pattern), value,
                        re.DOTALL) is not None


def test_like_matcher_agrees_with_sql_semantics() -> None:
    import random

    rng = random.Random(20260928)
    for _ in range(4000):  # exhaustive-style comparison with a reference regex translation
        value = "".join(rng.choice("abQ%_") for _ in range(rng.randint(0, 8)))
        pattern = "".join(rng.choice("abQ%_") for _ in range(rng.randint(0, 6)))
        assert vo_server.like_match(value, pattern) == _regex_like(value, pattern), (value, pattern)
    assert vo_server.like_match("GAIA_DR3", "gaia\\_%", case_insensitive=True) is False  # no ESCAPE in ADQL
    assert vo_server.like_match("Gaia DR3", "gaia%", case_insensitive=True)
    assert not vo_server.like_match("Gaia DR3", "gaia%")
    assert vo_server.like_match("", "%") and not vo_server.like_match("", "_")


def test_pathological_like_patterns_run_in_linear_time() -> None:
    # A backtracking regex needed 30 s for '%_%_%_%_%Q' on these ~280-character descriptions
    # and never finished for 25 '_%' groups; the matcher must stay fast and correct.
    for pattern in ("%_%_%_%_%Q", "%" + "_%" * 25 + "Q", "%" * 50_000 + "Q"):
        started = time.monotonic()
        result = adql(f"SELECT COUNT(*) AS n FROM TAP_SCHEMA.columns WHERE description LIKE '{pattern}'")
        assert time.monotonic() - started < 1.0, pattern[:40]
        assert result.rows == [(0,)]  # no description ends with 'Q'
    hits = adql("SELECT COUNT(*) AS n FROM TAP_SCHEMA.columns WHERE description LIKE '%_%_%_%_%.'").rows[0][0]
    total = adql("SELECT COUNT(*) AS n FROM TAP_SCHEMA.columns WHERE description LIKE '____%.'").rows[0][0]
    assert hits == total > 0


def test_adql_evaluation_time_is_bounded() -> None:
    tight = vo_server.VOConfig(max_eval_seconds=1e-9)
    with pytest.raises(vo_server.ADQLError, match="exceeded this service's limit"):
        adql("SELECT column_name FROM TAP_SCHEMA.columns WHERE description LIKE '%a%'", config=tight)


def test_binder_resolves_temporary_column_references_by_name() -> None:
    table = vo_server.TABLES_BY_NAME["astrosearch.matches"]
    binder = vo_server.Binder(table, vo_server.TableRef(("astrosearch", "matches"), (False, False), None, False))
    assert binder.type_of(vo_server.Col(("ra",), (False,), 0)) == "num"
    for name, kind in (("catalog", "str"), ("source_id", "str"), ("match_id", "str"), ("separation", "num")):
        # Temporary objects: their id() is reused after garbage collection.
        assert binder.type_of(vo_server.Col((name,), (False,), 0)) == kind, name
        assert binder.column(vo_server.Col((name,), (False,), 0)).name == name


@pytest.mark.parametrize(("query", "message"), [
    ("SELECT " + "9" * 5000 + " AS x FROM TAP_SCHEMA.schemas", "outside the 64-bit range"),
    ("SELECT 9223372036854775807 + 1 AS x FROM TAP_SCHEMA.schemas", "Integer overflow"),
    ("SELECT 9223372036854775808 AS x FROM TAP_SCHEMA.schemas", "Integer overflow"),
    ("SELECT 3037000500 * 3037000500 AS x FROM TAP_SCHEMA.schemas", "Integer overflow"),
    ("SELECT 1e999 AS x FROM TAP_SCHEMA.schemas", "outside the double-precision range"),
])
def test_out_of_range_numbers_are_adql_errors(client: Any, query: str, message: str) -> None:
    for fmt in ("votable", "csv"):
        response = client.get("/vo/tap/sync", params=tap_params(query, RESPONSEFORMAT=fmt))
        assert response.status_code == 400, response.text
        assert message in votable(response.content).resources[0].infos[0].content


def test_int64_limits_and_float_overflow_are_exact() -> None:
    row = adql("SELECT 9223372036854775807 AS hi, -9223372036854775808 AS lo, 1e308 * 10 AS inf, "
               "FLOOR(2.5) AS f, CEILING(-2.5) AS c FROM TAP_SCHEMA.schemas WHERE schema_name = 'TAP_SCHEMA'")
    values = row.as_dicts()[0]
    assert values["hi"] == 2**63 - 1 and values["lo"] == -(2**63) and values["inf"] is None
    assert values["f"] == 2.0 and isinstance(values["f"], float) and values["c"] == -2.0
    assert_valid_votable(vo_server.render_votable(row))


def test_circle_accepts_every_adql_argument_form(stub: RecordedCrossmatchService) -> None:
    forms = [
        f"CIRCLE('ICRS', {RA_3C273}, {DEC_3C273}, 0.0028)",
        f"CIRCLE({RA_3C273}, {DEC_3C273}, 0.0028)",
        f"CIRCLE('ICRS', POINT('ICRS', {RA_3C273}, {DEC_3C273}), 0.0028)",  # ADQL 2.1
        f"CIRCLE(POINT('ICRS', {RA_3C273}, {DEC_3C273}), 0.0028)",
    ]
    results = []
    for circle in forms:
        query = f"SELECT match_id FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), {circle})"
        results.append(sorted(run(vo_server.execute_adql(query, service=stub)).column_values("match_id")))
        assert stub.calls[-1]["radius_arcsec"] == pytest.approx(0.0028 * 3600)
    assert results[0] and all(r == results[0] for r in results)
    inside = adql("SELECT CONTAINS(POINT('ICRS', 10, 0.5), CIRCLE('ICRS', POINT('ICRS', 10, 0), 0.6)) AS c "
                  "FROM TAP_SCHEMA.schemas WHERE schema_name = 'TAP_SCHEMA'")
    assert inside.rows == [(1,)]
    with pytest.raises(vo_server.ADQLError, match="only ICRS is supported"):
        adql("SELECT 1 FROM TAP_SCHEMA.schemas WHERE 1 = CONTAINS(POINT(1, 2), CIRCLE('FK4', POINT(1, 2), 1))")


# ---------------------------------------------------------------------------
# astrosearch.matches through the recorded-row service
# ---------------------------------------------------------------------------


def test_matches_query_returns_3c273_counterparts(stub: RecordedCrossmatchService) -> None:
    result = run(vo_server.execute_adql(cone_adql(extra=" ORDER BY separation"), service=stub))
    rows = result.as_dicts()
    assert stub.calls == [{"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0, "catalogs": None}]
    by_id = {r["match_id"]: r for r in rows}
    gaia = by_id[f"gaia_dr3:{GAIA_3C273}"]
    assert gaia["separation"] == pytest.approx(0.0021, abs=0.002)
    assert gaia["wavelength"] == "optical" and gaia["epoch"] == 2016.0 and gaia["match_confidence"] > 0.99
    ned = by_id["ned:3C 273"]
    assert ned["redshift"] == pytest.approx(0.158339, abs=1e-5) and ned["object_type"] == "QSO"
    assert by_id["simbad:3C 273"]["separation"] < 0.01
    assert {r["wavelength"] for r in rows} >= {"optical", "infrared", "radio", "xray"}
    seps = [r["separation"] for r in rows]
    assert seps == sorted(seps) and max(seps) <= 10.0
    # The 3C 273 candidate set: one detection from each of 11 catalogs, optical to X-ray.
    core = [r for r in rows if r["group_id"] == gaia["group_id"]]
    assert {r["match_id"] for r in core} == {
        f"gaia_dr3:{GAIA_3C273}", "simbad:3C 273", "ned:3C 273", "panstarrs_dr2:110461872779253351",
        "sdss:1237651735760142397", "allwise:J122906.69+020308.6", "twomass_psc:12290669+0203085",
        "chandra:2CXO J122906.6+020308", "xmm:5XMM J122906.6+020308", "first:FIRST J122906.7+020308",
        "rosat:2RXS J122906.6+020309"}
    assert gaia["group_id"] == "group-1" and gaia["group_n_catalogs"] == 11
    assert gaia["group_n_catalogs"] == len(gaia["group_catalogs"].split(","))
    # Unrelated neighbours are NOT chained in: the NED absorption systems at 0.8", the
    # GALEX source at 2.7" and the Pan-STARRS objects at 6.9" and 8.8" have their own sets.
    for match_id in ("ned:[HB89] 1226+023 ABS01", "ned:GALEXMSC J122906.78+020306.1",
                     "panstarrs_dr2:110461872781485699", "panstarrs_dr2:110461872779676212"):
        assert by_id[match_id]["group_id"] != gaia["group_id"], match_id
    # NED lists 18 absorption systems on the 3C 273 sightline at ONE position (0.815" away, identical
    # coordinates): one detection, so they share one set of their own instead of 18 singletons.
    absorbers = {f"ned:[HB89] 1226+023 ABS{i:02d}" for i in range(1, 19)}
    abs_set = {r["match_id"] for r in rows if r["group_id"] == by_id["ned:[HB89] 1226+023 ABS01"]["group_id"]}
    assert abs_set == absorbers and by_id["ned:[HB89] 1226+023 ABS18"]["group_n_catalogs"] == 1
    assert len({r["group_id"] for r in rows}) == 7
    assert ("QUERY", cone_adql(extra=" ORDER BY separation")) in result.infos
    assert any(name == "citation" and value.startswith("gaia_dr3: Gaia Collaboration") for name, value in result.infos)


def test_contains_is_exact_and_catalog_constraints_are_pushed_down(stub: RecordedCrossmatchService) -> None:
    two = run(vo_server.execute_adql(cone_adql(radius_deg=2.0 / 3600), service=stub))
    assert two.rows and max(two.column_values("separation")) <= 2.0
    # Every catalog-level condition (catalog / wavelength) is used to skip archives.
    both = cone_adql(select="match_id, catalog, separation",
                     extra=" AND catalog IN ('gaia_dr3', 'ned', 'nvss') AND catalog <> 'ned' AND separation < 6 "
                           "ORDER BY separation")
    result = run(vo_server.execute_adql(both, service=stub))
    assert stub.calls[-1]["catalogs"] == ["gaia_dr3", "nvss"]
    assert set(result.column_values("catalog")) == {"gaia_dr3", "nvss"} and result.status == "OK"
    nvss = result.as_dicts()[-1]
    assert nvss["match_id"] == "nvss:NVSS J122906+020305" and nvss["separation"] == pytest.approx(5.58, abs=0.05)
    radio = run(vo_server.execute_adql(cone_adql(select="catalog", extra=" AND wavelength IN ('radio', 'xray')"),
                                       service=stub))
    registry = CatalogRegistry()
    expected = sorted(n for n, c in registry.enabled_catalogs().items() if c.wavelength in {"radio", "xray"})
    assert stub.calls[-1]["catalogs"] == expected and set(expected) >= {"first", "nvss", "chandra", "xmm"}
    assert set(radio.column_values("catalog")) <= set(expected)
    # A catalog-level condition that is NULL (unknown) for an archive excludes its rows too.
    run(vo_server.execute_adql(cone_adql(select="match_id", extra=" AND catalog IN ('gaia_dr3', NULL)"), service=stub))
    assert stub.calls[-1]["catalogs"] == ["gaia_dr3"]
    # DISTANCE form of the cone; the tightest of several cone terms is sent upstream.
    dist = (f"SELECT match_id FROM astrosearch.matches WHERE DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {RA_3C273}, "
            f"{DEC_3C273})) < 0.0003 AND 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {RA_3C273}, {DEC_3C273}, 0.002))")
    run(vo_server.execute_adql(dist, service=stub))
    assert stub.calls[-1]["radius_arcsec"] == pytest.approx(0.0003 * 3600)
    calls = len(stub.calls)
    none = run(vo_server.execute_adql(cone_adql(extra=" AND catalog = 'no_such_catalog'"), service=stub))
    assert none.rows == [] and len(stub.calls) == calls  # nothing to query upstream
    meta = run(vo_server.execute_adql(cone_adql(), service=stub, maxrec=0))
    assert meta.rows == [] and len(stub.calls) == calls and len(meta.columns) == len(vo_server.MATCHES_COLUMNS)
    assert meta.status == "OVERFLOW"  # DALI 1.1: MAXREC=0 carries the overflow indicator
    # A CIRCLE centred at negative RA is the same sky position (RA wraps to [0, 360)).
    wrapped = run(vo_server.execute_adql(
        "SELECT match_id FROM astrosearch.matches WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
        f"CIRCLE('ICRS', {RA_3C273 - 360.0}, {DEC_3C273}, {TEN_ARCSEC_DEG!r}))", service=stub))
    assert stub.calls[-1]["ra"] == pytest.approx(RA_3C273, abs=1e-9)
    assert f"gaia_dr3:{GAIA_3C273}" in wrapped.column_values("match_id")
    positive = run(vo_server.execute_adql(cone_adql(select="match_id"), service=stub))
    assert sorted(wrapped.column_values("match_id")) == sorted(positive.column_values("match_id"))


def test_where_clause_never_changes_group_values(stub: RecordedCrossmatchService) -> None:
    """group_* describe clusters over ALL archives; a catalog filter must not recompute them."""
    gaia_id = f"gaia_dr3:{GAIA_3C273}"
    cone_only = {r["match_id"]: r for r in run(vo_server.execute_adql(cone_adql(), service=stub)).as_dicts()}
    assert cone_only[gaia_id]["group_n_catalogs"] == 11
    everything = "allwise,chandra,first,gaia_dr3,ned,panstarrs_dr2,rosat,sdss,simbad,twomass_psc,xmm"
    assert cone_only[gaia_id]["group_catalogs"] == everything
    forms = [
        " AND catalog = 'gaia_dr3'",
        " AND catalog = 'gaia_dr3' AND group_n_catalogs >= 3",
        " AND (catalog = 'gaia_dr3' OR catalog = 'x') AND group_n_catalogs >= 3",
        " AND catalog IN ('gaia_dr3') AND wavelength = 'optical'",
    ]
    for extra in forms:
        result = run(vo_server.execute_adql(cone_adql(extra=extra), service=stub))  # SELECT * includes group_*
        assert stub.calls[-1]["catalogs"] is None, extra  # every archive queried: no push-down
        rows = result.as_dicts()
        assert [r["match_id"] for r in rows] == [gaia_id], extra
        assert rows[0]["group_n_catalogs"] == 11 and rows[0]["group_catalogs"] == everything, extra
        assert rows[0]["group_id"] == cone_only[gaia_id]["group_id"]
    for select in ("match_id, group_id", "match_id"):
        query = cone_adql(select=select, extra=" AND catalog = 'gaia_dr3' ORDER BY group_n_catalogs")
        run(vo_server.execute_adql(query, service=stub))
        assert stub.calls[-1]["catalogs"] is None  # group_* in the select list or ORDER BY
    run(vo_server.execute_adql(cone_adql(select="match_id, separation", extra=" AND catalog = 'gaia_dr3'"), service=stub))
    assert stub.calls[-1]["catalogs"] == ["gaia_dr3"]  # no group column: push-down is safe


def test_aggregate_over_matches(stub: RecordedCrossmatchService) -> None:
    row = run(vo_server.execute_adql(
        cone_adql(select="COUNT(*) AS n, COUNT(DISTINCT catalog) AS ncat, MIN(separation) AS best"), service=stub)).as_dicts()[0]
    all_rows = run(vo_server.execute_adql(cone_adql(), service=stub)).as_dicts()
    assert row["n"] == len(all_rows) and row["ncat"] == len({r["catalog"] for r in all_rows})
    assert row["best"] == min(r["separation"] for r in all_rows) < 0.001


def test_recorded_values_are_published_unchanged(stub: RecordedCrossmatchService) -> None:
    sources, _citations, _catalogs = load_recorded_sources()
    gaia_src = next(s for s in sources if s.catalog == "gaia_dr3" and s.source_id == GAIA_3C273)
    rows = {r["match_id"]: r for r in run(vo_server.execute_adql(cone_adql(), service=stub)).as_dicts()}
    gaia = rows[f"gaia_dr3:{GAIA_3C273}"]
    assert gaia["pmra"] == gaia_src.proper_motion_ra_masyr == pytest.approx(-0.022885672495060565)
    assert gaia["pmdec"] == gaia_src.proper_motion_dec_masyr == pytest.approx(0.09846797908921764)
    assert gaia["parallax"] == gaia_src.metadata["physical"]["parallax"] == pytest.approx(-0.015950290834234743)
    assert gaia["pos_error"] == gaia_src.positional_error_arcsec
    assert gaia["ra"] == gaia_src.ra and gaia["dec"] == gaia_src.dec
    # An off-centre row: SIMBAD [CME2001] 3C 273 1 at 4.54" with a 5" positional error has
    # the Gaussian score exp(-sep^2 / 2 sigma^2) = 0.662 (crossmatch.match_score).
    cme = rows["simbad:[CME2001] 3C 273 1"]
    assert cme["separation"] == pytest.approx(4.5412, abs=1e-3) and cme["pos_error"] == 5.0
    assert cme["match_confidence"] == pytest.approx(math.exp(-0.5 * (cme["separation"] / 5.0) ** 2), abs=1e-6)
    assert cme["match_confidence"] == pytest.approx(0.662, abs=0.001)
    # Recomputed from the published separation and position for every row.
    centre_ra, centre_dec = math.radians(RA_3C273), math.radians(DEC_3C273)
    for row in rows.values():
        ra, dec = math.radians(row["ra"]), math.radians(row["dec"])
        hav = math.sin((dec - centre_dec) / 2) ** 2 + math.cos(dec) * math.cos(centre_dec) * math.sin((ra - centre_ra) / 2) ** 2
        assert row["separation"] == pytest.approx(math.degrees(2 * math.asin(math.sqrt(hav))) * 3600, abs=1e-6)


def test_partial_and_total_upstream_failures() -> None:
    partial = RecordedCrossmatchService(failing=frozenset({"xmm", "chandra"}))
    result = run(vo_server.execute_adql(cone_adql(), service=partial))
    assert not {"xmm", "chandra"} & set(result.column_values("catalog"))
    warnings_ = [v for n, v in result.infos if n == "WARNING"]
    assert any(v.startswith("xmm failed (CatalogUnavailableError)") for v in warnings_) and len(warnings_) == 2
    # Rows are missing, so the result is flagged -- never QUERY_STATUS=OK.
    assert result.status == "OVERFLOW" and set(result.failed) == {"xmm", "chandra"} and not result.truncated
    incomplete = [v for n, v in result.infos if n == "INCOMPLETE"]
    assert incomplete == [("chandra: CatalogUnavailableError: HTTP 503 (simulated outage); "
                           "xmm: CatalogUnavailableError: HTTP 503 (simulated outage)")]
    dead = RecordedCrossmatchService(failing=frozenset(load_recorded_sources()[2]))
    with pytest.raises(vo_server.UpstreamError, match="All .* upstream archives failed"):
        run(vo_server.execute_adql(cone_adql(), service=dead))


def test_partial_failure_refuses_answers_that_need_every_row() -> None:
    """COUNT over gaia_dr3 + simbad with simbad down must not come back as 1 with status OK."""
    partial = RecordedCrossmatchService(failing=frozenset({"simbad"}))
    healthy = RecordedCrossmatchService()
    counting = cone_adql(select="COUNT(*) AS n", extra=" AND catalog IN ('gaia_dr3', 'simbad')")
    assert run(vo_server.execute_adql(counting, service=healthy)).rows == [(3,)]  # 1 Gaia + 2 SIMBAD rows
    with pytest.raises(vo_server.UpstreamError, match=r"aggregate .* simbad: CatalogUnavailableError"):
        run(vo_server.execute_adql(counting, service=partial))
    # An archive that the query excludes cannot make it incomplete.
    gaia_only = run(vo_server.execute_adql(cone_adql(select="COUNT(*) AS n", extra=" AND catalog = 'gaia_dr3'"),
                                           service=partial))
    assert gaia_only.rows == [(1,)] and gaia_only.status == "OK" and not gaia_only.incomplete
    for extra in (" AND catalog IN ('gaia_dr3', 'simbad') AND separation < 1",   # filter on a row column
                  " AND object_type = 'QSO'",
                  " ORDER BY separation DESC OFFSET 1",                           # ranking
                  " AND catalog = 'gaia_dr3' AND group_n_catalogs > 1"):          # group values need simbad too
        with pytest.raises(vo_server.UpstreamError, match="simbad"):
            run(vo_server.execute_adql(cone_adql(select="match_id", extra=extra), service=partial))
    plain = run(vo_server.execute_adql(cone_adql(select="match_id", extra=" AND catalog IN ('gaia_dr3', 'simbad')"),
                                       service=partial))
    assert plain.column_values("match_id") == [f"gaia_dr3:{GAIA_3C273}"] and plain.status == "OVERFLOW"


def test_truncated_archive_is_flagged_or_refused() -> None:
    """An archive cut at its row limit: rows kept are the nearest, flagged; exact answers refused."""
    truncating = RecordedCrossmatchService(truncate={"ned": 2})
    rows = run(vo_server.execute_adql(cone_adql(extra=" AND catalog = 'ned'"), service=truncating))
    assert len(rows.rows) == 2 and rows.status == "OVERFLOW" and set(rows.truncated) == {"ned"}
    assert max(rows.column_values("separation")) < 0.9
    warnings_ = [v for n, v in rows.infos if n == "WARNING"]
    assert "ned: truncated at the 2 nearest rows (simulated)" in warnings_  # the crossmatch provenance warning
    assert any(v.startswith("ned truncated: only the 2 nearest rows were fetched (complete only to 0.82 arcsec")
               for v in warnings_)
    assert_valid_votable(vo_server.render_votable(rows))
    statuses = [i.value for i in votable(vo_server.render_votable(rows)).resources[0].infos if i.name == "QUERY_STATUS"]
    assert statuses == ["OK", "OVERFLOW"]
    for select, extra in (("COUNT(*) AS n", " AND catalog = 'ned'"),
                          ("MAX(separation) AS m", ""),
                          ("match_id", " AND redshift > 0.1"),
                          ("TOP 3 match_id", " ORDER BY separation DESC"),
                          ("DISTINCT TOP 3 catalog, separation", " ORDER BY separation"),
                          ("TOP 3 match_id", " AND group_n_catalogs > 1 ORDER BY separation"),
                          ("TOP 11 match_id", " ORDER BY separation"),      # row 11 is NED's farthest kept row
                          ("TOP 3 match_id", " ORDER BY separation OFFSET 8")):
        with pytest.raises(vo_server.IncompleteResultError, match=r"ned: Truncated: only the 2 nearest rows .* Reduce"):
            run(vo_server.execute_adql(cone_adql(select=select, extra=extra), service=truncating))


def test_nearest_rows_of_a_truncated_cone_are_exact() -> None:
    """TOP n ... ORDER BY separation needs only the rows nearer than the n-th: when they all lie
    inside the radius to which the truncated archive is complete (NED: 0.815"), the answer is exact."""
    truncating = RecordedCrossmatchService(truncate={"ned": 2})
    healthy = RecordedCrossmatchService()
    distance = f"DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {RA_3C273}, {DEC_3C273}))"
    for select, extra in (("TOP 3 match_id, separation", " ORDER BY separation"),
                          ("TOP 10 match_id", " ORDER BY separation, catalog"),
                          ("TOP 3 match_id AS m, separation AS s", " ORDER BY s"),
                          ("TOP 3 separation, match_id", " ORDER BY 1"),
                          ("TOP 3 match_id", f" ORDER BY {distance}"),
                          ("TOP 2 match_id, redshift", " AND redshift > 0.1 ORDER BY separation"),
                          ("TOP 2 match_id", " ORDER BY separation OFFSET 1")):
        query = cone_adql(select=select, extra=extra)
        exact = run(vo_server.execute_adql(query, service=truncating))
        truth = run(vo_server.execute_adql(query, service=healthy))
        assert exact.rows == truth.rows and exact.rows, query
        assert exact.status == "OK" and not exact.incomplete, query
        assert any(n == "COMPLETENESS" and "within 0.82 arcsec" in v for n, v in exact.infos), query
        assert not any(n == "INCOMPLETE" for n, _v in exact.infos), query
    # The reviewer's case: the 3 nearest SIMBAD-or-Gaia-or-NED rows.
    nearest = run(vo_server.execute_adql(cone_adql(select="TOP 3 match_id", extra=" ORDER BY separation"),
                                         service=truncating))
    assert set(nearest.column_values("match_id")) == {"ned:3C 273", "simbad:3C 273", f"gaia_dr3:{GAIA_3C273}"}
    # group_* values may involve rows beyond the complete radius: rows exact, result still flagged.
    grouped = run(vo_server.execute_adql(cone_adql(select="TOP 3 match_id, group_id", extra=" ORDER BY separation"),
                                         service=truncating))
    assert grouped.status == "OVERFLOW" and grouped.truncated and len(grouped.rows) == 3
    # A failed archive has no completeness radius: refused.
    failing = RecordedCrossmatchService(failing=frozenset({"simbad"}))
    with pytest.raises(vo_server.UpstreamError, match="simbad"):
        run(vo_server.execute_adql(cone_adql(select="TOP 1 match_id", extra=" ORDER BY separation"), service=failing))
    # Served as CSV too, since the answer is complete.
    from fastapi.testclient import TestClient

    with TestClient(make_app(truncating)) as http:
        response = http.get("/vo/tap/sync", params=tap_params(cone_adql(select="TOP 3 match_id",
                                                                        extra=" ORDER BY separation"), RESPONSEFORMAT="csv"))
        assert response.status_code == 200 and response.headers["X-VO-Query-Status"] == "OK"
        assert len(response.text.splitlines()) == 4
    # Unaffected archives answer exactly.
    gaia = run(vo_server.execute_adql(cone_adql(select="COUNT(*) AS n", extra=" AND catalog = 'gaia_dr3'"),
                                      service=truncating))
    assert gaia.rows == [(1,)] and gaia.status == "OK"
    scs = run(vo_server.cone_search(truncating, RA_3C273, DEC_3C273, TEN_ARCSEC_DEG, catalogs=["ned", "gaia_dr3"]))
    assert scs.status == "OVERFLOW" and scs.truncated and ("INCOMPLETE", vo_server.incompleteness_summary(
        scs.truncated, {})) in scs.infos


def test_incomplete_results_in_every_output_format() -> None:
    from fastapi.testclient import TestClient

    service = RecordedCrossmatchService(failing=frozenset({"xmm", "chandra", "first"}))
    with TestClient(make_app(service)) as http:
        for fmt in ("csv", "tsv"):  # these formats cannot carry the warning: refuse, naming the archives
            response = http.get("/vo/tap/sync", params=tap_params(cone_adql(), RESPONSEFORMAT=fmt))
            assert response.status_code == 502
            message = votable(response.content).resources[0].infos[0].content
            assert "chandra: CatalogUnavailableError" in message and "first" in message and "xmm" in message
        payload = http.get("/vo/tap/sync", params=tap_params(cone_adql(), RESPONSEFORMAT="json"))
        assert payload.json()["query_status"] == "OVERFLOW" and payload.headers["X-VO-Query-Status"] == "OVERFLOW"
        assert "xmm" in payload.headers["X-VO-Incomplete"]
        assert sum(1 for i in payload.json()["infos"] if i["name"] == "WARNING") == 3
        vot = http.get("/vo/tap/sync", params=tap_params(cone_adql()))
        assert vot.status_code == 200 and vot.headers["X-VO-Query-Status"] == "OVERFLOW"
        assert [i.value for i in votable(vot.content).resources[0].infos if i.name == "QUERY_STATUS"] == ["OK", "OVERFLOW"]
        scs = http.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": TEN_ARCSEC_DEG})
        assert scs.headers["X-VO-Query-Status"] == "OVERFLOW"
        assert "OVERFLOW" in [i.value for i in scs_votable(scs.content).resources[0].infos if i.name == "QUERY_STATUS"]
        # Excluded archives do not matter: a complete CSV is served, with its status header.
        ok = http.get("/vo/tap/sync", params=tap_params(cone_adql(select="match_id", extra=" AND catalog = 'gaia_dr3'"),
                                                        RESPONSEFORMAT="csv"))
        assert ok.status_code == 200 and ok.headers["X-VO-Query-Status"] == "OK"
        assert ok.text.splitlines() == ["match_id", f"gaia_dr3:{GAIA_3C273}"]
    truncating = RecordedCrossmatchService(truncate={"ned": 2})
    with TestClient(make_app(truncating)) as http:
        response = http.get("/vo/tap/sync", params=tap_params(cone_adql(), RESPONSEFORMAT="csv"))
        assert response.status_code == 400 and "ned: Truncated" in votable(response.content).resources[0].infos[0].content


def test_match_rows_falls_back_to_counterparts_without_groups() -> None:
    record = {"counterparts": {"optical": [{"catalog": "gaia_dr3", "source_id": "1", "ra": 1.0, "dec": 2.0,
                                            "separation_arcsec": 0.5, "confidence": 0.9, "metadata": {"wavelength": "optical"}}]}}
    rows = vo_server.match_rows(record)
    assert rows[0]["match_id"] == "gaia_dr3:1" and rows[0]["wavelength"] == "optical"
    assert (rows[0]["group_id"], rows[0]["group_n_catalogs"], rows[0]["group_catalogs"]) == ("group-1", 1, "gaia_dr3")


# Registry wavelength labels of the catalogs used in synthetic rows (compilations are 'multi' etc.).
_WAVELENGTHS = {"gaia_dr3": "optical", "sdss": "optical", "panstarrs_dr2": "optical", "twomass_psc": "infrared",
                "allwise": "infrared", "rosat": "xray", "chandra": "xray", "first": "radio", "simbad": "multi",
                "ned": "extragalactic", "exoplanet_archive": "exoplanet"}
# Catalogs whose rows are single-epoch (apparent) positions (models registry 'single_epoch_positions').
_SINGLE_EPOCH = {"twomass_psc", "sdss", "first", "rosat"}


def _row(catalog: str, source_id: str, dx: float, dy: float, err: float | None, *, ra0: float = 150.0, dec0: float = 20.0,
         epoch: float | None = None, pm: tuple[float, float] | None = None, parallax: float | None = None,
         epoch_range: tuple[float, float] | None = None) -> dict[str, Any]:
    """A matches row offset (dx east, dy north) arcsec from (ra0, dec0)."""
    dec = dec0 + dy / 3600.0
    ra = ra0 + dx / 3600.0 / math.cos(math.radians(dec0))
    return {"match_id": f"{catalog}:{source_id}", "catalog": catalog, "source_id": source_id, "ra": ra, "dec": dec,
            "separation": math.hypot(dx, dy), "pos_error": err, "epoch": epoch,
            "pmra": pm[0] if pm else None, "pmdec": pm[1] if pm else None, "parallax": parallax,
            "wavelength": _WAVELENGTHS.get(catalog), "_single_epoch": catalog in _SINGLE_EPOCH,
            "_epoch_range": list(epoch_range) if epoch_range else None}


def _sets(rows: list[dict[str, Any]]) -> set[frozenset[str]]:
    by_group: dict[str, set[str]] = {}
    for row in rows:
        by_group.setdefault(row["group_id"], set()).add(row["match_id"])
    return {frozenset(v) for v in by_group.values()}


def test_candidate_sets_follow_positional_errors_without_chaining() -> None:
    # Two precise optical sources 2" apart, a ROSAT source (10" error) between them and an
    # unrelated star 3" away: union-find over max(radius, error) chained all four together.
    rows = [_row("gaia_dr3", "a", 0.0, 0.0, 0.001), _row("twomass_psc", "a", 0.05, 0.0, 0.06),
            _row("gaia_dr3", "b", 2.0, 0.0, 0.001), _row("twomass_psc", "b", 2.03, 0.02, 0.06),
            _row("rosat", "x", 0.6, 0.5, 10.0), _row("sdss", "far", 0.0, 3.0, 0.05)]
    vo_server.assign_groups(rows, 150.0, 20.0)
    sets = _sets(rows)
    assert frozenset({"gaia_dr3:a", "twomass_psc:a", "rosat:x"}) in sets  # ROSAT joins the nearest set only
    assert frozenset({"gaia_dr3:b", "twomass_psc:b"}) in sets and frozenset({"sdss:far"}) in sets
    for row in rows:  # at most one detection per catalog; counts consistent
        members = [r for r in rows if r["group_id"] == row["group_id"]]
        assert len({m["catalog"] for m in members}) == len(members) == row["group_n_catalogs"]
        assert row["group_catalogs"] == ",".join(sorted(m["catalog"] for m in members))
    assert rows[0]["group_id"] == "group-1"  # numbered by nearest member separation
    # 5-sigma apart (0.1" floor): separate; 2.5-sigma: together.
    far = [_row("gaia_dr3", "p", 0.0, 0.0, 0.0), _row("sdss", "q", 0.71, 0.0, 0.0)]
    near = [_row("gaia_dr3", "p", 0.0, 0.0, 0.0), _row("sdss", "q", 0.35, 0.0, 0.0)]
    assert len(_sets(vo_server.assign_groups(far, 150.0, 20.0))) == 2
    assert len(_sets(vo_server.assign_groups(near, 150.0, 20.0))) == 1
    # A catalog without errors gets 0.5": 1.5" apart is 2.9 sigma (joined), 2.5" is 4.8 sigma (not).
    assert len(_sets(vo_server.assign_groups([_row("gaia_dr3", "p", 0, 0, 0.001), _row("ned", "q", 1.5, 0, None)],
                                             150.0, 20.0))) == 1
    assert len(_sets(vo_server.assign_groups([_row("gaia_dr3", "p", 0, 0, 0.001), _row("ned", "q", 2.5, 0, None)],
                                             150.0, 20.0))) == 2


def test_candidate_sets_compare_moving_stars_at_a_common_epoch() -> None:
    """Barnard's star: Gaia DR3 (J2016.0, pm -801.551/+10362.394 mas/yr, parallax 546.976 mas) lies 166" from
    its J2000 position. Its 2MASS detection (J2000.405, a single-epoch apparent position) and AllWISE
    detection (mean epoch 2010.5) carry the real annual-parallax displacement, while an unrelated Gaia
    star sits near the 2MASS position."""
    from models import parallax_offset_arcsec

    ra0, dec0 = BARNARD
    pm, plx = (-801.551, 10362.394), 546.976

    def on_track(epoch: float, *, parallax_shift: bool = True) -> tuple[float, float]:
        x, y = pm[0] / 1000 * (epoch - 2000.0), pm[1] / 1000 * (epoch - 2000.0)
        if parallax_shift:
            east, north = parallax_offset_arcsec(ra0, dec0, plx, epoch)
            x, y = x + east, y + north
        return x, y

    tm_epoch, wise_epoch = 2000.405, 2010.5
    tm, wise = on_track(tm_epoch), on_track(wise_epoch)
    linear = on_track(tm_epoch, parallax_shift=False)
    assert math.hypot(tm[0] - linear[0], tm[1] - linear[1]) == pytest.approx(0.32, abs=0.01)  # parallax at J2000.405
    rows = [_row("gaia_dr3", "barnard", *on_track(2016.0, parallax_shift=False), 0.0001, epoch=2016.0, pm=pm,
                 parallax=plx, ra0=ra0, dec0=dec0),
            _row("simbad", "barnard", 0.0, 0.0, 0.0001, epoch=2000.0, pm=pm, parallax=plx, ra0=ra0, dec0=dec0),
            _row("twomass_psc", "barnard", *tm, 0.06, epoch=tm_epoch, ra0=ra0, dec0=dec0),
            _row("allwise", "barnard", *wise, 0.05, epoch=wise_epoch, ra0=ra0, dec0=dec0),
            _row("gaia_dr3", "field", linear[0] + 0.3, linear[1] + 1.0, 0.001, epoch=2016.0, ra0=ra0, dec0=dec0),
            _row("ned", "barnard", 0.0, 0.0, None, ra0=ra0, dec0=dec0)]            # no epoch: taken as J2000
    vo_server.assign_groups(rows, ra0, dec0)
    sets = _sets(rows)
    assert frozenset({"gaia_dr3:barnard", "simbad:barnard", "twomass_psc:barnard", "allwise:barnard",
                      "ned:barnard"}) in sets
    assert frozenset({"gaia_dr3:field"}) in sets


def test_candidate_sets_treat_co_located_entries_of_one_catalog_as_one_detection() -> None:
    """SIMBAD lists a host star and its planets at identical coordinates (float noise only): the star's
    survey detections must not go to whichever planet entry comes first."""
    noise = 1e-10 * 3600  # 1e-10 deg of floating-point noise, in arcsec
    pm = (29.766, -17.976)
    rows = [_row("simbad", "HD 209458b", noise, 0.0, 0.0005, epoch=2000.0, pm=pm, parallax=20.77),
            _row("simbad", "HD 209458", 0.0, 0.0, 0.0005, epoch=2000.0, pm=pm, parallax=20.77),
            _row("exoplanet_archive", "HD 209458 b", 0.45, -0.27, 0.1, epoch=2015.5),
            _row("twomass_psc", "22031077+1853036", 0.02, -0.01, 0.06, epoch=2000.77),
            _row("gaia_dr3", "1779546757669063552", 0.476, -0.288, 0.0001, epoch=2016.0, pm=pm, parallax=20.77)]
    vo_server.assign_groups(rows, 150.0, 20.0)
    assert len(_sets(rows)) == 1 and rows[0]["group_n_catalogs"] == 4
    assert rows[1]["group_catalogs"] == "exoplanet_archive,gaia_dr3,simbad,twomass_psc"
    # Distinct compilation entries whose ERRORS overlap stay distinct detections: a star (1 mas) and an
    # X-ray source 2" away with a 5" error are two SIMBAD objects, not one listed twice.
    pair = [_row("simbad", "star", 0.0, 0.0, 0.001), _row("simbad", "xray", 2.0, 0.0, 5.0)]
    vo_server.assign_groups(pair, 150.0, 20.0)
    assert pair[0]["group_id"] != pair[1]["group_id"]
    # Survey duplicates within their own errors are one detection (Pan-STARRS DR2 at Cyg X-1: two objects
    # 0.134" apart with errors 0.015" and 0.051"); resolved survey sources are not.
    dup = [_row("panstarrs_dr2", "150242995904152737", 0.0, 0.0, 0.015, epoch=2001.24),
           _row("panstarrs_dr2", "150242995901742318", -0.127, -0.043, 0.051, epoch=1998.36),
           _row("twomass_psc", "19582166+3512057", -0.127, -0.043, 0.06, epoch=1998.36),
           _row("panstarrs_dr2", "neighbour", 1.2, 0.0, 0.02, epoch=2012.0)]
    vo_server.assign_groups(dup, 150.0, 20.0)
    assert dup[0]["group_id"] == dup[1]["group_id"] == dup[2]["group_id"] != dup[3]["group_id"]
    assert dup[0]["group_n_catalogs"] == 2
    # Faint survey sources whose large errors overlap are NOT merged beyond GROUP_DUPLICATE_MAX_ARCSEC:
    # two ROSAT sources 30" apart with 15" errors are two resolved detections.
    rosat = [_row("rosat", "a", 0.0, 0.0, 15.0), _row("rosat", "b", 30.0, 0.0, 15.0)]
    vo_server.assign_groups(rosat, 150.0, 20.0)
    assert rosat[0]["group_id"] != rosat[1]["group_id"]
    # Detections with different proper motions are never merged, even at one position.
    movers = [_row("gaia_dr3", "a", 0.0, 0.0, 0.0001, epoch=2016.0, pm=(100.0, 0.0)),
              _row("gaia_dr3", "b", 0.0, 0.0, 0.0001, epoch=2016.0, pm=(-300.0, 50.0))]
    vo_server.assign_groups(movers, 150.0, 20.0)
    assert movers[0]["group_id"] != movers[1]["group_id"]


def test_candidate_sets_place_span_only_rows_on_the_moving_track() -> None:
    """A row without an epoch but with a known span (e.g. a Chandra detection, 1999.5-2022.0) is compared with
    the point of a fast star's track inside that span that passes closest to it."""
    pm = (6491.2, -5708.6)  # Kapteyn's star (mas/yr)
    t = 2012.8
    rows = [_row("simbad", "HD 33793", 0.0, 0.0, 0.0005, epoch=2000.0, pm=pm, parallax=254.2),
            _row("chandra", "2CXO", pm[0] / 1000 * (t - 2000) + 0.2, pm[1] / 1000 * (t - 2000) - 0.1, 0.16,
                 epoch_range=(1999.5, 2022.0)),
            _row("chandra", "late", pm[0] / 1000 * 30, pm[1] / 1000 * 30, 0.16, epoch_range=(1999.5, 2022.0))]
    vo_server.assign_groups(rows, 150.0, 20.0)
    assert rows[0]["group_id"] == rows[1]["group_id"] != rows[2]["group_id"]  # 2030 lies outside the span


# Live archive rows recorded through the real providers (tests/fixtures/vo_server/groups/*.json, see
# their "recorded" key); the expected identities are SIMBAD's own cross-identifications (sim-tap ident).
GROUP_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "vo_server" / "groups"
SIMBAD_IDENT_SETS = {
    "barnard": ("simbad:NAME Barnard's star", {"twomass_psc:17574849+0441405"}),
    "hd209458": ("simbad:HD 209458", {"gaia_dr3:1779546757669063552", "twomass_psc:22031077+1853036"}),
    "kapteyn": ("simbad:HD 33793", {"gaia_dr3:4810594479418041856", "twomass_psc:05114046-4501051"}),
    "cygx1": ("simbad:HD 226868", {"gaia_dr3:2059383668236814720", "twomass_psc:19582166+3512057"}),
    "proxima": ("simbad:NAME Proxima Centauri", {"gaia_dr3:5853498713190525696", "twomass_psc:14294291-6240465",
                                                 "allwise:J142937.35-624038.3"}),
}


def _replayed_rows(name: str) -> dict[str, dict[str, Any]]:
    payload = json.loads((GROUP_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    cone = payload["cone"]
    record = {"target": {"ra": cone["ra"], "dec": cone["dec"]}, "crossmatch_groups": [{"members": payload["members"]}]}
    return {r["match_id"]: r for r in vo_server.match_rows(record, centre=(cone["ra"], cone["dec"]))}


def _float_value(value: Any) -> float | None:
    """A float, or None for NULL/masked table values."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _assert_one_detection_per_catalog(rows: Any) -> None:
    """Every set holds one detection per catalog: rows of one catalog in a set are co-located (identical
    coordinates, or -- for surveys -- within their errors); group_n_catalogs counts the catalogs."""
    from models import haversine_arcsec

    by_group: dict[str, list[Any]] = {}
    for row in rows:
        by_group.setdefault(str(row["group_id"]), []).append(row)
    for group in by_group.values():
        assert int(group[0]["group_n_catalogs"]) == len({str(r["catalog"]) for r in group})
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if str(a["catalog"]) != str(b["catalog"]):
                    continue
                sep = haversine_arcsec(float(a["ra"]), float(a["dec"]), float(b["ra"]), float(b["dec"]))
                limit = vo_server.GROUP_SAME_POSITION_ARCSEC
                errors = [_float_value(a["pos_error"]), _float_value(b["pos_error"])]
                if str(a["wavelength"]) not in vo_server.COMPILATION_WAVELENGTHS and None not in errors:
                    limit = max(limit, min(vo_server.GROUP_K * math.hypot(*errors),  # type: ignore[arg-type]
                                           vo_server.GROUP_DUPLICATE_MAX_ARCSEC))
                assert sep <= limit, (str(a["match_id"]), str(b["match_id"]), sep)


@pytest.mark.parametrize("name", sorted(SIMBAD_IDENT_SETS))
def test_recorded_star_sets_match_simbad_cross_identifications(name: str) -> None:
    """Barnard's star, HD 209458, Kapteyn's star, Cyg X-1 and Proxima Cen: the star's own SIMBAD entry shares
    a set with the Gaia DR3 / 2MASS / AllWISE detections SIMBAD identifies with it -- not with a planet
    entry at the same coordinates, a Pan-STARRS duplicate, or nothing (annual parallax)."""
    rows = _replayed_rows(name)
    star, identified = SIMBAD_IDENT_SETS[name]
    star = next(m for m in rows if " ".join(m.split()) == star)  # SIMBAD keeps "HD  33793" double-spaced
    members = {m for m, r in rows.items() if r["group_id"] == rows[star]["group_id"]}
    assert identified <= members, (identified - members, sorted(members))
    for match_id in members:  # SIMBAD entries in the set (planets) sit at the star's coordinates
        if match_id.startswith("simbad:"):
            assert rows[match_id]["ra"] == pytest.approx(rows[star]["ra"], abs=1e-9)
    _assert_one_detection_per_catalog(rows.values())


def test_recorded_proxima_rows_need_the_parallax() -> None:
    """Proxima's 2MASS row (J2000.194) is 0.709" off the linear track but 0.052" off after removing the
    768 mas parallax (models.propagate_radec docstring): the grouping must model it."""
    from models import parallax_offset_arcsec, propagate_radec, tangent_offset_arcsec

    rows = _replayed_rows("proxima")
    gaia, tm = rows["gaia_dr3:5853498713190525696"], rows["twomass_psc:14294291-6240465"]
    ra, dec = propagate_radec(gaia["ra"], gaia["dec"], gaia["pmra"], gaia["pmdec"], gaia["epoch"], tm["epoch"])
    east, north = tangent_offset_arcsec(ra, dec, tm["ra"], tm["dec"])
    p_east, p_north = parallax_offset_arcsec(ra, dec, gaia["parallax"], tm["epoch"])
    assert math.hypot(east, north) == pytest.approx(0.709, abs=0.01)
    assert math.hypot(east - p_east, north - p_north) == pytest.approx(0.052, abs=0.01)
    assert tm["_single_epoch"] is True and rows["allwise:J142937.35-624038.3"]["_single_epoch"] is False
    # Grouped without the carriers' parallax, both detections fall out of Proxima's set (the reviewed bug).
    stripped = [dict(r, parallax=None) for r in rows.values()]
    vo_server.assign_groups(stripped, *PROXIMA)
    by_id = {r["match_id"]: r for r in stripped}
    proxima_set = by_id["gaia_dr3:5853498713190525696"]["group_id"]
    assert by_id["twomass_psc:14294291-6240465"]["group_id"] != proxima_set
    assert by_id["allwise:J142937.35-624038.3"]["group_id"] != proxima_set


def test_candidate_sets_scale_to_dense_cones() -> None:
    """3000 detections in a 180" cone (omega Cen density) are grouped in well under a second
    (crossmatch._group_matches' O(n^2) pair loop needed ~14 s for this many)."""
    import random

    rng = random.Random(5139)
    rows = []
    for catalog in ("gaia_dr3", "twomass_psc", "panstarrs_dr2"):
        for i in range(1000):
            r, theta = 180.0 * math.sqrt(rng.random()), rng.random() * 2 * math.pi
            rows.append(_row(catalog, str(i), r * math.cos(theta), r * math.sin(theta), 0.02 if catalog != "twomass_psc" else 0.08))
    started = time.perf_counter()
    vo_server.assign_groups(rows, 150.0, 20.0)
    assert time.perf_counter() - started < 1.5
    assert all(r["group_id"] for r in rows) and all(r["group_n_catalogs"] <= 3 for r in rows)


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def test_votable_serialization_validates_with_nulls_unicode_and_overflow() -> None:
    cols = [vo_server.MATCHES_COLUMNS[0], vo_server.MATCHES_COLUMNS[1], vo_server.MATCHES_COLUMNS[16],
            vo_server.MATCHES_COLUMNS[15]]
    result = vo_server.ResultTable("t", cols, [("a:1", 1.5, None, "Ω Cen"), ("b:2", None, 3, None)], "OVERFLOW",
                                   [("WARNING", "x <&> y"), ("WARNING", "second"), ("citation", "a: b")])
    body = vo_server.render_votable(result)
    assert_valid_votable(body)
    vot = votable(body)
    resource = vot.resources[0]
    assert [(i.name, i.value) for i in resource.infos] == [
        ("QUERY_STATUS", "OK"), ("WARNING", "x <&> y"), ("WARNING", "second"), ("citation", "a: b"), ("QUERY_STATUS", "OVERFLOW")]
    table = resource.tables[0]
    assert table.get_field_by_id_or_name("object_type").datatype == "unicodeChar"  # promoted for non-ASCII
    assert table.array["object_type"][0] == "Ω Cen"
    assert bool(table.array.mask["group_n_catalogs"][0]) and table.array["group_n_catalogs"][1] == 3
    assert bool(table.array.mask["ra"][1])
    assert table.get_field_by_id_or_name("ra").unit == "deg"


def test_error_votables() -> None:
    dali = vo_server.error_votable("bad query")
    assert_valid_votable(dali)
    info = votable(dali).resources[0].infos[0]
    assert (info.name, info.value, info.content) == ("QUERY_STATUS", "ERROR", "bad query")
    scs = vo_server.error_votable("Invalid RA", scs=True)
    assert_valid_votable(scs)
    root = votable(scs).infos[0]
    assert (root.ID, root.name, root.value) == ("Error", "Error", "Invalid RA")


def test_csv_tsv_json_serializers() -> None:
    result = adql("SELECT name, epoch, max_rows FROM astrosearch.catalogs WHERE name IN ('gaia_dr3', 'twomass_psc') ORDER BY name")
    csv_text = vo_server.render_csv(result).decode()
    lines = csv_text.splitlines()
    assert lines[0] == "name,epoch,max_rows" and lines[1] == "gaia_dr3,2016.0,2000"
    assert lines[2].startswith("twomass_psc,")
    tsv = vo_server.render_csv(result, delimiter="\t").decode().splitlines()
    assert tsv[0] == "name\tepoch\tmax_rows"
    payload = json.loads(vo_server.render_json(result))
    assert [m["name"] for m in payload["metadata"]] == ["name", "epoch", "max_rows"]
    assert payload["data"][0] == ["gaia_dr3", 2016.0, 2000] and payload["query_status"] == "OK"
    assert vo_server.resolve_format("text/csv; header=present") == ("csv", "text/csv;header=present")
    with pytest.raises(vo_server.VOParameterError, match="Unsupported RESPONSEFORMAT"):
        vo_server.resolve_format("fits")


# ---------------------------------------------------------------------------
# HTTP: Simple Cone Search
# ---------------------------------------------------------------------------


def test_scs_returns_valid_votable_with_required_columns(client: Any, stub: RecordedCrossmatchService) -> None:
    response = client.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": TEN_ARCSEC_DEG})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/xml")
    assert_valid_scs_votable(response.content)
    table = scs_votable(response.content).get_first_table()
    ucds = [f.ucd for f in table.fields]
    # SCS 1.03 section 2.2: exactly one FIELD each with ID_MAIN, POS_EQ_RA_MAIN, POS_EQ_DEC_MAIN.
    for ucd, name in (("ID_MAIN", "match_id"), ("POS_EQ_RA_MAIN", "ra"), ("POS_EQ_DEC_MAIN", "dec")):
        assert ucds.count(ucd) == 1 and table.fields[ucds.index(ucd)].name == name
    assert {f.name: f.ref for f in table.fields if f.ref} == {"ra": "icrs", "dec": "icrs"}  # COOSYS ICRS
    assert [f.name for f in table.fields] == [c.name for c in vo_server.MATCHES_COLUMNS if c.verb <= 2]
    ids = list(table.array["match_id"])
    assert f"gaia_dr3:{GAIA_3C273}" in ids and "simbad:3C 273" in ids
    assert stub.calls[-1]["radius_arcsec"] == 10.0  # SR (deg) * 3600, exactly
    assert all(sep <= 10.0 for sep in table.array["separation"])


def test_scs_parameter_names_are_case_insensitive_and_verb_selects_columns(client: Any) -> None:
    v1 = scs_votable(client.get("/vo/scs", params={"ra": RA_3C273, "dec": DEC_3C273, "sr": 0.001, "verb": 1}).content)
    assert [f.name for f in v1.get_first_table().fields] == ["match_id", "ra", "dec"]
    v3 = scs_votable(client.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": 0.001, "VERB": 3}).content)
    assert len(v3.get_first_table().fields) == len(vo_server.MATCHES_COLUMNS)


def test_scs_catalog_selection_and_zero_radius(client: Any, stub: RecordedCrossmatchService) -> None:
    body = client.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": TEN_ARCSEC_DEG,
                                         "CATALOGS": "nvss,first"}).content
    assert set(scs_votable(body).get_first_table().array["catalog"]) == {"nvss", "first"}
    assert stub.calls[-1]["catalogs"] == ["nvss", "first"]
    calls = len(stub.calls)
    empty = client.get("/vo/scs", params={"RA": 10, "DEC": 10, "SR": 0}).content
    assert_valid_scs_votable(empty)
    table = scs_votable(empty).get_first_table()
    assert len(table.array) == 0 and len(table.fields) == 10 and len(stub.calls) == calls
    # CATALOGS that names no catalog is an error, never a silent empty "OK" (pyvo reads only INFO Error).
    for value in (",", ",,", " , "):
        response = client.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": 0.0027, "CATALOGS": value})
        info = votable(response.content).infos[0]
        assert info.name == "Error" and "CATALOGS names no catalog" in info.value, value
        assert response.headers["X-VO-Query-Status"] == "ERROR" and len(stub.calls) == calls
    # An empty value means "not given": every enabled catalog is queried.
    client.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": 0.0027, "CATALOGS": ""})
    assert len(stub.calls) == calls + 1 and stub.calls[-1]["catalogs"] is None


@pytest.mark.parametrize(("params", "message"), [
    ({"DEC": 2.0, "SR": 0.01}, "Missing required parameter RA"),
    ({"RA": "abc", "DEC": 2.0, "SR": 0.01}, "Invalid RA value"),
    ({"RA": 10, "DEC": 91, "SR": 0.01}, "DEC must be within"),
    ({"RA": 400, "DEC": 1, "SR": 0.01}, "RA must be within"),
    ({"RA": 10, "DEC": 1, "SR": -1}, "radius must be positive"),
    ({"RA": 10, "DEC": 1, "SR": 5}, "exceeds this service's maximum"),
    ({"RA": 10, "DEC": 1, "SR": 0.01, "VERB": 7}, "VERB must be 1, 2 or 3"),
    ({"RA": 10, "DEC": 1, "SR": 0.01, "CATALOGS": "gaia_dr3,bogus"}, "Unknown or disabled catalog"),
])
def test_scs_errors_follow_the_spec(client: Any, params: dict[str, Any], message: str) -> None:
    response = client.get("/vo/scs", params=params)
    assert response.status_code == 200  # SCS 1.03 reports errors inside the VOTable
    assert_valid_votable(response.content)
    info = votable(response.content).infos[0]
    assert info.name == "Error" and message in info.value


def test_scs_total_upstream_failure_is_an_error_votable() -> None:
    from fastapi.testclient import TestClient

    dead = RecordedCrossmatchService(failing=frozenset(load_recorded_sources()[2]))
    with TestClient(make_app(dead)) as dead_client:
        body = dead_client.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": 0.001}).content
    info = votable(body).infos[0]
    assert info.name == "Error" and "upstream archives failed" in info.value and "HTTP 503" in info.value


# ---------------------------------------------------------------------------
# HTTP: TAP sync
# ---------------------------------------------------------------------------


def tap_params(query: str, **extra: Any) -> dict[str, Any]:
    return {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query, **extra}


def test_tap_sync_get_and_post_votable(client: Any) -> None:
    get = client.get("/vo/tap/sync", params=tap_params(cone_adql(extra=" ORDER BY separation")))
    assert get.status_code == 200 and get.headers["content-type"].startswith(vo_server.VOTABLE_MEDIA_TYPE)
    assert_valid_votable(get.content)
    table = votable(get.content).get_first_table()
    assert table.array["match_id"][0] in {"ned:3C 273", "simbad:3C 273", f"gaia_dr3:{GAIA_3C273}"}
    post = client.post("/vo/tap/sync", data={"request": "doQuery", "lang": "adql",
                                             "query": "SELECT name FROM astrosearch.catalogs ORDER BY name"})
    assert post.status_code == 200
    assert list(votable(post.content).get_first_table().array["name"]) == sorted(CatalogRegistry().catalogs)


def test_tap_sync_formats_and_overflow(client: Any) -> None:
    query = "SELECT name, max_rows FROM astrosearch.catalogs ORDER BY name"
    csv_resp = client.get("/vo/tap/sync", params=tap_params(query, RESPONSEFORMAT="csv"))
    assert csv_resp.headers["content-type"].startswith("text/csv") and csv_resp.text.splitlines()[0] == "name,max_rows"
    assert csv_resp.headers["X-VO-Query-Status"] == "OK"
    capped_csv = client.get("/vo/tap/sync", params=tap_params(query, RESPONSEFORMAT="csv", MAXREC=2))
    assert capped_csv.headers["X-VO-Query-Status"] == "OVERFLOW" and len(capped_csv.text.splitlines()) == 3
    tsv_resp = client.get("/vo/tap/sync", params=tap_params(query, FORMAT="text/tab-separated-values"))
    assert tsv_resp.text.splitlines()[0] == "name\tmax_rows"
    json_resp = client.get("/vo/tap/sync", params=tap_params(query, RESPONSEFORMAT="application/json", MAXREC=2))
    payload = json_resp.json()
    assert payload["query_status"] == "OVERFLOW" and len(payload["data"]) == 2
    vot = client.get("/vo/tap/sync", params=tap_params(query, MAXREC=2)).content
    assert_valid_votable(vot)
    assert [i.value for i in votable(vot).resources[0].infos if i.name == "QUERY_STATUS"] == ["OK", "OVERFLOW"]


@pytest.mark.parametrize(("params", "status", "message"), [
    ({"LANG": "ADQL"}, 400, "Missing required parameter QUERY"),
    ({"QUERY": "SELECT * FROM astrosearch.catalogs"}, 400, "Missing required parameter LANG"),
    ({"LANG": "PQL", "QUERY": "SELECT * FROM astrosearch.catalogs"}, 400, "Unsupported LANG"),
    ({"REQUEST": "getCapabilities", "LANG": "ADQL", "QUERY": "SELECT 1"}, 400, "only REQUEST=doQuery"),
    ({"LANG": "ADQL", "QUERY": "SELECT * FROM astrosearch.catalogs", "MAXREC": "-3"}, 400, "MAXREC must be"),
    ({"LANG": "ADQL", "QUERY": "SELECT * FROM astrosearch.catalogs", "RESPONSEFORMAT": "fits"}, 400, "Unsupported RESPONSEFORMAT"),
    ({"LANG": "ADQL", "QUERY": "SELECT * FROM astrosearch.catalogs", "UPLOAD": "t,param:t"}, 400, "UPLOAD is not supported"),
    ({"LANG": "ADQL", "QUERY": "SELECT * FROM astrosearch.matches"}, 400, "must constrain position"),
])
def test_tap_sync_errors_are_dali_error_documents(client: Any, params: dict[str, Any], status: int, message: str) -> None:
    response = client.get("/vo/tap/sync", params=params)
    assert response.status_code == status and response.headers["X-VO-Query-Status"] == "ERROR"
    assert_valid_votable(response.content)
    info = votable(response.content).resources[0].infos[0]
    assert info.name == "QUERY_STATUS" and info.value == "ERROR" and message in info.content


def test_tap_sync_upstream_failure_is_502() -> None:
    from fastapi.testclient import TestClient

    dead = RecordedCrossmatchService(failing=frozenset(load_recorded_sources()[2]))
    with TestClient(make_app(dead)) as dead_client:
        response = dead_client.get("/vo/tap/sync", params=tap_params(cone_adql()))
    assert response.status_code == 502
    assert "upstream archives failed" in votable(response.content).resources[0].infos[0].content


def test_tap_sync_multipart_post_and_upload_rejection(client: Any) -> None:
    files = {"REQUEST": (None, "doQuery"), "LANG": (None, "ADQL"),
             "QUERY": (None, "SELECT name FROM astrosearch.catalogs WHERE name = 'gaia_dr3'")}
    ok = client.post("/vo/tap/sync", files=files)
    assert ok.status_code == 200 and list(votable(ok.content).get_first_table().array["name"]) == ["gaia_dr3"]
    upload = client.post("/vo/tap/sync", files={**files, "t": ("t.xml", b"<VOTABLE/>", "application/x-votable+xml")})
    assert upload.status_code == 400 and "UPLOAD is not supported" in upload.text


# ---------------------------------------------------------------------------
# HTTP: VOSI documents
# ---------------------------------------------------------------------------


def test_vosi_capabilities_availability_tables_parse_with_pyvo(client: Any) -> None:
    from pyvo.io import vosi
    from pyvo.io.vosi import tapregext as tr

    caps = vosi.parse_capabilities(io.BytesIO(client.get("/vo/tap/capabilities").content))
    tap = next(c for c in caps if isinstance(c, tr.TableAccess))
    assert tap.interfaces[0].accessurls[0].content == "http://testserver/vo/tap"
    assert tap.outputlimit.default.content == 10000 and tap.outputlimit.hard.content == 100000
    assert {v.content for v in tap.languages[0].versions} == {"2.0", "2.1"}
    features = {f.form for lf in tap.languages[0].languagefeaturelists for f in lf}
    assert {"POINT", "CIRCLE", "CONTAINS", "DISTANCE", "OFFSET"} <= features
    # VOSI 1.1 (REC 2017-05-24, section 3): the 1.1 tables endpoint is ivo://ivoa.net/std/VOSI#tables-1.1.
    assert {c.standardid for c in caps} >= {"ivo://ivoa.net/std/TAP", "ivo://ivoa.net/std/VOSI#tables-1.1",
                                              "ivo://ivoa.net/std/VOSI#availability", "ivo://ivoa.net/std/VOSI#capabilities"}
    assert "ivo://ivoa.net/std/VOSI#tables" not in {c.standardid for c in caps}
    assert "rows nearest the cone centre, up to min(2000, 3000 / number of archives queried)" in \
        tap.languages[0].description
    avail = vosi.parse_availability(io.BytesIO(client.get("/vo/tap/availability").content))
    assert avail.available is True and avail.upsince is not None
    full = vosi.parse_tables(io.BytesIO(client.get("/vo/tap/tables").content))
    matches = full.get_table_by_name("astrosearch.matches")
    ra = next(c for c in matches.columns if c.name == "ra")
    assert (ra.ucd, ra.unit, ra.datatype.content) == ("pos.eq.ra;meta.main", "deg", "double")
    minimal = vosi.parse_tables(io.BytesIO(client.get("/vo/tap/tables", params={"detail": "min"}).content))
    assert not minimal.get_table_by_name("astrosearch.matches").columns
    single = vosi.parse_tables(io.BytesIO(client.get("/vo/tap/tables/TAP_SCHEMA.columns").content)).get_first_table()
    assert single.name == "TAP_SCHEMA.columns" and len(single.columns) == len(vo_server._TS_COLUMNS)
    assert client.get("/vo/tap/tables/nope.nothing").status_code == 404
    examples = client.get("/vo/tap/examples")
    assert examples.headers["content-type"].startswith("application/xhtml+xml") and 'property="query"' in examples.text


def test_tap_examples_all_execute(client: Any) -> None:
    for ident, _title, query in vo_server.EXAMPLES:
        response = client.get("/vo/tap/sync", params=tap_params(query))
        assert response.status_code == 200, (ident, response.text)
        assert_valid_votable(response.content)


# ---------------------------------------------------------------------------
# Official IVOA XML schemas (VOTable 1.4, VOSI, TAPRegExt, VODataService, UWS 1.1)
# ---------------------------------------------------------------------------

XSD_DIR = Path(__file__).resolve().parent / "fixtures" / "vo_server" / "xsd"


@pytest.fixture(scope="module")
def ivoa_schema():
    from lxml import etree

    return etree.XMLSchema(etree.parse(str(XSD_DIR / "ivoa_all.xsd")))


def assert_schema_valid(schema: Any, body: bytes) -> None:
    from lxml import etree

    doc = etree.ElementTree(etree.fromstring(body))
    assert schema.validate(doc), "\n".join(f"{e.line}: {e.message}" for e in schema.error_log)


def test_ivoa_schema_rejects_invalid_documents(ivoa_schema: Any) -> None:
    from lxml import etree

    bad = vo_server.capabilities_xml("http://x/vo/tap").replace(b"<language>", b"<bogus/><language>")
    assert not ivoa_schema.validate(etree.ElementTree(etree.fromstring(bad)))


def test_every_document_validates_against_the_ivoa_schemas(client: Any, ivoa_schema: Any) -> None:
    documents = {
        "capabilities": client.get("/vo/tap/capabilities").content,
        "availability": client.get("/vo/tap/availability").content,
        "tableset": client.get("/vo/tap/tables").content,
        "tableset-min": client.get("/vo/tap/tables", params={"detail": "min"}).content,
        "table": client.get("/vo/tap/tables/astrosearch.matches").content,
        "scs": client.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": TEN_ARCSEC_DEG, "VERB": 3}).content,
        "scs-error": client.get("/vo/scs", params={"RA": 1, "DEC": 99, "SR": 0.01}).content,
        "tap-overflow": client.get("/vo/tap/sync", params=tap_params(cone_adql(), MAXREC=3)).content,
        "tap-error": client.get("/vo/tap/sync", params=tap_params("SELECT * FROM nowhere")).content,
        "tap-aggregate": client.get("/vo/tap/sync", params=tap_params(cone_adql(select="COUNT(*), AVG(separation)"))).content,
    }
    done = client.post("/vo/tap/async", data={**tap_params("SELECT name FROM astrosearch.catalogs"), "PHASE": "RUN"},
                       follow_redirects=False).headers["location"]
    failed = client.post("/vo/tap/async", data={**tap_params("SELECT x FROM y"), "PHASE": "RUN", "RUNID": "a&b"},
                         follow_redirects=False).headers["location"]
    pending = client.post("/vo/tap/async", data=tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"),
                          follow_redirects=False).headers["location"]
    for url in (done, failed):
        deadline = time.monotonic() + 20
        while uws_phase(client.get(url, params={"WAIT": 5}).content) not in {"COMPLETED", "ERROR"}:
            assert time.monotonic() < deadline
    documents.update({
        "job-completed": client.get(done).content,
        "job-error": client.get(failed).content,
        "job-pending": client.get(pending).content,
        "job-list": client.get("/vo/tap/async").content,
        "job-result": client.get(done + "/results/result").content,
        "job-error-document": client.get(failed + "/error").content,
    })
    assert uws_phase(documents["job-completed"]) == "COMPLETED" and uws_phase(documents["job-error"]) == "ERROR"
    for name, body in documents.items():
        try:
            assert_schema_valid(ivoa_schema, body)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from None


# ---------------------------------------------------------------------------
# HTTP: UWS asynchronous jobs
# ---------------------------------------------------------------------------


def uws_phase(body: bytes) -> str:
    from pyvo.io import uws

    return uws.parse_job(io.BytesIO(body)).phase


def test_uws_async_job_lifecycle(client: Any) -> None:
    from pyvo.io import uws

    created = client.post("/vo/tap/async", data=tap_params(cone_adql(), RUNID="run-1"), follow_redirects=False)
    assert created.status_code == 303
    job_url = created.headers["location"]
    job = uws.parse_job(io.BytesIO(client.get(job_url).content))
    assert job.phase == "PENDING" and job.runid == "run-1"
    assert {p.id_: p.content for p in job.parameters}["query"] == cone_adql()
    assert client.get(job_url + "/phase").text == "PENDING"
    run_resp = client.post(job_url + "/phase", data={"PHASE": "RUN"}, follow_redirects=False)
    assert run_resp.status_code == 303
    deadline = time.monotonic() + 20
    while uws_phase(client.get(job_url, params={"WAIT": 5}).content) not in {"COMPLETED", "ERROR"}:
        assert time.monotonic() < deadline
    done = uws.parse_job(io.BytesIO(client.get(job_url).content))
    assert done.phase == "COMPLETED" and done.starttime is not None and done.endtime is not None
    result_url = done.results[0].href
    assert result_url.endswith("/results/result")
    result = client.get(result_url)
    assert_valid_votable(result.content)
    assert f"gaia_dr3:{GAIA_3C273}" in list(votable(result.content).get_first_table().array["match_id"])
    listing = uws.parse_job_list(io.BytesIO(client.get("/vo/tap/async").content))
    assert any(ref.jobid == job_url.rsplit("/", 1)[1] for ref in listing)
    assert client.delete(job_url, follow_redirects=False).status_code == 303
    assert client.get(job_url).status_code == 404


def test_uws_error_abort_and_parameter_updates(client: Any) -> None:
    from pyvo.io import uws

    bad = client.post("/vo/tap/async", data={**tap_params("SELECT * FROM astrosearch.matches"), "PHASE": "RUN"},
                      follow_redirects=False).headers["location"]
    deadline = time.monotonic() + 20
    while uws_phase(client.get(bad, params={"WAIT": 5}).content) != "ERROR":
        assert time.monotonic() < deadline
    job = uws.parse_job(io.BytesIO(client.get(bad).content))
    assert "must constrain position" in job.errorsummary.message.content
    err = client.get(bad + "/error")
    assert "must constrain position" in votable(err.content).resources[0].infos[0].content
    assert client.get(bad + "/results/result").status_code == 404

    pending = client.post("/vo/tap/async", data=tap_params("SELECT name FROM astrosearch.catalogs"),
                          follow_redirects=False).headers["location"]
    client.post(pending + "/parameters", data={"QUERY": "SELECT COUNT(*) AS n FROM astrosearch.catalogs"})
    assert "COUNT(*)" in client.get(pending + "/parameters").text
    client.post(pending + "/executionduration", data={"EXECUTIONDURATION": "30"})
    assert client.get(pending + "/executionduration").text == "30"
    client.post(pending + "/phase", data={"PHASE": "ABORT"})
    assert client.get(pending + "/phase").text == "ABORTED"
    assert client.post(pending + "/phase", data={"PHASE": "RUN"}).status_code == 400
    assert client.post(pending + "/phase", data={"PHASE": "FLY"}).status_code == 400
    assert client.post(pending, data={"ACTION": "DELETE"}, follow_redirects=False).status_code == 303
    assert client.get(pending + "/phase").status_code == 404
    assert client.get("/vo/tap/async/doesnotexist").status_code == 404


def _wait_for_phase(http: Any, url: str, phases: set[str], timeout: float = 20.0) -> str:
    deadline = time.monotonic() + timeout
    while (phase := http.get(url + "/phase").text) not in phases:
        assert time.monotonic() < deadline, phase
        time.sleep(0.02)
    return phase


def test_uws_wait_abort_and_execution_duration_on_running_jobs() -> None:
    from fastapi.testclient import TestClient

    slow = RecordedCrossmatchService(delay_s=1.5)
    with TestClient(make_app(slow)) as http:
        # WAIT blocks while the job is EXECUTING and returns as soon as it finishes.
        url = http.post("/vo/tap/async", data={**tap_params(cone_adql()), "PHASE": "RUN"},
                        follow_redirects=False).headers["location"]
        _wait_for_phase(http, url, {"EXECUTING"})
        started = time.monotonic()
        assert uws_phase(http.get(url).content) == "EXECUTING"  # without WAIT: immediate
        assert time.monotonic() - started < 0.5
        body = http.get(url, params={"WAIT": 30}).content
        elapsed = time.monotonic() - started
        assert uws_phase(body) == "COMPLETED" and 0.3 < elapsed < 10
        assert slow.completed == 1
        # ABORT of an EXECUTING job cancels its task: the crossmatch never completes.
        url = http.post("/vo/tap/async", data={**tap_params(cone_adql()), "PHASE": "RUN"},
                        follow_redirects=False).headers["location"]
        _wait_for_phase(http, url, {"EXECUTING"})
        assert http.post(url + "/phase", data={"PHASE": "ABORT"}, follow_redirects=False).status_code == 303
        assert http.get(url + "/phase").text == "ABORTED"
        time.sleep(2.0)  # longer than the 1.5 s the crossmatch would take
        assert http.get(url + "/phase").text == "ABORTED" and slow.completed == 1
        assert http.get(url + "/results/result").status_code == 404
        # EXECUTIONDURATION: a job running longer than its limit ends in ERROR.
        url = http.post("/vo/tap/async", data=tap_params(cone_adql()), follow_redirects=False).headers["location"]
        http.post(url + "/executionduration", data={"EXECUTIONDURATION": "1"})
        assert http.get(url + "/executionduration").text == "1"
        http.post(url + "/phase", data={"PHASE": "RUN"})
        assert _wait_for_phase(http, url, {"COMPLETED", "ERROR", "ABORTED"}) == "ERROR"
        assert "Execution duration of 1 s exceeded" in http.get(url + "/error").text
        assert slow.completed == 1


def test_uws_destruction_updates_and_expiry(client: Any) -> None:
    from datetime import UTC, datetime, timedelta

    url = client.post("/vo/tap/async", data=tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"),
                      follow_redirects=False).headers["location"]
    soon = (datetime.now(UTC) + timedelta(hours=1)).replace(microsecond=0)
    assert client.post(url + "/destruction", data={"DESTRUCTION": soon.strftime("%Y-%m-%dT%H:%M:%SZ")},
                       follow_redirects=False).status_code == 303
    assert client.get(url + "/destruction").text == soon.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    from pyvo.io import uws

    parsed_destruction = uws.parse_job(io.BytesIO(client.get(url).content)).destruction  # astropy Time (UTC)
    assert parsed_destruction.datetime == soon.replace(tzinfo=None)
    far = datetime.now(UTC) + timedelta(days=3650)
    client.post(url + "/destruction", data={"DESTRUCTION": far.isoformat()})
    retention = vo_server.VOConfig.from_env().retention_s
    capped = datetime.fromisoformat(client.get(url + "/destruction").text)
    assert capped < datetime.now(UTC) + timedelta(seconds=retention + 5)  # never beyond the retention period
    assert client.post(url + "/destruction", data={"DESTRUCTION": "yesterday"}).status_code == 400
    past = datetime.now(UTC) - timedelta(seconds=1)
    client.post(url + "/destruction", data={"DESTRUCTION": past.isoformat()})
    assert client.get(url).status_code == 404  # destroyed once its destruction time has passed


def test_xml_illegal_characters_never_break_documents(client: Any) -> None:
    from pyvo.io import uws

    bad = client.post("/vo/tap/async", data={**tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"), "RUNID": "a\x01b"},
                      follow_redirects=False)
    assert bad.status_code == 400 and "XML 1.0" in bad.text
    url = client.post("/vo/tap/async", data=tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"),
                      follow_redirects=False).headers["location"]
    update = client.post(url + "/parameters", data={"QUERY": "SELECT 'x\x02' FROM TAP_SCHEMA.schemas"})
    assert update.status_code == 400
    job = uws.parse_job(io.BytesIO(client.get(url).content))  # still well-formed
    assert {p.id_: p.content for p in job.parameters}["query"] == "SELECT 1 FROM TAP_SCHEMA.schemas"
    # A synchronous query may carry such characters in a string literal: they are replaced by U+FFFD.
    sync = client.get("/vo/tap/sync", params=tap_params("SELECT 'a\x01b' AS s FROM TAP_SCHEMA.schemas", RUNID="r\x03"))
    assert sync.status_code == 200
    assert_valid_votable(sync.content)
    table = votable(sync.content).get_first_table()
    assert list(table.array["s"]) == ["a�b", "a�b"]
    job = vo_server.UWSJob("id", {"QUERY": "x\x00y"}, datetime_now(), datetime_now(), 1, run_id="r\x1f",
                           phase="ERROR", error_message="bad \x07 thing")
    parsed = uws.parse_job(io.BytesIO(vo_server.job_xml(job, "http://x/vo/tap/async/id")))
    assert parsed.runid == "r�" and "�" in parsed.errorsummary.message.content


def datetime_now():
    from datetime import UTC, datetime

    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# pyvo against a real uvicorn server
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def serve(app: Any) -> Iterator[str]:
    import uvicorn

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            raise RuntimeError("uvicorn did not start")
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)


@pytest.fixture(scope="module")
def pyvo_server() -> Iterator[tuple[str, RecordedCrossmatchService]]:
    service = RecordedCrossmatchService()
    with serve(make_app(service)) as base:
        yield base, service


def test_pyvo_cone_search(pyvo_server: tuple[str, RecordedCrossmatchService]) -> None:
    import pyvo
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    base, _ = pyvo_server
    scs = pyvo.dal.SCSService(f"{base}/vo/scs")
    results = scs.search(pos=SkyCoord(RA_3C273, DEC_3C273, unit="deg"), radius=10 * u.arcsec)
    assert len(results) >= 20
    # pyvo's standard SCS accessors (SCSRecord.id / .pos look up ID_MAIN and POS_EQ_*_MAIN).
    assert results.fieldname_with_ucd("ID_MAIN") == "match_id"
    assert results.fieldname_with_ucd("POS_EQ_RA_MAIN") == "ra" and results.fieldname_with_ucd("POS_EQ_DEC_MAIN") == "dec"
    record = next(r for r in results if r.id == f"gaia_dr3:{GAIA_3C273}")
    assert record.pos.separation(SkyCoord(RA_3C273, DEC_3C273, unit="deg")).arcsec < 0.01
    assert results[0].id == "ned:3C 273" and results[0].pos.separation(
        SkyCoord(RA_3C273, DEC_3C273, unit="deg")).arcsec < 0.01  # nearest first
    assert all(r.pos is not None and r.id for r in results)
    table = results.to_table()
    assert table["separation"].unit == u.arcsec and float(table["separation"].max()) <= 10.0
    with pytest.raises(pyvo.dal.DALQueryError, match="exceeds this service's maximum"):
        scs.search(pos=(10.0, 5.0), radius=1.0)


def test_pyvo_tap_sync_async_and_metadata(pyvo_server: tuple[str, RecordedCrossmatchService]) -> None:
    import pyvo

    base, _ = pyvo_server
    tap = pyvo.dal.TAPService(f"{base}/vo/tap")
    assert tap.get_tap_capability().languages[0].name == "ADQL"
    assert tap.maxrec == 10000 and tap.hardlimit == 100000
    assert {"astrosearch.matches", "astrosearch.catalogs", "TAP_SCHEMA.columns"} <= set(tap.tables.keys())
    matches = tap.tables["astrosearch.matches"]  # fetched lazily from tables/<name> (detail=min listing)
    assert {c.name for c in matches.columns} == {c.name for c in vo_server.MATCHES_COLUMNS}
    result = tap.run_sync(cone_adql(select="match_id, catalog, separation, redshift", extra=" AND catalog = 'ned' "
                                                                                            "ORDER BY separation"))
    rows = result.to_table()
    assert rows["match_id"][0] == "ned:3C 273" and abs(float(rows["redshift"][0]) - 0.158339) < 1e-5
    counted = tap.run_async("SELECT COUNT(*) AS n FROM astrosearch.catalogs WHERE enabled = 1")
    assert int(counted.to_table()["n"][0]) == len(CatalogRegistry().enabled_catalogs())
    with pytest.raises(pyvo.dal.DALQueryError, match="Unknown column"):
        tap.run_sync("SELECT nothing FROM astrosearch.catalogs")
    capped = tap.run_sync("SELECT name FROM astrosearch.catalogs", maxrec=2)
    assert len(capped) == 2
    assert [i.value for i in capped.votable.resources[0].infos if i.name == "QUERY_STATUS"] == ["OK", "OVERFLOW"]
    assert len(tap.examples) == len(vo_server.EXAMPLES)


def test_pyvo_sees_incomplete_results() -> None:
    """Through pyvo, a truncated archive gives an overflow warning (rows) or an error (COUNT)."""
    import pyvo
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    truncating = RecordedCrossmatchService(truncate={"ned": 2})
    with serve(make_app(truncating)) as base:
        tap = pyvo.dal.TAPService(f"{base}/vo/tap")
        with pytest.warns(pyvo.dal.exceptions.DALOverflowWarning):
            rows = tap.run_sync(cone_adql(select="match_id", extra=" AND catalog = 'ned'"))
        assert len(rows) == 2
        with pytest.raises(pyvo.dal.DALQueryError, match="ned: Truncated"):
            tap.run_sync(cone_adql(select="COUNT(*) AS n", extra=" AND catalog = 'ned'"))
        # SCS 1.03 has no overflow indicator and pyvo's SCS client reads only INFO name="Error";
        # the DALI QUERY_STATUS and the WARNING INFOs are still in the document for clients.
        scs = pyvo.dal.SCSService(f"{base}/vo/scs")
        cone = scs.search(pos=SkyCoord(RA_3C273, DEC_3C273, unit="deg"), radius=10 * u.arcsec)
        assert sum(1 for r in cone if str(r["catalog"]) == "ned") == 2
        infos = [(i.name, i.value) for i in cone.votable.resources[0].infos]
        assert ("QUERY_STATUS", "OVERFLOW") in infos
        assert any(name == "WARNING" and value.startswith("ned truncated:") for name, value in infos)


# ---------------------------------------------------------------------------
# Real CrossmatchService: per-archive row limit, truncation, replayed archives
# ---------------------------------------------------------------------------

# Real archive responses for 3C 273 recorded with the VO row limit (TOP 2001 ...):
#   .venv/Scripts/python.exe tests/test_vo_server.py record-vo
VO_FIXTURES = "vo_server/3c273_vo"
VO_REPLAY_CATALOGS = ["gaia_dr3", "simbad", "ned", "first", "chandra"]


class CountingProvider:
    """Provider double for crossmatch.CrossmatchService: returns ``rows`` synthetic sources per
    catalog, spread from the cone centre outwards (like an archive's nearest-first answer)."""

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.seen_max_rows: dict[str, int] = {}

    async def query(self, catalog: Any, target: Any, radius_arcsec: float) -> Any:
        from providers import QueryResult

        self.seen_max_rows[catalog.name] = catalog.max_rows
        count = min(self.rows, catalog.max_rows + 1)  # like TOP max_rows+1
        sources = [CatalogSource(catalog.name, f"s{i}", target.ra, target.dec + (i + 1) * 0.001 / 3600, 0.1, {},
                                 {"wavelength": catalog.wavelength}, {})
                   for i in range(count)]
        return QueryResult(sources, {"max_rows": catalog.max_rows, "row_limit": catalog.max_rows,
                                     "archive_truncated": self.rows > catalog.max_rows})


def _counting_service(rows: int, *, registry_max_rows: int | None = None):
    """CrossmatchService over :class:`CountingProvider`; ``registry_max_rows`` replaces the
    registry's interactive per-catalog ``max_rows`` (200)."""
    from dataclasses import replace as dc_replace

    from crossmatch import CrossmatchService

    provider = CountingProvider(rows)
    registry = CatalogRegistry()
    if registry_max_rows is not None:
        registry._catalogs = {n: dc_replace(c, max_rows=registry_max_rows) for n, c in registry.catalogs.items()}
    providers = {name: provider for name in {c.provider for c in registry.catalogs.values()}}
    return CrossmatchService(registry, providers), provider


def test_vo_queries_raise_the_per_archive_row_limit() -> None:
    service, provider = _counting_service(rows=500)
    assert service.registry.get("gaia_dr3").max_rows == 200  # interactive default
    config = vo_server.VOConfig(catalog_row_limit=1000)
    result = run(vo_server.execute_adql(cone_adql(select="COUNT(*) AS n", extra=" AND catalog = 'gaia_dr3'"),
                                        service=service, config=config))
    assert provider.seen_max_rows == {"gaia_dr3": 1000}
    assert result.rows == [(500,)] and result.status == "OK"  # all 500 rows, not the nearest 200
    assert service.registry.get("gaia_dr3").max_rows == 200  # the shared registry is untouched
    # Services that are not CrossmatchService instances are used as they are.
    stub = RecordedCrossmatchService()
    assert vo_server.service_with_row_limit(stub, 1000) is stub


def test_real_crossmatch_truncation_is_detected() -> None:
    """CrossmatchService reports an archive holding more than max_rows rows as truncated."""
    service, provider = _counting_service(rows=50)
    config = vo_server.VOConfig(catalog_row_limit=20)
    rows = run(vo_server.execute_adql(cone_adql(select="match_id, separation", extra=" AND catalog = 'gaia_dr3'"),
                                      service=service, config=config))
    assert provider.seen_max_rows == {"gaia_dr3": 200}  # a limit below the registry's never lowers it
    assert rows.status == "OK" and len(rows.rows) == 50
    config = vo_server.VOConfig(catalog_row_limit=20)
    service, provider = _counting_service(rows=500)
    rows = run(vo_server.execute_adql(cone_adql(select="match_id, separation", extra=" AND catalog = 'gaia_dr3'"),
                                      service=service, config=config))
    assert rows.status == "OVERFLOW" and len(rows.rows) == 200 and set(rows.truncated) == {"gaia_dr3"}
    reach = max(rows.column_values("separation"))
    assert reach == pytest.approx(0.2, abs=1e-6)  # the 200 nearest rows reach 200 x 0.001"
    assert f"complete only to {reach:.2f} arcsec" in rows.truncated["gaia_dr3"]
    with pytest.raises(vo_server.IncompleteResultError, match="gaia_dr3: Truncated: only the 200 nearest rows"):
        run(vo_server.execute_adql(cone_adql(select="COUNT(*) AS n", extra=" AND catalog = 'gaia_dr3'"),
                                   service=service, config=config))


def test_scs_through_real_crossmatch_service_with_replayed_archives() -> None:
    import respx
    from fastapi.testclient import TestClient
    from fixture_io import load_exchanges, replay_side_effect

    app = make_app_without_service()
    exchanges = load_exchanges(VO_FIXTURES, VO_REPLAY_CATALOGS)
    assert {e.catalog for e in exchanges} == set(VO_REPLAY_CATALOGS)
    with respx.mock(assert_all_called=False) as router:
        router.route(host="testserver").pass_through()
        router.route().mock(side_effect=replay_side_effect(exchanges))  # strict: the exact recorded requests
        with TestClient(app) as http:
            body = http.get("/vo/scs", params={"RA": RA_3C273, "DEC": DEC_3C273, "SR": TEN_ARCSEC_DEG, "VERB": 3,
                                               "CATALOGS": ",".join(VO_REPLAY_CATALOGS)}).content
    assert_valid_scs_votable(body)
    vot = scs_votable(body)
    assert not [i for i in vot.resources[0].infos if i.name == "WARNING"]
    assert [i.value for i in vot.resources[0].infos if i.name == "QUERY_STATUS"] == ["OK"]
    rows = {r["match_id"]: r for r in vot.get_first_table().to_table()}
    # 3C 273 in the five archives is one candidate set.
    core = {f"gaia_dr3:{GAIA_3C273}", "simbad:3C 273", "ned:3C 273", "first:FIRST J122906.7+020308",
            "chandra:2CXO J122906.6+020308"}
    assert {str(rows[m]["group_id"]) for m in core} == {"group-1"}
    assert int(rows[f"gaia_dr3:{GAIA_3C273}"]["group_n_catalogs"]) == 5
    gaia = rows[f"gaia_dr3:{GAIA_3C273}"]
    assert float(gaia["separation"]) == pytest.approx(0.0021, abs=0.002) and float(gaia["epoch"]) == 2016.0
    assert float(rows["ned:3C 273"]["redshift"]) == pytest.approx(0.158339, abs=1e-5)
    assert float(rows["first:FIRST J122906.7+020308"]["separation"]) == pytest.approx(0.64, abs=0.05)
    assert float(rows["chandra:2CXO J122906.6+020308"]["separation"]) == pytest.approx(0.24, abs=0.02)
    assert {str(r["catalog"]) for r in rows.values()} == set(VO_REPLAY_CATALOGS)
    # The recorded requests asked each archive for its share of the VO row budget (+1 to detect
    # truncation): 5 archives share VO_MAX_CONE_ROWS = 3000, i.e. 600 each (VO_CATALOG_ROW_LIMIT 2000).
    config = vo_server.VOConfig.from_env()
    limit = vo_server.per_archive_row_limit(config.catalog_row_limit, config.max_cone_rows, len(VO_REPLAY_CATALOGS))
    assert limit == 600
    gaia_request = next(e for e in exchanges if e.catalog == "gaia_dr3")
    assert f"TOP {limit + 1} " in gaia_request.request_body.replace("+", " ")


def make_app_without_service(*, with_client: bool = True):
    """App whose router builds its own CrossmatchService (main.build_service) on first use."""
    import httpx
    from fastapi import FastAPI
    from helpers import shared_ssl_context

    app = FastAPI()
    app.include_router(vo_server.router)
    if with_client:
        app.state.client = httpx.AsyncClient(verify=shared_ssl_context(), timeout=30.0)
    return app


def test_fallback_service_shares_one_client_and_build_errors_are_tap_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient

    app = make_app_without_service(with_client=False)
    with TestClient(app) as http:
        assert http.get("/vo/tap/availability").status_code == 200
        client = app.state.vo_client
        providers = app.state.vo_service.providers.values()
        assert providers and all(p.client is client for p in providers)  # one pool, not one per provider
        http.get("/vo/tap/sync", params=tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"))
        assert app.state.vo_client is client
        http.portal.call(vo_server.aclose_vo_state, app)
        assert client.is_closed and app.state.vo_client is None

    import main

    def broken(**_: Any) -> Any:
        raise RuntimeError("registry file is invalid")

    monkeypatch.setattr(main, "build_service", broken)
    broken_app = make_app_without_service(with_client=False)
    with TestClient(broken_app) as http:
        response = http.get("/vo/tap/sync", params=tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"))
        assert response.status_code == 503
        assert "Service unavailable: RuntimeError: registry file is invalid" in \
            votable(response.content).resources[0].infos[0].content
        scs = votable(http.get("/vo/scs", params={"RA": 1, "DEC": 1, "SR": 0.001}).content)
        assert scs.infos[0].name == "Error" and "registry file is invalid" in scs.infos[0].value
        url = http.post("/vo/tap/async", data={**tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"), "PHASE": "RUN"},
                        follow_redirects=False).headers["location"]
        assert _wait_for_phase(http, url, {"COMPLETED", "ERROR"}) == "ERROR"
        assert "registry file is invalid" in http.get(url + "/error").text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="astrosearch")
    subparsers = parser.add_subparsers(dest="command")
    vo_server.register_cli(subparsers)
    return parser.parse_args(argv)


def test_cli_cone_adql_tables(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    stub = RecordedCrossmatchService()
    monkeypatch.setattr(vo_server, "_cli_service_factory", lambda _client: stub)
    args = cli_args(["vo", "cone", "--ra", str(RA_3C273), "--dec", str(DEC_3C273), "--sr", "0.0027777777777777779",
                     "--catalogs", "gaia_dr3,ned", "--format", "csv"])
    assert args.handler(args) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].split(",")[:3] == ["match_id", "ra", "dec"]
    assert any(line.startswith(f"gaia_dr3:{GAIA_3C273},") for line in out)
    assert stub.calls[-1] == {"ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0, "catalogs": ["gaia_dr3", "ned"]}

    target = tmp_path / "cone.xml"
    args = cli_args(["vo", "adql", cone_adql(extra=" AND catalog = 'simbad'"), "--output", str(target)])
    assert args.handler(args) == 0
    assert_valid_votable(target.read_bytes())
    assert "simbad:3C 273" in list(votable(target.read_bytes()).get_first_table().array["match_id"])

    args = cli_args(["vo", "adql", "SELECT nothing FROM astrosearch.catalogs"])
    assert args.handler(args) == 2
    assert "Unknown column 'nothing'" in capsys.readouterr().err

    args = cli_args(["vo", "cone", "--ra", "10", "--dec", "95", "--sr", "0.01"])
    assert args.handler(args) == 2 and "DEC must be within" in capsys.readouterr().err

    args = cli_args(["vo", "tables", "--columns"])
    assert args.handler(args) == 0
    listing = capsys.readouterr().out
    assert "astrosearch.matches" in listing and "pos.eq.ra;meta.main" in listing


# ---------------------------------------------------------------------------
# Resource bounds, protocol details and numerics (review regressions)
# ---------------------------------------------------------------------------


def test_tap_errors_carry_the_query_status_header(client: Any) -> None:
    for params, status in ((tap_params("SELECT nope FROM TAP_SCHEMA.schemas"), 400),
                           ({"QUERY": "SELECT 1 FROM TAP_SCHEMA.schemas"}, 400),
                           (tap_params(cone_adql(select="COUNT(*) AS n", extra=" AND catalog = 'bogus'")), 200)):
        response = client.get("/vo/tap/sync", params=params)
        assert response.status_code == status
        assert response.headers["X-VO-Query-Status"] == ("ERROR" if status >= 400 else "OK")
    from fastapi.testclient import TestClient

    dead = RecordedCrossmatchService(failing=frozenset(load_recorded_sources()[2]))
    with TestClient(make_app(dead)) as http:
        response = http.get("/vo/tap/sync", params=tap_params(cone_adql()))
        assert response.status_code == 502 and response.headers["X-VO-Query-Status"] == "ERROR"
        url = http.post("/vo/tap/async", data={**tap_params(cone_adql()), "PHASE": "RUN"},
                        follow_redirects=False).headers["location"]
        assert _wait_for_phase(http, url, {"COMPLETED", "ERROR"}) == "ERROR"
        error = http.get(url + "/error")
        assert error.headers["X-VO-Query-Status"] == "ERROR" and "upstream archives failed" in error.text
        bad = http.post("/vo/tap/async", data={**tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"), "RUNID": "a\x01b"})
        assert bad.status_code == 400 and bad.headers["X-VO-Query-Status"] == "ERROR"
        assert "XML 1.0" in votable(bad.content).resources[0].infos[0].content


def test_round_truncate_mod_follow_sql_numeric_semantics() -> None:
    """PostgreSQL (numeric): trunc(0.29, 2) = 0.29, round(1.005, 2) = 1.01, round(2.675, 2) = 2.68,
    round(1250, -2) = 1300, mod(-7, 3) = -1 -- binary scaling gave 0.28, 1.0 and 2.67."""
    result = adql(
        "SELECT TRUNCATE(0.29, 2) AS t, ROUND(1.005, 2) AS r, ROUND(2.675, 2) AS r2, ROUND(-1.005, 2) AS rn, "
        "TRUNCATE(-0.29, 2) AS tn, ROUND(1234.5, -2) AS rh, ROUND(1250, -2) AS ri, TRUNCATE(1299, -2) AS ti, "
        "ROUND(7) AS r7, MOD(7, 3) AS m, MOD(-7, 3) AS mn, MOD(7.5, 2) AS mf, MOD(7, 0) AS mz, "
        "ROUND(0.1, 400) AS fine, ROUND(1e300, 2) AS huge, ROUND(123.456, -1000) AS tiny, TRUNCATE(2.5) AS t0 "
        "FROM TAP_SCHEMA.schemas WHERE schema_name = 'TAP_SCHEMA'")
    row = result.as_dicts()[0]
    assert (row["t"], row["r"], row["r2"], row["rn"], row["tn"]) == (0.29, 1.01, 2.68, -1.01, -0.29)
    assert (row["rh"], row["ri"], row["ti"], row["r7"]) == (1200.0, 1300, 1200, 7)
    assert (row["m"], row["mn"], row["mf"], row["mz"]) == (1, -1, 1.5, None)
    assert isinstance(row["m"], int) and isinstance(row["ri"], int) and isinstance(row["t"], float)
    assert (row["fine"], row["huge"], row["tiny"], row["t0"]) == (0.1, 1e300, 0.0, 2.0)
    types = {c.name: c.datatype for c in result.columns}
    assert types["m"] == types["ri"] == types["ti"] == "long" and types["t"] == types["mf"] == "double"
    assert_valid_votable(vo_server.render_votable(result))


def test_aggregate_columns_keep_the_unit_and_ucd_of_their_argument(stub: RecordedCrossmatchService) -> None:
    from astropy import units as u
    from astropy.io.votable.ucd import check_ucd

    result = run(vo_server.execute_adql(cone_adql(
        select="AVG(separation) AS a, SUM(pos_error) AS s, MIN(separation) AS lo, MAX(ra) AS hi, COUNT(*) AS n, "
               "SUM(group_n_catalogs) AS g, AVG(epoch) AS e"), service=stub))
    cols = {c.name: c for c in result.columns}
    assert (cols["a"].unit, cols["a"].ucd, cols["a"].datatype) == ("arcsec", "pos.angDistance;stat.mean", "double")
    assert (cols["s"].unit, cols["s"].ucd) == ("arcsec", "stat.error;pos;arith.sum")
    assert (cols["lo"].unit, cols["lo"].ucd) == ("arcsec", "pos.angDistance;stat.min")
    assert (cols["hi"].unit, cols["hi"].ucd) == ("deg", "pos.eq.ra;stat.max")  # no longer claims meta.main
    assert (cols["e"].unit, cols["e"].ucd) == ("yr", "time.epoch;stat.mean")
    assert cols["g"].datatype == "long" and cols["n"].ucd == "meta.number"
    body = vo_server.render_votable(result)
    assert_valid_votable(body)
    table = votable(body).get_first_table().to_table()
    assert table["a"].unit == u.arcsec and table["hi"].unit == u.deg
    # Every UCD derivable from a published column is valid UCD1+ (or omitted).
    for table_def in vo_server.TABLES:
        for col in table_def.columns:
            for word in ("stat.min", "stat.max", "stat.mean", "arith.sum"):
                ucd = vo_server.derived_ucd(col.ucd, word)
                assert ucd is None or check_ucd(ucd, check_controlled_vocabulary=True), (col.name, ucd)


def test_adql_time_budget_is_honoured_inside_one_expensive_row() -> None:
    """The deadline was checked only every 64 rows: one LIKE of 30000 'a's against '%' + 15000 'a's + 'b'
    (~4.5e8 matcher steps) ran for minutes under a 0.5 s budget."""
    budget = vo_server.VOConfig(max_eval_seconds=0.5)
    query = f"SELECT schema_name FROM TAP_SCHEMA.schemas WHERE '{'a' * 30000}' LIKE '%{'a' * 15000}b'"
    started = time.perf_counter()
    with pytest.raises(vo_server.ADQLError, match="exceeded this service's limit of 0.5 s"):
        adql(query, config=budget)
    assert time.perf_counter() - started < 1.5
    # Strings are bounded, so no single expression can build an arbitrarily expensive value.
    def balanced(leaf: str, n: int) -> str:
        return leaf if n == 1 else f"({balanced(leaf, n // 2)}||{balanced(leaf, n - n // 2)})"

    started = time.perf_counter()
    with pytest.raises(vo_server.ADQLError, match="would build a string of .* at most 65536"):
        adql(f"SELECT column_name FROM TAP_SCHEMA.columns WHERE {balanced('description', 512)} LIKE '%Z'")
    with pytest.raises(vo_server.ADQLError, match="at most 65536"):
        adql(f"SELECT UPPER('{'ß' * 40000}') AS u FROM TAP_SCHEMA.schemas")  # 'SS': case mapping doubles it
    assert time.perf_counter() - started < 2.0
    # The budget also bounds aggregates and ORDER BY (keys are computed in the checked loop).
    tight = vo_server.VOConfig(max_eval_seconds=1e-9)
    for q in ("SELECT MAX(description) AS m FROM TAP_SCHEMA.columns",
              "SELECT column_name FROM TAP_SCHEMA.columns ORDER BY LOWER(description)"):
        with pytest.raises(vo_server.ADQLError, match="exceeded this service's limit"):
            adql(q, config=tight)


@pytest.mark.parametrize("query", [
    "SELECT TOP 1 " + "+".join(["1"] * 3000) + " AS s FROM TAP_SCHEMA.schemas",
    "SELECT name FROM astrosearch.catalogs WHERE " + "NOT " * 400 + "enabled = 1",
    "SELECT TOP 1 " + "ABS(" * 400 + "1" + ")" * 400 + " FROM TAP_SCHEMA.schemas",
    "SELECT name FROM astrosearch.catalogs ORDER BY " + "+".join(["enabled"] * 500),
    "SELECT name FROM astrosearch.catalogs WHERE " + "(" * 300 + "enabled = 1" + ")" * 300,
])
def test_deep_expressions_are_adql_errors_not_internal_errors(client: Any, query: str) -> None:
    response = client.post("/vo/tap/sync", data=tap_params(query))
    assert response.status_code == 400 and response.headers["X-VO-Query-Status"] == "ERROR"
    assert "too deeply nested" in votable(response.content).resources[0].infos[0].content


def test_moderately_deep_expressions_still_evaluate(client: Any) -> None:
    query = "SELECT TOP 1 " + "+".join(["1"] * 150) + " AS s FROM TAP_SCHEMA.schemas"
    assert vo_server.expression_depth(vo_server.parse_adql(query).items[0].expr) == 150
    response = client.post("/vo/tap/sync", data=tap_params(query))
    assert response.status_code == 200 and list(votable(response.content).get_first_table().array["s"]) == [150]


def test_multipart_part_with_an_unknown_charset_is_read_as_utf8(client: Any) -> None:
    boundary = "XyZ"
    parts = []
    for name, value, ctype in (("REQUEST", "doQuery", None), ("LANG", "ADQL", None),
                               ("QUERY", "SELECT name FROM astrosearch.catalogs WHERE name = 'gaia_dr3'",
                                "text/plain; charset=x-no-such-charset")):
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
        if ctype:
            head += f"Content-Type: {ctype}\r\n"
        parts.append(head + "\r\n" + value + "\r\n")
    body = ("".join(parts) + f"--{boundary}--\r\n").encode()
    headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
    sync = client.post("/vo/tap/sync", content=body, headers=headers)
    assert sync.status_code == 200 and list(votable(sync.content).get_first_table().array["name"]) == ["gaia_dr3"]
    created = client.post("/vo/tap/async", content=body, headers=headers, follow_redirects=False)
    assert created.status_code == 303
    assert vo_server._decode_part("é".encode(), "x-no-such-charset") == "é"
    assert vo_server._decode_part(b"\xff", "utf-8") == "�"


def test_row_budget_is_shared_by_the_archives_of_a_cone() -> None:
    assert vo_server.per_archive_row_limit(2000, 3000, 1) == 2000
    assert vo_server.per_archive_row_limit(2000, 3000, 2) == 1500
    assert vo_server.per_archive_row_limit(2000, 3000, 15) == 200
    assert vo_server.per_archive_row_limit(2000, None, 15) == 2000 and vo_server.per_archive_row_limit(5, 3, 15) == 1
    config = vo_server.VOConfig(catalog_row_limit=100, max_cone_rows=150)
    enabled = len(CatalogRegistry().enabled_catalogs())
    for catalogs, expected in ((None, 150 // enabled), (["gaia_dr3", "simbad"], 75), (["gaia_dr3"], 100)):
        service, provider = _counting_service(rows=500, registry_max_rows=1)
        result = run(vo_server.cone_search(service, RA_3C273, DEC_3C273, TEN_ARCSEC_DEG, catalogs=catalogs, config=config))
        assert set(provider.seen_max_rows.values()) == {expected}, catalogs
        assert len(result.rows) == expected * len(provider.seen_max_rows) <= 150 and result.status == "OVERFLOW"
    # astrosearch.catalogs.max_rows is what a single-archive cone fetches.
    service, provider = _counting_service(rows=5000)
    run(vo_server.execute_adql(cone_adql(select="match_id", extra=" AND catalog = 'gaia_dr3'"), service=service))
    published = adql("SELECT max_rows FROM astrosearch.catalogs WHERE name = 'gaia_dr3'").rows
    assert published == [(provider.seen_max_rows["gaia_dr3"],)] == [(2000,)]


class DenseProvider:
    """Archive double for a dense field: ``n`` sources uniform over the cone, returned nearest
    first and cut at TOP max_rows+1 like the TAP provider."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.seen_max_rows: dict[str, int] = {}

    async def query(self, catalog: Any, target: Any, radius_arcsec: float) -> Any:
        import random

        from providers import QueryResult

        self.seen_max_rows[catalog.name] = catalog.max_rows
        rng = random.Random(catalog.name)
        points = []
        for _ in range(self.n):
            r = radius_arcsec / 3600.0 * math.sqrt(rng.random())
            theta = rng.random() * 2 * math.pi
            points.append((r, target.ra + r * math.cos(theta) / math.cos(math.radians(target.dec)),
                           target.dec + r * math.sin(theta)))
        points.sort()
        sources = [CatalogSource(catalog.name, f"{catalog.name}-{i}", ra, dec, 0.1, {},
                                 {"wavelength": catalog.wavelength}, {})
                   for i, (_r, ra, dec) in enumerate(points[:catalog.max_rows + 1])]
        return QueryResult(sources, {"max_rows": catalog.max_rows, "row_limit": catalog.max_rows,
                                     "archive_truncated": self.n > catalog.max_rows})


def test_dense_cone_never_stalls_the_event_loop() -> None:
    """omega Cen-like cone: crossmatch._group_matches is O(n^2) (1200 rows: > 1 s); it and the
    serialization must run off the event loop, so other requests keep being served."""
    import httpx
    from fastapi import FastAPI

    from crossmatch import CrossmatchService

    registry = CatalogRegistry()
    provider = DenseProvider(5000)
    service = CrossmatchService(registry, {c.provider: provider for c in registry.catalogs.values()})
    app = FastAPI()
    app.include_router(vo_server.router)
    app.state.service, app.state.registry = service, registry
    app.state.vo_config = vo_server.VOConfig(catalog_row_limit=600, max_cone_rows=1200)

    async def scenario() -> tuple[float, float, httpx.Response, int]:
        gaps: list[float] = []
        done = asyncio.Event()
        served = 0

        async def heartbeat() -> None:
            last = time.perf_counter()
            while not done.is_set():
                await asyncio.sleep(0.02)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=None) as http:
            async def other_requests() -> None:  # the rest of the app keeps answering meanwhile
                nonlocal served
                while not done.is_set():
                    await http.get("/vo/tap/availability")
                    served += 1
                    await asyncio.sleep(0.05)

            beat = asyncio.create_task(heartbeat())
            others = asyncio.create_task(other_requests())
            started = time.perf_counter()
            response = await http.get("/vo/scs", params={"RA": 201.69683, "DEC": -47.47958, "SR": 0.05,
                                                         "CATALOGS": "gaia_dr3,twomass_psc"})
            elapsed = time.perf_counter() - started
            done.set()
            await beat
            await others
        return elapsed, max(gaps), response, served

    elapsed, stall, response, served = asyncio.run(scenario())
    assert provider.seen_max_rows == {"gaia_dr3": 600, "twomass_psc": 600}  # 1200 rows shared by 2 archives
    assert response.text.count("<TR>") == 1200 and response.headers["X-VO-Query-Status"] == "OVERFLOW"
    assert stall < 0.5, f"event loop stalled {stall:.2f} s during a {elapsed:.2f} s dense cone"
    assert served >= 2 and elapsed > stall


# ---------------------------------------------------------------------------
# Round-3 review regressions: planning off the loop, result bounds, cancellation, ADQL edge cases
# ---------------------------------------------------------------------------


def test_constant_conditions_are_planned_off_the_event_loop() -> None:
    """A costly column-free WHERE term (a pathological LIKE on literals) was evaluated on the event loop
    once per archive during planning: 5 concurrent requests froze the whole app for ~5 s."""
    import httpx

    stub = RecordedCrossmatchService()
    app = make_app(stub)
    app.state.vo_config = vo_server.VOConfig(max_eval_seconds=1.0)
    query = cone_adql(select="match_id", extra=f" AND '{'a' * 30000}' LIKE '%{'a' * 15000}b'")

    async def scenario() -> tuple[float, list[httpx.Response]]:
        gaps: list[float] = []
        done = asyncio.Event()

        async def heartbeat() -> None:
            last = time.perf_counter()
            while not done.is_set():
                await asyncio.sleep(0.01)
                now = time.perf_counter()
                gaps.append(now - last)
                last = now

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=None) as http:
            beat = asyncio.create_task(heartbeat())
            responses = await asyncio.gather(*(http.post("/vo/tap/sync", data=tap_params(query)) for _ in range(5)))
            done.set()
            await beat
        return max(gaps), list(responses)

    stall, responses = asyncio.run(scenario())
    assert stall < 0.5, f"event loop stalled {stall:.2f} s"
    for response in responses:
        assert response.status_code == 400
        # The budget reported is the one that applied: VO_ADQL_EVAL_SECONDS for the whole query.
        assert "exceeded this service's limit of 1 s" in votable(response.content).resources[0].infos[0].content
    assert stub.calls == []  # evaluated once per request at planning (never per archive), before any fetch
    # A constant condition that is not TRUE leaves nothing to fetch; one that is TRUE changes nothing.
    none = adql_on(stub, cone_adql(select="match_id", extra=" AND 1 = 0"))
    assert none.rows == [] and stub.calls == []
    kept = adql_on(stub, cone_adql(select="match_id", extra=" AND 'abc' LIKE 'a%'"))
    assert len(stub.calls) == 1 and kept.rows == adql_on(stub, cone_adql(select="match_id")).rows


def adql_on(service: Any, query: str) -> vo_server.ResultTable:
    return run(vo_server.execute_adql(query, service=service))


def test_result_size_is_bounded() -> None:
    """66 select items of 100 concatenated descriptions (a < 100 KB query) built a 30 MB response."""
    from fastapi.testclient import TestClient

    item = "(" + "||".join(["description"] * 100) + ")"
    query = "SELECT " + ", ".join(f"{item} AS c{i}" for i in range(66)) + " FROM TAP_SCHEMA.columns"
    assert len(query) < vo_server.MAX_QUERY_LENGTH
    app = make_app(RecordedCrossmatchService())
    app.state.vo_config = vo_server.VOConfig(max_result_bytes=2_000_000)
    with TestClient(app) as http:
        for fmt in ("csv", "votable"):
            started = time.perf_counter()
            response = http.post("/vo/tap/sync", data=tap_params(query, RESPONSEFORMAT=fmt))
            assert response.status_code == 400 and len(response.content) < 10_000
            assert "exceed this service's limit of 2000000 bytes" in votable(response.content).resources[0].infos[0].content
            assert time.perf_counter() - started < 5.0
        # The same bound applies to the serialized body (VOTable markup adds to the values).
        small = http.post("/vo/tap/sync", data=tap_params("SELECT column_name FROM TAP_SCHEMA.columns"))
        assert small.status_code == 200
        # Too many select items are refused before any evaluation.
        many = "SELECT " + ", ".join(["1"] * (vo_server.MAX_SELECT_ITEMS + 1)) + " FROM TAP_SCHEMA.schemas"
        refused = http.post("/vo/tap/sync", data=tap_params(many))
        assert refused.status_code == 400 and "at most 500" in refused.text
        # Oversized request bodies are refused (413) from their Content-Length, before being read.
        huge = http.post("/vo/tap/sync", content=b"QUERY=" + b"x" * (vo_server.MAX_REQUEST_BYTES + 1),
                         headers={"content-type": "application/x-www-form-urlencoded"})
        assert huge.status_code == 413 and huge.headers["X-VO-Query-Status"] == "ERROR"
        job = http.post("/vo/tap/async", data=tap_params("SELECT 1 FROM TAP_SCHEMA.schemas"),
                        follow_redirects=False).headers["location"]
        too_big = http.post(job + "/parameters", content=b"x" * (vo_server.MAX_REQUEST_BYTES + 1),
                            headers={"content-type": "application/x-www-form-urlencoded"})
        assert too_big.status_code == 413


def test_uws_result_store_is_bounded() -> None:
    """Completed async results are kept in memory: their total is capped, destroying the oldest
    finished jobs first, and a result larger than the whole store is a job error."""
    from fastapi.testclient import TestClient

    app = make_app(RecordedCrossmatchService())
    query = "SELECT column_name, description FROM TAP_SCHEMA.columns"
    with TestClient(app) as http:
        one = http.post("/vo/tap/sync", data=tap_params(query)).content
        app.state.vo_config = vo_server.VOConfig(uws_max_stored_bytes=int(len(one) * 2.5))
        urls = []
        for _ in range(4):
            url = http.post("/vo/tap/async", data={**tap_params(query), "PHASE": "RUN"},
                            follow_redirects=False).headers["location"]
            assert _wait_for_phase(http, url, {"COMPLETED", "ERROR"}) == "COMPLETED"
            urls.append(url)
        manager = app.state.vo_jobs
        assert manager.stored_bytes() <= manager.config.uws_max_stored_bytes
        assert [http.get(u).status_code for u in urls] == [404, 404, 200, 200]  # oldest destroyed first
        manager.config.uws_max_stored_bytes = len(one) // 2
        url = http.post("/vo/tap/async", data={**tap_params(query), "PHASE": "RUN"},
                        follow_redirects=False).headers["location"]
        assert _wait_for_phase(http, url, {"COMPLETED", "ERROR"}) == "ERROR"
        assert "exceeds this service's job result store" in http.get(url + "/error").text


class SlowProvider:
    """Archive double that takes ``delay_s`` per query and counts started/finished queries."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.started = 0
        self.finished = 0

    async def query(self, catalog: Any, target: Any, radius_arcsec: float) -> Any:
        from providers import QueryResult

        self.started += 1
        await asyncio.sleep(self.delay_s)
        self.finished += 1
        return QueryResult([CatalogSource(catalog.name, "s0", target.ra, target.dec, 0.1, {},
                                          {"wavelength": catalog.wavelength}, {})], {"max_rows": catalog.max_rows})


def test_uws_abort_cancels_the_real_crossmatch_and_its_archive_queries() -> None:
    """With a real CrossmatchService the crossmatch runs in a worker thread: ABORT must stop its archive
    queries (all 15 used to run to completion after the job was ABORTED)."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from crossmatch import CrossmatchService

    registry = CatalogRegistry()
    provider = SlowProvider(2.0)
    service = CrossmatchService(registry, {c.provider: provider for c in registry.catalogs.values()})
    app = FastAPI()
    app.include_router(vo_server.router)
    app.state.service, app.state.registry = service, registry
    with TestClient(app) as http:
        url = http.post("/vo/tap/async", data={**tap_params(cone_adql()), "PHASE": "RUN"},
                        follow_redirects=False).headers["location"]
        _wait_for_phase(http, url, {"EXECUTING"})
        deadline = time.monotonic() + 10
        while provider.started == 0:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        time.sleep(0.2)
        started = provider.started
        assert started > 1 and provider.finished == 0
        assert http.post(url + "/phase", data={"PHASE": "ABORT"}, follow_redirects=False).status_code == 303
        time.sleep(3.0)  # longer than the 2 s every archive query would take
        assert http.get(url + "/phase").text == "ABORTED"
        assert provider.finished == 0, (provider.started, provider.finished)
        assert provider.started == started  # nothing new was sent after the abort
        # Without an abort the same service completes normally (the cancellation is per request).
        url = http.post("/vo/tap/async", data={**tap_params(cone_adql(select="match_id", extra=" AND catalog = 'gaia_dr3'")),
                                               "PHASE": "RUN"}, follow_redirects=False).headers["location"]
        assert _wait_for_phase(http, url, {"COMPLETED", "ERROR"}, timeout=30.0) == "COMPLETED"
        assert provider.finished == 1


@pytest.mark.parametrize(("query", "message"), [
    ("SELECT 0x10 FROM TAP_SCHEMA.schemas", "'0' at position 8 is immediately followed by 'x10'"),
    ("SELECT 2abc FROM TAP_SCHEMA.schemas", "immediately followed by 'abc'"),
    ("SELECT DISTINCT provider FROM astrosearch.catalogs ORDER BY name",
     "For SELECT DISTINCT, ORDER BY expressions must appear in the select list"),
])
def test_review_adql_errors(client: Any, query: str, message: str) -> None:
    response = client.get("/vo/tap/sync", params=tap_params(query, RESPONSEFORMAT="csv"))
    assert response.status_code == 400
    assert message in votable(response.content).resources[0].infos[0].content


def test_distinct_order_by_selected_expressions_and_unique_names(client: Any) -> None:
    ordered = adql("SELECT DISTINCT provider AS p FROM astrosearch.catalogs AS c ORDER BY c.provider DESC")
    values = ordered.column_values("p")
    assert values == sorted(set(values), reverse=True)
    assert adql("SELECT DISTINCT LOWER(provider) FROM astrosearch.catalogs ORDER BY LOWER(provider)").rows
    # Generated names never collide with explicit ones: name, name, name AS name_2.
    result = adql("SELECT name, name, name AS name_2 FROM astrosearch.catalogs")
    assert [c.name for c in result.columns] == ["name", "name_3", "name_2"]
    table = votable(vo_server.render_votable(result)).get_first_table().to_table(use_names_over_ids=True)
    assert table.colnames == ["name", "name_3", "name_2"]
    assert vo_server._unique(["a", "A", "a_2"]) == ["a", "A_3", "a_2"]


def test_tsv_has_no_quoting(client: Any) -> None:
    query = "SELECT 'a\"b' AS q, 'x' || 'y' AS s FROM TAP_SCHEMA.schemas WHERE schema_name = 'TAP_SCHEMA'"
    response = client.get("/vo/tap/sync", params=tap_params(query, RESPONSEFORMAT="tsv"))
    assert response.status_code == 200 and response.text == 'q\ts\na"b\txy\n'
    rows = [line.split("\t") for line in response.text.splitlines()]
    assert rows == [["q", "s"], ['a"b', "xy"]]
    tab = vo_server.ResultTable("t", [vo_server.VOColumn("v", "char", "x", arraysize="*")], [("x\ny\tz\r",)], "OK",
                                [], None, {}, {})
    assert vo_server.render_csv(tab, delimiter="\t") == b"v\nx y z \n"  # one record per line, always


def test_xml_illegal_characters_in_identifiers_are_refused(client: Any) -> None:
    response = client.get("/vo/tap/sync", params=tap_params('SELECT 1 AS "a\x01b" FROM TAP_SCHEMA.schemas'))
    assert response.status_code == 400 and "XML 1.0" in votable(response.content).resources[0].infos[0].content
    # Column names and descriptions are sanitized by the VOTable writer as well.
    result = vo_server.ResultTable("t", [vo_server.VOColumn("a\x02b", "int", "d\x03")], [(1,)], "OK", [], None, {}, {})
    body = vo_server.render_votable(result)
    assert_valid_votable(body)
    assert votable(body).get_first_table().fields[0].name == "a\ufffdb"


def test_examples_document_uses_the_dali_11_vocabulary(client: Any) -> None:
    from lxml import etree

    body = client.get("/vo/tap/examples").content
    root = etree.fromstring(body)
    holders = root.xpath("//*[@vocab]")
    assert [h.get("vocab") for h in holders] == ["http://www.ivoa.net/rdf/examples#"]
    examples = root.xpath("//*[@typeof='example']")
    assert examples and all(e.getparent() is holders[0] or holders[0] in e.iterancestors() for e in examples)


# ---------------------------------------------------------------------------
# Live: real archives for 3C 273
# ---------------------------------------------------------------------------

NETWORK_ERRORS = ("CatalogUnavailableError", "QueryTimeoutError", "RateLimitedError")


def _network_failures(infos: list[tuple[str, str]]) -> dict[str, str]:
    """{catalog: error type} of the per-archive failure WARNINGs of a result (any error type)."""
    failed = {}
    for name, value in infos:
        if name == "WARNING" and " failed (" in value:
            catalog, rest = value.split(" failed (", 1)
            failed[catalog] = rest.split(")", 1)[0]
    return failed


_FAILURE_SEGMENT = re.compile(r"(?:(?<=: )|(?<=; ))([a-z0-9_]+): ([A-Z][A-Za-z0-9_]*): ")


def upstream_error_types(message: str) -> dict[str, str]:
    """{catalog: error type} parsed from an UpstreamError message ('catalog: ErrorType: message; ...')."""
    return dict(_FAILURE_SEGMENT.findall(message))


def skip_only_for_network_errors(exc: Exception) -> None:
    """Skip a live test when upstream archives were unreachable -- and ONLY then.

    Parser/model regressions (CatalogQueryError, ResponseParseError, KeyError, ...) or a
    truncated cone must fail the test, never skip it.
    """
    types = upstream_error_types(str(exc))
    if types and all(t in NETWORK_ERRORS for t in types.values()):
        pytest.skip(f"archives unreachable: {types}")
    raise exc


def test_live_skip_policy_only_skips_network_errors() -> None:
    network = ("All 2 upstream archives failed: gaia_dr3: CatalogUnavailableError: HTTP 503; "
               "simbad: QueryTimeoutError: Catalog simbad timed out after 60s")
    assert upstream_error_types(network) == {"gaia_dr3": "CatalogUnavailableError", "simbad": "QueryTimeoutError"}
    with pytest.raises(pytest.skip.Exception):
        skip_only_for_network_errors(RuntimeError(network))
    for message in (("All 2 upstream archives failed: gaia_dr3: CatalogQueryError: bad column; "
                     "simbad: CatalogUnavailableError: HTTP 503"),
                    "All 1 upstream archives failed: ned: KeyError: 'ra'",
                    "An aggregate needs every row of the cone, but ...: gaia_dr3: Truncated: only the 2000 nearest rows",
                    "upstream archives failed"):
        with pytest.raises(RuntimeError):
            skip_only_for_network_errors(RuntimeError(message))


@pytest.fixture(scope="module")
def live_server() -> Iterator[str]:
    from contextlib import asynccontextmanager

    import httpx
    from fastapi import FastAPI

    from main import build_service

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # the service must share the server's event loop
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as http:
            app.state.client = http
            app.state.service = build_service(client=http)
            app.state.registry = app.state.service.registry
            yield

    app = FastAPI(lifespan=lifespan)
    app.include_router(vo_server.router)
    with serve(app) as base:
        yield base


@pytest.mark.live
def test_live_cone_search_3c273_multiwavelength(live_server: str) -> None:
    import pyvo
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    catalogs = "gaia_dr3,simbad,ned,twomass_psc,first,chandra"
    query = pyvo.dal.SCSQuery(f"{live_server}/vo/scs", pos=SkyCoord(RA_3C273, DEC_3C273, unit="deg"), radius=10 * u.arcsec)
    query["CATALOGS"] = catalogs
    query["VERB"] = 3
    try:
        results = query.execute()
    except pyvo.dal.DALQueryError as exc:
        skip_only_for_network_errors(exc)
    infos = [(i.name, i.value) for i in results.votable.resources[0].infos]
    failed = _network_failures(infos)
    assert not {c: e for c, e in failed.items() if e not in NETWORK_ERRORS}, failed
    table = results.to_table()
    rows = {str(r["match_id"]): r for r in table}
    centre = SkyCoord(RA_3C273, DEC_3C273, unit="deg")
    expectations = {
        "gaia_dr3": (f"gaia_dr3:{GAIA_3C273}", 0.05),
        "simbad": ("simbad:3C 273", 0.05),
        "ned": ("ned:3C 273", 0.05),
        "twomass_psc": ("twomass_psc:12290669+0203085", 0.2),
        "first": ("first:FIRST J122906.7+020308", 1.0),
        "chandra": ("chandra:2CXO J122906.6+020308", 0.5),
    }
    checked = 0
    for catalog, (match_id, max_sep) in expectations.items():
        if catalog in failed:
            continue  # archive unreachable (network error) -- verified above
        assert match_id in rows, (match_id, sorted(rows))
        row = rows[match_id]
        assert float(row["separation"]) < max_sep
        assert SkyCoord(float(row["ra"]), float(row["dec"]), unit="deg").separation(centre).arcsec < max_sep + 0.05
        checked += 1
    if checked == 0:
        pytest.skip(f"every archive unreachable: {failed}")
    print(f"live 3C 273 cone: verified {checked}/{len(expectations)} archives; unreachable: {failed or 'none'}")
    if "ned" not in failed:
        assert float(rows["ned:3C 273"]["redshift"]) == pytest.approx(0.158339, abs=0.0005)  # z of 3C 273
    if "simbad" not in failed:
        assert str(rows["simbad:3C 273"]["object_type"]) in {"QSO", "BLL", "Bla"}
    wavebands = {str(r["wavelength"]) for r in table if float(r["separation"]) < 1.0}
    reachable = {c for c in expectations if c not in failed}
    if {"first"} <= reachable:
        assert "radio" in wavebands
    if {"chandra"} <= reachable:
        assert "xray" in wavebands


@pytest.mark.live
def test_live_tap_3c273_gaia_and_simbad(live_server: str) -> None:
    import pyvo

    tap = pyvo.dal.TAPService(f"{live_server}/vo/tap")
    query = cone_adql(select="match_id, catalog, separation, pmra, pmdec, parallax, object_type",
                      extra=" AND catalog IN ('gaia_dr3', 'simbad') ORDER BY separation")
    try:
        result = tap.run_sync(query)
    except pyvo.dal.DALQueryError as exc:
        skip_only_for_network_errors(exc)
    infos = [(i.name, i.value) for i in result.votable.resources[0].infos]
    failed = _network_failures(infos)
    assert not {c: e for c, e in failed.items() if e not in NETWORK_ERRORS}, failed
    rows = {str(r["match_id"]): r for r in result.to_table()}
    if "gaia_dr3" not in failed:
        gaia = rows[f"gaia_dr3:{GAIA_3C273}"]
        # A quasar at z = 0.158: Gaia DR3 parallax and proper motion consistent with zero (|pm| < 1 mas/yr).
        assert abs(float(gaia["parallax"])) < 0.2
        assert math.hypot(float(gaia["pmra"]), float(gaia["pmdec"])) < 1.0
        assert float(gaia["separation"]) < 0.05
    if "simbad" not in failed:
        assert float(rows["simbad:3C 273"]["separation"]) < 0.05


# Dense and high-proper-motion fields (positions: SIMBAD, ICRS J2000).
OMEGA_CEN = (201.69683, -47.47958)          # NGC 5139 core
BARNARD = (269.45207511, 4.69339088)        # Barnard's star; Gaia DR3 4472832130942575872
PROXIMA = (217.42894222, -62.67949019)      # Proxima Cen; Gaia DR3 5853498713190525696
GAIA_TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def _gaia_archive_count(ra: float, dec: float, radius_deg: float) -> int:
    """COUNT(*) of Gaia DR3 sources in a cone, asked of the ESA Gaia archive itself."""
    import httpx

    query = (f"SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), "
             f"CIRCLE('ICRS', {ra}, {dec}, {radius_deg}))")
    try:
        response = httpx.post(GAIA_TAP, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
                              timeout=120.0)
    except httpx.TransportError as exc:
        pytest.skip(f"Gaia archive unreachable: {exc}")
    if response.status_code >= 500:
        pytest.skip(f"Gaia archive unavailable: HTTP {response.status_code}")
    response.raise_for_status()
    return int(response.text.split()[-1])


def _run_live(tap: Any, query: str, **kw: Any) -> Any:
    import pyvo

    try:
        return tap.run_sync(query, **kw)
    except pyvo.dal.DALQueryError as exc:
        skip_only_for_network_errors(exc)


@pytest.mark.live
def test_live_dense_field_count_matches_the_gaia_archive(live_server: str) -> None:
    """omega Cen, 72" cone: COUNT through AstroSearch equals the Gaia archive's own COUNT (1240 in 2026)."""
    import pyvo

    ra, dec = OMEGA_CEN
    truth = _gaia_archive_count(ra, dec, 0.02)
    assert 1000 < truth < vo_server.VOConfig.from_env().catalog_row_limit  # dense, but within the row limit
    tap = pyvo.dal.TAPService(f"{live_server}/vo/tap")
    result = _run_live(tap, f"SELECT COUNT(*) AS n, MAX(separation) AS reach FROM astrosearch.matches WHERE "
                            f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.02)) "
                            "AND catalog = 'gaia_dr3'")
    row = result.to_table()[0]
    assert int(row["n"]) == truth
    assert 65.0 < float(row["reach"]) <= 72.0  # the rows fill the whole 72" cone
    assert [i.value for i in result.votable.resources[0].infos if i.name == "QUERY_STATUS"] == ["OK"]


@pytest.mark.live
def test_live_truncated_dense_cone_is_refused_or_flagged(live_server: str) -> None:
    """omega Cen, 180" cone: more Gaia sources than the per-archive row limit."""
    import pyvo

    ra, dec = OMEGA_CEN
    limit = vo_server.VOConfig.from_env().catalog_row_limit
    assert _gaia_archive_count(ra, dec, 0.05) > limit
    tap = pyvo.dal.TAPService(f"{live_server}/vo/tap")
    cone = f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.05)) AND catalog = 'gaia_dr3'"
    try:
        tap.run_sync(f"SELECT COUNT(*) AS n FROM astrosearch.matches WHERE {cone}")
    except pyvo.dal.DALQueryError as exc:
        if "Truncated" not in str(exc):
            skip_only_for_network_errors(exc)
        assert f"gaia_dr3: Truncated: only the {limit} nearest rows" in str(exc) and "Reduce the cone radius" in str(exc)
    else:
        raise AssertionError("COUNT over a truncated cone must not succeed")
    with pytest.warns(pyvo.dal.exceptions.DALOverflowWarning):
        rows = _run_live(tap, f"SELECT match_id, separation FROM astrosearch.matches WHERE {cone}")
    table = rows.to_table()
    assert len(table) == limit and float(table["separation"].max()) < 180.0
    infos = [(i.name, i.value) for i in rows.votable.resources[0].infos]
    assert any(n == "WARNING" and v.startswith(f"gaia_dr3 truncated: only the {limit} nearest rows") for n, v in infos)


@pytest.mark.live
def test_live_filtered_cone_finds_barnards_star(live_server: str) -> None:
    """parallax > 500 mas in a 180" cone at Barnard's star's J2000 position: exactly Barnard's star.

    Gaia DR3 4472832130942575872 (parallax 546.98 mas) lies 166" from its J2000 position
    at the Gaia epoch J2016.0: 10.39"/yr x 16 yr of proper motion.
    """
    import pyvo

    ra, dec = BARNARD
    tap = pyvo.dal.TAPService(f"{live_server}/vo/tap")
    result = _run_live(tap, "SELECT match_id, parallax, separation, pmra, pmdec, epoch FROM astrosearch.matches WHERE "
                            f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.05)) "
                            "AND catalog = 'gaia_dr3' AND parallax > 500")
    table = result.to_table()
    assert [str(m) for m in table["match_id"]] == ["gaia_dr3:4472832130942575872"]
    star = table[0]
    assert float(star["parallax"]) == pytest.approx(546.976, abs=0.05)
    assert float(star["pmra"]) == pytest.approx(-801.55, abs=0.1) and float(star["pmdec"]) == pytest.approx(10362.39, abs=0.1)
    moved = math.hypot(float(star["pmra"]), float(star["pmdec"])) / 1000 * (float(star["epoch"]) - 2000.0)
    assert float(star["separation"]) == pytest.approx(moved, abs=1.0) and 160 < float(star["separation"]) < 172
    assert [i.value for i in result.votable.resources[0].infos if i.name == "QUERY_STATUS"] == ["OK"]


@pytest.mark.live
def test_live_proxima_gaia_row_is_not_lost(live_server: str) -> None:
    """Proxima Cen, 72" cone: the Gaia DR3 row (J2016, ~62" from the J2000 position) is present."""
    import pyvo

    ra, dec = PROXIMA
    tap = pyvo.dal.TAPService(f"{live_server}/vo/tap")
    result = _run_live(tap, "SELECT match_id, catalog, separation, pmra, pmdec, epoch FROM astrosearch.matches WHERE "
                            f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.02)) "
                            "AND catalog IN ('gaia_dr3', 'simbad') ORDER BY separation")
    infos = [(i.name, i.value) for i in result.votable.resources[0].infos]
    failed = _network_failures(infos)
    assert not {c: e for c, e in failed.items() if e not in NETWORK_ERRORS}, failed
    assert not any(n == "INCOMPLETE" and "Truncated" in v for n, v in infos), infos
    rows = {str(r["match_id"]): r for r in result.to_table()}
    if "gaia_dr3" not in failed:
        proxima = rows["gaia_dr3:5853498713190525696"]
        moved = math.hypot(float(proxima["pmra"]), float(proxima["pmdec"])) / 1000 * (float(proxima["epoch"]) - 2000.0)
        assert float(proxima["separation"]) == pytest.approx(moved, abs=1.0) and 60 < float(proxima["separation"]) < 64
    if "simbad" not in failed:
        nearest = min(rows.values(), key=lambda r: float(r["separation"]))
        assert str(nearest["catalog"]) == "simbad" and float(nearest["separation"]) < 0.1
        assert float(nearest["pmra"]) == pytest.approx(-3781.741, abs=1.0)


def _live_scs(live_server: str, ra: float, dec: float, sr_deg: float, catalogs: str) -> tuple[Any, dict[str, str]]:
    """pyvo SCS query against the live server: (table rows by match_id, {failed catalog: error type})."""
    import pyvo

    query = pyvo.dal.SCSQuery(f"{live_server}/vo/scs", pos=(ra, dec), radius=sr_deg)
    query["CATALOGS"] = catalogs
    query["VERB"] = 3
    try:
        results = query.execute()
    except pyvo.dal.DALQueryError as exc:
        skip_only_for_network_errors(exc)
    infos = [(i.name, i.value) for i in results.votable.resources[0].infos]
    failed = _network_failures(infos)
    assert not {c: e for c, e in failed.items() if e not in NETWORK_ERRORS}, failed
    assert not any(n == "INCOMPLETE" and "Truncated" in v for n, v in infos), infos
    # pyvo's SCS 1.03 accessors work on every record.
    assert all(r.id and r.pos is not None for r in results)
    return {str(r["match_id"]): r for r in results.to_table()}, failed


def _members(rows: Any, match_id: str) -> set[str]:
    group = str(rows[match_id]["group_id"])
    return {m for m, r in rows.items() if str(r["group_id"]) == group}


@pytest.mark.live
def test_live_m87_cone_is_split_into_candidate_sets(live_server: str) -> None:
    """M87, 10": the nucleus, the jet and the surrounding globular clusters/field objects are
    distinct candidate sets (union-find chained all 163 detections into one 'object-1')."""
    rows, failed = _live_scs(live_server, 187.7059308, 12.3911233, 10 / 3600.0, "simbad,ned,gaia_dr3,chandra")
    if {"simbad", "ned"} & set(failed):
        pytest.skip(f"archives unreachable: {failed}")
    groups: dict[str, list[str]] = {}
    for row in rows.values():
        groups.setdefault(str(row["group_id"]), []).append(str(row["catalog"]))
    assert len(groups) > 10, len(groups)
    _assert_one_detection_per_catalog(rows.values())  # one detection per catalog per set
    nucleus = _members(rows, "simbad:M 87")
    assert "ned:Messier 087" in nucleus and len(nucleus) >= 2
    assert int(rows["simbad:M 87"]["group_n_catalogs"]) == len(nucleus) < len(rows) / 4
    jet = next(m for m in rows if m.startswith("simbad:") and "M87 Jet" in m)
    assert _members(rows, jet).isdisjoint(nucleus)


@pytest.mark.live
def test_live_61_cyg_sets_follow_proper_motion_not_proximity(live_server: str) -> None:
    """61 Cyg A (pm 5.28"/yr): SIMBAD (J2000) is 0.005" from the cone centre and Gaia DR3
    1872046609345556480 (J2016, parallax 286.0 mas) 84.5" away; the unrelated Gaia DR3
    1872047227813442688 (parallax 1.1 mas) at 2.1" must not share its set."""
    rows, failed = _live_scs(live_server, 316.72475, 38.749417, 0.026, "gaia_dr3,simbad,twomass_psc")
    if {"gaia_dr3", "simbad"} & set(failed):
        pytest.skip(f"archives unreachable: {failed}")
    star_a = _members(rows, "simbad:* 61 Cyg A")
    assert "gaia_dr3:1872046609345556480" in star_a
    assert float(rows["gaia_dr3:1872046609345556480"]["parallax"]) == pytest.approx(285.995, abs=0.05)
    assert float(rows["gaia_dr3:1872046609345556480"]["separation"]) == pytest.approx(84.5, abs=1.0)
    assert "gaia_dr3:1872047227813442688" not in star_a
    assert float(rows["gaia_dr3:1872047227813442688"]["parallax"]) < 5
    star_b = _members(rows, "simbad:* 61 Cyg B")
    assert "gaia_dr3:1872046574983497216" in star_b and star_b.isdisjoint(star_a)
    if "twomass_psc" not in failed:  # 2MASS saw 61 Cyg A 8" from its J2000 position (epoch 1998.5)
        assert "twomass_psc:21065341+3844529" in star_a


def _simbad_identifiers(main_id: str) -> set[str]:
    """SIMBAD's own cross-identifications of an object (sim-tap ``ident``), the ground truth for its set."""
    import httpx

    query = ("SELECT i.id FROM ident AS i JOIN basic AS b ON b.oid = i.oidref WHERE b.main_id = '"
             + main_id.replace("'", "''") + "'")
    try:
        response = httpx.post("https://simbad.cds.unistra.fr/simbad/sim-tap/sync", timeout=120.0,
                              data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query})
    except httpx.TransportError as exc:
        pytest.skip(f"SIMBAD unreachable: {exc}")
    if response.status_code >= 500:
        pytest.skip(f"SIMBAD unavailable: HTTP {response.status_code}")
    response.raise_for_status()
    return {line.strip().strip('"') for line in response.text.splitlines()[1:] if line.strip()}


# (SIMBAD main_id, cone (ra, dec, radius deg), catalogs, {match_id: SIMBAD identifier it must equal})
LIVE_STAR_SETS = {
    "barnard": ("NAME Barnard's star", (269.4520749, 4.6933639, 6 / 3600), "simbad,twomass_psc",
                {"twomass_psc:17574849+0441405": "2MASS J17574849+0441405"}),
    "hd209458": ("HD 209458", (330.79488, 18.88431, 5 / 3600), "simbad,gaia_dr3,twomass_psc,exoplanet_archive,panstarrs_dr2",
                 {"gaia_dr3:1779546757669063552": "Gaia DR3 1779546757669063552",
                  "twomass_psc:22031077+1853036": "2MASS J22031077+1853036"}),
    "proxima": ("NAME Proxima Centauri", (217.42895, -62.67948, 70 / 3600), "gaia_dr3,simbad,twomass_psc,allwise",
                {"gaia_dr3:5853498713190525696": "Gaia DR3 5853498713190525696",
                 "twomass_psc:14294291-6240465": "2MASS J14294291-6240465",
                 "allwise:J142937.35-624038.3": "WISEA J142937.35-624038.3"}),
}


@pytest.mark.live
@pytest.mark.parametrize("name", sorted(LIVE_STAR_SETS))
def test_live_star_sets_match_simbad_cross_identifications(live_server: str, name: str) -> None:
    """Barnard's star (planets b-e at its coordinates), HD 209458 (planet b at its coordinates) and Proxima Cen
    (768 mas parallax): the star's SIMBAD entry shares its set with the detections SIMBAD identifies with it."""
    main_id, (ra, dec, sr), catalogs, expected = LIVE_STAR_SETS[name]
    identifiers = _simbad_identifiers(main_id)
    assert set(expected.values()) <= identifiers, set(expected.values()) - identifiers  # SIMBAD's ground truth
    rows, failed = _live_scs(live_server, ra, dec, sr, catalogs)
    if "simbad" in failed:
        pytest.skip(f"SIMBAD unreachable: {failed}")
    star = next(m for m in rows if " ".join(m.removeprefix("simbad:").split()) == main_id and m.startswith("simbad:"))
    members = _members(rows, star)
    for match_id in expected:
        if match_id.split(":", 1)[0] not in failed:
            assert match_id in members, (match_id, sorted(members))
    _assert_one_detection_per_catalog(rows.values())


@pytest.mark.live
def test_live_nearest_rows_of_a_truncated_dense_cone_match_simbad(live_server: str) -> None:
    """Sgr A*, 36" cone: SIMBAD holds more than the row limit, yet the 3 nearest SIMBAD objects are
    exact -- identical to SIMBAD's own TAP answer -- while COUNT is refused."""
    import httpx
    import pyvo

    ra, dec = 266.41681662, -29.00782497
    tap = pyvo.dal.TAPService(f"{live_server}/vo/tap")
    cone = f"1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.01)) AND catalog = 'simbad'"
    result = _run_live(tap, f"SELECT TOP 3 match_id, separation FROM astrosearch.matches WHERE {cone} ORDER BY separation")
    infos = [(i.name, i.value) for i in result.votable.resources[0].infos]
    assert any(n == "WARNING" and v.startswith("simbad truncated") for n, v in infos)  # the cone IS truncated
    assert [i.value for i in result.votable.resources[0].infos if i.name == "QUERY_STATUS"] == ["OK"]
    assert any(n == "COMPLETENESS" for n, _v in infos)
    ours = [str(m).removeprefix("simbad:") for m in result.to_table()["match_id"]]
    try:
        truth = httpx.post("https://simbad.cds.unistra.fr/simbad/sim-tap/sync", timeout=120.0, data={
            "REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
            "QUERY": f"SELECT TOP 3 main_id, DISTANCE(POINT('ICRS', ra, dec), POINT('ICRS', {ra}, {dec})) AS d FROM basic "
                     f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.001)) ORDER BY d"})
    except httpx.TransportError as exc:
        pytest.skip(f"SIMBAD unreachable: {exc}")
    if truth.status_code >= 500:
        pytest.skip(f"SIMBAD unavailable: HTTP {truth.status_code}")
    names = [line.rsplit(",", 1)[0].strip('"') for line in truth.text.strip().splitlines()[1:]]
    assert ours == names and ours[0] == "NAME Sgr A*"
    with pytest.raises(pyvo.dal.DALQueryError, match="simbad: Truncated"):
        tap.run_sync(f"SELECT COUNT(*) AS n FROM astrosearch.matches WHERE {cone}")


# ---------------------------------------------------------------------------
# Fixture recording (offline, from the raw archive responses in tests/fixtures/3c273)
# ---------------------------------------------------------------------------


async def _record() -> None:
    import os

    os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
    os.environ["PROVIDER_REQUESTS_PER_SECOND"] = "1000"
    os.environ.pop("REDIS_URL", None)
    import respx
    from fixture_io import load_exchanges, replay_side_effect
    from helpers import make_service, offline_client

    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        async with offline_client() as http:
            record = await make_service(http).crossmatch(RA_3C273, DEC_3C273, radius_arcsec=10.0)
    if record.failures:
        raise SystemExit(f"replay failures: {record.failures}")
    sources, citations = [], {}
    for name, result in record.catalog_results.items():
        if result.get("citation"):
            citations[name] = result["citation"]
        for src in result["sources"]:
            sources.append({
                "catalog": src.catalog, "source_id": src.source_id, "ra": src.ra, "dec": src.dec,
                "positional_error_arcsec": src.positional_error_arcsec, "data": {},
                "metadata": {"wavelength": src.metadata.get("wavelength"), "physical": src.metadata.get("physical", {}),
                             "citation": src.metadata.get("citation")},
                "provenance": {"derived_from": f"tests/fixtures/3c273/{name}.json"},
                "epoch": src.epoch, "proper_motion_ra_masyr": src.proper_motion_ra_masyr,
                "proper_motion_dec_masyr": src.proper_motion_dec_masyr,
                "position_uncertainty_arcsec": src.position_uncertainty_arcsec,
            })
    payload = {
        "target": "3C 273", "ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0,
        "note": "Catalog rows parsed by providers.py from the real archive responses recorded in tests/fixtures/3c273.",
        "catalogs": sorted(record.catalog_results), "citations": citations, "sources": sources,
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {len(sources)} rows from {len(record.catalog_results)} catalogs to {FIXTURE}")


async def _record_vo() -> None:
    """Record the real archive exchanges of one VO cone (3C 273, 10", VO row limit) for replay."""
    import os

    os.environ["PROVIDER_CACHE_TTL_SECONDS"] = "0"
    os.environ.pop("REDIS_URL", None)
    import httpx
    from fixture_io import FIXTURES, _request_text, match_tokens, redact
    from helpers import make_service, shared_ssl_context

    registry = CatalogRegistry()
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    config = vo_server.VOConfig.from_env()
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True, verify=shared_ssl_context(),
                                 event_hooks={"response": [hook]}) as http:
        fetched = await vo_server.fetch_match_rows(make_service(http), RA_3C273, DEC_3C273, TEN_ARCSEC_DEG,
                                                   catalogs=VO_REPLAY_CATALOGS, row_limit=config.catalog_row_limit,
                                                   max_total_rows=config.max_cone_rows)
    if fetched.failed or fetched.truncated:
        raise SystemExit(f"incomplete recording: {fetched.failed} {fetched.truncated}")
    folder = FIXTURES / VO_FIXTURES
    folder.mkdir(parents=True, exist_ok=True)
    for name in VO_REPLAY_CATALOGS:
        tokens = match_tokens(registry.get(name))
        mine = [(rq, rs) for rq, rs in log if any(tok in _request_text(rq) for tok in tokens)]
        if not mine:
            raise SystemExit(f"no exchange recorded for {name}")
        for old in folder.glob(f"{name}.*.body"):
            old.unlink()
        exchanges = []
        for idx, (request, response) in enumerate(mine):
            (folder / f"{name}.{idx}.body").write_bytes(redact(response.content))
            exchanges.append({"method": request.method, "url": str(request.url),
                              "request_body": request.content.decode("utf-8", "replace") if request.content else "",
                              "status_code": response.status_code,
                              "content_type": response.headers.get("content-type", ""), "match": tokens})
        meta = {"catalog": name, "target": "3c273", "ra": RA_3C273, "dec": DEC_3C273, "radius_arcsec": 10.0,
                "note": "Recorded through vo_server.fetch_match_rows with the VO per-archive row limit.",
                "exchanges": exchanges}
        (folder / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{name:<10} {len(mine)} exchange(s)")
    print(f"{len(fetched.rows)} rows; recorded to {folder}")


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        asyncio.run(_record())
    elif sys.argv[1:] == ["record-vo"]:
        asyncio.run(_record_vo())
    else:
        print(__doc__)

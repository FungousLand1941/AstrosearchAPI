"""Offline: mirroring through the REAL MAST, SDSS SqlSearch and IRSA Gator adapters.

A mock archive (respx) answers every request in the service's own wire format -- MAST
catalogs API JSON (``info`` + ``data``, TOP ``pagesize`` rows sorted by ``distance`` in
degrees), SkyServer SqlSearch JSON (``TOP N`` rows of ``fGetNearbyObjEq`` with
``dist_arcsec``, several rows per objID when an object has several spectra) and Gator IPAC
tables (the whole cone, unordered, with ``dist`` in arcsec and ``angle`` in degrees E of N
printed with 6 decimals) -- computed from a synthetic catalog with astropy's (Vincenty)
separations and position angles. The real adapters (``providers.MASTProvider`` etc.) parse
them; the mirror, the store and LocalProvider must then give exactly the adapters' remote
answer, including the query-relative distance columns each archive computes.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import math
import re

import astropy.units as u
import httpx
import numpy as np
import pytest
import respx
from astropy.coordinates import SkyCoord
from astropy.table import MaskedColumn, Table
from helpers import offline_client
from test_skycache import cap_points

from models import DEFAULT_CATALOGS, IRSA_GATOR, catalog_from_dict, validate_target
from providers import CacheManager, provider_map
from skycache import CoverageError, LocalProvider, SkyCache, mirror_region

CENTER = (150.25, 20.5)  # few decimals: the adapters print centres with 9 decimals
MIRROR_RADIUS_DEG = 0.12
# (ra, dec, radius_arcsec): radii exact in the adapters' printed units (deg / arcmin / arcsec).
CONES = [(150.25, 20.5, 36.0), (150.28, 20.47, 72.0), (150.22, 20.53, 18.0), (150.25, 20.5, 288.0)]

# Tolerances for the columns archives compute relative to the query centre, in the column's own
# unit: MAST 'distance' (deg) and SDSS 'dist_arcsec' are full-precision floats (haversine vs
# Vincenty: ~1e-16 deg); Gator prints dist (arcsec) and angle (deg) with 6 decimals.
TOLERANCES = {"distance": 1e-12, "dist_arcsec": 1e-8, "dist": 1.5e-6, "angle": 1.5e-6}


@pytest.fixture(autouse=True)
def _fast_guards(monkeypatch):
    monkeypatch.setenv("PROVIDER_REQUESTS_PER_SECOND", "10000")


class Sky:
    """A synthetic catalog: positions (rounded to 1e-7 deg) and per-row values."""

    def __init__(self, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        ra, dec = cap_points(CENTER[0], CENTER[1], MIRROR_RADIUS_DEG + 0.05, n, seed=seed)
        self.ra = np.round(ra, 7)
        self.dec = np.round(dec, 7)
        self.n = n
        self.rng = rng

    def separations(self, ra0: float, dec0: float) -> tuple[np.ndarray, np.ndarray]:
        centre = SkyCoord(ra0 * u.deg, dec0 * u.deg)
        points = SkyCoord(self.ra * u.deg, self.dec * u.deg)
        return centre.separation(points).deg, centre.position_angle(points).deg


# ---------------------------------------------------------------------------
# MAST catalogs API (Pan-STARRS DR2 mean objects)
# ---------------------------------------------------------------------------


MAST_INFO = {  # as the real service labels them (raMeanErr 'deg' and distance 'arcsec' are wrong)
    "objID": ("long", "NULL", "meta.id;meta.main"), "objName": ("char", "NULL", "meta.id;meta.main"),
    "raMean": ("double", "deg", "pos.eq.ra;meta.main"), "decMean": ("double", "deg", "pos.eq.dec;meta.main"),
    "raMeanErr": ("float", "deg", "stat.error"), "decMeanErr": ("float", "deg", "stat.error"),
    "epochMean": ("double", "d", "time.epoch"), "nDetections": ("short", "NULL", "meta.number"),
    "qualityFlag": ("unsignedByte", "NULL", "meta.code.qual"), "iMeanKronMag": ("float", "mag", "phot.mag"),
    "distance": ("double", "arcsec", "NULL"),
}


def _mast_info(name: str) -> dict:
    kind, unit, ucd = MAST_INFO.get(name, ("float", "mag", "phot.mag"))
    return {"name": name, "type": kind, "datatype": kind, "unit": unit, "ucd": ucd, "description": name}


class MastArchive:
    def __init__(self, sky: Sky) -> None:
        self.sky = sky
        rng = sky.rng
        n = sky.n
        self.obj_id = (108_000_000_000_000_000 + np.arange(n) * 104_729).tolist()
        self.ndet = rng.integers(1, 40, n)  # rows with nDetections <= 1 are filtered server-side
        self.values = {
            "raMeanErr": rng.uniform(0.005, 0.2, n).round(6), "decMeanErr": rng.uniform(0.005, 0.2, n).round(6),
            "epochMean": rng.uniform(55200, 56900, n).round(5), "qualityFlag": rng.integers(0, 64, n),
        }
        self.calls = 0

    def row(self, i: int, columns: list[str], distance: float) -> list:
        out = []
        for c in columns:
            if c == "objID":
                out.append(self.obj_id[i])
            elif c == "objName":
                out.append(f"PSO J{self.sky.ra[i]:.4f}{self.sky.dec[i]:+.4f}")
            elif c == "raMean":
                out.append(float(self.sky.ra[i]))
            elif c == "decMean":
                out.append(float(self.sky.dec[i]))
            elif c == "nDetections":
                out.append(int(self.ndet[i]))
            elif c == "distance":
                out.append(distance)
            elif c in self.values:
                v = self.values[c][i]
                out.append(int(v) if c == "qualityFlag" else float(v))
            else:  # magnitudes / errors, with the -999 sentinel for missing bands
                v = ((sum(map(ord, c)) * 31 + i * 7) % 1000) / 100.0 + 12.0
                out.append(-999.0 if (i + len(c)) % 9 == 0 else round(v, 4))
        return out

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        q = request.url.params
        ra0, dec0, radius = float(q["ra"]), float(q["dec"]), float(q["radius"])
        columns = json.loads(q["columns"])
        assert q["sort_by"] == "distance"
        page = int(q["pagesize"])
        min_det = int(q.get("nDetections.gt", "-1"))
        sep, _pa = self.sky.separations(ra0, dec0)
        idx = np.flatnonzero((sep <= radius) & (self.ndet > min_det))
        idx = idx[np.lexsort((np.asarray(self.obj_id)[idx], sep[idx]))][:page]
        body = {"info": [_mast_info(c) for c in columns], "data": [self.row(int(i), columns, float(sep[i])) for i in idx]}
        return httpx.Response(200, json=body)


# ---------------------------------------------------------------------------
# SDSS SkyServer SqlSearch
# ---------------------------------------------------------------------------


class SdssArchive:
    NEARBY = re.compile(r"fGetNearbyObjEq\(([-0-9.]+), ([-0-9.]+), ([-0-9.]+)\)")

    def __init__(self, sky: Sky) -> None:
        self.sky = sky
        rng = sky.rng
        n = sky.n
        self.obj_id = (1_237_651_735_000_000_000 + np.arange(n) * 7_919).tolist()
        # Objects with 2 spectra appear twice (LEFT JOIN SpecObj), same objID and position.
        self.spectra = [2 if i % 17 == 0 else 1 for i in range(n)]
        self.mjd = rng.integers(51_000, 54_500, n)
        self.err = rng.uniform(0.02, 0.2, (n, 2)).round(6)
        self.mag = rng.uniform(14, 22, (n, 5)).round(5)
        self.calls = 0

    def rows(self, i: int, dist_arcsec: float) -> list[dict]:
        out = []
        for k in range(self.spectra[i]):
            has_spec = self.spectra[i] > 1 or i % 5 == 0
            out.append({
                "objID": self.obj_id[i], "dist_arcsec": dist_arcsec, "ra": float(self.sky.ra[i]),
                "dec": float(self.sky.dec[i]), "raErr": float(self.err[i, 0]), "decErr": float(self.err[i, 1]),
                "mjd": int(self.mjd[i]), "type": 6 if i % 3 else 3, "clean": i % 2, "mode": 1,
                "psfMag_u": float(self.mag[i, 0]), "psfMag_g": float(self.mag[i, 1]), "psfMag_r": float(self.mag[i, 2]),
                "psfMag_i": float(self.mag[i, 3]), "psfMag_z": float(self.mag[i, 4]), "psfMagErr_r": 0.01,
                "modelMag_r": float(self.mag[i, 2]) + 0.1,
                "photoz": None if i % 4 == 0 else round(0.1 + (i % 50) / 100.0, 4), "photozErr": None,
                "specz": round(0.05 * (k + 1) + i * 1e-5, 6) if has_spec else None,
                "speczErr": 1e-4 if has_spec else None,
                "specClass": ("GALAXY" if k == 0 else "QSO") if has_spec else "",
            })
        return out

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        sql = request.url.params["cmd"]
        top = int(re.search(r"SELECT TOP (\d+)", sql).group(1))
        ra0, dec0, radius_arcmin = (float(v) for v in self.NEARBY.search(sql).groups())
        sep, _pa = self.sky.separations(ra0, dec0)
        idx = np.flatnonzero(sep * 60.0 <= radius_arcmin)
        idx = idx[np.lexsort((np.asarray(self.obj_id)[idx], sep[idx]))]
        rows = [r for i in idx for r in self.rows(int(i), float(sep[i] * 60.0) * 60.0)][:top]
        return httpx.Response(200, json=[{"TableName": "Table1", "Rows": rows},
                                         {"TableName": "SqlQuery", "Rows": [{"query": sql}]}])


# ---------------------------------------------------------------------------
# IRSA Gator (IPAC table of the whole cone, unordered)
# ---------------------------------------------------------------------------


class GatorArchive:
    def __init__(self, sky: Sky) -> None:
        self.sky = sky
        rng = sky.rng
        n = sky.n
        self.designation = [f"{i:08d}+{(i * 37) % 10_000_000:07d}" for i in range(n)]
        self.err = np.column_stack([rng.uniform(0.06, 0.3, n), rng.uniform(0.05, 0.25, n)]).round(2)
        self.err_ang = rng.integers(0, 180, n)
        self.jdate = rng.uniform(2_450_600.5, 2_451_900.5, n).round(4)
        self.mags = rng.uniform(8, 16.5, (n, 3)).round(3)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        q = request.url.params
        assert q["spatial"] == "cone" and q["radunits"] == "arcsec" and q["outfmt"] == "1"
        ra0, dec0 = (float(v) for v in q["objstr"].split())
        radius = float(q["radius"])
        selcols = q["selcols"].split(",")
        sep, pa_ = self.sky.separations(ra0, dec0)
        idx = np.flatnonzero(sep * 3600.0 <= radius)
        idx = idx[np.random.default_rng(len(idx)).permutation(idx.size)]  # Gator order is arbitrary
        table = Table()
        units = {"ra": u.deg, "dec": u.deg, "err_maj": u.arcsec, "err_min": u.arcsec, "err_ang": u.deg,
                 "j_m": u.mag, "h_m": u.mag, "k_m": u.mag, "j_msigcom": u.mag, "h_msigcom": u.mag, "k_msigcom": u.mag}
        for col in selcols:
            if col == "designation":
                values = [self.designation[i] for i in idx]
            elif col == "ra":
                values = self.sky.ra[idx]
            elif col == "dec":
                values = self.sky.dec[idx]
            elif col in ("err_maj", "err_min"):
                values = self.err[idx, 0 if col == "err_maj" else 1]
            elif col == "err_ang":
                values = self.err_ang[idx]
            elif col in ("j_m", "h_m", "k_m"):
                values = self.mags[idx, "jhk".index(col[0])]
            elif col == "jdate":
                values = self.jdate[idx]
            elif col in ("ph_qual", "cc_flg"):
                values = ["AAA" if i % 3 else "AUU" for i in idx]
            elif col == "ext_key":  # extended-source key: null for most point sources
                values = MaskedColumn(np.asarray(idx, dtype=np.int64), mask=np.asarray(idx % 4 != 0, dtype=bool))
            else:  # *_msigcom: null for the U(pper limit) bands
                values = MaskedColumn(np.full(idx.size, 0.02), mask=np.asarray(idx % 3 == 0, dtype=bool))
            table[col] = values
            if col in units:
                table[col].unit = units[col]
        table["dist"] = sep[idx] * 3600.0
        table["dist"].unit = u.arcsec
        table["angle"] = pa_[idx]
        table["angle"].unit = u.deg
        for col in ("ra", "dec"):
            table[col].format = ".7f"
        for col in ("dist", "angle"):
            table[col].format = ".6f"
        out = io.StringIO()
        table.write(out, format="ascii.ipac")
        text = "\\fixlen = T\n" + out.getvalue()
        return httpx.Response(200, text=text, headers={"content-type": "text/plain"})


def gator_definition():
    entry = copy.deepcopy(DEFAULT_CATALOGS["twomass_psc"])
    entry.update({"provider": "irsa_gator", "endpoint": IRSA_GATOR, "description": "2MASS PSC via IRSA Gator"})
    for key in ("fallback", "format", "distance"):
        entry["parameters"].pop(key, None)
    return catalog_from_dict("twomass_gator", entry)


ARCHIVES = {
    "mast": (lambda: catalog_from_dict("panstarrs_dr2", copy.deepcopy(DEFAULT_CATALOGS["panstarrs_dr2"])),
             MastArchive, "catalogs.mast.stsci.edu", 1500),
    "sdss": (lambda: catalog_from_dict("sdss", copy.deepcopy(DEFAULT_CATALOGS["sdss"])),
             SdssArchive, "skyserver.sdss.org", 1500),
    "gator": (gator_definition, GatorArchive, "irsa.ipac.caltech.edu", 1500),
}


def assert_local_equals_remote(local, remote) -> None:
    """Same sources, order, data (query-relative columns within TOLERANCES), metadata and meta."""
    assert [s.source_id for s in local] == [s.source_id for s in remote]
    for a, b in zip(local, remote):
        da, db = a.as_dict(), b.as_dict()
        pa_, pb_ = da.pop("provenance"), db.pop("provenance")
        assert pa_["provider"] == "skycache" and pa_["source_id"] == pb_["source_id"]
        for name, tol in TOLERANCES.items():
            if name in db["data"]:
                assert name in da["data"], f"query-relative column {name!r} missing locally"
                assert math.isclose(da["data"].pop(name), db["data"].pop(name), rel_tol=0.0, abs_tol=tol), name
        assert da == db
    for key in ("status", "row_count", "raw_row_count", "truncated", "archive_truncated", "excess_row_count",
                "pad_row_count", "requested_radius_arcsec", "query_radius_arcsec", "cone", "warnings", "citation",
                "row_limit", "max_rows", "dropped_rows", "filtered_rows", "epoch_span"):
        assert local.meta[key] == remote.meta[key], key
    for key in ("pad_sources", "excess_sources"):
        assert [s.source_id for s in local.meta[key]] == [s.source_id for s in remote.meta[key]], key


@pytest.mark.parametrize("kind", sorted(ARCHIVES))
def test_mirror_through_real_adapter_local_equals_remote(kind, tmp_path):
    make_definition, archive_cls, host, n = ARCHIVES[kind]
    definition = make_definition()
    archive = archive_cls(Sky(n, seed=len(kind)))
    store = SkyCache(tmp_path / "store", partition_rows=250)

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host=host).mock(side_effect=archive)
            async with offline_client() as client:
                providers = provider_map(client, cache=CacheManager(None))
                report = await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=MIRROR_RADIUS_DEG,
                                             store=store, providers=providers, tile_rows=120)
                remote = providers[definition.provider]
                local = LocalProvider(store)
                pairs = []
                for ra, dec, radius in CONES:
                    target = validate_target(ra, dec)
                    pairs.append((await local.query(definition, target, radius),
                                  await remote.query(definition, target, radius)))
                return report, pairs

    report, pairs = asyncio.run(scenario())
    assert report.tiles_failed == 0 and report.region_covered_fraction == 1.0
    assert any("split into" in w for w in report.warnings) and report.queries > 5  # tiled mirror
    assert report.providers_used == [definition.provider]
    for got, want in pairs:
        assert got.meta["skycache"]["hit"] is True
        assert_local_equals_remote(got, want)
    counts = [len(want) for _got, want in pairs]
    assert min(counts) >= 1 and max(counts) == definition.max_rows  # the 288" cone is truncated at max_rows
    rows = store.load_rows(definition.name)
    if kind == "sdss":
        # Multi-spectrum objects keep both rows exactly once, although overlapping tiles refetched them.
        per_id: dict[str, int] = {}
        for row in rows:
            per_id[row.source_id] = per_id.get(row.source_id, 0) + 1
        expected = {str(archive.obj_id[i]): archive.spectra[i] for i in range(archive.sky.n)
                    if str(archive.obj_id[i]) in per_id}
        assert per_id == expected and max(per_id.values()) == 2
    else:
        ids = [r.source_id for r in rows]
        assert len(ids) == len(set(ids))
    if kind == "mast":  # server-side filter nDetections > 1: filtered rows never enter the store
        stored = {r.source_id for r in rows}
        assert not stored & {str(archive.obj_id[i]) for i in np.flatnonzero(archive.ndet <= 1)}
    # Query-relative columns are never stored (recomputed per query).
    assert not {"distance", "dist_arcsec", "dist", "angle", "match_dist"} & set(rows[0].data)


def test_gator_local_returns_whole_cone_like_gator(tmp_path):
    """Gator answers with the whole cone: local excess rows match remote ones, not just TOP N+1."""
    definition = gator_definition()
    definition.max_rows = 20
    archive = GatorArchive(Sky(900, seed=5))
    store = SkyCache(tmp_path / "store")

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="irsa.ipac.caltech.edu").mock(side_effect=archive)
            async with offline_client() as client:
                providers = provider_map(client, cache=CacheManager(None))
                await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=0.08, store=store,
                                    providers=providers, tile_rows=5000)
                target = validate_target(*CENTER)
                return (await LocalProvider(store).query(definition, target, 144.0),
                        await providers["irsa_gator"].query(definition, target, 144.0))

    got, want = asyncio.run(scenario())
    assert len(want.meta["excess_sources"]) > 1
    assert_local_equals_remote(got, want)


def test_mirror_region_outside_coverage_still_refused(tmp_path):
    definition = catalog_from_dict("sdss", copy.deepcopy(DEFAULT_CATALOGS["sdss"]))
    archive = SdssArchive(Sky(600, seed=9))
    store = SkyCache(tmp_path / "store")

    async def scenario():
        with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
            router.get(host="skyserver.sdss.org").mock(side_effect=archive)
            async with offline_client() as client:
                providers = provider_map(client, cache=CacheManager(None))
                await mirror_region(definition, ra=CENTER[0], dec=CENTER[1], radius_deg=0.05, store=store,
                                    providers=providers)
        with pytest.raises(CoverageError) as err:
            await LocalProvider(store).query(definition, validate_target(CENTER[0], CENTER[1] + 0.05), 30.0)
        return err.value

    assert asyncio.run(scenario()).reason == "partially_covered"

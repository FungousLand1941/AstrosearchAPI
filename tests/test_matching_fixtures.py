"""Recorded real-archive cones for the Bayesian matching engine (crowded fields).

Recording (live network, one cone query per catalogue)::

    .venv/Scripts/python.exe tests/test_matching_fixtures.py record

stores ``tests/fixtures/matching/<key>/<catalog>.json`` + bodies with the helpers of
``fixture_io`` (same format, strict request replay). The offline tests below replay them.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import respx

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import fixture_io
from fixture_io import load_exchanges, replay_side_effect
from helpers import make_service, offline_client

# A crowded bulge field near Baade's window (l = 4.16, b = -3.30): Gaia DR3 holds 268 sources
# within 30" (~95 per arcmin^2), so a 20" cone holds ~120 stars seen by several surveys.
CROWDED_KEY = "bulge_field"
CROWDED = {"ra": 272.0, "dec": -27.0, "radius_arcsec": 20.0}
CROWDED_CATALOGS = ["gaia_dr3", "twomass_psc", "allwise", "panstarrs_dr2"]
FIXTURE_DIR = f"matching/{CROWDED_KEY}"


def _register() -> None:
    # fixture_io records targets listed in its dictionaries: add ours at run time.
    fixture_io.EPOCH_TARGETS.setdefault(CROWDED_KEY, dict(CROWDED))


async def record_crowded() -> None:
    _register()
    for name in CROWDED_CATALOGS:
        catalog, records, outcome = await fixture_io._record_catalog(name, CROWDED_KEY, CROWDED["ra"], CROWDED["dec"])
        if records:
            fixture_io._save(FIXTURE_DIR, catalog, records, coords_key=CROWDED_KEY)
        print(f"{CROWDED_KEY:<14} {catalog:<16} {outcome}")


async def crowded_record(**kwargs):
    exchanges = load_exchanges(FIXTURE_DIR)
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(exchanges))
        async with offline_client() as client:
            service = make_service(client)
            return await service.crossmatch(CROWDED["ra"], CROWDED["dec"], radius_arcsec=CROWDED["radius_arcsec"],
                                            catalogs=CROWDED_CATALOGS, **kwargs)


def independent_members(group: dict) -> list[dict]:
    return [m for m in group["members"] if m["coincident_with"] is None]


@pytest.mark.skipif(not (fixture_io.FIXTURES / FIXTURE_DIR).exists(), reason="crowded-field fixtures not recorded")
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


@pytest.mark.skipif(not (fixture_io.FIXTURES / FIXTURE_DIR).exists(), reason="crowded-field fixtures not recorded")
async def test_crowded_field_densities_are_local() -> None:
    record = await crowded_record()
    densities = record.provenance["association"]["densities"]
    # The Galactic plane is far denser in Gaia than the all-sky mean (43,900 per deg^2):
    # the density used by the prior comes from the cone, not the catalogue average.
    assert densities["gaia_dr3"]["density_per_deg2"] > 1.5 * densities["gaia_dr3"]["sky_mean_per_deg2"]
    assert densities["gaia_dr3"]["method"] == "gamma_poisson"


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        asyncio.run(record_crowded())
    else:
        print(__doc__)

"""Live tests for sed.py against the real archives (run with ``pytest -m live``).

Astrophysical truths asserted: 3C 273 is a quasar at z = 0.158 (Schmidt 1963); M87 is a (radio-loud,
AGN-hosting) giant elliptical at z = 0.0043; Vega and HD 209458 are stars. Tests skip only when a service
is unreachable (network error, timeout, HTTP 5xx); wrong or empty answers fail.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import httpx
import pytest

import sed
from models import ObjectResolutionError

pytestmark = pytest.mark.live

UNREACHABLE = {"CatalogUnavailableError", "QueryTimeoutError", "RateLimitedError"}
SDSS_QSO = (180.01108, 33.171292)


def all_unreachable(failures: list[dict[str, Any]]) -> bool:
    """True when every crossmatch failure is an outage (network error, timeout, rate limit, HTTP 5xx)."""
    return bool(failures) and all(f.get("error_type") in UNREACHABLE for f in failures)


async def build(tmp_path: Path, **kwargs: Any) -> dict[str, Any]:
    filters = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    try:
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            return await sed.build_sed(client=client, filters=filters, **kwargs)
    except httpx.TransportError as exc:
        pytest.skip(f"archive unreachable: {exc}")
    except sed.SEDUpstreamError as exc:
        # Skip only for outages; parse/query errors (ResponseParseError, CatalogQueryError, ...) are regressions.
        if exc.failures and all_unreachable(exc.failures):
            pytest.skip(f"archive unreachable: {exc}")
        raise
    except ObjectResolutionError as exc:
        if str(exc).startswith("Sesame request failed"):
            pytest.skip(f"Sesame unreachable: {exc}")
        raise


def require_catalogs(result: dict[str, Any], catalogs: list[str]) -> None:
    """Skip when a catalog the assertion depends on was unreachable (never when it answered empty)."""
    down = [f for f in result["failures"] if f.get("catalog") in catalogs and f.get("error_type") in UNREACHABLE]
    if down:
        pytest.skip(f"unreachable: {[(f['catalog'], f['error_type']) for f in down]}")


def band(result: dict[str, Any], facility: str, name: str) -> dict[str, Any]:
    hits = [p for p in result["points"] if p["facility"].startswith(facility) and p["band"] == name]
    assert hits, f"missing {facility} {name}"
    return hits[0]


async def test_3c273_is_a_quasar_at_z_0158(tmp_path: Path) -> None:
    result = await build(tmp_path, name="3C 273")
    require_catalogs(result, ["gaia_dr3", "allwise", "ned", "simbad", "first", "twomass_psc"])
    assert result["classification"]["label"] == "qso", result["classification"]
    z = result["redshift"]
    assert z["kind"] == "spec"
    assert z["value"] == pytest.approx(0.158, abs=0.001)
    # Mid-IR AGN colour (Stern et al. 2012) from live AllWISE.
    w1, w2 = band(result, "WISE", "W1"), band(result, "WISE", "W2")
    assert w1["magnitude"] - w2["magnitude"] >= 0.8
    # 3C 273 is a ~37-55 Jy source at 1.4 GHz (FIRST integrated / NVSS).
    assert 20.0 < band(result, "FIRST", "1.4 GHz")["flux_jy"] < 80.0
    # Ks ~ 10 mag (Vega) -> ~0.07 Jy with the SVO 2MASS zero point (666.8 Jy).
    ks = band(result, "2MASS", "Ks")
    assert ks["zero_point_jy"] == pytest.approx(666.8)
    assert ks["flux_jy"] == pytest.approx(666.8 * 10 ** (-0.4 * ks["magnitude"]), rel=1e-9)
    assert 0.03 < ks["flux_jy"] < 0.15
    assert any(p["regime"] == "xray" for p in result["points"])
    assert result["gaia_extra"] is not None and result["gaia_extra"]["classprob_dsc_combmod_quasar"] > 0.5


async def test_m87_is_a_galaxy_hosting_an_agn_at_z_0004(tmp_path: Path) -> None:
    result = await build(tmp_path, name="M87")
    require_catalogs(result, ["ned", "simbad", "first", "allwise"])
    assert result["classification"]["label"] in {"galaxy", "agn"}, result["classification"]
    z = result["redshift"]
    # A spectroscopic value, or NED's vetted value whose zflag ('UUN') does not state the technique.
    assert z["kind"] == "spec" or (z["source"], z["kind"]) == ("ned", None)
    assert z["reliable"] is True
    assert z["value"] == pytest.approx(0.0043, abs=0.0005)
    evidence = " | ".join(result["classification"]["evidence"])
    assert "radio-loud" in evidence  # Virgo A
    assert band(result, "FIRST", "1.4 GHz")["flux_jy"] > 50.0


async def test_vega_is_a_star(tmp_path: Path) -> None:
    result = await build(tmp_path, name="Vega")
    require_catalogs(result, ["simbad"])
    cls = result["classification"]
    assert cls["label"] == "star", cls
    assert cls["scores"]["star"] > 0.9
    # Vega defines 0 mag: its SIMBAD V (0.03) is ~3500 Jy with the SVO Johnson V zero point (3617.5 Jy).
    v = band(result, "SIMBAD", "V")
    assert 3000.0 < v["flux_jy"] < 3700.0
    assert "Doppler" in result["redshift"].get("note", "")


async def test_hd209458_is_a_star_from_gaia_parallax(tmp_path: Path) -> None:
    # Exoplanet Archive J2015.5 position (Gaia DR3 1779546757669063552); HD 209458 has parallax ~20.7 mas.
    result = await build(tmp_path, ra=330.79502, dec=18.88432, radius_arcsec=5.0)
    require_catalogs(result, ["gaia_dr3"])
    assert result["classification"]["label"] == "star", result["classification"]
    evidence = " | ".join(result["classification"]["evidence"])
    assert "significant parallax" in evidence
    g = band(result, "Gaia DR3", "G")
    assert 7.0 < g["magnitude"] < 7.6 and g["flux_err_jy"] is not None


async def test_svo_live_zero_points_match_embedded_table(tmp_path: Path) -> None:
    catalog = sed.FilterCatalog(cache_path=tmp_path / "svo.json")
    ids = ["2MASS/2MASS.Ks", "WISE/WISE.W1", "GAIA/GAIA3.G", "SLOAN/SDSS.u"]
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        await catalog.prefetch(ids, client)
    network = [fid for fid, err in catalog.errors.items() if "ConnectError" in err or "Timeout" in err or "HTTP 5" in err]
    if network:
        pytest.skip(f"SVO unreachable: {network}")
    assert not catalog.errors, catalog.errors
    for fid in ids:
        info = catalog.info(fid)
        assert info.metadata_origin == "svo"
        assert info.svo_vega_zero_point_jy == pytest.approx(sed.EMBEDDED_FILTERS[fid]["ZeroPoint"], rel=1e-9)
        assert info.wavelength_eff_um == pytest.approx(sed.EMBEDDED_FILTERS[fid]["WavelengthEff"] / 1e4, rel=1e-9)
    assert catalog.info("2MASS/2MASS.Ks").zero_point_jy == pytest.approx(666.8)
    assert catalog.info("SLOAN/SDSS.u").zero_point_jy == 3631.0


async def test_sdss_specobj_live() -> None:
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            cands = await sed.fetch_sdss_redshifts(client, *SDSS_QSO)
    except (httpx.TransportError, sed.SEDUpstreamError) as exc:
        if isinstance(exc, sed.SEDUpstreamError) and "HTTP 5" not in str(exc):
            raise
        pytest.skip(f"SkyServer unreachable: {exc}")
    spec = [c for c in cands if c["kind"] == "spec" and c["reliable"]]
    assert spec, cands
    assert spec[0]["value"] == pytest.approx(1.008554, abs=1e-4)
    assert spec[0]["spec_class"] == "QSO"
    assert math.isfinite(spec[0]["error"])


# ---------------------------------------------------------------------------
# Round-1 review regressions (live truths)
# ---------------------------------------------------------------------------


def require_lookups(result: dict[str, Any], keys: list[str]) -> None:
    """Skip when a supplementary lookup the assertion depends on hit a network error / 5xx (never on parse errors)."""
    for note in result["notes"]:
        for key in keys:
            if note.startswith(f"{key} lookup failed") and any(t in note for t in ("ConnectError", "Timeout", "HTTP 5",
                                                                                      "RemoteProtocolError", "ReadError")):
                pytest.skip(f"supplementary lookup unreachable: {note}")


async def test_ngc5548_flagged_sdss_spectrum_loses_to_ned(tmp_path: Path) -> None:
    # SDSS SpecObj 2394812531219654656 (z = 0.016266) has zWarning = 16; NED's spectroscopic z = 0.017175.
    result = await build(tmp_path, name="NGC 5548")
    require_catalogs(result, ["ned", "sdss"])
    require_lookups(result, ["sdss", "sdss_object"])
    z = result["redshift"]
    assert (z["kind"], z["source"], z["reliable"]) == ("spec", "ned", True)
    assert z["value"] == pytest.approx(0.0172, abs=5e-4)
    flagged = [c for c in z["candidates"] if c["source"] in {"sdss", "sdss_specobj"}]
    assert flagged and all(c["reliable"] is False and c["z_warning"] != 0 for c in flagged)
    # SDSS galaxy photometry in one aperture (modelMag) in every band.
    sdss = [p for p in result["points"] if p["facility"] == "SDSS"]
    assert len(sdss) == 5 and all(any("modelMag" in n for n in p["notes"]) for p in sdss)
    assert "radio-loud" not in " | ".join(result["classification"]["evidence"])  # radio-quiet Seyfert 1


async def test_star_with_photometric_simbad_redshift(tmp_path: Path) -> None:
    # 2MASS J10005438+0227239: Galactic star (Gaia parallax/error ~ 235) with a COSMOS photo-z of 0.3 in SIMBAD.
    result = await build(tmp_path, ra=150.22660940485, dec=2.45665712683, radius_arcsec=5.0)
    require_catalogs(result, ["gaia_dr3", "simbad"])
    require_lookups(result, ["simbad", "gaia"])
    assert result["classification"]["label"] == "star"
    simbad = next(c for c in result["redshift"]["candidates"] if c["source"] == "simbad")
    assert simbad["kind"] == "photo" and simbad["nature"] == "p"
    assert result["redshift"]["kind"] != "spec" and result["redshift"]["conflict"] is True


async def test_cygnus_a_radio_lobes(tmp_path: Path) -> None:
    # Cygnus A: ~1.6 kJy at 1.4 GHz in two NVSS lobes ~45-52 arcsec from the nucleus.
    result = await build(tmp_path, name="Cygnus A", radius_arcsec=60.0)
    require_catalogs(result, ["nvss"])
    nvss = band(result, "NVSS", "1.4 GHz")
    assert 1000.0 < nvss["flux_jy"] < 2500.0
    assert "radio-loud" in " | ".join(result["classification"]["evidence"])
    assert result["classification"]["label"] in {"agn", "galaxy"}


async def test_ngc1068_is_radio_quiet(tmp_path: Path) -> None:
    result = await build(tmp_path, name="NGC 1068")
    require_catalogs(result, ["nvss", "first", "simbad"])
    evidence = " | ".join(result["classification"]["evidence"])
    assert "radio-loud" not in evidence and "radio-quiet" in evidence
    assert result["classification"]["label"] in {"agn", "galaxy"}


async def test_simbad_and_sdss_supplementary_live() -> None:
    try:
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            star = await sed.fetch_simbad_extra(client, "2MASS J10005438+0227239")
            quasar = await sed.fetch_simbad_extra(client, "3C 273")
            galaxy = await sed.fetch_sdss_object(client, "1237665532785786979")  # NGC 5548
    except (httpx.TransportError, sed.SEDUpstreamError) as exc:
        if isinstance(exc, sed.SEDUpstreamError) and "HTTP 5" not in str(exc):
            raise
        pytest.skip(f"archive unreachable: {exc}")
    assert star is not None and star["rvz_nature"] == "p"
    assert quasar is not None and sed.simbad_redshift_kind(quasar["rvz_nature"]) == "spec"
    assert quasar["rvz_redshift"] == pytest.approx(0.158, abs=0.002)
    assert quasar["fluxes"]["V"]["flux"] is not None
    assert galaxy is not None and galaxy["type"] == 3
    assert all(galaxy[f"modelMagErr_{b}"] is not None for b in "ugriz")
    spec = next(s for s in galaxy["spectra"] if s["specObjID"] == "2394812531219654656")
    assert spec["zWarning"] != 0 and spec["z"] == pytest.approx(0.01627, abs=1e-4)


# ---------------------------------------------------------------------------
# Round-2 review regressions (live truths)
# ---------------------------------------------------------------------------


def evidence_text(result: dict[str, Any]) -> str:
    return " | ".join(result["classification"]["evidence"])


@pytest.mark.parametrize(("name", "z_true"), [("3C 351", 0.3715), ("QSO B1422+231", 3.62)])
async def test_quasar_redshift_is_not_an_absorbers(tmp_path: Path, name: str, z_true: float) -> None:
    # 3C 351: NED lists the absorber '[HB89] 1704+608 ABS01' (z = 0.2216) nearer than the quasar row (z = 0.3715);
    # B1422+231: '[PBW92] B1422+231 ABS01' (z = 3.5375) at the quasar position (z = 3.62).
    result = await build(tmp_path, name=name)
    require_catalogs(result, ["ned", "simbad"])
    z = result["redshift"]
    assert z["kind"] == "spec" and z["reliable"] is True
    assert z["value"] == pytest.approx(z_true, abs=0.01), z
    ned = next(m for m in result["members"] if m["catalog"] == "ned")
    assert "ABS" not in ned["source_id"]
    assert result["classification"]["label"] == "qso", result["classification"]


async def test_centaurus_a_is_a_galaxy_not_a_quasar(tmp_path: Path) -> None:
    # NGC 5128: the nearest radio galaxy; its Gaia counterpart is a G ~ 21 knot, its integrated V = 6.84.
    result = await build(tmp_path, name="NGC 5128")
    require_catalogs(result, ["simbad", "ned", "allwise", "gaia_dr3"])
    assert result["classification"]["label"] in {"galaxy", "agn"}, result["classification"]
    text = evidence_text(result)
    assert "Gaia G as V proxy" not in text
    assert result["redshift"]["value"] == pytest.approx(0.0018, abs=0.0003)


@pytest.mark.parametrize("name", ["3C 279", "PKS 0118-272"])
async def test_blazar_proper_motion_is_not_galactic(tmp_path: Path, name: str) -> None:
    result = await build(tmp_path, name=name)
    require_catalogs(result, ["gaia_dr3", "simbad", "ned"])
    require_lookups(result, ["gaia"])
    assert "moving, Galactic" not in evidence_text(result)
    assert result["gaia_extra"]["astrometric_excess_noise_sig"] > 2.0
    assert result["classification"]["label"] == "qso", result["classification"]


async def test_aldebaran_saturated_wise_is_flagged(tmp_path: Path) -> None:
    # Aldebaran (K5 III, Ks ~ -3.0): AllWISE W1/W2/W3 are ph_qual 'U' "limits" at saturated brightnesses (5.6, 1.3,
    # -2.8 mag, brighter than the nominal saturation limits 8, 7, 3.8): failed extractions, flagged. W4 = -2.93
    # (ph_qual A) is a valid profile-fit measurement: profile fitting of the unsaturated wings is reliable to -4.0 mag
    # in W4 (All-Sky Explanatory Supplement VI.3.d; the 0.4 mag limit is the APERTURE-photometry one), and it lies on
    # the Rayleigh-Jeans tail of Ks (tests/test_sed_round2.py: 124 Jy vs 105 Jy).
    result = await build(tmp_path, name="Aldebaran")
    require_catalogs(result, ["allwise", "twomass_psc", "simbad"])
    for name in ("W1", "W2", "W3"):
        p = band(result, "WISE", name)
        assert p["is_upper_limit"] and p["quality_warning"], (name, p)
        assert any("nominal saturation limit" in w and "failed extraction" in w for w in p["warnings"]), p["warnings"]
    w4 = band(result, "WISE", "W4")
    assert not w4["is_upper_limit"] and not w4["quality_warning"], w4["warnings"]
    assert -4.0 < w4["magnitude"] < 0.4
    ks = band(result, "2MASS", "Ks")
    rj = ks["flux_jy"] * (ks["wavelength_um"] / w4["wavelength_um"]) ** 2
    assert rj / sed.WISE_RJ_MARGIN < w4["flux_jy"] < rj * sed.WISE_RJ_MARGIN
    assert "WISE W1-W2" not in " | ".join(result["classification"]["evidence"])  # flagged bands feed no colour rule
    assert result["classification"]["label"] == "star"


async def test_ngc4472_redshift_is_nearly_981_km_s(tmp_path: Path) -> None:
    # NGC 4472 (M49): NED z = 0.003272 (cz = 981 km/s), zflag 'UUN'; SIMBAD's quality-E z must not win over it.
    result = await build(tmp_path, name="NGC 4472")
    require_catalogs(result, ["ned", "simbad"])
    z = result["redshift"]
    assert z["reliable"] is True and z["value"] == pytest.approx(0.00327, abs=0.0002), z
    assert result["classification"]["label"] in {"galaxy", "agn"}

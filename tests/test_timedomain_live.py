"""Live truth tests for timedomain.py (run with ``pytest -m live``) and the fixture recorder.

Targets and catalogued truths:

* CSS J132708.3+384442 = ZTF J132708.32+384442.3 = Gaia DR3 1476301553808173824,
  an RRab star with P = 0.62974 d (AAVSO VSX; Catalina/Drake et al. 2014),
  0.6297706 d (Chen et al. 2020, ApJS 249, 18, ZTF periodic variables) and
  pf = 0.62974 d (Gaia DR3 vari_rrlyrae, Clementini et al. 2023).
* RR Lyr (TIC 159717514; TIC position from MAST): P = 0.566788 d (Kolenberg et al.
  2011, MNRAS 411, 878, Kepler photometry).
* 3C 273: optically variable quasar (e.g. ZTF/ASAS-SN monitoring; Soldi et al. 2008).
* SDSS Stripe 82 standard star at (10.742019, +1.126138), r = 15.817 with rms 0.006
  mag over 11 SDSS epochs (Ivezic et al. 2007, AJ 134, 973, catalog J/AJ/134/973);
  Gaia DR3 2549372572635491072, not in VSX: a photometrically quiet star.
* (4) Vesta and (1) Ceres positions from JPL Horizons (live) must be matched by SkyBoT.
* Eclipsing binaries from Chen et al. (2020, ApJS 249, 18; J/ApJS/249/18/table2, checked
  live): ZTFJ014057.73+582241.5, type EA, P = 1.8083775 d (rAmp 1.075 mag), and
  ZTFJ030001.97+010340.8, type EW (W UMa), P = 0.3858416 d. Both have their
  single-harmonic Lomb-Scargle peak at P/2; the pipeline must report the orbital period.
* Gaia DR3 387837952011193088 (GAPS field, G = 14.02, phot_variable_flag NOT_AVAILABLE):
  a source with GAPS epoch photometry (Evans et al. 2023, A&A 674, A4) that was flagged
  variable (G chi^2/dof = 9.9) with the uncalibrated pipeline errors.
* SDSS Stripe 82 standard at (321.620483, -0.892541), r = 14.467, rchi2 0.1 over 10
  epochs (Ivezic et al. 2007), whose NEOWISE visits alternate with latent-image
  (persistence) contaminated ones: constant in all default surveys.
* 61 Cyg A (TIC 165602000; 61 Cyg B = TIC 165602023 lies 30.7" away): a 45" cone holds
  three SPOC 2-min targets; only the nearest may be used.
* Barnard's star by name: the Sesame J2000 position and proper motion (-801.551,
  10362.394 mas/yr) must be propagated to J2016.0 to find Gaia DR3 4472832130942575872.

Only network errors, timeouts and HTTP 5xx skip; wrong physics fails.

Recording fixtures for the offline suite (polite: one pass, a few minutes):

    .venv/Scripts/python.exe tests/test_timedomain_live.py record [case ...]
"""

from __future__ import annotations

import asyncio
import gzip
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import timedomain as td

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "timedomain"

RRAB = (201.7846706228748, 38.74506826403494)  # Gaia DR3 1476301553808173824 (J2016.0; pm negligible here)
RRAB_PERIOD_DAYS = 0.62974
RR_LYR = (291.366301347666, 42.7843585093162)  # TIC 159717514
RR_LYR_PERIOD_DAYS = 0.566788
QSO_3C273 = (187.2779154, 2.0523883)  # SIMBAD J2000
QUIET_STAR = (10.742019, 1.126138)  # Ivezic et al. 2007 standard
VESTA_EPOCH_MJD = 60000.0  # 2023-02-25 00:00 UTC
CERES_EPOCH_MJD = 60400.0  # 2024-03-30 00:00 UTC
EA_STAR = (25.24055, 58.3782)  # ZTFJ014057.73+582241.5 (Chen et al. 2020)
EA_PERIOD_DAYS = 1.8083775
EW_STAR = (45.00823, 1.06134)  # ZTFJ030001.97+010340.8 (Chen et al. 2020)
EW_PERIOD_DAYS = 0.3858416
GAPS_QUIET = (9.572303162281901, 44.2737627337779)  # Gaia DR3 387837952011193088 (J2016.0)
GAPS_QUIET_ID = "387837952011193088"
PERSISTENCE_STD = (321.620483, -0.892541)  # Ivezic et al. 2007 standard, r = 14.467
CYG61_A = (316.72475, 38.74942)  # 61 Cyg A (TIC 165602000)
BARNARD_NAME = "Barnard" + chr(39) + "s star"
BARNARD_GAIA_ID = "4472832130942575872"
# Round-3 review targets:
CM_DRA_PERIOD_DAYS = 1.26838965  # AAVSO VSX; detached EB of two equal M dwarfs (pm 1.6 "/yr)
EA2_STAR = (12.077042, 62.975306)  # ZTFJ004818.49+625831.1 (Chen et al. 2020), EA with equal minima
EA2_PERIOD_DAYS = 1.2128074  # Chen et al. 2020 (Gaia DR3 vari_eclipsing_binary 1.21293, ASAS-SN 1.2129)
# Further Chen et al. (2020) EA systems whose Lomb-Scargle peak lies at P/2 or P/4 (review):
EA_MORE: tuple[tuple[str, float, float, float], ...] = (
    ("ZTFJ002817.81+590638.2", 7.074208, 59.110611, 1.9450768),
    ("ZTFJ000720.05+674633.7", 1.833542, 67.776028, 1.491812),
    ("ZTFJ004403.14+634418.9", 11.013083, 63.738583, 1.7500288),
    ("ZTFJ001511.81+644858.2", 3.799208, 64.816167, 3.6804866),
)
V350_LYR_PERIOD_DAYS = 0.5942378  # AAVSO VSX (Kepler RRab); Gaia DR3 vari_rrlyrae pf 0.5942437
V445_LYR_PERIOD_DAYS = 0.513075  # AAVSO VSX (Kepler RRab)
ALGOL_PERIOD_DAYS = 2.867343  # AAVSO VSX
QSO_J0747 = (116.987458, 45.757694)  # SDSS J074756.99+454527.7 (DR16Q quasar)
TAU_CET_TIC = "419015728"
# Gaia DR3 vari_agn sources for which the review found "reliable" Gaia-only periods (the
# first five), or which kept one after the first round-3 fix (the last three).
GAIA_AGN_IDS = ("3831352754950287360", "1009994308781760384", "3078743574088717440", "6613458777742792064",
                "5502412727530020224", "6378923189372478592", "1481944178762850688", "3787945998686214272")
# Round-4 review targets:
# Saturated high proper-motion stars whose ZTF tracks cross unrelated stars (Barnard's star
# r ~ 8.5, 10.4 "/yr; Groombridge 1830 V = 6.4, 7.1 "/yr): no ZTF light curve is theirs.
GROOMBRIDGE_1830_NAME = "Groombridge 1830"
# Landolt (1992) standard PG1323-086B (V = 13.406; J2000 position and pm of Landolt 2009):
# seven catflags = 0 r epochs of MJD 58964-58965 are 0.5-1.7 mag faint with bad PSF fits.
PG1323_086B = (201.46104583, -8.84863889)
# Kepler Blazhko RRab V354 Lyr (Gaia DR3 vari_rrlyrae pf 0.5616916 d; Benko et al. 2010, MNRAS
# 409, 1585: 0.5617 d) and the Blazhko RRc ZTFJ080626.57+343028.2 (Chen et al. 2020: 0.3183006 d;
# Gaia DR3 p1_o 0.3183067 d): modulated pulsators whose periods must not be doubled.
V354_LYR_PERIOD_DAYS = 0.5616916
RRC_BLAZHKO = (121.61071, 34.50786)
RRC_BLAZHKO_PERIOD_DAYS = 0.3183006
# ZTF long-period variable ZTFJ125600.42+164558.5 (Gaia DR3 vari_long_period_variable 155.5 d;
# AAVSO VSX 154.9 d): a real periodic variable with < 20 cycles in the baseline.
LPV_ZTF = (194.00176, 16.76627)
LPV_PERIOD_DAYS = 155.2
# Gaia DR3 vari_rrlyrae RRab stars (LMC field) with Gaia-only light curves (J2016.0 positions).
GAIA_RRAB = (70.7481574926497, -65.31461034529983)  # Gaia DR3 4663209996596648704
GAIA_RRAB_ID = "4663209996596648704"
GAIA_RRAB_PERIOD_DAYS = 0.5528192923231825  # pf
# Gaia DR3 4663847575916438784 (RRab, pf 0.4937024 d): half of its 86 G transits fall in the first
# 6 days (ecliptic-pole scanning), leaving a 0.22 phase gap in that half.
GAIA_RRAB_EPSL = (74.71630889919842, -65.04350643535167)
GAIA_RRAB_EPSL_PERIOD_DAYS = 0.49370240907711
# WASP-12 b transits every 1.0914203 d (Collins et al. 2017, AJ 153, 78); 1.5 % deep.
WASP12_PERIOD_DAYS = 1.0914203
# Gaia DR3 vari_eclipsing_binary 4655469709637879296 (porb 2.021 d; LMC field, J2016.0): 31 G
# transits, a few in eclipse; a chance frequency aligning them must not become a period.
GAIA_SPARSE_EB = (73.56372079022016, -69.07427109450667)
GAIA_SPARSE_EB_ID = "4655469709637879296"
# Gaia DR3 vari_agn 6378923189372478592 (J2016.0): a 6.68-d red-noise peak (FAP 0.006) that stays
# phase-coherent across 48 transits; unconfirmable Gaia-only periods need FAP < 1e-3.
GAIA_AGN_COHERENT = (347.88483638384696, -74.04014531386096)
# Round-5 review targets (positions: Gaia DR3 J2016.0, checked live 2026-09):
# Gaia DR3 vari_eclipsing_binary sources with sparse G light curves whose LS peak is at P/2 and
# whose secondary eclipse is visible in the G transits: the orbital period is P, never 2P.
#  1974933547345251968 = ASASSN-V J215955.32+460714.3 (VSX 0.704444 d; Gaia 1/0.7044 d; ZTF 0.7044518 d)
#  2637447302310594944 = CSS J232034.2-031415 (VSX 0.510527 d; Gaia frequency 1.95875 c/d)
GAIA_EB_PARITY: dict[str, tuple[float, float, float]] = {
    "1974933547345251968": (329.98045905264337, 46.12061689148052, 0.704444),
    "2637447302310594944": (350.1427754791999, -3.2378231106818616, 0.510527),
    "5233671057555506176": (175.37314439942438, -70.03333668433515, 1.78185),
    "5653489198602121472": (130.52563103399274, -24.47207793477145, 0.715871),
    "4316725467221293696": (295.1670784779848, 12.428881651070881, 1.0735978),
}
# Gaia DR3 vari_cepheid stars (pf from the Gaia DR3 table) whose Gaia-only periods were rejected
# although correct (non-sinusoidal light curves; few effective points), and a Gaia EW binary.
GAIA_CEPHEIDS: dict[str, tuple[float, float, float]] = {
    "2049531563002338432": (288.98477982010263, 34.45223709711877, 1.6011972711525537),  # T2CEP
    "4655167172163949184": (74.22303194643543, -69.63179683595858, 3.059788254691241),  # DCEP
    "4661366184375995520": (75.01707959980222, -67.9684452335219, 3.140137898085693),  # DCEP
    "2686850313960864256": (323.368588924551, -0.7987392774283647, 15.56513887355756),  # DCEP
    "4041710426226283776": (266.77355044909024, -34.629471514497354, 12.312218550291652),  # T2CEP
}
GAIA_EW = (147.87115720302134, -13.001520710690393)  # Gaia DR3 5690870910317297408
GAIA_EW_ID = "5690870910317297408"
GAIA_EW_PERIOD_DAYS = 0.3209272  # Gaia DR3 vari_eclipsing_binary orbital frequency 3.1159710 c/d
# Stochastic (red-noise) Seyfert 1 Mrk 817: ZTF must adopt no period.
MRK_817_NAME = "Mrk 817"
# Chen et al. (2020) Miras with Gaia DR3 vari_long_period_variable periods:
MIRA_320 = (292.32525, 22.30642)  # ZTFJ192918.06+221823.1, Gaia DR3 LPV 325.2 +- 13.6 d
MIRA_320_PERIOD_DAYS = 325.2
MIRA_206 = (270.38054, -7.56775)  # ZTFJ180131.33-073403.9, Gaia DR3 LPV 206.0 +- 1.9 d
MIRA_206_PERIOD_DAYS = 206.0
# Giant planets transiting M dwarfs (single transits 7-11 % deep): the orbital period is P.
TOI_519_PERIOD_DAYS = 1.2652328  # Parviainen et al. 2021; Kagetani et al. 2023
TOI_5205_PERIOD_DAYS = 1.630757  # Kanodia et al. 2023, AJ 165, 120
# GJ 3622 (M6.5 V, pm 1.53 "/yr, parallax 220 mas): ZTF assigns new object IDs along its track.
GJ_3622_NAME = "GJ 3622"
# Chen et al. (2020) delta Scuti ZTFJ210655.21+462600.4, P = 0.0743561 d, chi^2/dof < 1 in ZTF.
DSCT_ZTF = (316.73004, 46.43344)
DSCT_PERIOD_DAYS = 0.0743561


# ---------------------------------------------------------------------------
# Cases shared by the live tests and the fixture recorder
# ---------------------------------------------------------------------------


async def case_rrab(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*RRAB, radius_arcsec=3.0, surveys="ztf,neowise,gaia", client=client, use_cache=False)


async def case_3c273(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*QSO_3C273, radius_arcsec=3.0, surveys="ztf,neowise,gaia", client=client, use_cache=False)


async def case_quiet(client: httpx.AsyncClient) -> td.LightCurveResult:
    # Default survey set (ztf, neowise, gaia).
    return await td.get_lightcurves(*QUIET_STAR, radius_arcsec=3.0, client=client, use_cache=False)


async def case_rr_lyr_tess(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*RR_LYR, radius_arcsec=3.0, surveys="tess", client=client, tess_max_sectors=1,
                                    use_cache=False)


async def case_ea(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*EA_STAR, radius_arcsec=3.0, surveys="ztf,gaia", client=client, use_cache=False)


async def case_ew(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*EW_STAR, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_gaps_quiet(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*GAPS_QUIET, radius_arcsec=1.0, surveys="gaia", client=client, use_cache=False)


async def case_persistence_std(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*PERSISTENCE_STD, radius_arcsec=3.0, client=client, use_cache=False)


async def case_tess_61cyg(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*CYG61_A, radius_arcsec=45.0, surveys="tess", client=client, tess_max_sectors=2,
                                    use_cache=False)


async def case_barnard_name(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name(BARNARD_NAME, client=client, radius_arcsec=3.0, surveys="gaia",
                                            use_cache=False)


async def case_rrab_neowise_wide(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*RRAB, radius_arcsec=60.0, surveys="neowise", client=client, period=False,
                                    use_cache=False)


async def _asteroid_case(client: httpx.AsyncClient, command: str, epoch: float) -> tuple[td.EphemerisPoint, td.SolarSystemResult]:
    eph = (await td.horizons_ephemeris(command, epoch, client=client))[0]
    sky = await td.solar_system_objects(eph.ra, eph.dec, epoch_mjd=epoch, radius_arcsec=600.0, client=client)
    return eph, sky


async def case_vesta(client: httpx.AsyncClient):
    return await _asteroid_case(client, "4;", VESTA_EPOCH_MJD)


async def case_ceres(client: httpx.AsyncClient):
    return await _asteroid_case(client, "1;", CERES_EPOCH_MJD)


async def case_skybot_empty(client: httpx.AsyncClient) -> td.SolarSystemResult:
    # Near the north ecliptic pole region (dec +80): no catalogued body within 30".
    return await td.solar_system_objects(180.0, 80.0, epoch_mjd=VESTA_EPOCH_MJD, radius_arcsec=30.0, client=client)


async def case_name_3c273_gaia(client: httpx.AsyncClient) -> td.LightCurveResult:
    ra, dec, _resolved = await td.resolve_name("3C 273", client)
    return await td.get_lightcurves(ra, dec, radius_arcsec=3.0, surveys="gaia", client=client, name="3C 273",
                                    use_cache=False)


async def case_ztf_bad_collection(client: httpx.AsyncClient) -> Any:
    try:
        return await td.fetch_ztf(client, *QUIET_STAR, 3.0, collection="ztf_dr999")
    except td.UpstreamServiceError as exc:
        return exc


async def case_rr_lyr_ztf10(client: httpx.AsyncClient) -> td.LightCurveResult:
    # RR Lyr (V = 7.2) is saturated in ZTF: its own objects have no good epochs.
    return await td.get_lightcurves(*RR_LYR, radius_arcsec=10.0, surveys="ztf", client=client, use_cache=False)


async def case_cm_dra_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("CM Dra", client=client, radius_arcsec=3.0, surveys="ztf", use_cache=False)


async def case_ea2(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*EA2_STAR, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_v350_lyr_gaia(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("V350 Lyr", client=client, radius_arcsec=3.0, surveys="gaia",
                                            use_cache=False)


async def case_v445_lyr_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("V445 Lyr", client=client, radius_arcsec=3.0, surveys="ztf",
                                            use_cache=False)


async def case_tau_cet_tess(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("tau Cet", client=client, radius_arcsec=3.0, surveys="tess",
                                            tess_max_sectors=1, use_cache=False)


async def case_oj287_name(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("OJ 287", client=client, use_cache=False)


async def case_qso_j0747(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*QSO_J0747, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_barnard_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name(BARNARD_NAME, client=client, radius_arcsec=3.0, surveys="ztf",
                                            use_cache=False)


async def case_groombridge1830_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name(GROOMBRIDGE_1830_NAME, client=client, radius_arcsec=3.0, surveys="ztf",
                                            use_cache=False)


async def case_pg1323_086b_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*PG1323_086B, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_v354_lyr_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("V354 Lyr", client=client, radius_arcsec=3.0, surveys="ztf",
                                            use_cache=False)


async def case_rrc_blazhko_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*RRC_BLAZHKO, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_lpv_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*LPV_ZTF, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_gaia_rrab(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*GAIA_RRAB, radius_arcsec=1.0, surveys="gaia", client=client, use_cache=False)


async def case_gaia_sparse_eb(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*GAIA_SPARSE_EB, radius_arcsec=1.0, surveys="gaia", client=client,
                                    use_cache=False)


async def case_gaia_rrab_epsl(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*GAIA_RRAB_EPSL, radius_arcsec=1.0, surveys="gaia", client=client, use_cache=False)


async def case_gaia_agn_coherent(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*GAIA_AGN_COHERENT, radius_arcsec=1.0, surveys="gaia", client=client,
                                    use_cache=False)


async def case_wasp12_tess(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("WASP-12", client=client, radius_arcsec=3.0, surveys="tess",
                                            tess_max_sectors=1, use_cache=False)


def _gaia_case(ra: float, dec: float) -> Callable[[httpx.AsyncClient], Awaitable[td.LightCurveResult]]:
    async def case(client: httpx.AsyncClient) -> td.LightCurveResult:
        return await td.get_lightcurves(ra, dec, radius_arcsec=1.0, surveys="gaia", client=client, use_cache=False)
    return case


async def case_mrk817_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name(MRK_817_NAME, client=client, radius_arcsec=3.0, surveys="ztf",
                                            use_cache=False)


async def case_mira_320_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*MIRA_320, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_mira_206_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*MIRA_206, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


async def case_toi519_tess(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("TOI-519", client=client, radius_arcsec=3.0, surveys="tess",
                                            tess_max_sectors=2, use_cache=False)


async def case_toi5205_tess(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name("TOI-5205", client=client, radius_arcsec=3.0, surveys="tess",
                                            tess_max_sectors=2, use_cache=False)


async def case_gj3622_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves_by_name(GJ_3622_NAME, client=client, radius_arcsec=3.0, surveys="ztf",
                                            use_cache=False)


async def case_dsct_ztf(client: httpx.AsyncClient) -> td.LightCurveResult:
    return await td.get_lightcurves(*DSCT_ZTF, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)


CASES: dict[str, Callable[[httpx.AsyncClient], Awaitable[Any]]] = {
    "rrab_css_j132708": case_rrab,
    "qso_3c273": case_3c273,
    "quiet_s82_standard": case_quiet,
    "rr_lyr_tess": case_rr_lyr_tess,
    "vesta_2023": case_vesta,
    "ceres_2024": case_ceres,
    "skybot_empty": case_skybot_empty,
    "ztf_bad_collection": case_ztf_bad_collection,
    "name_3c273_gaia": case_name_3c273_gaia,
    "ea_ztfj0140": case_ea,
    "ew_ztfj0300": case_ew,
    "gaps_quiet": case_gaps_quiet,
    "persistence_std": case_persistence_std,
    "tess_61cyg": case_tess_61cyg,
    "barnard_name": case_barnard_name,
    "rrab_neowise_wide": case_rrab_neowise_wide,
    "rr_lyr_ztf10": case_rr_lyr_ztf10,
    "cm_dra_ztf": case_cm_dra_ztf,
    "ea_ztfj0048": case_ea2,
    "v350_lyr_gaia": case_v350_lyr_gaia,
    "v445_lyr_ztf": case_v445_lyr_ztf,
    "tau_cet_tess": case_tau_cet_tess,
    "qso_j0747_ztf": case_qso_j0747,
    "oj287_name": case_oj287_name,
    "barnard_ztf": case_barnard_ztf,
    "groombridge1830_ztf": case_groombridge1830_ztf,
    "pg1323_086b_ztf": case_pg1323_086b_ztf,
    "v354_lyr_ztf": case_v354_lyr_ztf,
    "rrc_blazhko_ztf": case_rrc_blazhko_ztf,
    "lpv_ztf": case_lpv_ztf,
    "gaia_rrab": case_gaia_rrab,
    "wasp12_tess": case_wasp12_tess,
    "gaia_sparse_eb": case_gaia_sparse_eb,
    "gaia_agn_coherent": case_gaia_agn_coherent,
    "gaia_rrab_epsl": case_gaia_rrab_epsl,
    **{f"gaia_eb_{sid}": _gaia_case(ra, dec) for sid, (ra, dec, _p) in GAIA_EB_PARITY.items()},
    **{f"gaia_cep_{sid}": _gaia_case(ra, dec) for sid, (ra, dec, _p) in GAIA_CEPHEIDS.items()},
    "gaia_ew_5690870910317297408": _gaia_case(*GAIA_EW),
    "mrk817_ztf": case_mrk817_ztf,
    "mira_320_ztf": case_mira_320_ztf,
    "mira_206_ztf": case_mira_206_ztf,
    "toi519_tess": case_toi519_tess,
    "toi5205_tess": case_toi5205_tess,
    "gj3622_ztf": case_gj3622_ztf,
    "dsct_ztf": case_dsct_ztf,
}


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


def _run(case: Callable[[httpx.AsyncClient], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        async with httpx.AsyncClient(follow_redirects=True, timeout=300.0) as client:
            return await case(client)
    return asyncio.run(go())


def _skip_if_unreachable(exc: td.UpstreamServiceError) -> None:
    if exc.retryable:
        pytest.skip(f"upstream unreachable: {exc}")
    raise exc


def _live(case: Callable[[httpx.AsyncClient], Awaitable[Any]]) -> Any:
    try:
        return _run(case)
    except td.UpstreamServiceError as exc:
        _skip_if_unreachable(exc)


def _require_survey(result: td.LightCurveResult, survey: str) -> None:
    for failure in result.failures:
        if failure["survey"] == survey:
            if failure["retryable"]:
                pytest.skip(f"{survey} unreachable: {failure['error']}")
            pytest.fail(f"{survey} failed: {failure['error']}")


@pytest.fixture(scope="module")
def rrab_result() -> td.LightCurveResult:
    return _live(case_rrab)


@pytest.fixture(scope="module")
def qso_result() -> td.LightCurveResult:
    return _live(case_3c273)


pytestmark = pytest.mark.live
MIN_N_QUIET = 20


def within(value: float, truth: float, rel: float) -> bool:
    return abs(value - truth) / truth < rel


def test_live_rrab_ztf_period_within_one_percent(rrab_result: td.LightCurveResult) -> None:
    _require_survey(rrab_result, "ztf")
    ztf = [p for p in rrab_result.period_search.per_series if p.series.startswith("ztf:")]
    assert ztf, "no ZTF periodogram"
    best_ztf = min(ztf, key=lambda p: p.false_alarm_probability)
    assert within(best_ztf.best_period_days, RRAB_PERIOD_DAYS, 0.01)
    assert best_ztf.false_alarm_probability < 1e-10 and best_ztf.reliable
    assert best_ztf.harmonic_test["doubled"] is False  # a pulsator: no period doubling
    best = rrab_result.period_search.best
    assert best is not None and within(best.best_period_days, RRAB_PERIOD_DAYS, 0.001)
    for band in ("ztf:g", "ztf:r"):
        m = rrab_result.variability[band]
        assert m.n >= 50 and m.is_variable is True
        assert 0.3 < m.amplitude_5_95 < 1.5  # RRab V amplitudes 0.5-1.3 mag; Chen+2020 gAmp 0.775 (half-range)


def test_live_rrab_gaia_epoch_photometry(rrab_result: td.LightCurveResult) -> None:
    _require_survey(rrab_result, "gaia")
    keys = {s.key for s in rrab_result.series}
    assert {"gaia:G", "gaia:BP", "gaia:RP"} <= keys
    g = next(s for s in rrab_result.series if s.key == "gaia:G")
    assert g.source_ids == ["Gaia DR3 1476301553808173824"]
    m = rrab_result.variability["gaia:G"]
    assert m.n >= 50 and m.is_variable is True
    # Intensity-averaged catalogue G = 15.116; a magnitude average of a pulsator is a bit fainter.
    assert 14.9 < m.weighted_mean < 15.5
    gaia_p = next(p for p in rrab_result.period_search.per_series if p.series == "gaia:G")
    assert within(gaia_p.best_period_days, RRAB_PERIOD_DAYS, 0.01)
    # Gaia times lie within the DR3 epoch-photometry window (2014-07-25 .. 2017-05-28).
    t, _, _ = g.good_arrays()
    assert 56863 < t.min() and t.max() < 57902


def test_live_rrab_neowise_visits(rrab_result: td.LightCurveResult) -> None:
    _require_survey(rrab_result, "neowise")
    w1 = next(s for s in rrab_result.series if s.key == "neowise:W1")
    assert len(w1.points) >= 10
    assert all(p.n and p.n >= 3 for p in w1.points)
    span = w1.points[-1].mjd - w1.points[0].mjd
    assert span > 8 * 365.25  # NEOWISE-R: Dec 2013 - Aug 2024
    assert 12.0 < rrab_result.variability["neowise:W1"].weighted_mean < 15.5


def test_live_3c273_variable_in_ztf(qso_result: td.LightCurveResult) -> None:
    _require_survey(qso_result, "ztf")
    ztf = {k: m for k, m in qso_result.variability.items() if k.startswith("ztf:") and m.n >= 20}
    assert ztf, "3C 273 has no ZTF light curve"
    assert any(m.is_variable for m in ztf.values()), {k: m.evidence for k, m in ztf.items()}
    assert qso_result.as_dict()["variability"]["summary"]["is_variable"] is True


def test_live_3c273_neowise(qso_result: td.LightCurveResult) -> None:
    _require_survey(qso_result, "neowise")
    w1 = next(s for s in qso_result.series if s.key == "neowise:W1")
    assert len(w1.points) >= 15
    # AllWISE W1 of 3C 273 is ~8.2 mag (Vega); NEOWISE per-visit means cluster near it.
    assert 7.5 < qso_result.variability["neowise:W1"].weighted_mean < 9.0


def test_live_name_resolution_gaia_source() -> None:
    result = _live(case_name_3c273_gaia)
    _require_survey(result, "gaia")
    # 3C 273 = Gaia DR3 3700386905605055360 (see tests/test_live_canary.py).
    assert result.provenance["gaia"]["source_id"] == "3700386905605055360"
    assert result.provenance["gaia"]["separation_arcsec"] < 0.1
    assert result.target["name"] == "3C 273"


def test_live_quiet_standard_star_not_variable() -> None:
    result = _live(case_quiet)
    for survey in ("ztf", "neowise", "gaia"):
        _require_survey(result, survey)
    ztf = {k: m for k, m in result.variability.items() if k.startswith("ztf:")}
    assert {"ztf:g", "ztf:r"} <= set(ztf)
    assert {"neowise:W1", "neowise:W2"} <= set(result.variability)
    for key, m in result.variability.items():
        if m.n >= MIN_N_QUIET or key.startswith("neowise:"):
            assert m.is_variable is not True, (key, m.evidence)
    assert 15.6 < result.variability["ztf:r"].weighted_mean < 16.1  # SDSS r = 15.817
    assert not any(k.startswith("gaia:") for k in result.variability)
    assert any("no published epoch photometry" in n for n in result.notes)
    assert result.period_search.best is None
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False
    json.dumps(result.as_dict(), allow_nan=False)  # strict RFC 8259 JSON


def test_live_eclipsing_binary_ea_variable_with_orbital_period() -> None:
    result = _live(case_ea)
    _require_survey(result, "ztf")
    for band in ("ztf:g", "ztf:r"):
        m = result.variability[band]
        assert m.n >= 300 and m.is_variable is True, m.evidence
        assert m.amplitude_5_95 > 0.5  # Chen+2020 rAmp 1.075 mag
    best = result.period_search.best
    assert best is not None and best.series.startswith("ztf:")
    assert within(best.best_period_days, EA_PERIOD_DAYS, 0.01), best
    assert best.harmonic_test["doubled"] is True
    assert result.as_dict()["variability"]["summary"]["is_variable"] is True


def test_live_eclipsing_binary_ew_orbital_period() -> None:
    result = _live(case_ew)
    _require_survey(result, "ztf")
    assert all(result.variability[b].is_variable for b in ("ztf:g", "ztf:r"))
    best = result.period_search.best
    assert best is not None
    # The W UMa minima differ enough in ZTF (O'Connell effect) for the 2P test to decide.
    assert within(best.best_period_days, EW_PERIOD_DAYS, 0.01), (best.best_period_days, best.harmonic_test)
    assert best.alternative_period_days is not None and within(best.alternative_period_days, EW_PERIOD_DAYS / 2, 0.01)


def test_live_gaia_gaps_constant_source_not_variable() -> None:
    result = _live(case_gaps_quiet)
    _require_survey(result, "gaia")
    assert result.provenance["gaia"]["source_id"] == GAPS_QUIET_ID
    g = result.variability["gaia:G"]
    assert g.n >= 20 and g.is_variable is False, g.evidence
    assert result.variability["gaia:BP"].is_variable is None  # BP/RP are not decisive
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False


def test_live_neowise_persistence_standard_not_variable() -> None:
    result = _live(case_persistence_std)
    for survey in ("ztf", "neowise"):
        _require_survey(result, survey)
    for key in ("neowise:W1", "neowise:W2"):
        m = result.variability[key]
        assert m.n >= 8 and m.is_variable is False, (key, m.evidence)
    w1 = next(s for s in result.series if s.key == "neowise:W1")
    assert w1.metadata["n_visits_rejected"] >= 3  # latent-image visits dropped
    assert result.as_dict()["variability"]["summary"]["is_variable"] is False


def test_live_tess_uses_only_the_nearest_spoc_target() -> None:
    result = _live(case_tess_61cyg)
    _require_survey(result, "tess")
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.source_ids == ["TIC 165602000"]  # 61 Cyg A only
    assert "165602023" in result.provenance["tess"]["other_tics_in_cone"]  # 61 Cyg B, 30.7"
    assert len(set(tess.metadata["sectors"])) == len(tess.metadata["sectors"]) == 2
    assert set(tess.metadata["flux_columns"].values()) == {"PDCSAP_FLUX"}


def test_live_barnard_by_name_follows_proper_motion() -> None:
    result = _live(case_barnard_name)
    _require_survey(result, "gaia")
    assert result.target["pm_dec_masyr"] == pytest.approx(10362.394, abs=1)
    assert result.provenance["gaia"]["source_id"] == BARNARD_GAIA_ID
    assert result.provenance["gaia"]["separation_arcsec"] < 0.5
    assert any("propagated" in n for n in result.notes)


def test_live_neowise_wide_cone_does_not_adopt_neighbours(rrab_result: td.LightCurveResult) -> None:
    wide = _live(case_rrab_neowise_wide)
    _require_survey(wide, "neowise")
    _require_survey(rrab_result, "neowise")
    w1_wide = next(s for s in wide.series if s.key == "neowise:W1")
    w1_narrow = next(s for s in rrab_result.series if s.key == "neowise:W1")
    assert w1_wide.metadata["n_exposures_used"] == w1_narrow.metadata["n_exposures_used"]
    assert [round(p.value, 6) for p in w1_wide.points] == [round(p.value, 6) for p in w1_narrow.points]


def test_live_rr_lyr_tess_period() -> None:
    result = _live(case_rr_lyr_tess)
    _require_survey(result, "tess")
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.source_ids == ["TIC 159717514"]
    p = next(p for p in result.period_search.per_series if p.series == "tess:TESS")
    assert within(p.best_period_days, RR_LYR_PERIOD_DAYS, 0.01)
    assert p.period_error_days is not None and p.period_error_days < 0.01 * RR_LYR_PERIOD_DAYS
    assert min(pt.value for pt in tess.points) > 0  # physical (SAP fallback when PDCSAP is over-corrected)
    assert result.variability["tess:TESS"].is_variable is True


@pytest.mark.parametrize("case,number,name", [(case_vesta, 4, "Vesta"), (case_ceres, 1, "Ceres")])
def test_live_skybot_matches_horizons(case, number: int, name: str) -> None:
    eph, sky = _live(case)
    assert name in eph.target
    match = [o for o in sky.objects if o.number == number]
    assert match, [o.name for o in sky.objects]
    obj = match[0]
    assert obj.name == name and obj.type == "asteroid"
    # Independent ephemerides (JPL DE441/SB441 vs IMCCE INPOP) agree to ~arcsec.
    assert obj.separation_arcsec < 5.0
    assert eph.v_mag is not None and obj.v_mag is not None and abs(eph.v_mag - obj.v_mag) < 0.6
    assert sky.objects[0].number == number  # the brightest body sits at the cone centre


def test_live_skybot_empty_field() -> None:
    result = _live(case_skybot_empty)
    assert result.objects == []
    assert result.provenance["http_status"] == 204


def test_live_ztf_bad_collection_is_reported() -> None:
    outcome = _run(case_ztf_bad_collection)
    assert isinstance(outcome, td.UpstreamServiceError), outcome
    if outcome.retryable:
        pytest.skip(f"ZTF unreachable: {outcome}")
    assert outcome.service == "ztf" and "ztf_objects_dr999" in outcome.message


# ---------------------------------------------------------------------------
# Round-3 review: identity, eclipsing binaries, red noise, Gaia doubling, TESS
# ---------------------------------------------------------------------------


def test_live_rr_lyr_ztf_cone_adopts_no_neighbour() -> None:
    result = _live(case_rr_lyr_ztf10)
    _require_survey(result, "ztf")
    assert [s for s in result.series if s.survey == "ztf"] == []
    assert result.period_search is None or result.period_search.best is None
    assert any("no ZTF light curve for the target" in n for n in result.notes)
    assert result.provenance["ztf"]["neighbour_object_ids"]  # the 17.9-mag neighbour 7.9" away is listed


def test_live_cm_dra_high_proper_motion_eclipsing_binary() -> None:
    result = _live(case_cm_dra_ztf)
    _require_survey(result, "ztf")
    g = next(s for s in result.series if s.key == "ztf:g")
    assert len(g.points) > 300  # its own g object, found along the proper-motion track
    best = result.period_search.best
    assert best is not None and best.period_error_days < 1e-3
    assert abs(best.best_period_days - CM_DRA_PERIOD_DAYS) < 5 * best.period_error_days


@pytest.mark.parametrize("name,ra,dec,period", [("ZTFJ004818.49+625831.1", *EA2_STAR, EA2_PERIOD_DAYS), *EA_MORE])
def test_live_equal_minima_eclipsing_binaries_orbital_period(name: str, ra: float, dec: float, period: float) -> None:
    async def case(client: httpx.AsyncClient) -> td.LightCurveResult:
        return await td.get_lightcurves(ra, dec, radius_arcsec=3.0, surveys="ztf", client=client, use_cache=False)

    result = _live(case)
    _require_survey(result, "ztf")
    best = result.period_search.best
    assert best is not None, name
    assert within(best.best_period_days, period, 1e-3), (name, best.best_period_days)
    assert best.alternative_period_days is not None and within(best.alternative_period_days, period / 2, 2e-3)


def test_live_blazar_oj287_has_no_period() -> None:
    result = _live(case_oj287_name)
    for survey in ("ztf", "gaia"):
        _require_survey(result, survey)
    assert result.summary_is_variable() is True
    assert result.period_search.best is None
    g = next(p for p in result.period_search.per_series if p.series == "gaia:G")
    assert not g.reliable  # the 0.0833-d Gaia 'period' is the 3rd harmonic of the 6-h spin


def test_live_quasars_have_no_period() -> None:
    result = _live(case_qso_j0747)
    _require_survey(result, "ztf")
    assert result.period_search.best is None
    gaia_3c273 = _live(case_name_3c273_gaia)
    _require_survey(gaia_3c273, "gaia")
    assert gaia_3c273.period_search.best is None


def test_live_gaia_vari_agn_sources_have_no_period() -> None:
    async def case(client: httpx.AsyncClient) -> list[td.LightCurveResult]:
        query = ("SELECT source_id, ra, dec FROM gaiadr3.gaia_source WHERE source_id IN ("
                 + ", ".join(GAIA_AGN_IDS) + ")")
        response = await td._tap_query(client, td.GAIA_TAP_SYNC_URL, query, service="gaia", timeout=120.0)
        rows = td._csv_rows(response.text, required=("source_id", "ra", "dec"), service="gaia")
        assert len(rows) == len(GAIA_AGN_IDS)
        return [await td.get_lightcurves(float(r["ra"]), float(r["dec"]), radius_arcsec=1.0, surveys="gaia",
                                         client=client, use_cache=False) for r in rows]

    results = _live(case)
    for result in results:
        _require_survey(result, "gaia")
    adopted = [(r.provenance["gaia"]["source_id"], r.period_search.best.best_period_days)
               for r in results if r.period_search.best is not None]
    assert adopted == []


def test_live_v350_lyr_gaia_only_period_not_doubled_by_rp() -> None:
    result = _live(case_v350_lyr_gaia)
    _require_survey(result, "gaia")
    best = result.period_search.best
    assert best is not None and best.series == "gaia:G"
    assert within(best.best_period_days, V350_LYR_PERIOD_DAYS, 1e-3)
    assert best.harmonic_test["doubled"] is False


def test_live_v445_lyr_daily_alias_not_adopted() -> None:
    result = _live(case_v445_lyr_ztf)
    _require_survey(result, "ztf")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, V445_LYR_PERIOD_DAYS, 1e-3)
    for p in result.period_search.per_series:
        if p.reliable:
            assert within(p.best_period_days, V445_LYR_PERIOD_DAYS, 1e-2), (p.series, p.best_period_days)


def test_live_algol_tess_period_error_covers_the_catalogue_period() -> None:
    async def case(client: httpx.AsyncClient) -> td.LightCurveResult:
        return await td.get_lightcurves_by_name("Algol", client=client, surveys="tess", use_cache=False)

    result = _live(case)
    _require_survey(result, "tess")
    best = result.period_search.best
    assert best is not None and best.period_error_days is not None
    assert abs(best.best_period_days - ALGOL_PERIOD_DAYS) < 3 * best.period_error_days, (
        best.best_period_days, best.period_error_days)
    assert within(best.best_period_days, ALGOL_PERIOD_DAYS, 3e-3)


def test_live_tau_ceti_tess_not_variable() -> None:
    result = _live(case_tau_cet_tess)
    _require_survey(result, "tess")
    tess = next(s for s in result.series if s.key == "tess:TESS")
    assert tess.source_ids == [f"TIC {TAU_CET_TIC}"]
    m = result.variability["tess:TESS"]
    assert m.is_variable is False, m.evidence
    assert result.period_search.best is None


# ---------------------------------------------------------------------------
# Round-4 review: moving-target identity, modulated pulsators, coherence, transits, PSF fits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [case_barnard_ztf, case_groombridge1830_ztf], ids=["barnard", "groombridge1830"])
def test_live_saturated_high_proper_motion_stars_get_no_ztf_light_curve(case) -> None:
    result = _live(case)
    _require_survey(result, "ztf")
    assert [s for s in result.series if s.survey == "ztf"] == [], [(s.key, s.source_ids) for s in result.series]
    assert result.provenance["ztf"]["target_object_ids"] == []
    assert any("no ZTF light curve for the target" in n for n in result.notes)


def test_live_landolt_standard_pg1323_086b_not_variable() -> None:
    result = _live(case_pg1323_086b_ztf)
    _require_survey(result, "ztf")
    for key, m in result.variability.items():
        if m.n >= MIN_N_QUIET:
            assert m.is_variable is False, (key, m.evidence)
    assert result.summary_is_variable() is False


@pytest.mark.parametrize("case,period", [(case_v354_lyr_ztf, V354_LYR_PERIOD_DAYS),
                                         (case_rrc_blazhko_ztf, RRC_BLAZHKO_PERIOD_DAYS)], ids=["V354Lyr", "RRc"])
def test_live_blazhko_pulsators_are_not_doubled(case, period: float) -> None:
    result = _live(case)
    _require_survey(result, "ztf")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, period, 1e-3), best and best.best_period_days
    assert best.harmonic_test["doubled"] is False


def test_live_long_period_variable_is_adopted() -> None:
    result = _live(case_lpv_ztf)
    _require_survey(result, "ztf")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, LPV_PERIOD_DAYS, 0.02)


def test_live_gaia_only_rrab_is_adopted() -> None:
    result = _live(case_gaia_rrab)
    _require_survey(result, "gaia")
    best = result.period_search.best
    assert best is not None and best.series == "gaia:G"
    assert within(best.best_period_days, GAIA_RRAB_PERIOD_DAYS, 2e-5)
    assert result.period_search.multiband is None  # BP/RP are not summed


def test_live_gaia_rrab_with_clustered_transits_is_adopted() -> None:
    result = _live(case_gaia_rrab_epsl)
    _require_survey(result, "gaia")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, GAIA_RRAB_EPSL_PERIOD_DAYS, 2e-5)


def test_live_coherent_gaia_agn_peak_is_not_adopted() -> None:
    result = _live(case_gaia_agn_coherent)
    _require_survey(result, "gaia")
    assert result.period_search.best is None


def test_live_sparse_gaia_eclipsing_binary_has_no_chance_period() -> None:
    result = _live(case_gaia_sparse_eb)
    _require_survey(result, "gaia")
    assert result.provenance["gaia"]["source_id"] == GAIA_SPARSE_EB_ID
    assert result.period_search.best is None


def test_live_wasp12_transit_period_not_doubled() -> None:
    result = _live(case_wasp12_tess)
    _require_survey(result, "tess")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, WASP12_PERIOD_DAYS, 1e-4), best.best_period_days
    assert best.harmonic_test["doubled"] is False
    assert best.alternative_period_days is not None and within(best.alternative_period_days, 2 * WASP12_PERIOD_DAYS, 1e-4)


# Round-5 review targets.


@pytest.mark.parametrize("source_id", ["1974933547345251968", "2637447302310594944"])
def test_live_sparse_gaia_eclipsing_binaries_keep_their_orbital_period(source_id: str) -> None:
    _ra, _dec, truth = GAIA_EB_PARITY[source_id]
    result = _live(CASES[f"gaia_eb_{source_id}"])
    _require_survey(result, "gaia")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, truth, 1e-4), best and best.best_period_days
    assert "eclipse_shape" not in str(best.harmonic_test["doubled_by"])


@pytest.mark.parametrize("source_id", ["2049531563002338432", "2686850313960864256"])
def test_live_gaia_only_cepheids_are_adopted(source_id: str) -> None:
    result = _live(CASES[f"gaia_cep_{source_id}"])
    _require_survey(result, "gaia")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, GAIA_CEPHEIDS[source_id][2], 1e-4)


def test_live_mrk817_gets_no_ztf_period() -> None:
    result = _live(case_mrk817_ztf)
    _require_survey(result, "ztf")
    assert result.period_search is not None and result.period_search.best is None


def test_live_mira_near_the_annual_window_is_adopted() -> None:
    result = _live(case_mira_320_ztf)
    _require_survey(result, "ztf")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, MIRA_320_PERIOD_DAYS, 0.03)


def test_live_toi519_giant_planet_period_not_doubled() -> None:
    result = _live(case_toi519_tess)
    _require_survey(result, "tess")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, TOI_519_PERIOD_DAYS, 1e-3), best and best.best_period_days
    assert best.harmonic_test["shape"]["transit_like"] is True


def test_live_gj3622_ztf_keeps_every_object_on_its_track() -> None:
    result = _live(case_gj3622_ztf)
    _require_survey(result, "ztf")
    r = next(s for s in result.series if s.key == "ztf:r")
    assert len(r.source_ids) >= 2 and max(p.mjd for p in r.points if p.flag == 0) > 60500
    assert result.target["parallax_mas"] is not None and result.target["parallax_mas"] > 200


def test_live_delta_scuti_with_chi2_below_one_is_adopted() -> None:
    result = _live(case_dsct_ztf)
    _require_survey(result, "ztf")
    best = result.period_search.best
    assert best is not None and within(best.best_period_days, DSCT_PERIOD_DAYS, 1e-4)


# ---------------------------------------------------------------------------
# Fixture recorder
# ---------------------------------------------------------------------------


async def _record_case(name: str) -> str:
    log: list[tuple[httpx.Request, httpx.Response]] = []

    async def hook(response: httpx.Response) -> None:
        await response.aread()
        log.append((response.request, response))

    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True, event_hooks={"response": [hook]}) as client:
        try:
            outcome = await CASES[name](client)
            summary = type(outcome).__name__
            if isinstance(outcome, td.LightCurveResult) and outcome.failures:
                summary += f" FAILURES {outcome.failures}"
        except Exception as exc:  # noqa: BLE001 - recorded anyway: the offline suite asserts on it
            summary = f"raised {type(exc).__name__}: {exc}"
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for old in FIXTURES.glob(f"{name}.*.body.gz"):
        old.unlink()
    exchanges = []
    for idx, (request, response) in enumerate(log):
        (FIXTURES / f"{name}.{idx}.body.gz").write_bytes(gzip.compress(response.content, mtime=0))
        try:
            body = request.content.decode("utf-8", "replace") if request.content else ""
        except httpx.RequestNotRead:  # redirected follow-up requests carry no body
            body = ""
        exchanges.append({
            "method": request.method, "url": str(request.url), "request_body": body,
            "status_code": response.status_code, "content_type": response.headers.get("content-type", ""),
            "location": response.headers.get("location", ""),
        })
    (FIXTURES / f"{name}.json").write_text(json.dumps({"case": name, "exchanges": exchanges}, indent=2), encoding="utf-8")
    return f"{name:<22} {len(exchanges)} exchange(s); {summary}"


async def record(names: list[str]) -> None:
    # One case at a time: concurrent recordings of large light curves ran out of memory.
    for name in names:
        print(await _record_case(name), flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] != "record":
        print(__doc__)
        raise SystemExit(0)
    asyncio.run(record(args[1:] or list(CASES)))

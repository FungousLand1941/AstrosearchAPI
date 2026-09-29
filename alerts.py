"""Live transient alert ingestion (ALeRCE, Fink/ZTF, Fink/LSST) with automatic crossmatch enrichment.

Brokers (public REST APIs, verified live 2026-09-28):

* **ALeRCE** (Foerster et al. 2021, AJ 161, 242) -- ``https://api.alerce.online/ztf/v1``.
  ``GET /objects/`` takes ``classifier``, ``class``, ``ranking``, ``probability``,
  ``firstmjd``/``lastmjd`` (repeated twice = [min, max] range), ``page``/``page_size``
  (at least 1000 rows per page), ``count``, ``order_by``/``order_mode`` (swagger at
  ``/ztf/v1/swagger.json``). It answers one row per *classifier version* that ranks the class
  first (live: ZTF18actadei twice for lc_classifier AGN, 0.84372 and 0.497556); rows are merged per
  object and a repeated object takes the probability of its newest version
  (``/objects/{oid}/probabilities``), pages being read until ``limit + 1`` distinct objects are
  seen. Positions are the object mean (``meanra``/``meandec``). The
  photometry of the alert is the object's latest detection from
  ``GET /objects/{oid}/detections`` (``magpsf``, ``sigmapsf``, ``fid``, ``isdiffpos``,
  ``candid``); ``/magstats`` is not used because its ``maglast`` silently includes
  negative-subtraction detections (verified: ZTF18abablcx, r maglast = an isdiffpos -1
  detection). Early classes come from the stamp classifier (Carrasco-Davis et al. 2021,
  AJ 162, 231); light-curve classes from the hierarchical random forest (Sanchez-Saez et
  al. 2021, AJ 161, 141).
* **Fink/ZTF** (Moller et al. 2021, MNRAS 501, 3272) -- ``https://api.ztf.fink-portal.org/api/v1``
  (the former ``api.fink-portal.org`` host no longer answers). ``/latests`` returns the
  newest ``n`` alerts (sorted by ``i:jd``, newest first; verified) of a Fink class between
  ``startdate`` and ``stopdate``. Dates must be ISO UTC strings with 1 s resolution and are
  compared with the exact alert time (verified: a stopdate 0.997 s before an alert excludes
  it): MJD/JD values are documented but answer HTTP 500 (verified), and ``v:`` (derived)
  columns cannot be requested via ``columns`` (HTTP 500), so the class is the requested one.
  ``class`` also accepts SIMBAD classes without their ``(SIMBAD) `` prefix (verified with
  ``RRLyrae``). Class scores follow the Fink filter definitions in
  ``fink_filters/ztf/livestream/*/filter.py``: SN candidate = SuperNNova
  (Moller & de Boissiere 2020) ``snn_snia_vs_nonia`` or ``snn_sn_vs_all`` > 0.5, and its
  probability is ``snn_sn_vs_all`` (the Ia-vs-non-Ia score is a subclass score among SNe); Early
  SN Ia = ``rf_snia_vs_nonia`` (Leoni et al. 2022, A&A 663, A13); Kilonova =
  ``rf_kn_vs_nonkn`` (Biswas et al. 2023); Microlensing = LIA ``mulens`` (Godines et al. 2019).
* **Fink/LSST** -- ``https://api.lsst.fink-portal.org/api/v1`` ``/tags`` (Rubin alerts,
  schema lsst.v11), sorted by ``midpointMjdTai`` newest first; ``startdate``/``stopdate``
  are compared with ``midpointMjdTai`` itself (a TAI time; verified), so the UTC window is
  converted to TAI. Tags whose ``/tags`` listing says ``"API support": false`` answer
  HTTP 400 and are rejected as invalid input. ``psfFlux`` is the difference-image PSF flux
  in nJy (schema doc); it is converted to an AB magnitude with the Oke & Gunn (1983, ApJ 266,
  713) definition m_AB = -2.5 log10(f_nu / erg s^-1 cm^-2 Hz^-1) - 48.60, i.e.
  m = 31.4 - 2.5 log10(f / nJy) (48.60 corresponds to 3630.8 Jy). ``midpointMjdTai`` is
  converted TAI -> UTC. CATS broad classes (Fraga et al. 2024, arXiv:2404.08798) are taken
  from ``clf_cats_class``.

**Negative detections.** A difference-image detection can be negative (the source was
brighter in the reference image). For every broker ``magpsf`` is the magnitude of the
*absolute* difference flux (the ZTF convention; for LSST computed from |psfFlux|) and
``is_negative`` carries the sign (ZTF ``isdiffpos`` 'f'/'0'/-1, LSST ``isNegative``), so a
fading variable star is never shown as a brightening.

**Complete ingestion.** ``limit`` caps the alerts stored per poll. Every broker is asked for
``limit + 1`` objects newest first (Fink pages further back in time when several alerts
belong to one object), so an over-full window is detected, never silently cut: the poll
reports ``truncated`` with the MJD ``boundary`` below which alerts were not fetched, and a
default-window poll (``alerts watch``) records the unfetched part as a *backlog* window in
the ``alert_cursors`` table and ingests it on the following polls before moving on. A row with
a malformed value is skipped (or a malformed field left empty) with a warning; an alert whose
photometry request failed is stored without photometry and filled in by a later poll.

Every stored alert is auto-crossmatched with :class:`crossmatch.CrossmatchService`
(bounded concurrency) in a small cone (default 2") against Gaia DR3, SIMBAD and NED, plus a
host-galaxy search. Enrichment:

* ``host``: galaxies are searched with a *server-side* galaxy-type filter (SIMBAD ``otype``,
  NED ``prefphytype``) within the host radius (default 60"), so stars and HII regions cannot
  use up the row limit, and in HyperLEDA (Paturel et al. 2003, A&A 412, 45; VizieR VII/237;
  every PGC type: galaxies 'G', galaxies in multiple systems 'GM' and multiple systems 'M',
  e.g. M86, IC 10) for galaxies within 4 D25 semi-major axes of the alert (up to 6 deg away,
  covering the LMC). The host follows the directional light radius (DLR) method of Sullivan et
  al. (2006, ApJ 648, 868) and Gupta et al. (2016, AJ 152, 154): d_DLR = separation / r(theta),
  with r the D25 ellipse radius toward the alert (HyperLEDA's B1950 position angles are
  precessed to ICRS). An alert inside a D25 ellipse (d_DLR <= 1) is assigned to the galaxy with
  the smallest d_DLR (e.g. SN 2014J -> M82, 58" from its nucleus); else to the galaxy with the
  smallest d_DLR <= 4, outside its ellipse (SN 2023bee -> NGC 2708 at 1.48, SN 2018aoz -> NGC 3923
  at 1.45), unless a catalogued galaxy without a D25 size lies nearer than that galaxy's light
  radius; else to the nearest catalogued galaxy, never to an entry named only by a transient
  designation (NED lists e.g. the CV 'AT 2017abr' as a galaxy): such a case is reported as
  ``ambiguous_transient_entry``. Host names prefer major catalogues (Messier, NGC, IC, UGC,
  PGC...) over fibre/sub-component entries and transient-host labels ('SN 1994I HOST').
  The projected offset in kpc uses a redshift-independent Cosmicflows-4 distance (Tully et
  al. 2023, ApJ 944, 94) for hosts with z < 0.01 or no redshift (e.g. SN 1987A -> LMC,
  1.0 kpc), else the CF4 distance of the host's group (Tully 2015, AJ 149, 171 membership; e.g.
  M100 and M86 -> Virgo, 16.2 Mpc), else the Planck 2018 distances (Planck Collaboration 2020,
  A&A 641, A6) of the CMB-frame redshift -- the group's CMB velocity when the host is in a group
  -- uncertain by ~v_pec / cz (300 km/s, or the group's velocity dispersion); below
  cz_CMB = 1500 km/s without a CF4 distance no offset is given (None, with the reason in the
  evidence). Truncated cones or failed catalogs make the search ``incomplete`` (never a false
  ``none_within_radius``).
* ``known_star`` means *Galactic* star. The astrometry of a Gaia DR3 counterpart is first
  vetted: it is not a star's when its parallax is < -3 sigma, when Gaia's DSC gives P(galaxy) +
  P(quasar) > 0.5 (Delchambre et al. 2023), when it is a Gaia galaxy candidate, or when a
  SIMBAD/NED galaxy/AGN entry lies within 1.5" and the single-star model fits it poorly -- a
  galaxy nucleus, AGN or star cluster, whose formally significant proper motions are spurious
  (M87 7.5 mas/yr at 12 sigma, NGC 3783 0.24 mas/yr at 15 sigma). Otherwise it is Galactic when
  its parallax / parallax_error >= 5 (Bailer-Jones 2015, PASP 127, 994) -- inside a D25 ellipse
  only if >= 10 sigma, or G < 19, or RUWE < 1.4 (Rybizki et al. 2022, MNRAS 510, 2597; Lindegren
  et al. 2021, A&A 649, A2); when it is a well-behaved point source (RUWE < 1.4, excess-noise
  significance <= 2) whose proper motion (>= 5 sigma) exceeds 750 km/s at the associated host's
  distance (3.2 mas/yr at the LMC distance when unknown); or when it *is* a catalogued star
  (stellar-type entry at its position) inside a galaxy of known distance with M_G < -10
  (Humphreys & Davidson 1979). The broker's own Gaia xmatch parallax (no RUWE/G) counts at >= 5
  sigma outside galaxies and >= 10 sigma inside them (never for a broker galaxy/AGN match), and
  defers to the counterpart search's row of the same source. NED '!'-prefixed (Milky Way)
  stellar types are Galactic, NED 'exG*' extragalactic. Without such evidence, a stellar-type
  source inside a galaxy with |z| < 0.01 (D < ~43 Mpc, the distance to which individual stars --
  Cepheids, novae, X-ray binaries -- are catalogued, cf. the SH0ES Cepheid hosts, Riess et al.
  2022, ApJL 934, L7) is an *extragalactic* star (``known_star`` False, e.g. M31N 2008-12a,
  IC 10 X-1 or an M82 X-ray binary); outside every D25 ellipse but associated with (d_DLR <= 4),
  or within the host radius of, such a nearby galaxy the answer is unknown (None, e.g. the SN
  impostor SN 2009ip near NGC 7259); inside a galaxy of unknown redshift unknown (None); far from
  any galaxy, Galactic. ``stellar_counterpart`` reports the stellar-type match itself.
* ``known_variable``: a catalogued variable source at the position (SIMBAD variable-star
  ``otypedef`` types, codes and labels including '_Candidate' labels as emitted by Fink,
  blazars, NED ``V*``/``Nova``/``Flare*``, the broker's Gaia DR3 variability flag); it may be
  extragalactic (the evidence says so). ``known_agn``: a catalogued AGN/QSO/Seyfert/blazar at
  the position (SIMBAD 'G > AGN' types, NED 'QSO', the broker's SIMBAD match) -- AGN flares are
  the main 'known variable' contaminant of extragalactic alert streams.
* ``is_new``: no catalogued source other than the transient itself within the match radius;
  None unless every match catalog answered. The transient itself is an entry of a transient
  type (SN, GRB, GW event...) or with a transient designation (SN/AT/TNS or survey names)
  that the catalogue does not type as a star or variable: catalogued CVs and novae with
  discovery names ('MASTER OT J...', 'PNV J...', 'ZTF18aayefwp' CV*) are prior sources.

An enrichment is ``done`` when every query answered, ``partial`` when some failed (flags
that depend on a failed catalog are None) and ``failed`` when no counterpart catalog
answered; ``partial``/``failed`` rows are retried by later polls (up to
``MAX_CROSSMATCH_ATTEMPTS``), and a failed retry never overwrites an earlier complete
result.

Alerts persist in the SQLite (or PostgreSQL) metadata database used by
:class:`datasets.MetadataStore` (``ALERTS_DATABASE_URL`` overrides it for both the API and
the CLI), in tables ``alerts`` and ``alert_cursors`` owned by this module, deduplicated on
``broker:object_id``; polling is idempotent and an older detection never replaces a newer one.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, ClassVar, Literal

import httpx
from astropy.time import Time
from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from crossmatch import AdvancedQuery, CrossmatchService
from datasets import MetadataStore
from models import CatalogRegistry, QueryPlan, UnifiedRecord, catalog_from_dict, haversine_arcsec, validate_target

# stdlib logging (stderr), like models/providers: keeps CLI stdout clean for --format json.
logger = logging.getLogger("astrosearch.alerts")

# ---------------------------------------------------------------------------
# Constants (all verified against the services' docs or live responses)
# ---------------------------------------------------------------------------

ALERCE_API = "https://api.alerce.online/ztf/v1"
ALERCE_OBJECT_URL = "https://alerce.online/object/{object_id}"
FINK_ZTF_API = "https://api.ztf.fink-portal.org/api/v1"
FINK_ZTF_OBJECT_URL = "https://ztf.fink-portal.org/{object_id}"
FINK_LSST_API = "https://api.lsst.fink-portal.org/api/v1"
FINK_LSST_OBJECT_URL = "https://lsst.fink-portal.org/{object_id}"
VIZIER_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"

JD_MJD_OFFSET = 2400000.5
MJD_EPOCH = datetime(1858, 11, 17, tzinfo=UTC)  # MJD 0 = 1858-11-17T00:00 (86400 s days)
# ZTF filter ids in the alert packets: 1 = ZTF-g, 2 = ZTF-r, 3 = ZTF-i (Bellm et al. 2019, PASP 131, 018002).
ZTF_FILTERS: dict[int, str] = {1: "g", 2: "r", 3: "i"}
# AB magnitudes (Oke & Gunn 1983): m_AB = -2.5 log10(f_nu [erg s^-1 cm^-2 Hz^-1]) - 48.60.
# 1 nJy = 1e-32 erg s^-1 cm^-2 Hz^-1, so m_AB = 80 - 48.60 - 2.5 log10(f_nu / nJy) = 31.4 - 2.5 log10(f_nu / nJy).
AB_OFFSET_CGS = 48.60
NJY_AB_ZERO_POINT_MAG = -2.5 * math.log10(1e-32) - AB_OFFSET_CGS
# 2.5 / ln(10): magnitude error from a fractional flux error.
MAG_PER_FRACTIONAL_FLUX = 2.5 / math.log(10.0)

# Fink/ZTF derived classes -> the score columns their filters threshold (see module doc).
FINK_ZTF_CLASS_SCORES: dict[str, tuple[str, ...]] = {
    "SN candidate": ("d:snn_snia_vs_nonia", "d:snn_sn_vs_all"),
    "Early SN Ia candidate": ("d:rf_snia_vs_nonia",),
    "Kilonova candidate": ("d:rf_kn_vs_nonkn",),
    "Microlensing candidate": ("d:mulens",),
    "SLSN candidate": ("d:slsn_score",),
}
# The score that is the probability of the class itself (reported as Alert.probability). For
# 'SN candidate' it is SuperNNova's SN-vs-all score: snn_snia_vs_nonia is an Ia-vs-non-Ia score
# *among* supernovae (Moller & de Boissiere 2020), not P(SN) (live: ZTF26abtnjsl has
# snn_sn_vs_all 0.25 and snn_snia_vs_nonia 0.75). Both raw scores stay in extra['scores'].
FINK_ZTF_CLASS_PROBABILITY: dict[str, str] = {
    "SN candidate": "d:snn_sn_vs_all",
    "Early SN Ia candidate": "d:rf_snia_vs_nonia",
    "Kilonova candidate": "d:rf_kn_vs_nonkn",
    "Microlensing candidate": "d:mulens",
    "SLSN candidate": "d:slsn_score",
}
FINK_ZTF_COLUMNS = (
    "i:objectId,i:candid,i:ra,i:dec,i:jd,i:magpsf,i:sigmapsf,i:fid,i:isdiffpos,i:jdstarthist,i:ndethist,"
    "i:drb,i:classtar,i:sgscore1,i:distpsnr1,d:snn_snia_vs_nonia,d:snn_sn_vs_all,d:rf_snia_vs_nonia,"
    "d:rf_kn_vs_nonkn,d:mulens,d:slsn_score,d:cdsxmatch,d:tns,d:DR3Name,d:Plx,d:e_Plx,d:gaiaVarFlag,"
    "d:mangrove_HyperLEDA_name,d:mangrove_ang_dist,d:mangrove_lum_dist,d:roid"
)
FINK_LSST_COLUMNS = (
    "r:diaObjectId,r:diaSourceId,r:ra,r:dec,r:midpointMjdTai,r:band,r:psfFlux,r:psfFluxErr,r:reliability,"
    "r:extendedness,r:snr,r:isNegative,f:clf_cats_class,f:clf_cats_score,f:clf_snnSnVsOthers_score,"
    "f:clf_earlySNIa_score,f:xm_simbad_otype,f:xm_gaiadr3_DR3Name,f:xm_gaiadr3_Plx,f:xm_gaiadr3_e_Plx,"
    "f:xm_gaiadr3_VarFlag,f:xm_mangrove_HyperLEDA_name,f:xm_mangrove_ang_dist,f:xm_tns_fullname,"
    "f:xm_tns_type,f:xm_tns_redshift,f:xm_legacydr8_zphot"
)
# CATS broad classes (Fink/LSST schema doc of f:clf_cats_class; -1 = not processed).
CATS_CLASSES: dict[int, str] = {11: "SN-like", 12: "Fast", 13: "Long", 21: "Periodic", 22: "Non-periodic"}
# Placeholder strings Fink uses for "no value" in cross-match columns.
FINK_NULLS = {"", "nan", "none", "null", "unknown", "fail"}
FINK_SIMBAD_PREFIX = "(SIMBAD) "

# Poll sizes: ``limit`` alerts per poll (one more is requested to detect an over-full window).
MAX_POLL_LIMIT = 500
FINK_MAX_ROWS = 1000  # rows per /latests or /tags request when paging back in time
MAX_PAGES = 20  # requests per Fink poll

DEFAULT_MATCH_RADIUS_ARCSEC = 2.0
DEFAULT_HOST_RADIUS_ARCSEC = 60.0
DEFAULT_MATCH_CATALOGS: tuple[str, ...] = ("gaia_dr3", "simbad", "ned")
DEFAULT_HOST_CATALOGS: tuple[str, ...] = ("ned", "simbad")
# parallax / parallax_error at or above which a Gaia counterpart is taken to be a Galactic star.
PARALLAX_SNR_STAR = 5.0
# Gaia DR3 RUWE below which the astrometric solution is well behaved (Lindegren et al. 2021).
GAIA_RUWE_MAX = 1.4
# NED/SIMBAD list the same galaxy several times (e.g. 2MASX, SDSS, CGCG and PGC entries of
# AT 2018cow's host lie within 1.6" of each other): entries this close are one host.
HOST_ALIAS_ARCSEC = 2.5
# HyperLEDA D25 search: the LMC (PGC 17223, logD25 = 3.81) has a 5.4 deg semi-major axis.
D25_SEARCH_RADIUS_DEG = 6.0
# d_DLR <= 1: inside the D25 isophote (Sullivan et al. 2006; Gupta et al. 2016).
DLR_INSIDE = 1.0
# Host association limit: Gupta et al. (2016, AJ 152, 154) assign the galaxy with the smallest
# d_DLR when d_DLR < 4 and call the SN hostless otherwise.
DLR_HOST_MAX = 4.0
# NED/SIMBAD entry of a HyperLEDA galaxy: PGC 2003 centres differ from NED's by a few arcsec
# (M31: 2.5", NGC 4993: 7.3"), more for large galaxies: max(10", a/4), at most 60".
D25_ALIAS_MIN_ARCSEC = 10.0
D25_ALIAS_FRACTION = 0.25
D25_ALIAS_MAX_ARCSEC = 60.0
# |z| below which catalogued individual stars can belong to the galaxy (D < ~43 Mpc; see module doc).
LOCAL_VOLUME_MAX_Z = 0.01
# A position change larger than this fraction of the match radius triggers a new crossmatch.
REMATCH_FRACTION = 0.5
# Crossmatch attempts per alert before a partial/failed enrichment stops being retried.
MAX_CROSSMATCH_ATTEMPTS = 5
# Catalog error types that mean "service unreachable" (network error, timeout, HTTP 5xx/429).
UNREACHABLE_ERROR_TYPES = frozenset({"CatalogUnavailableError", "RateLimitedError", "QueryTimeoutError"})
# Two MJDs closer than this (~9 ms) are the same instant.
MJD_EPS = 1e-7

# SIMBAD object types from the ``otypedef`` table (SIMBAD TAP, queried 2026-09-28): one
# "code|label|hierarchy path" entry per type of the stellar ("*"), galaxy ("G", "IG", "PaG")
# and transient branches. SIMBAD answers carry the code (TAP basic.otype); Fink's
# cross-match (d:cdsxmatch, f:xm_simbad_otype) carries the label, including the
# "<label>_Candidate" labels of the '?' codes (verified live: 'CataclyV*_Candidate').
SIMBAD_OTYPEDEF: tuple[tuple[str, str, tuple[str, ...]], ...] = tuple(
    (code, label, tuple(path.split(" > ")))
    for code, label, path in (entry.split("|") for entry in (  # noqa: SIM905 - compact otypedef dump
        "var|Variable|var;*|Star|*;**|**|* > **;**?|**_Candidate|* > **;BY*|BYDraV*|* > ** > BY*;"
        "BY?|BYDraV*_Candidate|* > ** > BY*;CV*|CataclyV*|* > ** > CV*;CV?|CataclyV*_Candidate|* > ** > CV*;"
        "No*|Nova|* > ** > CV* > No*;No?|Nova_Candidate|* > ** > CV* > No*;EB*|EclBin|* > ** > EB*;"
        "EB?|EclBin_Candidate|* > ** > EB*;El*|EllipVar|* > ** > El*;El?|EllipVar_Candidate|* > ** > El*;"
        "RS*|RSCVnV*|* > ** > RS*;RS?|RSCVnV*_Candidate|* > ** > RS*;SB*|SB*|* > ** > SB*;"
        "SB?|SB*_Candidate|* > ** > SB*;Sy*|Symbiotic*|* > ** > Sy*;Sy?|Symbiotic*_Candidate|* > ** > Sy*;"
        "XB*|XrayBin|* > ** > XB*;XB?|XrayBin_Candidate|* > ** > XB*;HXB|HighMassXBin|* > ** > XB* > HXB;"
        "HX?|HighMassXBin_Candidate|* > ** > XB* > HXB;LXB|LowMassXBin|* > ** > XB* > LXB;"
        "LX?|LowMassXBin_Candidate|* > ** > XB* > LXB;Em*|EmLine*|* > Em*;Ev*|Evolved*|* > Ev*;"
        "Ev?|Evolved*_Candidate|* > Ev*;AB*|AGB*|* > Ev* > AB*;AB?|AGB*_Candidate|* > Ev* > AB*;"
        "Mi*|Mira|* > Ev* > AB* > Mi*;Mi?|Mira_Candidate|* > Ev* > AB* > Mi*;C*|C*|* > Ev* > C*;"
        "C*?|C*_Candidate|* > Ev* > C*;Ce*|Cepheid|* > Ev* > Ce*;Ce?|Cepheid_Candidate|* > Ev* > Ce*;"
        "cC*|ClassicalCep|* > Ev* > Ce* > cC*;HB*|HorBranch*|* > Ev* > HB*;"
        "HB?|HorBranch*_Candidate|* > Ev* > HB*;RR*|RRLyrae|* > Ev* > HB* > RR*;"
        "RR?|RRLyrae_Candidate|* > Ev* > HB* > RR*;HS*|HotSubdwarf|* > Ev* > HS*;"
        "HS?|HotSubdwarf_Candidate|* > Ev* > HS*;LP*|LongPeriodV*|* > Ev* > LP*;"
        "LP?|LongPeriodV*_Candidate|* > Ev* > LP*;OH*|OH/IR*|* > Ev* > OH*;"
        "OH?|OH/IR*_Candidate|* > Ev* > OH*;PN|PlanetaryNeb|* > Ev* > PN;"
        "PN?|PlanetaryNeb_Candidate|* > Ev* > PN;RG*|RGB*|* > Ev* > RG*;RB?|RGB*_Candidate|* > Ev* > RG*;"
        "RV*|RVTauV*|* > Ev* > RV*;RV?|RVTauV*_Candidate|* > Ev* > RV*;S*|S*|* > Ev* > S*;"
        "S*?|S*_Candidate|* > Ev* > S*;WD*|WhiteDwarf|* > Ev* > WD*;WD?|WhiteDwarf_Candidate|* > Ev* > WD*;"
        "WV*|Type2Cep|* > Ev* > WV*;WV?|Type2Cep_Candidate|* > Ev* > WV*;pA*|post-AGB*|* > Ev* > pA*;"
        "pA?|post-AGB*_Candidate|* > Ev* > pA*;HV*|HighVel*|* > HV*;LM*|Low-Mass*|* > LM*;"
        "LM?|Low-Mass*_Candidate|* > LM*;BD*|BrownD*|* > LM* > BD*;BD?|BrownD*_Candidate|* > LM* > BD*;"
        "MS*|MainSequence*|* > MS*;MS?|MainSequence*_Candidate|* > MS*;BS*|BlueStraggler|* > MS* > BS*;"
        "BS?|BlueStraggler_Candidate|* > MS* > BS*;SX*|SXPheV*|* > MS* > BS* > SX*;Be*|Be*|* > MS* > Be*;"
        "Be?|Be*_Candidate|* > MS* > Be*;dS*|delSctV*|* > MS* > dS*;gD*|gammaDorV*|* > MS* > gD*;"
        "Ma*|Massiv*|* > Ma*;Ma?|Massiv*_Candidate|* > Ma*;N*|Neutron*|* > Ma* > N*;"
        "N*?|Neutron*_Candidate|* > Ma* > N*;Psr|Pulsar|* > Ma* > N* > Psr;bC*|bCepV*|* > Ma* > bC*;"
        "bC?|bCepV*_Candidate|* > Ma* > bC*;sg*|Supergiant|* > Ma* > sg*;"
        "sg?|Supergiant_Candidate|* > Ma* > sg*;s*b|BlueSG|* > Ma* > sg* > s*b;"
        "s?b|BlueSG_Candidate|* > Ma* > sg* > s*b;WR*|WolfRayet*|* > Ma* > sg* > s*b > WR*;"
        "WR?|WolfRayet*_Candidate|* > Ma* > sg* > s*b > WR*;s*r|RedSG|* > Ma* > sg* > s*r;"
        "s?r|RedSG_Candidate|* > Ma* > sg* > s*r;s*y|YellowSG|* > Ma* > sg* > s*y;"
        "s?y|YellowSG_Candidate|* > Ma* > sg* > s*y;PM*|HighPM*|* > PM*;Pe*|ChemPec*|* > Pe*;"
        "Pe?|ChemPec*_Candidate|* > Pe*;RC*|RCrBV*|* > Pe* > RC*;RC?|RCrBV*_Candidate|* > Pe* > RC*;"
        "a2*|alf2CVnV*|* > Pe* > a2*;a2?|alf2CVnV*_Candidate|* > Pe* > a2*;Pl|Planet|* > Pl;"
        "Pl?|Planet_Candidate|* > Pl;SN*|Supernova|* > SN*;SN?|Supernova_Candidate|* > SN*;"
        "V*|Variable*|* > V*;V*?|Variable*_Candidate|* > V*;Er*|Eruptive*|* > V* > Er*;"
        "Er?|Eruptive*_Candidate|* > V* > Er*;Ir*|IrregularV*|* > V* > Ir*;Pu*|PulsV*|* > V* > Pu*;"
        "Pu?|PulsV*_Candidate|* > V* > Pu*;Ro*|RotV*|* > V* > Ro*;Ro?|RotV*_Candidate|* > V* > Ro*;"
        "Y*O|YSO|* > Y*O;Y*?|YSO_Candidate|* > Y*O;Ae*|Ae*|* > Y*O > Ae*;Ae?|Ae*_Candidate|* > Y*O > Ae*;"
        "Or*|OrionV*|* > Y*O > Or*;TT*|TTauri*|* > Y*O > TT*;TT?|TTauri*_Candidate|* > Y*O > TT*;"
        "out|Outflow|* > Y*O > out;of?|Outflow_Candidate|* > Y*O > out;HH|HerbigHaroObj|* > Y*O > out > HH;"
        "G|Galaxy|G;G?|Galaxy_Candidate|G;AGN|AGN|G > AGN;AG?|AGN_Candidate|G > AGN;LIN|LINER|G > AGN > LIN;"
        "QSO|QSO|G > AGN > QSO;Q?|QSO_Candidate|G > AGN > QSO;Bla|Blazar|G > AGN > QSO > Bla;"
        "Bz?|Blazar_Candidate|G > AGN > QSO > Bla;BLL|BLLac|G > AGN > QSO > Bla > BLL;"
        "BL?|BLLac_Candidate|G > AGN > QSO > Bla > BLL;SyG|Seyfert|G > AGN > SyG;"
        "Sy1|Seyfert1|G > AGN > SyG > Sy1;Sy2|Seyfert2|G > AGN > SyG > Sy2;rG|RadioG|G > AGN > rG;"
        "EmG|EmissionG|G > EmG;GiC|GtowardsCl|G > GiC;BiC|BrightestCG|G > GiC > BiC;"
        "GiG|GtowardsGroup|G > GiG;GiP|GinPair|G > GiP;H2G|HIIG|G > H2G;LSB|LowSurfBrghtG|G > LSB;"
        "SBG|StarburstG|G > SBG;bCG|BlueCompactG|G > bCG;IG|InteractingG|IG;PaG|PairG|PaG;"
        "rB|radioBurst|Rad > rB;ev|Transient|ev;gB|gammaBurst|gam > gB;GWE|GravWaveEvent|grv > GWE;"
        "Lev|LensingEv|grv > Lev;LeG|LensedG|grv > gLS > LeI > LeG"
    ).split(";"))
)
# Hierarchy nodes defined by intrinsic or eclipsing variability (every type below them is variable).
SIMBAD_VARIABLE_NODES = frozenset({
    "var", "V*", "CV*", "EB*", "El*", "RS*", "BY*", "Sy*", "XB*", "Mi*", "Ce*", "RR*", "LP*", "RV*", "WV*", "SX*",
    "dS*", "gD*", "bC*", "RC*", "a2*", "Or*", "TT*",
})
# Transient events: supernovae ("* > SN*"), "ev", GRBs, GW and lensing events, radio bursts.
SIMBAD_TRANSIENT_NODES = frozenset({"SN*", "ev", "gB", "GWE", "Lev", "rB"})
# Stellar-branch types that are not stars: planets and YSO outflows / Herbig-Haro objects.
SIMBAD_NONSTELLAR_NODES = frozenset({"Pl", "out"})


def _otype_names(predicate: Callable[[str, tuple[str, ...]], bool]) -> frozenset[str]:
    """Codes and labels of the SIMBAD types whose (code, path) satisfy ``predicate``."""
    return frozenset(name for code, label, path in SIMBAD_OTYPEDEF if predicate(code, path) for name in (code, label))


SIMBAD_TRANSIENT_TYPES = _otype_names(lambda code, path: bool(SIMBAD_TRANSIENT_NODES & set(path))) | frozenset({
    "SN", "Candidate_SN*", "SN*_Candidate",  # legacy spellings (Fink /classes)
})
SIMBAD_STAR_TYPES = _otype_names(lambda code, path: path[0] == "*" and not (
    (SIMBAD_TRANSIENT_NODES | SIMBAD_NONSTELLAR_NODES) & set(path)))
SIMBAD_VARIABLE_TYPES = _otype_names(lambda code, path: bool(SIMBAD_VARIABLE_NODES & set(path)))
SIMBAD_GALAXY_TYPES = _otype_names(lambda code, path: path[0] in {"G", "IG", "PaG"} or code == "LeG")
# Active nuclei ("G > AGN" branch: AGN, LINER, QSO, blazars, Seyferts, radio galaxies) and blazars,
# whose variability is part of their definition (Bla/BLL, Urry & Padovani 1995, PASP 107, 803).
SIMBAD_AGN_TYPES = _otype_names(lambda code, path: "AGN" in path)
SIMBAD_BLAZAR_TYPES = _otype_names(lambda code, path: "Bla" in path)
# The otype codes (TAP basic.otype holds codes) used as the server-side host filter:
# every galaxy code except lensed images (LeG).
SIMBAD_HOST_OTYPES: tuple[str, ...] = (
    "G", "G?", "AGN", "AG?", "LIN", "QSO", "Q?", "Bla", "Bz?", "BLL", "BL?", "rG", "SyG", "Sy1", "Sy2", "bCG",
    "EmG", "GiC", "BiC", "GiG", "GiP", "H2G", "LSB", "SBG", "IG", "PaG",
)
# Legacy SIMBAD labels still emitted by Fink's cross-match (d:cdsxmatch; listed by Fink /api/v1/classes).
FINK_LEGACY_GALAXY_LABELS = frozenset({
    "BlueCompG", "G_Candidate", "GinCl", "GinGroup", "GinPair", "HII_G", "LSB_G", "Possible_G", "Seyfert_1", "Seyfert_2",
})
FINK_LEGACY_AGN_LABELS = frozenset({"Seyfert_1", "Seyfert_2"})
FINK_LEGACY_VARIABLE_LABELS = frozenset({
    "BYDra", "Candidate_Cepheid", "Candidate_CV*", "Candidate_EB*", "Candidate_Mi*", "Candidate_Nova", "Candidate_RRLyr",
    "Candidate_Symb*", "Candidate_TTau*", "Candidate_HMXB", "Candidate_LMXB", "Candidate_XB*", "Candidate_LP*",
    "Cepheid_Candidate", "CV*_Candidate", "deltaCep", "EB*_Candidate", "Erupt*RCrB", "gammaDor", "HMXB", "HMXB_Candidate",
    "Irregular_V*", "LMXB", "LMXB_Candidate", "LP*_Candidate", "LPV*", "Mi*_Candidate", "Orion_V*", "PulsV*bCep",
    "PulsV*delSct", "PulsV*RVTau", "pulsV*SX", "PulsV*WVir", "RCrB_Candidate", "RotV*alf2CVn", "RRLyr", "RRLyr_Candidate",
    "RSCVn", "Symb*_Candidate", "TTau*", "TTau*_Candidate", "V*_Candidate", "XB", "XB*_Candidate",
})
FINK_LEGACY_STAR_LABELS = FINK_LEGACY_VARIABLE_LABELS | frozenset({
    "Candidate_**", "**_Candidate", "Ae*_Candidate", "AGB*_Candidate", "Be*_Candidate", "BlueSG*", "brownD*",
    "brownD*_Candidate", "BSG*_Candidate", "BSS_Candidate", "C*_Candidate", "Candidate_Ae*", "Candidate_AGB*",
    "Candidate_Be*", "Candidate_brownD*", "Candidate_BSG*", "Candidate_BSS", "Candidate_C*", "Candidate_HB*",
    "Candidate_Hsd", "Candidate_low-mass*", "Candidate_OH", "Candidate_post-AGB*", "Candidate_RGB*", "Candidate_RSG*",
    "Candidate_S*", "Candidate_SG*", "Candidate_WD*", "Candidate_WR*", "Candidate_YSG*", "Candidate_YSO",
    "HB*_Candidate", "Hsd_Candidate", "low-mass*", "low-mass*_Candidate", "OH/IR", "OH/IR*", "Pec*", "post-AGB*",
    "post-AGB*_Candidate", "RedSG*", "RGB*_Candidate", "RSG*_Candidate", "S*_Candidate", "SG*", "SG*_Candidate",
    "WD*_Candidate", "WR*_Candidate", "YellowSG*", "YSG*_Candidate", "YSO_Candidate",
})
# NED object types (NED "Object Type" codes, https://ned.ipac.caltech.edu/help/ui/nearposn-list_objecttypes;
# verified in live NEDTAP.objdir answers). A '!' prefix marks a Galactic (Milky Way) object.
NED_GALAXY_TYPES = frozenset({"G", "GPair", "GTrpl", "G_Lens", "QSO"})
NED_STAR_TYPES = frozenset({"*", "**", "V*", "WD*", "WR*", "C*", "Red*", "Blue*", "Flare*", "Psr", "Nova", "exG*"})
NED_VARIABLE_TYPES = frozenset({"V*", "Nova", "Flare*"})
# "Extragalactic star (not a member of an identified galaxy)": explicitly not Galactic.
NED_EXTRAGALACTIC_STAR_TYPES = frozenset({"exG*"})
NED_TRANSIENT_TYPES = frozenset({"SN", "GRB"})
NED_AGN_TYPES = frozenset({"QSO"})
NED_GALACTIC_PREFIX = "!"

# Transient designations (IAU/TNS names and the discovery names of transient surveys). NED
# lists some host galaxies under the transient's name (AT2019dsg: 'AT 2019dsg', type G), and
# SIMBAD/NED list the transient itself (e.g. 'SN 2018cow', 'ASASSN-14li', 'GW170817').
# Persistent-source names of the same surveys ('ZTF J1901+1458', 'ASASSN-V J...') do not match.
TRANSIENT_NAME_PATTERNS: tuple[re.Pattern[str], ...] = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^(SN|AT|TDE)\s?\d{4}[a-z]{1,3}$",  # IAU / TNS: SN 2011fe, AT 2019dsg, SN 1987A
    r"^(GRB|GW|GrW|FRB)\s?\d{6,8}[a-z]?$",  # GRB 170817A, GW170817, SIMBAD 'GrW 170817', FRB 20180916B
    # Survey discovery names, with or without a space (SIMBAD writes 'ATLAS 17kol', TNS 'ATLAS17kol').
    r"^ASASSN-\d{2}[a-z]{1,4}$",
    r"^ATLAS\s?\d{2}[a-z]{1,4}$",
    r"^PS1-\d{2}[a-z]{1,4}$",
    r"^PS\s?\d{2}[a-z]{1,4}$",
    r"^Gaia\s?\d{2}[a-z]{1,4}$",
    r"^ZTF\s?\d{2}[a-z]{7}$",
    r"^i?PTF\s?\d{2}[a-z]{1,4}$",
    r"^LSQ\s?\d{2}[a-z]{1,4}$",
    r"^DES\d{2}[a-z]\d[a-z]{1,4}$",
    r"^(PSN|PNV|TCP)\s?J\d",  # CBAT unconfirmed transients / novae
    r"^MASTER\s?OT\s?J\d",
    r"^OGLE-\d{4}-(SN|TR|NOVA)-\d+$",
))
# NED names a galaxy after a transient it hosted ('SN 1994I HOST' is M51, 'SN 1993J HOST' M81).
TRANSIENT_HOST_LABEL = re.compile(r"^(SN|AT)\s?\d{4}\w*\s+HOST$", re.IGNORECASE)
# Designations of catalogue-wide surveys / major galaxy catalogues, preferred as host names.
MAJOR_GALAXY_NAME = re.compile(r"^((M|Messier|NGC|IC|UGC|UGCA|PGC|LEDA|ESO|MCG|CGCG|Z)\s?[-+]?\d|NAME\s)", re.IGNORECASE)

BrokerName = Literal["alerce", "fink", "fink_lsst"]


def is_transient_designation(name: Any) -> bool:
    """True for a transient's own designation (TNS/IAU or survey discovery name)."""
    text = " ".join(str(name or "").split())
    return bool(text) and any(p.match(text) for p in TRANSIENT_NAME_PATTERNS)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Alert:
    """One broker alert normalized to a common schema.

    ``mjd`` is the (UTC) MJD of the alert's detection; ``magpsf`` the PSF magnitude of the
    absolute difference-image flux in ``band`` (AB for LSST; ZTF photometric system for
    ZTF) and ``is_negative`` its sign (True: the source is fainter than in the reference
    image). None means the broker did not report it.
    """

    broker: str
    object_id: str
    ra: float
    dec: float
    mjd: float
    magpsf: float | None
    band: str | None
    classification: str | None
    probability: float | None
    url: str
    survey: str = "ztf"
    magpsf_err: float | None = None
    first_mjd: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    is_negative: bool | None = None

    @property
    def alert_id(self) -> str:
        return f"{self.broker}:{self.object_id}"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["id"] = self.alert_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Alert:
        neg = data.get("is_negative")
        return cls(
            broker=str(data["broker"]),
            object_id=str(data["object_id"]),
            ra=float(data["ra"]),
            dec=float(data["dec"]),
            mjd=float(data["mjd"]),
            magpsf=_float(data.get("magpsf")),
            band=data.get("band"),
            classification=data.get("classification"),
            probability=_float(data.get("probability")),
            url=str(data.get("url") or ""),
            survey=str(data.get("survey") or "ztf"),
            magpsf_err=_float(data.get("magpsf_err")),
            first_mjd=_float(data.get("first_mjd")),
            extra=dict(data.get("extra") or {}),
            is_negative=None if neg is None else bool(neg),
        )


@dataclass(slots=True)
class HostCandidate:
    """A catalogued galaxy chosen as the alert's host."""

    name: str
    catalog: str
    ra: float
    dec: float
    separation_arcsec: float
    object_type: str | None
    redshift: float | None
    projected_offset_kpc: float | None = None
    redshift_source: str | None = None
    aliases: list[str] = field(default_factory=list)
    method: str = "nearest"  # d25_ellipse | dlr_outside_d25 | nearest
    d_dlr: float | None = None
    pgc: int | None = None
    d25_semi_major_arcsec: float | None = None
    distance_mpc: float | None = None  # angular-diameter distance used for projected_offset_kpc
    distance_modulus: float | None = None
    distance_method: str | None = None  # cosmicflows4 | cosmicflows4_group | hubble_flow_planck18
    distance_uncertainty_fraction: float | None = None
    redshift_cmb: float | None = None  # the CMB-frame redshift behind a Hubble-flow distance
    velocity_frame: str | None = None  # cmb | cmb_group | heliocentric
    group: dict[str, Any] | None = None  # Tully (2015) group with its Cosmicflows-4 distance

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlertEnrichment:
    """Crossmatch summary and astrophysical flags for one alert."""

    status: str  # done | partial | failed
    match_radius_arcsec: float
    host_radius_arcsec: float
    catalogs: list[str]
    catalog_status: dict[str, str] = field(default_factory=dict)
    counterparts: list[dict[str, Any]] = field(default_factory=list)
    transient_designations: list[str] = field(default_factory=list)
    host: dict[str, Any] | None = None
    host_candidates: list[dict[str, Any]] = field(default_factory=list)
    d25_galaxies: list[dict[str, Any]] = field(default_factory=list)
    # found | none_within_radius | incomplete | failed | not_searched | not_applicable_star | ambiguous_transient_entry
    host_status: str = "not_searched"
    host_search_complete: bool | None = None
    known_star: bool | None = None
    stellar_counterpart: bool | None = None
    known_variable: bool | None = None
    known_agn: bool | None = None  # coincident with a catalogued AGN / QSO / blazar
    is_new: bool | None = None
    evidence: list[str] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    exception: str | None = None  # set only when the enrichment itself raised (a bug, not an outage)
    elapsed_ms: float | None = None
    crossmatched_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FetchResult:
    """Alerts of one broker window (at most ``limit``, newest first) and whether the window was cut."""

    alerts: list[Alert]
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False
    # When truncated: alerts at or before this UTC MJD (of the window column) were not all fetched.
    boundary_mjd: float | None = None
    requests: int = 0


@dataclass(slots=True)
class PollResult:
    """Outcome of one broker poll."""

    broker: str
    since_mjd: float | None
    until_mjd: float
    options: dict[str, Any] = field(default_factory=dict)
    window: str = "explicit"  # explicit | new | backlog
    fetched: int = 0
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    crossmatched: int = 0
    crossmatch_partial: int = 0
    crossmatch_failed: int = 0
    retried: int = 0
    # Alerts whose crossmatch was due (fetched and not done, plus retried rows); with
    # crossmatch_deferred they are enriched by a background task after the poll returned.
    crossmatch_queued: int = 0
    crossmatch_deferred: bool = False
    # Fetched alerts not re-crossmatched because MAX_CROSSMATCH_ATTEMPTS were already made.
    crossmatch_capped: int = 0
    truncated: bool = False
    boundary_mjd: float | None = None
    backlog: dict[str, float] | None = None
    alert_ids: list[str] = field(default_factory=list)
    new_alert_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BrokerError(RuntimeError):
    """A broker API could not be reached or answered with an error / unparseable payload.

    ``unreachable`` is True only for transport errors, timeouts and HTTP 5xx/429 (an
    outage); a payload that cannot be parsed is a real failure (``unreachable`` False).
    """

    def __init__(self, broker: str, message: str, status_code: int | None = None, *, unreachable: bool = False) -> None:
        super().__init__(f"{broker}: {message}")
        self.broker = broker
        self.status_code = status_code
        self.unreachable = unreachable


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _float(value: Any) -> float | None:
    """Finite float or None (Fink sends placeholders such as 'nan', 'None', -999)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lower() in FINK_NULLS:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in FINK_NULLS else text


def _score(value: Any) -> float | None:
    """A classifier score in [0, 1]; Fink uses -1 for 'not computed'."""
    out = _float(value)
    return out if out is not None and 0.0 <= out <= 1.0 else None


def _valid_position(ra: float | None, dec: float | None) -> bool:
    return ra is not None and dec is not None and 0.0 <= ra < 360.0 and -90.0 <= dec <= 90.0


def ztf_band(fid: Any) -> str | None:
    """ZTF filter name of an alert filter id (1/2/3 -> g/r/i, another integer -> its text).

    None for a missing or malformed id (a string such as 'g', NaN, a non-integer): an
    upstream glitch in one field must not abort the parsing of the whole answer.
    """
    value = _float(fid)
    if value is None or not value.is_integer():
        return None
    return ZTF_FILTERS.get(int(value), str(int(value)))


# Per-row parsing errors of an upstream answer (a row with a malformed value is skipped with
# a warning instead of turning the whole poll into a 422/500).
_ROW_ERRORS = (TypeError, ValueError, KeyError, OverflowError, AttributeError, IndexError)


def negative_from_isdiffpos(value: Any) -> bool | None:
    """Sign of a ZTF difference detection: isdiffpos 't'/'1'/1 -> False, 'f'/'0'/-1/0 -> True."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return None if not math.isfinite(value) else value <= 0
    text = str(value or "").strip().lower()
    if text in {"t", "1", "true"}:
        return False
    if text in {"f", "0", "-1", "false"}:
        return True
    return None


def now_mjd() -> float:
    """Current UTC time as an MJD."""
    return float(Time.now().utc.mjd)


def mjd_to_iso(mjd: float, *, round_up: bool = False) -> str:
    """'YYYY-MM-DD HH:MM:SS' for an MJD (the 1 s date format Fink accepts), floored or ceiled.

    Uses 86400 s days (the MJD definition), so it also formats a TAI MJD as a TAI date.
    """
    seconds = float(mjd) * 86400.0
    whole = math.ceil(seconds - 1e-6) if round_up else math.floor(seconds + 1e-6)
    return (MJD_EPOCH + timedelta(seconds=whole)).strftime("%Y-%m-%d %H:%M:%S")


def tai_mjd_to_utc(mjd_tai: float) -> float:
    """Convert an MJD in TAI (Rubin midpointMjdTai) to UTC (TAI - UTC = 37 s since 2017)."""
    return float(Time(float(mjd_tai), format="mjd", scale="tai").utc.mjd)


def tai_mjds_to_utc(mjds_tai: Sequence[float]) -> list[float]:
    """Vectorised :func:`tai_mjd_to_utc`: one astropy Time for a whole page of rows (a
    Time per row costs ~0.2 ms, i.e. seconds of blocked event loop for a 20-page walk)."""
    if not mjds_tai:
        return []
    return [float(v) for v in Time(list(mjds_tai), format="mjd", scale="tai").utc.mjd]


def utc_mjd_to_tai(mjd_utc: float) -> float:
    """Convert a UTC MJD to TAI."""
    return float(Time(float(mjd_utc), format="mjd", scale="utc").tai.mjd)


def njy_to_ab_mag(flux_njy: float | None, flux_err_njy: float | None = None) -> tuple[float | None, float | None]:
    """AB magnitude (and error) of a positive flux in nJy; (None, None) for non-positive flux."""
    if flux_njy is None or flux_njy <= 0:
        return None, None
    mag = NJY_AB_ZERO_POINT_MAG - 2.5 * math.log10(flux_njy)
    err = MAG_PER_FRACTIONAL_FLUX * flux_err_njy / flux_njy if flux_err_njy is not None and flux_err_njy >= 0 else None
    return mag, err


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _check_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_POLL_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_POLL_LIMIT}")
    return limit


async def _get_json(
    client: httpx.AsyncClient, broker: str, url: str, params: Sequence[tuple[str, str]] | Mapping[str, str], timeout: float
) -> Any:
    """GET a broker endpoint and decode JSON; every failure becomes :class:`BrokerError`."""
    query = params if isinstance(params, Mapping) else tuple(params)
    try:
        response = await client.get(url, params=query, timeout=timeout)
    except httpx.HTTPError as exc:
        raise BrokerError(broker, f"network error contacting {url}: {exc.__class__.__name__}: {exc}",
                          unreachable=True) from exc
    if response.status_code >= 400:
        raise BrokerError(broker, f"HTTP {response.status_code} from {url}: {response.text[:300]}", response.status_code,
                          unreachable=response.status_code >= 500 or response.status_code == 429)
    try:
        return response.json()
    except ValueError as exc:
        raise BrokerError(broker, f"non-JSON answer from {url}: {response.text[:300]!r}", response.status_code) from exc


# Class lists (/classifiers/, /classes) change rarely: cached per URL so that quiet watch
# cycles (an empty window triggers class validation) do not re-download them.
CLASS_LIST_TTL_SECONDS = 6 * 3600.0
_CLASS_LIST_CACHE: dict[str, tuple[float, Any]] = {}


def clear_class_list_cache() -> None:
    """Forget the cached broker class lists (tests; or after a broker adds classes)."""
    _CLASS_LIST_CACHE.clear()


async def _class_list(client: httpx.AsyncClient, broker: str, url: str, timeout: float) -> tuple[Any, bool]:
    """The (cached) JSON class list at ``url``; returns (payload, fetched_now)."""
    cached = _CLASS_LIST_CACHE.get(url)
    if cached is not None and time.monotonic() - cached[0] < CLASS_LIST_TTL_SECONDS:
        return cached[1], False
    payload = await _get_json(client, broker, url, {}, timeout)
    _CLASS_LIST_CACHE[url] = (time.monotonic(), payload)
    return payload, True


# ---------------------------------------------------------------------------
# Broker Clients
# ---------------------------------------------------------------------------


class AlerceBroker:
    """ALeRCE ZTF object API client (see module docstring for the verified parameters)."""

    name = "alerce"
    survey = "ztf"
    default_options: ClassVar[dict[str, Any]] = {"classifier": "stamp_classifier", "class_name": "SN", "mjd_field": "firstmjd"}

    def __init__(self, base_url: str = ALERCE_API, *, timeout: float = 90.0, concurrency: int = 4) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.concurrency = max(1, concurrency)

    def object_params(
        self,
        *,
        since_mjd: float,
        until_mjd: float,
        limit: int,
        classifier: str | None = "stamp_classifier",
        class_name: str | None = "SN",
        mjd_field: str = "firstmjd",
    ) -> list[tuple[str, str]]:
        """Query parameters for ``GET /objects/`` (``limit`` rows, newest first within the MJD window)."""
        if mjd_field not in {"firstmjd", "lastmjd"}:
            raise ValueError("mjd_field must be 'firstmjd' or 'lastmjd'")
        params: list[tuple[str, str]] = []
        if classifier:
            params += [("classifier", classifier), ("ranking", "1")]
        if class_name:
            params.append(("class", class_name))
        params += [
            (mjd_field, f"{since_mjd:.6f}"),
            (mjd_field, f"{until_mjd:.6f}"),
            ("page_size", str(int(limit))),
            ("order_by", mjd_field),
            ("order_mode", "DESC"),
        ]
        return params

    @staticmethod
    def parse_objects(payload: Any) -> tuple[list[Alert], list[str]]:
        """Alerts from an ``/objects/`` answer, one per object (photometry filled later from the detections).

        ALeRCE answers one row per *classifier version* that ranks the requested class
        first (verified live: ``lc_classifier`` AGN rows of ZTF18actadei with probabilities
        0.84372 (``hierarchical_rf_1.1.0``) and 0.497556 (``lc_classifier_1.1.13``); the rows
        carry no version). Rows of one ``oid`` are merged here deterministically: the highest
        probability is kept and every row's probability is listed in
        ``extra['classifier_rows']``; :meth:`fetch` then replaces the choice by the newest
        classifier version (``/objects/{oid}/probabilities``). Rows with malformed values are
        skipped with a warning.
        """
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise BrokerError("alerce", f"unexpected /objects/ payload: {str(payload)[:200]}")
        by_oid: dict[str, Alert] = {}
        warnings: list[str] = []
        for item in payload["items"]:
            if not isinstance(item, dict):
                raise BrokerError("alerce", f"unexpected /objects/ item: {str(item)[:200]}")
            try:
                oid = _text(item.get("oid"))
                ra, dec, mjd = _float(item.get("meanra")), _float(item.get("meandec")), _float(item.get("lastmjd"))
                if not oid or not _valid_position(ra, dec) or mjd is None:
                    warnings.append(f"alerce: skipped object without id/position/mjd: {str(item)[:120]}")
                    continue
                assert ra is not None and dec is not None
                probability = _score(item.get("probability"))
                row = {"class": _text(item.get("class")), "probability": probability}
                known = by_oid.get(oid)
                if known is not None:
                    known.extra["classifier_rows"].append(row)
                    if probability is not None and (known.probability is None or probability > known.probability):
                        known.probability = probability
                    continue
                by_oid[oid] = Alert(
                    broker="alerce",
                    object_id=oid,
                    ra=ra,
                    dec=dec,
                    mjd=mjd,
                    magpsf=None,
                    band=None,
                    classification=_text(item.get("class")),
                    probability=probability,
                    url=ALERCE_OBJECT_URL.format(object_id=oid),
                    survey="ztf",
                    first_mjd=_float(item.get("firstmjd")),
                    extra={
                        "classifier": item.get("classifier"),
                        "classifier_rows": [row],
                        "ndet": item.get("ndet"),
                        "ndethist": _float(item.get("ndethist")),
                        "stellar": item.get("stellar"),
                        "sigmara": _float(item.get("sigmara")),
                        "sigmadec": _float(item.get("sigmadec")),
                    },
                )
            except _ROW_ERRORS as exc:
                warnings.append(f"alerce: skipped a malformed /objects/ row ({exc.__class__.__name__}: {exc}): "
                                f"{str(item)[:120]}")
        for alert in by_oid.values():
            rows = alert.extra["classifier_rows"]
            if len(rows) > 1:
                alert.extra["classifier_choice"] = "max_probability"  # refined by fetch() when possible
            else:
                alert.extra.pop("classifier_rows")
        return list(by_oid.values()), warnings

    @staticmethod
    def version_key(version: str) -> tuple[tuple[int, ...], str]:
        """Sort key of an ALeRCE classifier version: its trailing x.y.z number, then the name
        ('lc_classifier_1.1.13' > 'hierarchical_rf_1.1.0'; 'stamp_classifier_1.0.4' > '..._1.0.0')."""
        found = re.search(r"(\d+(?:\.\d+)*)\s*$", version or "")
        numbers = tuple(int(p) for p in found.group(1).split(".")) if found else ()
        return numbers, version or ""

    @classmethod
    def choose_version(cls, alert: Alert, probabilities: Any, classifier: str | None) -> bool:
        """Adopt the probability of the newest classifier version that ranks the alert's class
        first (from ``/objects/{oid}/probabilities``); returns False when none does."""
        if not isinstance(probabilities, list):
            raise BrokerError("alerce", f"unexpected probabilities payload for {alert.object_id}: "
                                        f"{str(probabilities)[:200]}")
        ranked = [p for p in probabilities if isinstance(p, dict) and p.get("ranking") == 1
                  and (classifier is None or p.get("classifier_name") == classifier)
                  and p.get("class_name") == alert.classification and _score(p.get("probability")) is not None]
        if not ranked:
            return False
        newest = max(ranked, key=lambda p: cls.version_key(str(p.get("classifier_version") or "")))
        alert.probability = _score(newest["probability"])
        alert.extra["classifier_version"] = newest.get("classifier_version")
        alert.extra["classifier_versions"] = {
            str(p.get("classifier_version")): _score(p.get("probability")) for p in ranked}
        alert.extra["classifier_choice"] = "newest_version"
        return True

    @staticmethod
    def apply_detections(alert: Alert, detections: Any) -> Alert:
        """Fill the photometry from the object's latest detection (magpsf, sigmapsf, band, sign, candid).

        Any malformed payload raises :class:`BrokerError` (not unreachable): the caller keeps
        the alert without photometry and a later poll fills it in.
        """
        if not isinstance(detections, list):
            raise BrokerError("alerce", f"unexpected detections payload for {alert.object_id}: {str(detections)[:200]}")
        try:
            rows = [d for d in detections if isinstance(d, dict) and _float(d.get("mjd")) is not None]
            if not rows:
                return alert
            latest = max(rows, key=lambda d: float(d["mjd"]))
            per_band: dict[str, dict[str, Any]] = {}
            for d in sorted(rows, key=lambda d: float(d["mjd"])):
                band = ztf_band(d.get("fid")) or "?"
                entry = per_band.setdefault(band, {"ndet": 0})
                entry.update({"ndet": entry["ndet"] + 1, "last_mjd": _float(d.get("mjd")),
                              "last_magpsf": _float(d.get("magpsf")),
                              "last_is_negative": negative_from_isdiffpos(d.get("isdiffpos"))})
            extra = {
                "candid": str(latest["candid"]) if latest.get("candid") is not None else None,
                "isdiffpos": latest.get("isdiffpos"),
                "detection_mjd": _float(latest.get("mjd")),
                "n_detections": len(rows),
                "n_negative_detections": sum(1 for d in rows if negative_from_isdiffpos(d.get("isdiffpos"))),
                "bands": per_band,
            }
            if latest.get("fid") is not None and ztf_band(latest.get("fid")) is None:
                extra["malformed_fid"] = str(latest.get("fid"))[:40]
        except _ROW_ERRORS as exc:
            raise BrokerError("alerce", f"malformed detections for {alert.object_id}: {exc.__class__.__name__}: {exc}"
                              ) from exc
        alert.magpsf = _float(latest.get("magpsf"))
        alert.magpsf_err = _float(latest.get("sigmapsf"))
        alert.band = ztf_band(latest.get("fid"))
        alert.is_negative = negative_from_isdiffpos(latest.get("isdiffpos"))
        alert.extra.update(extra)
        return alert

    async def validate_class(self, client: httpx.AsyncClient, classifier: str | None, class_name: str | None) -> bool:
        """Raise ValueError for a classifier/class ALeRCE does not know (checked via /classifiers/).

        Returns True when the class list was downloaded (False: cached or nothing to check).
        """
        if not classifier and not class_name:
            return False
        url = f"{self.base_url}/classifiers/"
        payload, fetched = await _class_list(client, self.name, url, self.timeout)
        if not isinstance(payload, list):
            _CLASS_LIST_CACHE.pop(url, None)
            raise BrokerError(self.name, f"unexpected /classifiers/ payload: {str(payload)[:200]}")
        merged: dict[str, set[str]] = {}  # a classifier can be listed once per version
        for c in payload:
            if isinstance(c, dict):
                merged.setdefault(str(c.get("classifier_name")), set()).update(c.get("classes") or [])
        if classifier and classifier not in merged:
            raise ValueError(f"Unknown ALeRCE classifier {classifier!r}; known: {sorted(merged)}")
        if class_name:
            pool = merged[classifier] if classifier else set().union(*merged.values())
            if class_name not in pool:
                raise ValueError(f"Unknown ALeRCE class {class_name!r} for classifier {classifier!r}; known: {sorted(pool)}")
        return fetched

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        since_mjd: float,
        until_mjd: float,
        limit: int = 20,
        classifier: str | None = "stamp_classifier",
        class_name: str | None = "SN",
        mjd_field: str = "firstmjd",
        with_photometry: bool = True,
    ) -> FetchResult:
        """The ``limit`` newest objects whose ``mjd_field`` lies in [since, until], with their last detection.

        Pages of ``limit + 1`` rows are requested until ``limit + 1`` *distinct* objects are
        seen (ALeRCE repeats an object once per classifier version, see :meth:`parse_objects`)
        or the window is exhausted: more than ``limit`` objects means the window holds more
        (``truncated``; ``boundary_mjd`` is the ``mjd_field`` value of the oldest kept object).
        A repeated object takes the probability of its newest classifier version.
        """
        _check_limit(limit)
        page_size = limit + 1
        collected: dict[str, Alert] = {}
        warnings: list[str] = []
        requests = 0
        exhausted = False
        last_row_mjd: float | None = None
        for page in range(1, MAX_PAGES + 1):
            params = self.object_params(since_mjd=since_mjd, until_mjd=until_mjd, limit=page_size,
                                        classifier=classifier, class_name=class_name, mjd_field=mjd_field)
            if page > 1:
                params.append(("page", str(page)))
            payload = await _get_json(client, self.name, f"{self.base_url}/objects/", params, self.timeout)
            requests += 1
            alerts, page_warnings = self.parse_objects(payload)
            warnings.extend(w for w in page_warnings if w not in warnings)
            for alert in alerts:
                known = collected.get(alert.object_id)
                if known is None:
                    collected[alert.object_id] = alert
                    continue
                # The same object on two pages (one row per classifier version).
                rows = known.extra.setdefault("classifier_rows", [{"class": known.classification,
                                                                    "probability": known.probability}])
                rows.extend(alert.extra.get("classifier_rows") or [{"class": alert.classification,
                                                                    "probability": alert.probability}])
                known.extra["classifier_choice"] = "max_probability"
                if alert.probability is not None and (known.probability is None or alert.probability > known.probability):
                    known.probability = alert.probability
            items = [i for i in payload["items"] if isinstance(i, dict)]
            if items:
                last_row_mjd = _float(items[-1].get(mjd_field))
            if len(payload["items"]) < page_size:
                exhausted = True
                break
            if len(collected) > limit:
                break
        else:
            warnings.append(f"alerce: stopped after {MAX_PAGES} pages of repeated objects")
        ordered = list(collected.values())  # newest first (order_by mjd_field DESC)
        kept = ordered[:limit]
        truncated = len(ordered) > limit or not exhausted
        boundary = None
        if len(ordered) > limit and kept:
            last = kept[-1]
            boundary = last.first_mjd if mjd_field == "firstmjd" else last.mjd
        elif truncated:
            boundary = last_row_mjd if last_row_mjd is not None else until_mjd
        if not kept and await self.validate_class(client, classifier, class_name):
            requests += 1
        semaphore = asyncio.Semaphore(self.concurrency)
        repeated = [a for a in kept if len(a.extra.get("classifier_rows") or []) > 1]

        async def version(alert: Alert) -> None:
            async with semaphore:
                try:
                    found = await _get_json(client, self.name, f"{self.base_url}/objects/{alert.object_id}/probabilities",
                                            {"classifier": classifier} if classifier else {}, self.timeout)
                    if not self.choose_version(alert, found, classifier):
                        warnings.append(f"alerce: {alert.object_id}: no classifier version ranks {alert.classification!r} "
                                        "first; kept the highest probability of its rows")
                except (BrokerError, *_ROW_ERRORS) as exc:
                    warnings.append(f"alerce: classifier versions of {alert.object_id} unavailable ({exc}); kept the "
                                    "highest probability of its rows")

        async def photometry(alert: Alert) -> None:
            async with semaphore:
                try:
                    found = await _get_json(client, self.name, f"{self.base_url}/objects/{alert.object_id}/detections",
                                            {}, self.timeout)
                    self.apply_detections(alert, found)
                except (BrokerError, *_ROW_ERRORS) as exc:
                    warnings.append(f"alerce: no photometry for {alert.object_id}: {exc}")

        if repeated:
            await asyncio.gather(*(version(a) for a in repeated))
            requests += len(repeated)
        if with_photometry and kept:
            await asyncio.gather(*(photometry(a) for a in kept))
            requests += len(kept)
        return FetchResult(kept, warnings, truncated, boundary, requests)


class _FinkWalkBack:
    """Shared paging of Fink's newest-first endpoints (/latests, /tags) back in time.

    Each request asks for ``n`` rows ending at ``stopdate``; the next page ends at the oldest
    row's time (rounded up to the second, since Fink dates have 1 s resolution), so rows of
    the same second are fetched twice and deduplicated. ``n`` doubles (up to
    ``FINK_MAX_ROWS``) when a page brings no new object (one object with many alerts, e.g. a
    periodic variable) and when a page makes no progress (more than ``n`` rows in one second).

    When the walk stops before the window is exhausted, ``boundary_mjd`` is where fetching
    stopped: the oldest kept object's newest alert when more than ``limit`` objects were
    found, else the oldest row time reached (after ``MAX_PAGES`` requests), since every
    alert newer than that row was collected.
    """

    name: str
    timeout: float

    def _params(self, *, start: str, stop: str, n: int, class_name: str) -> dict[str, str]:
        raise NotImplementedError

    def _window_dates(self, since_mjd: float, until_mjd: float) -> tuple[str, str]:
        raise NotImplementedError

    def _row_time(self, row: dict[str, Any]) -> float | None:
        """Row time in the scale Fink compares startdate/stopdate with."""
        raise NotImplementedError

    def _to_utc(self, row_time: float) -> float:
        """A :meth:`_row_time` value as a UTC MJD."""
        return row_time

    def _parse(self, payload: Any, class_name: str) -> tuple[list[Alert], list[str]]:
        raise NotImplementedError

    async def _request(self, client: httpx.AsyncClient, params: dict[str, str]) -> Any:
        raise NotImplementedError

    async def _walk(self, client: httpx.AsyncClient, *, since_mjd: float, until_mjd: float, limit: int,
                    class_name: str) -> FetchResult:
        _check_limit(limit)
        start, stop = self._window_dates(since_mjd, until_mjd)
        n = min(limit + 1, FINK_MAX_ROWS)
        collected: dict[str, Alert] = {}
        warnings: list[str] = []
        exhausted = False
        requests = 0
        prev_oldest: float | None = None
        while requests < MAX_PAGES:
            payload = await self._request(client, self._params(start=start, stop=stop, n=n, class_name=class_name))
            requests += 1
            alerts, page_warnings = self._parse(payload, class_name)
            for w in page_warnings:
                if w not in warnings:
                    warnings.append(w)
            before = len(collected)
            for alert in alerts:
                known = collected.get(alert.object_id)
                if known is None or alert.mjd > known.mjd:
                    collected[alert.object_id] = alert
            if len(payload) < n:
                exhausted = True
                break
            if len(collected) > limit:
                break
            times = [t for t in (self._row_time(r) for r in payload if isinstance(r, dict)) if t is not None]
            oldest = min(times) if times else None
            if oldest is None:
                raise BrokerError(self.name, "a full page without any row time: cannot page back")
            if prev_oldest is not None and oldest >= prev_oldest - MJD_EPS:
                if n >= FINK_MAX_ROWS:
                    warnings.append(f"{self.name}: more than {FINK_MAX_ROWS} alerts share one second at MJD {oldest:.6f}; "
                                    "paging stopped there")
                    break
                n = min(2 * n, FINK_MAX_ROWS)
                continue
            prev_oldest = oldest
            stop = mjd_to_iso(oldest, round_up=True)
            if len(collected) == before:  # a page of already-seen objects: page back faster
                n = min(2 * n, FINK_MAX_ROWS)
        else:
            warnings.append(f"{self.name}: stopped after {MAX_PAGES} requests")
        ordered = sorted(collected.values(), key=lambda a: a.mjd, reverse=True)
        kept = ordered[:limit]
        truncated = not exhausted or len(ordered) > limit
        boundary: float | None = None
        if len(ordered) > limit:
            boundary = kept[-1].mjd
        elif truncated:
            # Every alert newer than the oldest row reached was collected: resume from there.
            boundary = self._to_utc(prev_oldest) if prev_oldest is not None else (kept[-1].mjd if kept else until_mjd)
        return FetchResult(kept, warnings, truncated, boundary, requests)


class FinkZTFBroker(_FinkWalkBack):
    """Fink/ZTF ``/latests`` client (alerts of one Fink derived or SIMBAD class)."""

    name = "fink"
    survey = "ztf"
    default_options: ClassVar[dict[str, Any]] = {"class_name": "SN candidate"}

    def __init__(self, base_url: str = FINK_ZTF_API, *, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def latests_params(self, *, since_mjd: float, until_mjd: float, limit: int, class_name: str) -> dict[str, str]:
        start, stop = self._window_dates(since_mjd, until_mjd)
        return self._params(start=start, stop=stop, n=limit, class_name=class_name)

    def _params(self, *, start: str, stop: str, n: int, class_name: str) -> dict[str, str]:
        return {"class": class_name, "n": str(int(n)), "startdate": start, "stopdate": stop, "columns": FINK_ZTF_COLUMNS}

    def _window_dates(self, since_mjd: float, until_mjd: float) -> tuple[str, str]:
        return mjd_to_iso(since_mjd), mjd_to_iso(until_mjd, round_up=True)

    def _row_time(self, row: dict[str, Any]) -> float | None:
        jd = _float(row.get("i:jd"))
        return jd - JD_MJD_OFFSET if jd is not None else None

    def _parse(self, payload: Any, class_name: str) -> tuple[list[Alert], list[str]]:
        return self.parse_latests(payload, class_name)

    async def _request(self, client: httpx.AsyncClient, params: dict[str, str]) -> Any:
        payload = await _get_json(client, self.name, f"{self.base_url}/latests", params, self.timeout)
        if not isinstance(payload, list):
            raise BrokerError(self.name, f"unexpected /latests payload: {str(payload)[:200]}")
        return payload

    @staticmethod
    def parse_latests(payload: Any, class_name: str) -> tuple[list[Alert], list[str]]:
        """Alerts from a ``/latests`` answer; one per object (its latest alert). Rows with
        malformed values are skipped with a warning."""
        if not isinstance(payload, list):
            raise BrokerError("fink", f"unexpected /latests payload: {str(payload)[:200]}")
        probability_column = FINK_ZTF_CLASS_PROBABILITY.get(class_name)
        by_object: dict[str, Alert] = {}
        warnings: list[str] = []
        for row in payload:
            if not isinstance(row, dict):
                raise BrokerError("fink", f"unexpected /latests row: {str(row)[:200]}")
            try:
                alert = FinkZTFBroker._parse_row(row, class_name, probability_column, warnings)
            except _ROW_ERRORS as exc:
                warnings.append(f"fink: skipped a malformed /latests row ({exc.__class__.__name__}: {exc}): "
                                f"{str(row)[:120]}")
                continue
            if alert is None:
                continue
            known = by_object.get(alert.object_id)
            if known is None or alert.mjd > known.mjd:
                by_object[alert.object_id] = alert
        return list(by_object.values()), warnings

    @staticmethod
    def _parse_row(row: dict[str, Any], class_name: str, probability_column: str | None,
                   warnings: list[str]) -> Alert | None:
        oid = _text(row.get("i:objectId"))
        ra, dec, jd = _float(row.get("i:ra")), _float(row.get("i:dec")), _float(row.get("i:jd"))
        if not oid or not _valid_position(ra, dec) or jd is None:
            warnings.append(f"fink: skipped alert without id/position/jd: {str(row)[:120]}")
            return None
        assert ra is not None and dec is not None
        fid = row.get("i:fid")
        band = ztf_band(fid)
        if fid is not None and band is None:
            warnings.append(f"fink: {oid}: malformed filter id {str(fid)[:20]!r}; band unknown")
        jdstart = _float(row.get("i:jdstarthist"))
        return Alert(
            broker="fink",
            object_id=oid,
            ra=ra,
            dec=dec,
            mjd=jd - JD_MJD_OFFSET,
            magpsf=_float(row.get("i:magpsf")),
            band=band,
            classification=class_name,
            probability=_score(row.get(probability_column)) if probability_column else None,
            url=FINK_ZTF_OBJECT_URL.format(object_id=oid),
            survey="ztf",
            magpsf_err=_float(row.get("i:sigmapsf")),
            first_mjd=jdstart - JD_MJD_OFFSET if jdstart is not None else None,
            is_negative=negative_from_isdiffpos(row.get("i:isdiffpos")),
            extra={
                "candid": str(row["i:candid"]) if row.get("i:candid") is not None else None,
                "isdiffpos": row.get("i:isdiffpos"),
                "ndethist": row.get("i:ndethist"),
                "drb": _float(row.get("i:drb")),
                "classtar": _float(row.get("i:classtar")),
                "sgscore1": _float(row.get("i:sgscore1")),
                "distpsnr1_arcsec": _float(row.get("i:distpsnr1")),
                "probability_score": probability_column.split(":", 1)[1] if probability_column else None,
                "scores": {c.split(":", 1)[1]: _score(row.get(c)) for c in (
                    "d:snn_snia_vs_nonia", "d:snn_sn_vs_all", "d:rf_snia_vs_nonia", "d:rf_kn_vs_nonkn",
                    "d:mulens", "d:slsn_score") if c in row},
                "simbad_otype": _text(row.get("d:cdsxmatch")),
                # d:tns is the TNS classification (e.g. 'SN Ia', 'CV'), not the TNS name:
                # stored as tns_type like Fink/LSST's f:xm_tns_type ('tns' holds names only).
                "tns_type": _text(row.get("d:tns")),
                "gaia_dr3_name": _text(row.get("d:DR3Name")),
                "gaia_parallax_mas": _float(row.get("d:Plx")),
                "gaia_parallax_error_mas": _float(row.get("d:e_Plx")),
                "gaia_var_flag": row.get("d:gaiaVarFlag"),
                "mangrove_hyperleda_name": _text(row.get("d:mangrove_HyperLEDA_name")),
                "mangrove_ang_dist": _float(row.get("d:mangrove_ang_dist")),
                "mangrove_lum_dist": _float(row.get("d:mangrove_lum_dist")),
                "roid": row.get("d:roid"),
            },
        )

    async def validate_class(self, client: httpx.AsyncClient, class_name: str) -> bool:
        """Raise ValueError when ``class_name`` is not a Fink/ZTF class (Fink answers [] silently).

        ``/classes`` lists SIMBAD classes with a ``(SIMBAD) `` prefix, but ``/latests`` also
        accepts the bare SIMBAD name (verified: ``RRLyrae``), so both spellings are valid.
        Returns True when the class list was downloaded (False: cached).
        """
        url = f"{self.base_url}/classes"
        payload, fetched = await _class_list(client, self.name, url, self.timeout)
        if not isinstance(payload, dict):
            _CLASS_LIST_CACHE.pop(url, None)
            raise BrokerError(self.name, f"unexpected /classes payload: {str(payload)[:200]}")
        known = {str(c) for group in payload.values() if isinstance(group, list) for c in group}
        bare = {c.removeprefix(FINK_SIMBAD_PREFIX) for c in known if c.startswith(FINK_SIMBAD_PREFIX)}
        if class_name not in known and class_name not in bare:
            raise ValueError(f"Unknown Fink/ZTF class {class_name!r} (see {self.base_url}/classes)")
        return fetched

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        since_mjd: float,
        until_mjd: float,
        limit: int = 20,
        class_name: str = "SN candidate",
    ) -> FetchResult:
        """The ``limit`` objects with the newest alerts of ``class_name`` in [since, until]."""
        result = await self._walk(client, since_mjd=since_mjd, until_mjd=until_mjd, limit=limit, class_name=class_name)
        if not result.alerts and await self.validate_class(client, class_name):
            result.requests += 1
        return result


class FinkLSSTBroker(_FinkWalkBack):
    """Fink/LSST (Rubin) ``/tags`` client."""

    name = "fink_lsst"
    survey = "lsst"
    default_options: ClassVar[dict[str, Any]] = {"class_name": "extragalactic_new_candidate"}

    def __init__(self, base_url: str = FINK_LSST_API, *, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def tags_params(self, *, since_mjd: float, until_mjd: float, limit: int, tag: str) -> dict[str, str]:
        start, stop = self._window_dates(since_mjd, until_mjd)
        return self._params(start=start, stop=stop, n=limit, class_name=tag)

    def _params(self, *, start: str, stop: str, n: int, class_name: str) -> dict[str, str]:
        return {"tag": class_name, "n": str(int(n)), "startdate": start, "stopdate": stop, "columns": FINK_LSST_COLUMNS}

    def _window_dates(self, since_mjd: float, until_mjd: float) -> tuple[str, str]:
        # Fink compares the dates with midpointMjdTai: express the UTC window in TAI.
        return mjd_to_iso(utc_mjd_to_tai(since_mjd)), mjd_to_iso(utc_mjd_to_tai(until_mjd), round_up=True)

    def _row_time(self, row: dict[str, Any]) -> float | None:
        return _float(row.get("r:midpointMjdTai"))

    def _to_utc(self, row_time: float) -> float:
        return tai_mjd_to_utc(row_time)

    def _parse(self, payload: Any, class_name: str) -> tuple[list[Alert], list[str]]:
        return self.parse_tags(payload, class_name)

    async def _request(self, client: httpx.AsyncClient, params: dict[str, str]) -> Any:
        url = f"{self.base_url}/tags"
        try:
            response = await client.get(url, params=params, timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise BrokerError(self.name, f"network error contacting {url}: {exc.__class__.__name__}: {exc}",
                              unreachable=True) from exc
        # An unknown tag, or a tag served only by the Livestream ("API support": false in
        # /tags), is answered with HTTP 400 and a plain-text explanation: invalid input.
        if response.status_code == 400:
            raise ValueError(f"Fink/LSST rejected tag {params.get('tag')!r}: {' '.join(response.text.split())[:400]}")
        if response.status_code >= 400:
            raise BrokerError(self.name, f"HTTP {response.status_code} from {url}: {response.text[:300]}", response.status_code,
                              unreachable=response.status_code >= 500 or response.status_code == 429)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BrokerError(self.name, f"non-JSON answer from {url}: {response.text[:300]!r}") from exc
        if not isinstance(payload, list):
            raise BrokerError(self.name, f"unexpected /tags payload: {str(payload)[:200]}")
        return payload

    @staticmethod
    def parse_tags(payload: Any, tag: str) -> tuple[list[Alert], list[str]]:
        """Alerts from a ``/tags`` answer; one per diaObject (its latest diaSource)."""
        if not isinstance(payload, list):
            raise BrokerError("fink_lsst", f"unexpected /tags payload: {str(payload)[:200]}")
        by_object: dict[str, Alert] = {}
        warnings: list[str] = []
        if not all(isinstance(row, dict) for row in payload):
            bad = next(row for row in payload if not isinstance(row, dict))
            raise BrokerError("fink_lsst", f"unexpected /tags row: {str(bad)[:200]}")
        tai_times = sorted({t for t in (_float(row.get("r:midpointMjdTai")) for row in payload) if t is not None})
        utc_of = dict(zip(tai_times, tai_mjds_to_utc(tai_times), strict=True))
        for row in payload:
            try:
                alert = FinkLSSTBroker._parse_row(row, tag, utc_of, warnings)
            except _ROW_ERRORS as exc:
                warnings.append(f"fink_lsst: skipped a malformed /tags row ({exc.__class__.__name__}: {exc}): "
                                f"{str(row)[:120]}")
                continue
            if alert is None:
                continue
            known = by_object.get(alert.object_id)
            if known is None or alert.mjd > known.mjd:
                by_object[alert.object_id] = alert
        return list(by_object.values()), warnings

    @staticmethod
    def _parse_row(row: dict[str, Any], tag: str, utc_of: dict[float, float], warnings: list[str]) -> Alert | None:
        oid = row.get("r:diaObjectId")
        ra, dec, mjd_tai = _float(row.get("r:ra")), _float(row.get("r:dec")), _float(row.get("r:midpointMjdTai"))
        if oid in (None, "", 0) or not _valid_position(ra, dec) or mjd_tai is None:
            warnings.append(f"fink_lsst: skipped alert without id/position/mjd: {str(row)[:120]}")
            return None
        assert ra is not None and dec is not None
        object_id = str(oid)  # 64-bit ids: keep as text (exceeds 2**53)
        mjd = utc_of[mjd_tai]
        flux, flux_err = _float(row.get("r:psfFlux")), _float(row.get("r:psfFluxErr"))
        flag = row.get("r:isNegative")
        negative = bool(flag) if isinstance(flag, bool) else (flux < 0 if flux is not None else None)
        # Magnitude of the absolute difference flux (the ZTF magpsf convention); the sign is is_negative.
        mag, mag_err = njy_to_ab_mag(abs(flux) if flux is not None else None, flux_err)
        cats_value = _float(row.get("f:clf_cats_class"))
        cats_code = int(cats_value) if cats_value is not None and cats_value.is_integer() else None
        cats_label = CATS_CLASSES.get(cats_code) if cats_code is not None else None
        var_flag = _text(row.get("f:xm_gaiadr3_VarFlag"))
        return Alert(
            broker="fink_lsst",
            object_id=object_id,
            ra=ra,
            dec=dec,
            mjd=mjd,
            magpsf=mag,
            band=_text(row.get("r:band")),
            classification=cats_label or tag,
            probability=_score(row.get("f:clf_cats_score")) if cats_label else None,
            url=FINK_LSST_OBJECT_URL.format(object_id=object_id),
            survey="lsst",
            magpsf_err=mag_err,
            first_mjd=None,
            is_negative=negative,
            extra={
                "tag": tag,
                "dia_source_id": str(row["r:diaSourceId"]) if row.get("r:diaSourceId") is not None else None,
                "midpoint_mjd_tai": mjd_tai,
                "psf_flux_njy": flux,
                "psf_flux_err_njy": flux_err,
                "is_negative": flag,
                "reliability": _float(row.get("r:reliability")),
                "extendedness": _float(row.get("r:extendedness")),
                "snr": _float(row.get("r:snr")),
                "cats_class": cats_code,
                "cats_score": _score(row.get("f:clf_cats_score")),
                "scores": {
                    "snn_sn_vs_others": _score(row.get("f:clf_snnSnVsOthers_score")),
                    "early_snia": _score(row.get("f:clf_earlySNIa_score")),
                },
                "simbad_otype": _text(row.get("f:xm_simbad_otype")),
                "gaia_dr3_name": _text(row.get("f:xm_gaiadr3_DR3Name")),
                "gaia_parallax_mas": _float(row.get("f:xm_gaiadr3_Plx")),
                "gaia_parallax_error_mas": _float(row.get("f:xm_gaiadr3_e_Plx")),
                "gaia_var_flag": int(var_flag) if var_flag is not None and var_flag.lstrip("-").isdigit() else None,
                "mangrove_hyperleda_name": _text(row.get("f:xm_mangrove_HyperLEDA_name")),
                "mangrove_ang_dist": _float(row.get("f:xm_mangrove_ang_dist")),
                "tns": _text(row.get("f:xm_tns_fullname")),
                "tns_type": _text(row.get("f:xm_tns_type")),
                "tns_redshift": _float(row.get("f:xm_tns_redshift")),
                "legacydr8_zphot": _float(row.get("f:xm_legacydr8_zphot")),
            },
        )

    async def fetch(
        self,
        client: httpx.AsyncClient,
        *,
        since_mjd: float,
        until_mjd: float,
        limit: int = 20,
        class_name: str = "extragalactic_new_candidate",
    ) -> FetchResult:
        """The ``limit`` diaObjects with the newest diaSources of tag ``class_name`` in [since, until] (UTC)."""
        return await self._walk(client, since_mjd=since_mjd, until_mjd=until_mjd, limit=limit, class_name=class_name)


BROKERS: dict[str, Callable[[], AlerceBroker | FinkZTFBroker | FinkLSSTBroker]] = {
    "alerce": AlerceBroker,
    "fink": FinkZTFBroker,
    "fink_lsst": FinkLSSTBroker,
}


def get_broker(name: str) -> AlerceBroker | FinkZTFBroker | FinkLSSTBroker:
    try:
        return BROKERS[name]()
    except KeyError:
        raise ValueError(f"Unknown broker {name!r}; choose from {sorted(BROKERS)}") from None


def broker_info() -> list[dict[str, Any]]:
    """Static description of the supported brokers and their default filters."""
    return [
        {"name": "alerce", "survey": "ztf", "api": ALERCE_API, "default_options": AlerceBroker.default_options,
         "class_option": "ALeRCE class of 'classifier' (see /classifiers/)", "mjd_field": ["firstmjd", "lastmjd"]},
        {"name": "fink", "survey": "ztf", "api": FINK_ZTF_API, "default_options": FinkZTFBroker.default_options,
         "class_option": "Fink/ZTF class (see /api/v1/classes; SIMBAD classes also without '(SIMBAD) ')",
         "scored_classes": sorted(FINK_ZTF_CLASS_SCORES)},
        {"name": "fink_lsst", "survey": "lsst", "api": FINK_LSST_API, "default_options": FinkLSSTBroker.default_options,
         "class_option": "Fink/LSST tag with API support (see /api/v1/tags)"},
    ]


def normalize_options(broker: str, options: Mapping[str, Any] | None) -> dict[str, Any]:
    """The broker's default options overridden by the given (non-None) ones; unknown options raise ValueError."""
    client_broker = get_broker(broker)
    given = {k: v for k, v in (options or {}).items() if v is not None}
    allowed = set(client_broker.default_options)
    unknown = set(given) - allowed
    if unknown:
        raise ValueError(f"Options {sorted(unknown)} are not supported by broker {broker!r}")
    return {**client_broker.default_options, **given}


async def fetch_alerts(
    client: httpx.AsyncClient,
    broker: str,
    *,
    since_mjd: float,
    until_mjd: float,
    limit: int = 20,
    options: dict[str, Any] | None = None,
) -> FetchResult:
    """Fetch the ``limit`` newest normalized alerts of one broker window (options: classifier, class_name, mjd_field)."""
    _check_limit(limit)
    opts = normalize_options(broker, options)
    client_broker = get_broker(broker)
    if isinstance(client_broker, AlerceBroker):
        return await client_broker.fetch(client, since_mjd=since_mjd, until_mjd=until_mjd, limit=limit,
                                         classifier=opts.get("classifier"), class_name=opts.get("class_name"),
                                         mjd_field=opts.get("mjd_field", "firstmjd"))
    return await client_broker.fetch(client, since_mjd=since_mjd, until_mjd=until_mjd, limit=limit,
                                     class_name=str(opts["class_name"]))


# ---------------------------------------------------------------------------
# Crossmatch Enrichment
# ---------------------------------------------------------------------------

HYPERLEDA = "hyperleda_d25"
# HyperLEDA / PGC 2003 on VizieR (VII/237, ReadMe): logD25 = log10 of the B-band 25 mag/arcsec^2
# isophotal diameter in units of 0.1 arcmin, logR25 = log10(major / minor axis), PA = "Adopted
# 1950-position angle" of the major axis (deg, N through E, relative to the B1950 north; see
# :func:`pa_b1950_to_icrs`). OType is 'G' (galaxy), 'GM' (galaxy in a multiple system) or 'M'
# (multiple system); all three are galaxies (live counts 970678 / 8444 / 4139; 'M' holds e.g.
# M86 = NGC 4406, IC 10 and NGC 1260). Verified: M31 (PGC 2557) logD25 = 3.30 -> 199.5'.
HYPERLEDA_DEFINITION: dict[str, Any] = {
    "enabled": True,
    "provider": "tap",
    "wavelength": "optical",
    "endpoint": VIZIER_TAP,
    "table": '"VII/237/pgc"',
    "description": "HyperLEDA / PGC 2003 galaxy D25 isophotal ellipses (VizieR VII/237)",
    "parameters": {
        "columns": ["PGC", "RAJ2000", "DEJ2000", "OType", "logD25", "logR25", "PA", "ANames"],
        "id_field": "PGC",
        "ra_field": "RAJ2000",
        "dec_field": "DEJ2000",
        "format": "json",
        "distance": "deg",
    },
    "epoch": 2000.0,
    "citation": "Paturel et al. 2003, A&A 412, 45 (2003A&A...412...45P)",
    "acknowledgement": "We acknowledge the usage of the HyperLEDA database (http://leda.univ-lyon1.fr).",
    "max_rows": 200,
    "timeout_seconds": 60.0,
    "coverage": "all-sky galaxy catalog (983k galaxies)",
}
HYPERLEDA_OTYPE_FILTER = "(OType LIKE 'G%' OR OType LIKE 'M%')"
# Arcsec per unit of 10**logD25 for the semi-major axis: D25 = 0.1' * 10**logD25 = 6" * 10**logD25.
D25_SEMI_MAJOR_ARCSEC_PER_UNIT = 3.0

COSMICFLOWS4 = "cosmicflows4"
# Cosmicflows-4 redshift-independent distance moduli of individual galaxies, keyed by PGC
# (Tully et al. 2023, ApJ 944, 94; VizieR J/ApJ/944/94/table2: PGC, DM, e_DM, Vcmb). Verified
# live: LMC (PGC 17223) DM = 18.469, M31 (PGC 2557) 24.366, M82 (PGC 28655) 27.74, M101
# (PGC 50063) 29.151, IC 10 (PGC 1305) 24.45, NGC 4993 (PGC 45657) 32.974.
COSMICFLOWS4_DEFINITION: dict[str, Any] = {
    "enabled": True,
    "provider": "tap",
    "wavelength": "optical",
    "endpoint": VIZIER_TAP,
    "table": '"J/ApJ/944/94/table2"',
    "description": "Cosmicflows-4 galaxy distances (VizieR J/ApJ/944/94)",
    "parameters": {
        "columns": ["PGC", "DM", "e_DM", "Vcmb", "RAJ2000", "DEJ2000"],
        "id_field": "PGC",
        "ra_field": "RAJ2000",
        "dec_field": "DEJ2000",
        "format": "json",
        "distance": "deg",
    },
    "epoch": 2000.0,
    "citation": "Tully et al. 2023, ApJ 944, 94 (2023ApJ...944...94T)",
    "acknowledgement": "This research has made use of the Cosmicflows-4 distances (Tully et al. 2023).",
    "max_rows": 5,
    "timeout_seconds": 60.0,
    "coverage": "56k galaxies with redshift-independent distances",
}
# CF4 and HyperLEDA positions of one PGC galaxy agree to < 1" (verified: LMC, SMC, M31, NGC 4993).
CF4_SEARCH_RADIUS_ARCSEC = 60.0

# Cosmicflows-4 *group* distances (VizieR J/ApJ/944/94/groups: DMzp, the group's weighted
# distance modulus on the TRGB/Cepheid/maser zero point, and V3k, the group velocity in the CMB
# frame) for galaxies without an individual distance, via their group membership in the 2MASS
# K < 11.75 group catalogue of Tully (2015, AJ 149, 171; VizieR J/AJ/149/171 table4 members,
# table3 groups: sigV, the line-of-sight velocity dispersion, and R2t, the projected second-
# turnaround radius). One ADQL join; verified live: M100 (PGC 40153, not in CF4 table2) -> Virgo
# (nest 100002, PGC1 41220 = M49, sigV 670 km/s, R2t 1.44 Mpc), CF4 DMzp 31.048 (16.2 Mpc).
COSMICFLOWS4_GROUPS = "cosmicflows4_groups"
COSMICFLOWS4_GROUPS_DEFINITION: dict[str, Any] = {
    "enabled": True,
    "provider": "tap",
    "wavelength": "optical",
    "endpoint": VIZIER_TAP,
    "table": ('"J/AJ/149/171/table4" AS m JOIN "J/AJ/149/171/table3" AS n ON m.Nest = n.Nest '
              'LEFT OUTER JOIN "J/ApJ/944/94/groups" AS g ON g."1PGC" = n.PGC1'),
    "description": "Tully (2015) 2MASS group membership joined to Cosmicflows-4 group distances",
    "parameters": {
        "columns": ["m.PGC", "m.Nest", "n.PGC1", "n.Nmb", "n.sigV", "n.R2t", "g.Ngal", "g.DMzp", "g.e_DMzp", "g.V3k"],
        "id_field": "m.PGC",
        "ra_field": 'm."_RA.icrs"',
        "dec_field": 'm."_DE.icrs"',
        "format": "json",
        "distance": "deg",
    },
    "epoch": 2000.0,
    "citation": "Tully 2015, AJ 149, 171 (2015AJ....149..171T); Tully et al. 2023, ApJ 944, 94",
    "acknowledgement": "This research has made use of the Cosmicflows-4 group distances (Tully et al. 2023).",
    "max_rows": 5,
    "timeout_seconds": 60.0,
    "coverage": "2MASS K < 11.75 galaxies (cz < ~24000 km/s)",
}
# Largest group velocity of the Tully (2015) catalogue (VizieR range of <Vcmba>: 7..23996 km/s):
# beyond it a group lookup cannot succeed and is not made.
GROUP_CATALOG_MAX_CZ_KMS = 24000.0

# Extra Gaia DR3 columns of the counterpart search (the base registry has pmra/pmdec, parallax,
# RUWE and G): the proper-motion errors and correlation for the proper-motion test, and the
# source-quality columns that tell a star's astrometry from the spurious 5-parameter solution of
# an extended source (galaxy nucleus, AGN, star cluster), see AlertEnricher._gaia_verdict.
GAIA_PM_ERROR_COLUMNS: tuple[str, ...] = ("pmra_error", "pmdec_error", "pmra_pmdec_corr")
GAIA_QUALITY_COLUMNS: tuple[str, ...] = (
    "astrometric_excess_noise", "astrometric_excess_noise_sig", "in_galaxy_candidates",
    "classprob_dsc_combmod_star", "classprob_dsc_combmod_galaxy", "classprob_dsc_combmod_quasar",
)

# --- Galactic-star criteria (see AlertEnricher._gaia_verdict) ---
# Inside a galaxy a parallax this significant is taken as real whatever the RUWE / magnitude.
PARALLAX_SNR_SECURE = 10.0
# Spurious Gaia parallaxes in crowded fields are overwhelmingly faint (Rybizki et al. 2022,
# MNRAS 510, 2597): a 5-10 sigma parallax inside a galaxy counts for G < 19 or RUWE < 1.4.
GAIA_BRIGHT_G = 19.0
# A proper motion (>= 5 sigma) above what any galaxy can show is Galactic. The largest
# heliocentric transverse velocity of a Local Group galaxy is the LMC's ~450 km/s (1.9 mas/yr
# at 49.6 kpc; Gaia Collaboration, Helmi et al. 2018, A&A 616, A12); with internal rotation
# (<~ 300 km/s) no star of a galaxy moves faster than 750 km/s across the line of sight.
PM_SNR_STAR = 5.0
MAX_GALAXY_TRANSVERSE_KMS = 750.0
KMS_PER_KPC_MASYR = 4.740470463  # 1 AU/yr in km/s: v_t = 4.74 * mu[mas/yr] * d[kpc]
LMC_DISTANCE_KPC = 49.59  # Pietrzynski et al. 2019, Nature 567, 200
# Without a host distance the limit is set at the LMC distance (the nearest galaxy with a D25 ellipse
# that alerts fall on in numbers), i.e. 3.19 mas/yr.
PM_MAX_UNKNOWN_DISTANCE = MAX_GALAXY_TRANSVERSE_KMS / (KMS_PER_KPC_MASYR * LMC_DISTANCE_KPC)
# The most luminous stars reach M ~ -10 (Humphreys & Davidson 1979, ApJ 232, 409). A catalogued
# star brighter than that at the host's distance is a Galactic foreground star. (Luminosity
# alone is not used: nuclei and star clusters of nearby galaxies are brighter, e.g. G1 in M31.)
M_G_BRIGHTEST_STAR = -10.0
# --- Is a Gaia DR3 source's astrometry that of a star? ---
# The 5-parameter solution of an extended source (galaxy nucleus, AGN host, star cluster) can have
# a formally significant but spurious proper motion and parallax (live: M87's nucleus 7.5 mas/yr
# at 12 sigma, NGC 4395's 2.8 mas/yr at 17 sigma, NGC 3783's 0.24 mas/yr at 15 sigma). The
# astrometry of a source is never used as Galactic evidence when
#  * its parallax is below -3 sigma: unphysical, the solution is spurious (nuclei of M87 -3.4,
#    NGC 4395 -4.1, NGC 3783 -4.7, NGC 3516 -10.2 sigma);
GAIA_NEGATIVE_PARALLAX_SNR = -3.0
#  * Gaia DR3's DSC-Combmod classifier (Delchambre et al. 2023, A&A 674, A31) gives it
#    P(galaxy) + P(quasar) > 0.5 (M87 0.999, NGC 4258 1.0, ASASSN-14ko's host 0.9995);
GAIA_DSC_EXTRAGALACTIC_MIN = 0.5
#  * it is in Gaia DR3's galaxy_candidates (Gaia Collaboration, Bailer-Jones et al. 2023, A&A 674, A41);
#  * a SIMBAD/NED galaxy, AGN or QSO entry lies within 1.5" of it and its astrometry is not that of a
#    well-behaved point source (below): it is that galaxy's nucleus. (The bright Seyfert 1 nuclei of
#    NGC 7469, NGC 6814, NGC 4151 and NGC 3227 have DSC P(star) ~ 1 but excess-noise significances of
#    7-400; catalogue positions of nuclei differ from Gaia's photocentre by up to 1.5": M106's by
#    1.50". A well-fitted star blended with a catalogued galaxy keeps its evidence: the 9.8 mas/yr
#    common-proper-motion pair under ZTF26abxsysn, which NED types as a galaxy.)
GALAXY_COINCIDENCE_ARCSEC = 1.5
# The proper-motion test and the luminosity test further need a well-behaved *point source*:
# RUWE < 1.4 (Lindegren et al. 2021, A&A 649, A2) and astrometric_excess_noise_sig <= 2 (a larger
# value means the excess noise is significant, i.e. the source is not fitted by the single-star
# model; Lindegren et al. 2012, A&A 538, A78).
GAIA_EXCESS_NOISE_SIG_MAX = 2.0
# A stellar-type SIMBAD/NED entry is the Gaia source itself (for the luminosity test) when it lies
# within 0.5" of the Gaia position after the proper-motion drift between the catalogue's J2000
# positions and Gaia DR3's epoch J2016.0.
SAME_SOURCE_ARCSEC = 0.5
GAIA_DR3_EPOCH_YR = 2016.0
CATALOGUE_EPOCH_YR = 2000.0

# --- Distances for projected offsets ---
C_KMS = 299792.458
# Peculiar velocities (~300 km/s) make redshift distances uncertain by 300/cz; below cz = 1500
# km/s (>= 20%, and meaningless for Local Group galaxies whose cz is dominated by the solar and
# peculiar motion) no redshift distance is used. Below z = 0.01 a Cosmicflows-4 distance is preferred.
PECULIAR_VELOCITY_KMS = 300.0
HUBBLE_FLOW_MIN_CZ_KMS = 1500.0
# Redshift distances use the CMB-frame redshift: the heliocentric one is corrected for the solar
# dipole motion, 369.82 km/s towards Galactic (l, b) = (264.021, 48.253) deg (Planck Collaboration
# 2020, A&A 641, A1, Table 3), and the distances follow Davis et al. (2019, MNRAS 490, 2948):
# D_M from z_CMB, D_A = D_M / (1 + z_helio), D_L = D_M (1 + z_helio).
CMB_DIPOLE_KMS = 369.82
CMB_DIPOLE_L_DEG = 264.021
CMB_DIPOLE_B_DEG = 48.253
# Note on scales: Cosmicflows-4 distances are on the TRGB/Cepheid zero point (H0 = 74.6 km/s/Mpc,
# Tully et al. 2023) while Hubble-flow distances use Planck 2018 (H0 = 67.66): redshift distances are
# ~10% larger than CF4 ones at the same velocity; the adopted method is reported with every host.

_COSMOLOGY: Any = None


def _planck18() -> Any:
    """astropy's Planck 2018 cosmology (imported once; ~1 s, so warmed in a thread by the enricher)."""
    global _COSMOLOGY
    if _COSMOLOGY is None:
        from astropy.cosmology import Planck18

        _COSMOLOGY = Planck18
    return _COSMOLOGY


def _sql_list(values: Sequence[str]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


class HostSearchRegistry(CatalogRegistry):
    """A copy of a registry for the host search: SIMBAD and NED keep only galaxy-type rows
    (server-side ``WHERE``, so stars/HII regions cannot use up the row limit), plus HyperLEDA
    and the Cosmicflows-4 galaxy and group distances."""

    HOST_FILTERS: ClassVar[dict[str, str]] = {
        "simbad": f"b.otype IN ({_sql_list(SIMBAD_HOST_OTYPES)})",
        "ned": f"prefphytype IN ({_sql_list(sorted(NED_GALAXY_TYPES))})",
    }

    def __init__(self, base: CatalogRegistry) -> None:  # derived registry: nothing to load
        self.registry_path = None
        catalogs = dict(base.catalogs)
        for name, where in self.HOST_FILTERS.items():
            if name in catalogs:
                cat = catalogs[name]
                prior = cat.parameters.get("where")
                combined = f"({prior}) AND ({where})" if prior else where
                catalogs[name] = replace(cat, parameters={**cat.parameters, "where": combined})
        catalogs[HYPERLEDA] = catalog_from_dict(HYPERLEDA, HYPERLEDA_DEFINITION)
        catalogs[COSMICFLOWS4] = catalog_from_dict(COSMICFLOWS4, COSMICFLOWS4_DEFINITION)
        catalogs[COSMICFLOWS4_GROUPS] = catalog_from_dict(COSMICFLOWS4_GROUPS, COSMICFLOWS4_GROUPS_DEFINITION)
        self._catalogs = catalogs

    def reload(self) -> None:
        """Derived from another registry: nothing to reload."""


class MatchSearchRegistry(CatalogRegistry):
    """A copy of a registry for the counterpart search: Gaia DR3 also returns the proper-motion
    errors and correlation (``GAIA_PM_ERROR_COLUMNS``) and the source-quality columns
    (``GAIA_QUALITY_COLUMNS``) of the Galactic-star tests."""

    def __init__(self, base: CatalogRegistry) -> None:  # derived registry: nothing to load
        self.registry_path = None
        catalogs = dict(base.catalogs)
        gaia = catalogs.get("gaia_dr3")
        if gaia is not None:
            columns = list(gaia.parameters.get("columns") or [])
            missing = [c for c in (*GAIA_PM_ERROR_COLUMNS, *GAIA_QUALITY_COLUMNS) if c not in columns]
            if columns and missing:
                catalogs["gaia_dr3"] = replace(gaia, parameters={**gaia.parameters, "columns": columns + missing})
        self._catalogs = catalogs

    def reload(self) -> None:
        """Derived from another registry: nothing to reload."""


def position_angle_deg(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Position angle (deg, North through East, in [0, 360)) of point 2 seen from point 1."""
    a1, d1, a2, d2 = (math.radians(v) for v in (ra1, dec1, ra2, dec2))
    y = math.sin(a2 - a1) * math.cos(d2)
    x = math.cos(d1) * math.sin(d2) - math.sin(d1) * math.cos(d2) * math.cos(a2 - a1)
    return math.degrees(math.atan2(y, x)) % 360.0


def pa_b1950_to_icrs(ras: Sequence[float], decs: Sequence[float], pas: Sequence[float | None]) -> list[float | None]:
    """ICRS position angles of HyperLEDA's B1950 ("1950-position angle") major axes.

    The north direction of the FK4 B1950 frame is rotated with respect to the ICRS one by
    precession (~0.28 deg x sin(RA) / cos(Dec), more near the poles). Each galaxy's J2000
    position is transformed to FK4 (equinox B1950), offset by 1' along the B1950 PA there and
    transformed back; the ICRS PA is that of the offset point. Vectorised (astropy).
    """
    idx = [i for i, pa in enumerate(pas) if pa is not None]
    out: list[float | None] = [None] * len(pas)
    if not idx:
        return out
    from astropy import units as u
    from astropy.coordinates import FK4, SkyCoord

    icrs = SkyCoord([ras[i] for i in idx] * u.deg, [decs[i] for i in idx] * u.deg, frame="icrs")
    fk4 = icrs.transform_to(FK4(equinox="B1950"))
    stepped = fk4.directional_offset_by([float(pas[i]) for i in idx] * u.deg, 1.0 * u.arcmin).transform_to("icrs")
    for i, pa in zip(idx, icrs.position_angle(stepped).deg, strict=True):
        out[i] = float(pa) % 360.0
    return out


def d25_ellipse(log_d25: float | None, log_r25: float | None) -> tuple[float, float] | None:
    """(semi-major, semi-minor) axes in arcsec of a HyperLEDA D25 isophote; None without logD25."""
    if log_d25 is None:
        return None
    a = D25_SEMI_MAJOR_ARCSEC_PER_UNIT * 10.0 ** log_d25
    b = a / 10.0 ** log_r25 if log_r25 is not None and log_r25 >= 0 else a
    return a, b


def directional_light_radius(a: float, b: float, galaxy_pa_deg: float | None, direction_pa_deg: float) -> float:
    """Radius (arcsec) of an ellipse (semi-axes a >= b, major axis at ``galaxy_pa_deg``) toward ``direction_pa_deg``.

    r(theta) = a b / sqrt((b cos theta)^2 + (a sin theta)^2), theta measured from the major
    axis (Sullivan et al. 2006; Gupta et al. 2016). Without a position angle an elongated
    ellipse cannot be oriented and its circularised radius sqrt(a b) is used.
    """
    if galaxy_pa_deg is None or a == b:
        return math.sqrt(a * b)
    theta = math.radians(direction_pa_deg - galaxy_pa_deg)
    return a * b / math.hypot(b * math.cos(theta), a * math.sin(theta))


def proper_motion_significance(pmra: float | None, pmdec: float | None, pmra_error: float | None,
                               pmdec_error: float | None, corr: float | None = None) -> tuple[float | None, float | None]:
    """(total proper motion in mas/yr, its significance sqrt(mu^T C^-1 mu)) from Gaia's pm and covariance."""
    if pmra is None or pmdec is None:
        return None, None
    total = math.hypot(pmra, pmdec)
    if not pmra_error or not pmdec_error:
        return total, None
    rho = corr if corr is not None and abs(corr) < 1.0 else 0.0
    # Inverse of [[sa^2, rho sa sd], [rho sa sd, sd^2]] applied to (pmra, pmdec).
    xa, xd = pmra / pmra_error, pmdec / pmdec_error
    chi2 = (xa * xa - 2.0 * rho * xa * xd + xd * xd) / (1.0 - rho * rho)
    return total, math.sqrt(max(chi2, 0.0))


def _counterpart_summary(source: dict[str, Any]) -> dict[str, Any]:
    """Compact, JSON-safe summary of one crossmatch counterpart."""
    data = source.get("data") or {}
    physical = source.get("physical") or {}
    catalog = source.get("catalog")
    summary: dict[str, Any] = {
        "catalog": catalog,
        "source_id": source.get("source_id"),
        "ra": _float(source.get("ra")),
        "dec": _float(source.get("dec")),
        "separation_arcsec": _float(source.get("separation_arcsec")),
        "object_type": _text(data.get("otype") or data.get("prefphytype") or physical.get("object_type")),
        "redshift": _float(data.get("rvz_redshift") if catalog == "simbad" else data.get("z", physical.get("redshift"))),
    }
    if catalog == "gaia_dr3":
        plx, err = _float(data.get("parallax")), _float(data.get("parallax_error"))
        pm, pm_sig = proper_motion_significance(_float(data.get("pmra")), _float(data.get("pmdec")),
                                                _float(data.get("pmra_error")), _float(data.get("pmdec_error")),
                                                _float(data.get("pmra_pmdec_corr")))
        p_gal, p_qso = _float(data.get("classprob_dsc_combmod_galaxy")), _float(data.get("classprob_dsc_combmod_quasar"))
        summary.update({
            "parallax_mas": plx,
            "parallax_error_mas": err,
            "parallax_over_error": plx / err if plx is not None and err else None,
            "ruwe": _float(data.get("ruwe")),
            "g_mag": _float(data.get("phot_g_mean_mag")),
            "pm_masyr": pm,
            "pm_over_error": pm_sig,
            "astrometric_excess_noise_sig": _float(data.get("astrometric_excess_noise_sig")),
            "in_galaxy_candidates": _flag(data.get("in_galaxy_candidates")),
            "dsc_p_star": _float(data.get("classprob_dsc_combmod_star")),
            "dsc_p_extragalactic": (p_gal or 0.0) + (p_qso or 0.0) if p_gal is not None or p_qso is not None else None,
        })
    return summary


def _flag(value: Any) -> bool | None:
    """A boolean column (JSON true/false, 1/0 or 't'/'f' text); None when missing."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) if math.isfinite(value) else None
    text = str(value).strip().lower()
    return True if text in {"true", "t", "1"} else False if text in {"false", "f", "0"} else None


def _ned_type(otype: Any) -> tuple[str, bool]:
    """(NED type without its '!' prefix, True when NED marks it as a Galactic object)."""
    text = str(otype or "")
    return (text[1:], True) if text.startswith(NED_GALACTIC_PREFIX) else (text, False)


def _is_galaxy(entry: dict[str, Any]) -> bool:
    otype = entry.get("object_type")
    if entry.get("catalog") == "ned":
        return _ned_type(otype)[0] in NED_GALAXY_TYPES
    return otype in SIMBAD_GALAXY_TYPES


def _is_stellar(entry: dict[str, Any]) -> bool:
    otype = entry.get("object_type")
    if entry.get("catalog") == "ned":
        return _ned_type(otype)[0] in NED_STAR_TYPES
    return entry.get("catalog") == "simbad" and otype in SIMBAD_STAR_TYPES


def _is_variable(entry: dict[str, Any]) -> bool:
    otype = entry.get("object_type")
    if entry.get("catalog") == "ned":
        return _ned_type(otype)[0] in NED_VARIABLE_TYPES
    return entry.get("catalog") == "simbad" and otype in SIMBAD_VARIABLE_TYPES


def _is_agn(entry: dict[str, Any]) -> bool:
    """A catalogued active nucleus (SIMBAD AGN-branch types; NED 'QSO')."""
    otype = entry.get("object_type")
    if entry.get("catalog") == "ned":
        return _ned_type(otype)[0] in NED_AGN_TYPES
    return entry.get("catalog") == "simbad" and otype in SIMBAD_AGN_TYPES


def _is_blazar(entry: dict[str, Any]) -> bool:
    return entry.get("catalog") == "simbad" and entry.get("object_type") in SIMBAD_BLAZAR_TYPES


def _is_ned_galactic_star(entry: dict[str, Any]) -> bool:
    base, galactic = _ned_type(entry.get("object_type"))
    return entry.get("catalog") == "ned" and galactic and base in NED_STAR_TYPES - NED_EXTRAGALACTIC_STAR_TYPES


def _is_extragalactic_star(entry: dict[str, Any]) -> bool:
    return entry.get("catalog") == "ned" and _ned_type(entry.get("object_type"))[0] in NED_EXTRAGALACTIC_STAR_TYPES


def _is_transient(entry: dict[str, Any]) -> bool:
    """True for the transient itself: a transient catalogue type, or a transient-style designation
    that the catalogue does not type as a star or variable.

    Catalogued CVs and novae often carry discovery designations (SIMBAD 'MASTER OT J073857.06+182648.2'
    CV?, 'PNV J...' No*, 'ZTF18aayefwp' CV*): they are prior sources and stay counterparts.
    """
    otype = entry.get("object_type")
    if entry.get("catalog") == "ned" and _ned_type(otype)[0] in NED_TRANSIENT_TYPES:
        return True
    if entry.get("catalog") == "simbad" and otype in SIMBAD_TRANSIENT_TYPES:
        return True
    if not is_transient_designation(entry.get("source_id")):
        return False
    return not (_is_stellar(entry) or _is_variable(entry))


_CMB_APEX_ICRS: tuple[float, float] | None = None


def cmb_dipole_velocity_kms(ra: float, dec: float) -> float:
    """Line-of-sight component (km/s) of the Sun's velocity relative to the CMB towards (ra, dec).

    Positive towards the dipole apex (Planck 2020 solar dipole, see ``CMB_DIPOLE_KMS``):
    a CMB-frame velocity is cz_CMB ~ cz_helio + this value.
    """
    global _CMB_APEX_ICRS
    if _CMB_APEX_ICRS is None:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        apex = SkyCoord(l=CMB_DIPOLE_L_DEG * u.deg, b=CMB_DIPOLE_B_DEG * u.deg, frame="galactic").icrs
        _CMB_APEX_ICRS = (float(apex.ra.deg), float(apex.dec.deg))
    separation_deg = haversine_arcsec(ra, dec, *_CMB_APEX_ICRS) / 3600.0
    return CMB_DIPOLE_KMS * math.cos(math.radians(separation_deg))


def cmb_redshift(z_helio: float, ra: float, dec: float) -> float:
    """CMB-frame redshift of a heliocentric one: 1 + z_CMB = (1 + z_helio) / (1 - v_r / c), with
    v_r the solar dipole velocity towards the source (Davis et al. 2011, ApJ 741, 67, eq. 11)."""
    return (1.0 + z_helio) / (1.0 - cmb_dipole_velocity_kms(ra, dec) / C_KMS) - 1.0


def host_distance(redshift: float | None, cf4: tuple[float, float] | None = None, *, ra: float | None = None,
                  dec: float | None = None, group: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Distance of a host for projected offsets and luminosities, or None (see the module docstring).

    Returns {"angular_diameter_mpc", "distance_modulus", "method", "fractional_uncertainty", ...}:

    * ``cosmicflows4``: the galaxy's own Cosmicflows-4 distance modulus ``cf4 = (DM, e_DM)`` when
      the redshift is unknown or z < 0.01 (D_A = D_L / (1+z)^2);
    * ``cosmicflows4_group``: else, in the same range, the CF4 distance modulus of the galaxy's
      group (``group``: dm, e_dm, r2t_mpc from :meth:`AlertEnricher._cf4_group`), uncertain by
      e_DM and by the group's depth (~ its projected second-turnaround radius R2t);
    * ``hubble_flow_planck18``: else, for cz_CMB >= 1500 km/s, the Planck 2018 distances of the
      CMB-frame redshift -- the group's CMB velocity (``group['v3k_kms']``, free of the
      intra-group velocity dispersion) or the galaxy's own redshift corrected for the solar
      dipole when ``ra``/``dec`` are given (heliocentric otherwise). Uncertain by v_pec / cz_CMB
      with v_pec = 300 km/s, or the group's velocity dispersion when larger (Virgo: 670 km/s).
    """
    local = redshift is None or redshift < LOCAL_VOLUME_MAX_Z
    z_obs = redshift if redshift is not None and redshift > 0 else 0.0
    ln10_5 = math.log(10.0) / 5.0
    if cf4 is not None and local:
        dm, e_dm = cf4
        d_l = 10.0 ** (dm / 5.0 - 5.0)  # Mpc
        return {"angular_diameter_mpc": d_l / (1.0 + z_obs) ** 2, "distance_modulus": dm, "method": "cosmicflows4",
                "fractional_uncertainty": ln10_5 * e_dm if e_dm is not None else None}
    group = dict(group or {})
    if local and group.get("dm") is not None:
        dm = float(group["dm"])
        d_l = 10.0 ** (dm / 5.0 - 5.0)
        e_dm = _float(group.get("e_dm")) or 0.0
        depth = (_float(group.get("r2t_mpc")) or 0.0) / d_l
        return {"angular_diameter_mpc": d_l / (1.0 + z_obs) ** 2, "distance_modulus": dm,
                "method": "cosmicflows4_group", "fractional_uncertainty": math.hypot(ln10_5 * e_dm, depth),
                "group": group}
    if redshift is None or redshift <= 0 or redshift >= 20.0:
        return None
    sigma_v = _float(group.get("sigma_v_kms"))
    peculiar = PECULIAR_VELOCITY_KMS
    if group.get("v3k_kms") is not None:
        z_cmb, frame = float(group["v3k_kms"]) / C_KMS, "cmb_group"
    else:
        if ra is not None and dec is not None:
            z_cmb, frame = cmb_redshift(redshift, ra, dec), "cmb"
        else:
            z_cmb, frame = redshift, "heliocentric"
        if sigma_v is not None and sigma_v > peculiar:
            peculiar = sigma_v  # a group member's own velocity scatters by the group's dispersion
    if z_cmb <= 0 or C_KMS * z_cmb < HUBBLE_FLOW_MIN_CZ_KMS:
        return None
    from astropy import units as u

    cosmo = _planck18()
    d_m = float(cosmo.comoving_transverse_distance(z_cmb).to(u.Mpc).value)
    d_l = d_m * (1.0 + redshift)
    return {"angular_diameter_mpc": d_m / (1.0 + redshift), "distance_modulus": 5.0 * math.log10(d_l * 1e5),
            "method": "hubble_flow_planck18", "fractional_uncertainty": peculiar / (C_KMS * z_cmb),
            "redshift_cmb": z_cmb, "velocity_frame": frame, "peculiar_velocity_kms": peculiar,
            **({"group": group} if group else {})}


def projected_offset_kpc(separation_arcsec: float, redshift: float | None,
                         cf4: tuple[float, float] | None = None, **position: Any) -> float | None:
    """Projected separation in kpc at the host distance of :func:`host_distance` (None without one);
    ``position`` (ra, dec, group) is passed on to it."""
    dist = host_distance(redshift, cf4, **position)
    if dist is None:
        return None
    return float(dist["angular_diameter_mpc"] * 1000.0 * math.radians(separation_arcsec / 3600.0))


def _name_rank(entry: dict[str, Any]) -> tuple[int, int, int, int, int, int, float]:
    """Preferred display entry of a galaxy: not a transient designation, not a transient-host
    label ('SN 1994I HOST'), not a sub-component ('IC 0010:[CDA2003] 17-B'), a major catalogue
    name (Messier/NGC/IC/UGC/PGC/...), not a candidate type ('G?'), with a redshift, then nearest."""
    name = " ".join(str(entry.get("source_id") or "").split())
    otype = str(entry.get("object_type") or "")
    return (int(is_transient_designation(name)), int(bool(TRANSIENT_HOST_LABEL.match(name))), int(":" in name),
            int(not MAJOR_GALAXY_NAME.match(name)), int(otype.endswith("?")), int(entry.get("redshift") is None),
            float(entry.get("separation_arcsec") or 0.0))


def group_host_candidates(galaxies: list[dict[str, Any]], alias_arcsec: float = HOST_ALIAS_ARCSEC) -> list[dict[str, Any]]:
    """Merge catalogue entries of one galaxy (within ``alias_arcsec`` of the group's first entry).

    ``galaxies`` must be sorted by separation. Each group keeps the position and separation
    of its nearest entry, is named after the entry preferred by :func:`_name_rank` (a transient
    designation only when no other name exists: NED lists some hosts under the transient's
    name, e.g. 'AT 2019dsg'; such groups are flagged ``transient_named``), lists the other
    entries as ``aliases`` and adopts the first redshift found among its members
    (``redshift_source`` names the entry), then computes the (Hubble-flow) projected offset.
    """
    groups: list[dict[str, Any]] = []
    for gal in galaxies:
        for grp in groups:
            if haversine_arcsec(grp["ra"], grp["dec"], gal["ra"], gal["dec"]) <= alias_arcsec:
                grp["members"].append(gal)
                break
        else:
            groups.append({**gal, "members": [gal]})
    for grp in groups:
        members = grp.pop("members")
        named = min(members, key=_name_rank)
        grp.update({k: named[k] for k in ("catalog", "source_id", "object_type")})
        if str(named.get("object_type") or "").endswith("?"):
            definite = next((m for m in members if m.get("object_type") and not str(m["object_type"]).endswith("?")), None)
            if definite is not None:
                grp["object_type"] = definite["object_type"]
        grp["aliases"] = [f"{m['catalog']}:{m['source_id']}" for m in members if m is not named]
        grp["transient_named"] = all(is_transient_designation(m.get("source_id")) for m in members)
        with_z = next((m for m in [named, *members] if m.get("redshift") is not None), None)
        grp["redshift"] = with_z["redshift"] if with_z else None
        grp["redshift_source"] = f"{with_z['catalog']}:{with_z['source_id']}" if with_z else None
        grp["projected_offset_kpc"] = projected_offset_kpc(grp["separation_arcsec"] or 0.0, grp["redshift"])
    return groups


def _flatten(record: UnifiedRecord) -> list[dict[str, Any]]:
    rows = [src for sources in record.counterparts.values() for src in sources]
    return sorted(rows, key=lambda s: float(s.get("separation_arcsec") or 0.0))


def _catalog_status(record: UnifiedRecord) -> dict[str, str]:
    return {name: str(res.get("status")) for name, res in record.catalog_results.items()}


def _galaxies(record: UnifiedRecord) -> list[dict[str, Any]]:
    return [e for e in (_counterpart_summary(s) for s in _flatten(record))
            if _is_galaxy(e) and e["ra"] is not None and e["dec"] is not None]


def _alias_reach(gal: dict[str, Any]) -> float:
    return min(max(D25_ALIAS_MIN_ARCSEC, D25_ALIAS_FRACTION * (gal["semi_major_arcsec"] or 0.0)), D25_ALIAS_MAX_ARCSEC)


def _catalog_name_key(name: Any) -> str:
    """Comparable form of a galaxy designation: 'IC 0010' / 'IC10' -> 'IC10', 'NGC 0224' -> 'NGC224'."""
    text = re.sub(r"\s+", "", str(name or "")).upper()
    return re.sub(r"(?<=[A-Z])0+(?=\d)", "", text)


def _pretty_leda_name(name: str) -> str:
    """HyperLEDA ANames ('IC10', 'NGC3034', 'ESO56-115') with the usual space ('IC 10')."""
    return re.sub(r"^([A-Za-z]+)(?=[-+]?\d)", r"\1 ", name)


# NGC numbers of the Messier galaxies (HyperLEDA's ANames list NGC, not Messier, designations;
# M102 is omitted: its identification is disputed).
MESSIER_GALAXY_NGC: dict[int, int] = {
    31: 224, 32: 221, 33: 598, 49: 4472, 51: 5194, 58: 4579, 59: 4621, 60: 4649, 61: 4303, 63: 5055, 64: 4826,
    65: 3623, 66: 3627, 74: 628, 77: 1068, 81: 3031, 82: 3034, 83: 5236, 84: 4374, 85: 4382, 86: 4406, 87: 4486,
    88: 4501, 89: 4552, 90: 4569, 91: 4548, 94: 4736, 95: 3351, 96: 3368, 98: 4192, 99: 4254, 100: 4321, 101: 5457,
    104: 4594, 105: 3379, 106: 4258, 108: 3556, 109: 3992, 110: 205,
}


def _name_keys(name: Any) -> set[str]:
    """Comparable keys of a designation, a Messier galaxy's NGC number included ('M 83' -> M83, NGC5236)."""
    key = _catalog_name_key(name)
    keys = {key}
    found = re.fullmatch(r"M(?:ESSIER)?(\d+)", key)
    if found and int(found.group(1)) in MESSIER_GALAXY_NGC:
        keys.add(f"NGC{MESSIER_GALAXY_NGC[int(found.group(1))]}")
    return keys


def _names_hyperleda_galaxy(group: dict[str, Any], gal: dict[str, Any]) -> bool:
    """True when a NED/SIMBAD galaxy group carries one of the HyperLEDA galaxy's names."""
    wanted = {_catalog_name_key(n) for n in gal.get("hyperleda_names") or []}
    names = [group.get("source_id"), *(str(a).split(":", 1)[-1] for a in group.get("aliases") or [])]
    return bool(wanted & {k for n in names for k in _name_keys(n)})


class AlertEnricher:
    """Crossmatch an alert (counterparts, host galaxy, star/variable/new flags) via CrossmatchService."""

    def __init__(
        self,
        service: CrossmatchService,
        *,
        match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
        host_radius_arcsec: float = DEFAULT_HOST_RADIUS_ARCSEC,
        catalogs: Sequence[str] = DEFAULT_MATCH_CATALOGS,
        host_catalogs: Sequence[str] = DEFAULT_HOST_CATALOGS,
        parallax_snr: float = PARALLAX_SNR_STAR,
        d25_search: bool = True,
    ) -> None:
        if not math.isfinite(match_radius_arcsec) or match_radius_arcsec <= 0:
            raise ValueError("match_radius_arcsec must be positive")
        if not math.isfinite(host_radius_arcsec) or host_radius_arcsec < 0:
            raise ValueError("host_radius_arcsec must be >= 0")
        self.service = service
        self.match_radius_arcsec = float(match_radius_arcsec)
        self.host_radius_arcsec = float(host_radius_arcsec)
        enabled = service.registry.enabled_catalogs()
        self.catalogs = [c for c in catalogs if c in enabled]
        self.host_catalogs = [c for c in host_catalogs if c in enabled]
        self.parallax_snr = parallax_snr
        self.d25_search = bool(d25_search) and self.host_radius_arcsec > 0
        # The derived services inherit the injected service's executor limits (timeout and the
        # operator's CATALOG_TIMEOUT_CAP_SECONDS cap), association parameters and concurrency.
        executor = service.executor
        derived: dict[str, Any] = {"timeout": executor.timeout, "timeout_cap": getattr(executor, "timeout_cap", None)}
        if getattr(service, "association_config", None) is not None:
            derived["association_config"] = service.association_config
        if getattr(service, "max_concurrency", None) is not None:
            derived["max_concurrency"] = service.max_concurrency
        self.match_service = CrossmatchService(MatchSearchRegistry(service.registry), service.providers, **derived)
        self.host_service = CrossmatchService(HostSearchRegistry(service.registry), service.providers, **derived)

    async def _crossmatch(self, service: CrossmatchService, ra: float, dec: float, catalogs: list[str],
                          radius: float) -> UnifiedRecord:
        # Positions are compared as given (no epoch): alert positions are current, and an
        # epoch-widened cone (up to ~10.5"/yr x gap) would be far larger than the 2" match.
        query = AdvancedQuery.from_dict({
            "ra": ra, "dec": dec, "radius_arcsec": radius, "catalogs": catalogs,
            "min_confidence": 0.0, "proper_motion": False,
        })
        return await service.crossmatch(ra, dec, query=query)

    async def _d25(self, alert: Alert) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool]:
        """HyperLEDA galaxies within the host radius or near enough to host the alert by their D25 ellipse.

        The server keeps rows with DLR_HOST_MAX * 3" * 10**logD25 (4 semi-major axes) >= separation,
        i.e. every galaxy the alert could be associated with at d_DLR <= 4 (Gupta et al. 2016);
        the exact ellipse distance is computed here. Returns (galaxies sorted by d_DLR, failure,
        truncated).
        """
        distance = f"DISTANCE(POINT('ICRS', RAJ2000, DEJ2000), POINT('ICRS', {alert.ra:.9f}, {alert.dec:.9f}))"
        where = (f"{HYPERLEDA_OTYPE_FILTER} AND ({distance} <= {self.host_radius_arcsec / 3600.0:.10f} OR "
                 f"{DLR_HOST_MAX * D25_SEMI_MAJOR_ARCSEC_PER_UNIT:g} * POWER(10, logD25) >= 3600.0 * {distance})")
        definition = self.host_service.registry.get(HYPERLEDA)
        plan = QueryPlan(HYPERLEDA, definition.provider, definition.endpoint, {"where": where},
                         D25_SEARCH_RADIUS_DEG * 3600.0, definition.wavelength)
        successes, failures = await self.host_service.executor.execute([plan], validate_target(alert.ra, alert.dec))
        if failures:
            return [], failures[0].as_dict(), False
        rows = list(successes[0][1])
        truncated = bool(getattr(successes[0][1], "meta", {}).get("truncated", False))
        pas_1950 = [_float((src.data or {}).get("PA")) for src in rows]
        pas_icrs = await asyncio.to_thread(pa_b1950_to_icrs, [src.ra for src in rows], [src.dec for src in rows], pas_1950)
        galaxies: list[dict[str, Any]] = []
        for src, pa_1950, pa in zip(rows, pas_1950, pas_icrs, strict=True):
            data = src.data or {}
            axes = d25_ellipse(_float(data.get("logD25")), _float(data.get("logR25")))
            sep = haversine_arcsec(src.ra, src.dec, alert.ra, alert.dec)
            pgc = data.get("PGC", src.source_id)
            entry: dict[str, Any] = {
                "catalog": HYPERLEDA, "source_id": f"PGC {pgc}", "pgc": int(pgc) if str(pgc).isdigit() else None,
                "otype": _text(data.get("OType")),
                "hyperleda_names": [a for a in str(data.get("ANames") or "").split() if a],
                "ra": src.ra, "dec": src.dec, "separation_arcsec": sep, "semi_major_arcsec": None,
                "semi_minor_arcsec": None, "pa_deg": pa, "pa_b1950_deg": pa_1950, "dlr_arcsec": None, "d_dlr": None,
            }
            if axes is not None:
                a, b = axes
                radius = directional_light_radius(a, b, pa, position_angle_deg(src.ra, src.dec, alert.ra, alert.dec))
                entry.update({"semi_major_arcsec": a, "semi_minor_arcsec": b, "dlr_arcsec": radius,
                              "d_dlr": sep / radius if radius > 0 else None})
            galaxies.append(entry)
        galaxies.sort(key=lambda g: (g["d_dlr"] is None, g["d_dlr"] if g["d_dlr"] is not None else 0.0, g["separation_arcsec"]))
        return galaxies, None, truncated

    async def _cf4_distance(self, pgc: int, ra: float, dec: float) -> tuple[float, float] | None:
        """Cosmicflows-4 (DM, e_DM) of PGC ``pgc`` (None when CF4 has no distance); raises on failure."""
        definition = self.host_service.registry.get(COSMICFLOWS4)
        plan = QueryPlan(COSMICFLOWS4, definition.provider, definition.endpoint, {"where": f"PGC = {int(pgc)}"},
                         CF4_SEARCH_RADIUS_ARCSEC, definition.wavelength)
        successes, failures = await self.host_service.executor.execute([plan], validate_target(ra, dec))
        if failures:
            failure = failures[0]
            raise RuntimeError(f"{failure.error_type}: {failure.message}")
        for src in successes[0][1]:
            data = src.data or {}
            if str(data.get("PGC", src.source_id)) == str(pgc) and _float(data.get("DM")) is not None:
                return float(data["DM"]), _float(data.get("e_DM")) or 0.0
        return None

    async def _cf4_group(self, pgc: int, ra: float, dec: float) -> dict[str, Any] | None:
        """Group of PGC ``pgc`` in Tully (2015) with its Cosmicflows-4 group distance (when CF4 has one);
        None when the galaxy is in no group of that catalogue; raises on failure."""
        definition = self.host_service.registry.get(COSMICFLOWS4_GROUPS)
        plan = QueryPlan(COSMICFLOWS4_GROUPS, definition.provider, definition.endpoint, {"where": f"m.PGC = {int(pgc)}"},
                         CF4_SEARCH_RADIUS_ARCSEC, definition.wavelength)
        successes, failures = await self.host_service.executor.execute([plan], validate_target(ra, dec))
        if failures:
            failure = failures[0]
            raise RuntimeError(f"{failure.error_type}: {failure.message}")
        for src in successes[0][1]:
            data = src.data or {}
            if str(data.get("PGC", src.source_id)) != str(pgc):
                continue
            nest, pgc1 = _float(data.get("Nest")), _float(data.get("PGC1"))
            return {
                "catalog": "Tully 2015 (J/AJ/149/171) + Cosmicflows-4 groups (J/ApJ/944/94)",
                "nest": int(nest) if nest is not None else None, "pgc1": int(pgc1) if pgc1 is not None else None,
                "n_members": _float(data.get("Nmb")), "sigma_v_kms": _float(data.get("sigV")),
                "r2t_mpc": _float(data.get("R2t")), "n_distances": _float(data.get("Ngal")),
                "dm": _float(data.get("DMzp")), "e_dm": _float(data.get("e_DMzp")), "v3k_kms": _float(data.get("V3k")),
            }
        return None

    async def enrich(self, alert: Alert) -> AlertEnrichment:
        """Run the counterpart, host-cone and D25 searches concurrently and derive the flags."""
        started = time.perf_counter()
        result = AlertEnrichment(
            status="done",
            match_radius_arcsec=self.match_radius_arcsec,
            host_radius_arcsec=self.host_radius_arcsec,
            catalogs=list(self.catalogs),
        )
        search_host = bool(self.host_catalogs) and self.host_radius_arcsec > 0
        jobs: dict[str, Awaitable[Any]] = {"cosmology": asyncio.to_thread(_planck18)}
        if self.catalogs:
            jobs["match"] = self._crossmatch(self.match_service, alert.ra, alert.dec, self.catalogs, self.match_radius_arcsec)
        if search_host:
            jobs["host"] = self._crossmatch(self.host_service, alert.ra, alert.dec, self.host_catalogs, self.host_radius_arcsec)
        if self.d25_search:
            jobs["d25"] = self._d25(alert)
        outcomes = dict(zip(jobs, await asyncio.gather(*jobs.values(), return_exceptions=True), strict=True))
        match_outcome, host_outcome, d25_outcome = outcomes.get("match"), outcomes.get("host"), outcomes.get("d25")

        # --- counterparts within the match radius ---
        answered: set[str] = set()
        match_failed = False
        if isinstance(match_outcome, UnifiedRecord):
            result.catalog_status = _catalog_status(match_outcome)
            result.failures = list(match_outcome.failures)
            answered = {n for n, s in result.catalog_status.items() if s != "failed"}
            match_failed = len(answered) < len(self.catalogs)
            for src in _flatten(match_outcome):
                entry = _counterpart_summary(src)
                if _is_transient(entry):
                    if str(entry["source_id"]).lower() not in {d.lower() for d in result.transient_designations}:
                        result.transient_designations.append(str(entry["source_id"]))
                    continue
                if is_transient_designation(entry["source_id"]):
                    result.evidence.append(
                        f"{str(entry['catalog']).upper()} {entry['source_id']} has a transient-style designation but is "
                        f"catalogued as type {entry['object_type']}: a known prior source, not the transient itself")
                result.counterparts.append(entry)
        elif isinstance(match_outcome, BaseException):
            match_failed = True
            result.error = f"counterpart crossmatch failed: {match_outcome.__class__.__name__}: {match_outcome}"
            result.failures.append({"catalog": "counterparts", "status": "failed",
                                    "error_type": match_outcome.__class__.__name__, "message": str(match_outcome)})
        if self.catalogs and not answered:
            # Nothing is known about the position: keep the alert queued for another attempt.
            result.status = "failed"
            if result.error is None:
                result.error = "every counterpart catalog failed: " + "; ".join(
                    f"{f.get('catalog')}: {f.get('error_type')}" for f in result.failures)

        # --- host search: galaxy-filtered cone + HyperLEDA D25 ellipses ---
        host_failed, host_truncated, host_cone_answered = False, False, False
        groups: list[dict[str, Any]] = []
        if isinstance(host_outcome, UnifiedRecord):
            groups = group_host_candidates(_galaxies(host_outcome))
            host_answered = [n for n, r in host_outcome.catalog_results.items() if r.get("status") != "failed"]
            host_cone_answered = bool(host_answered)
            host_failed = len(host_answered) < len(self.host_catalogs)
            for name, res in host_outcome.catalog_results.items():
                if res.get("truncated"):
                    host_truncated = True
                    detail = "; ".join(res.get("warnings") or [])
                    result.evidence.append(f"host search in {name} truncated" + (f": {detail}" if detail else ""))
            result.failures.extend(dict(f, search="host") for f in host_outcome.failures)
        elif isinstance(host_outcome, BaseException):
            host_failed = True
            result.failures.append({"catalog": "host", "search": "host", "status": "failed",
                                    "error_type": host_outcome.__class__.__name__, "message": str(host_outcome)})
        d25: list[dict[str, Any]] = []
        d25_failed = False
        if isinstance(d25_outcome, tuple):
            d25, d25_failure, d25_truncated = d25_outcome
            if d25_failure is not None:
                d25_failed = True
                result.failures.append(dict(d25_failure, search="host"))
            if d25_truncated:
                host_truncated = True
                result.evidence.append("HyperLEDA D25 search truncated at 200 rows")
        elif isinstance(d25_outcome, BaseException):
            d25_failed = True
            result.failures.append({"catalog": HYPERLEDA, "search": "host", "status": "failed",
                                    "error_type": d25_outcome.__class__.__name__, "message": str(d25_outcome)})
        distance_failed = False
        if search_host:
            identity_failed = await self._choose_host(alert, result, groups, d25)
            host_failed = host_failed or identity_failed
            complete = not (host_failed or d25_failed or host_truncated)
            result.host_search_complete = complete
            if result.host is None:
                if result.host_status == "ambiguous_transient_entry":
                    pass
                elif complete:
                    result.host_status = "none_within_radius"
                elif host_cone_answered or (self.d25_search and not d25_failed):
                    result.host_status = "incomplete"
                else:
                    result.host_status = "failed"
            else:
                distance_failed = await self._host_distance(result)
                if not complete:
                    result.evidence.append("host search incomplete (a catalog failed or a cone was truncated): "
                                           "a better host may exist")

        self._flags(alert, result, answered, d25_answered=self.d25_search and not d25_failed)
        if result.known_star and result.host is not None:
            # A Galactic star has no extragalactic host: galaxies nearby are background objects.
            result.evidence.append(f"host candidate {result.host['name']} not adopted: the alert is a Galactic star")
            result.host = None
            result.host_status = "not_applicable_star"
        if result.status != "failed" and (match_failed or host_failed or d25_failed or distance_failed):
            result.status = "partial"
            result.error = "some catalogs failed: " + "; ".join(
                f"{f.get('catalog')}: {f.get('error_type')}" for f in result.failures)
        result.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        result.crossmatched_at = _utcnow()
        return result

    async def _choose_host(self, alert: Alert, result: AlertEnrichment, groups: list[dict[str, Any]],
                           d25: list[dict[str, Any]]) -> bool:
        """Choose the host galaxy by the directional light radius (Sullivan et al. 2006; Gupta et al. 2016).

        1. Inside a D25 ellipse (d_DLR <= 1): the galaxy with the smallest d_DLR (method 'd25_ellipse').
        2. Else the galaxy with the smallest d_DLR <= DLR_HOST_MAX = 4, outside its D25 ellipse
           (method 'dlr_outside_d25'; e.g. SN 2023bee -> NGC 2708 at d_DLR 1.4, SN 2018aoz -> NGC 3923
           at 1.1) -- unless a catalogued galaxy without a D25 size (whose own d_DLR cannot be
           computed) lies nearer to the alert than that galaxy's light radius and outside its D25
           ellipse: that galaxy may be the host, and rule 3 applies (the D25 candidate is recorded
           in the evidence).
        3. Else the nearest catalogued galaxy of the host cone (method 'nearest').

        Galaxy entries named only by a transient designation (or by one of the alert's own
        designations) are not adopted by the nearest-galaxy rule: the catalogue may list the
        transient itself as a galaxy (e.g. NED 'AT 2017abr', type G, for a Galactic CV). When
        they are the only candidates, ``host_status`` is 'ambiguous_transient_entry'.
        Returns True when the NED/SIMBAD identity lookup of a D25 host failed.
        """
        for gal in d25:
            # A cone entry is the HyperLEDA galaxy when it lies at its centre (catalogue centres
            # differ by a few arcsec) or, further out (large galaxies), carries one of its names:
            # otherwise a fragment near a big galaxy's centre (e.g. 'EZOA J0020+59', 45" from
            # IC 10's) would stand for the galaxy; the identity is then resolved at the centre.
            # Among several such entries, one carrying a HyperLEDA name or the best catalogue name
            # wins over a fibre/fragment entry nearer to HyperLEDA's centre (M83: 'M 83', not
            # '6dFGS gJ133700.5-295200').
            reach = _alias_reach(gal)
            near = [(haversine_arcsec(g["ra"], g["dec"], gal["ra"], gal["dec"]), g) for g in groups]
            near = [(d, g) for d, g in near if d <= reach and g.get("pgc") is None
                    and (d <= D25_ALIAS_MIN_ARCSEC or _names_hyperleda_galaxy(g, gal))]
            if near:
                grp = min(near, key=lambda t: (not _names_hyperleda_galaxy(t[1], gal), _name_rank(t[1])[:-1], t[0]))[1]
                grp.update({"pgc": gal["pgc"], "d_dlr": gal["d_dlr"], "d25_semi_major_arcsec": gal["semi_major_arcsec"]})
                grp["aliases"].append(f"{HYPERLEDA}:{gal['source_id']}")
                gal["matched"] = f"{grp['catalog']}:{grp['source_id']}"
        result.host_candidates = [dict(g) for g in groups[:5]]
        result.d25_galaxies = [dict(g) for g in d25[:5]]
        own = set(result.transient_designations)
        usable = [g for g in groups if not g.get("transient_named") and str(g["source_id"]) not in own]
        inside = [g for g in d25 if g["d_dlr"] is not None and g["d_dlr"] <= DLR_INSIDE]
        outer = [g for g in d25 if g["d_dlr"] is not None and DLR_INSIDE < g["d_dlr"] <= DLR_HOST_MAX]
        chosen: dict[str, Any] | None = None
        method = "nearest"
        if inside:
            chosen, method = inside[0], "d25_ellipse"
        elif outer:
            gal = outer[0]
            blockers = [g for g in usable if g.get("pgc") is None and g["separation_arcsec"] < gal["dlr_arcsec"]
                        and not _inside_d25(g, gal)]
            names = " ".join(gal["hyperleda_names"][:2])
            if blockers:
                near_gal = blockers[0]
                result.evidence.append(
                    f"{gal['source_id']}{f' ({names})' if names else ''} (d_DLR = {gal['d_dlr']:.2f}, outside its D25 "
                    f"ellipse) not adopted: {near_gal['catalog']}:{near_gal['source_id']} (no D25 size) lies nearer "
                    f"({near_gal['separation_arcsec']:.1f}\") than that galaxy's light radius ({gal['dlr_arcsec']:.0f}\") "
                    "and may be the host")
            else:
                chosen, method = gal, "dlr_outside_d25"
        best: dict[str, Any] | None = None
        identity_failed = False
        if chosen is not None:
            gal = chosen
            best = next((g for g in groups if g.get("pgc") is not None and g.get("pgc") == gal["pgc"]), None)
            if best is None:
                best, identity_failed = await self._resolve_d25_galaxy(alert, gal, result)
            off_centre = haversine_arcsec(best["ra"], best["dec"], gal["ra"], gal["dec"])
            unnamed = off_centre > D25_ALIAS_MIN_ARCSEC and not _names_hyperleda_galaxy(best, gal)
            if best.get("catalog") != HYPERLEDA and (best.get("transient_named") or unnamed):
                # The galaxy is established by HyperLEDA: name and place it after HyperLEDA. The
                # NED/SIMBAD entry (named after a transient, or off the D25 centre without a common
                # name, e.g. 'EZOA J0020+59' 31" from IC 10's centre) is kept as an alias for its redshift.
                label = _pretty_leda_name(gal["hyperleda_names"][0]) if gal["hyperleda_names"] else gal["source_id"]
                result.evidence.append(
                    f"{best['catalog']}:{best['source_id']} ({off_centre:.0f}\" from the D25 centre) supplies the redshift "
                    f"of {label}; the host is named and placed after HyperLEDA")
                best = {**best, "catalog": HYPERLEDA, "source_id": label, "ra": gal["ra"], "dec": gal["dec"],
                        "separation_arcsec": gal["separation_arcsec"],
                        "aliases": [*best.get("aliases", []), f"{best['catalog']}:{best['source_id']}"]}
            names = " ".join(gal["hyperleda_names"][:2])
            if method == "d25_ellipse":
                result.evidence.append(
                    f"inside the D25 ellipse of {gal['source_id']}{f' ({names})' if names else ''}: d_DLR = "
                    f"{gal['d_dlr']:.2f} (semi-major axis {gal['semi_major_arcsec']:.0f}\", separation "
                    f"{gal['separation_arcsec']:.1f}\")")
            else:
                result.evidence.append(
                    f"outside every D25 ellipse; nearest in light radii: {gal['source_id']}"
                    f"{f' ({names})' if names else ''} at d_DLR = {gal['d_dlr']:.2f} <= {DLR_HOST_MAX:g} (Gupta et al. "
                    f"2016; semi-major axis {gal['semi_major_arcsec']:.0f}\", separation {gal['separation_arcsec']:.1f}\")")
        else:
            skipped = [g for g in groups if g not in usable]
            best = usable[0] if usable else None
            for grp in skipped:
                if best is None or grp["separation_arcsec"] < best["separation_arcsec"]:
                    result.evidence.append(
                        f"galaxy entry {grp['catalog']}:{grp['source_id']} (type {grp['object_type']}, z={grp['redshift']}) "
                        f"at {grp['separation_arcsec']:.2f}\" is named only by a transient designation: the catalogue "
                        "may list the transient itself as a galaxy; not adopted as host")
            if best is None and skipped:
                result.host_status = "ambiguous_transient_entry"
        if best is None:
            return identity_failed
        host = HostCandidate(
            name=str(best["source_id"]), catalog=str(best["catalog"]), ra=float(best["ra"]), dec=float(best["dec"]),
            separation_arcsec=float(best["separation_arcsec"]), object_type=best.get("object_type"),
            redshift=best.get("redshift"), projected_offset_kpc=best.get("projected_offset_kpc"),
            redshift_source=best.get("redshift_source"), aliases=list(best.get("aliases") or []), method=method,
            d_dlr=best.get("d_dlr"), pgc=best.get("pgc"), d25_semi_major_arcsec=best.get("d25_semi_major_arcsec"),
        )
        result.host = host.as_dict()
        result.host_status = "found"
        return identity_failed

    async def _host_distance(self, result: AlertEnrichment) -> bool:
        """Set the host's distance and projected offset (:func:`host_distance`); True when a CF4 lookup failed.

        A host with a PGC number and z < 0.01 (or no redshift) takes its Cosmicflows-4 distance,
        else the CF4 distance of its group; a host with a Hubble-flow distance takes the CMB-frame
        velocity of its group when it is in one (cz < GROUP_CATALOG_MAX_CZ_KMS).
        """
        host = result.host
        assert host is not None
        z = host.get("redshift")
        pgc = host.get("pgc")
        cf4: tuple[float, float] | None = None
        group: dict[str, Any] | None = None
        failed = False
        ra, dec = float(host["ra"]), float(host["dec"])
        if pgc is not None and (z is None or z < LOCAL_VOLUME_MAX_Z):
            try:
                cf4 = await self._cf4_distance(int(pgc), ra, dec)
            except Exception as exc:  # noqa: BLE001 - the host stands without its redshift-independent distance
                failed = True
                result.failures.append({"catalog": COSMICFLOWS4, "search": "host_distance", "status": "failed",
                                        "error_type": exc.__class__.__name__, "message": str(exc)})
        if pgc is not None and cf4 is None and not failed and (z is None or C_KMS * z < GROUP_CATALOG_MAX_CZ_KMS):
            try:
                group = await self._cf4_group(int(pgc), ra, dec)
            except Exception as exc:  # noqa: BLE001 - the host stands without its group distance
                failed = True
                result.failures.append({"catalog": COSMICFLOWS4_GROUPS, "search": "host_distance", "status": "failed",
                                        "error_type": exc.__class__.__name__, "message": str(exc)})
        dist = host_distance(z, cf4, ra=ra, dec=dec, group=group)
        offset = None
        if dist is not None:
            offset = dist["angular_diameter_mpc"] * 1000.0 * math.radians(host["separation_arcsec"] / 3600.0)
            host.update({"distance_mpc": dist["angular_diameter_mpc"], "distance_modulus": dist["distance_modulus"],
                         "distance_method": dist["method"],
                         "distance_uncertainty_fraction": dist.get("fractional_uncertainty"),
                         "redshift_cmb": dist.get("redshift_cmb"), "velocity_frame": dist.get("velocity_frame")})
        if group is not None:
            host["group"] = group
        host["projected_offset_kpc"] = offset
        text = f", {offset:.2f} kpc projected" if offset is not None else ""
        result.evidence.append(
            f"host {host['name']} ({host['catalog']}, type {host['object_type']}, z={z}) at "
            f"{host['separation_arcsec']:.2f}\"{text} [{host['method']}]")
        group_label = ""
        if group:
            members = group.get("n_members")
            dispersion = (f", {members:.0f} members, sigma_v = {group['sigma_v_kms']:.0f} km/s"
                          if members and members > 1 and group.get("sigma_v_kms") else ", a single-galaxy group")
            group_label = f"group {group['nest']} of Tully (2015) (dominant galaxy PGC {group['pgc1']}{dispersion})"
        if dist is not None and dist["method"] == "cosmicflows4":
            result.evidence.append(f"host distance {dist['angular_diameter_mpc']:.3g} Mpc from the Cosmicflows-4 distance "
                                   f"modulus {dist['distance_modulus']:.2f} (Tully et al. 2023)")
        elif dist is not None and dist["method"] == "cosmicflows4_group":
            result.evidence.append(
                f"host distance {dist['angular_diameter_mpc']:.3g} Mpc from the Cosmicflows-4 distance modulus "
                f"{dist['distance_modulus']:.2f} of its {group_label} (no distance of its own), uncertain by "
                f"~{100.0 * dist['fractional_uncertainty']:.0f}% (group e_DM and depth R2t = {group['r2t_mpc']} Mpc)")
        elif dist is not None:
            velocity = {"cmb_group": f"the CMB-frame velocity of its {group_label}",
                        "cmb": "its CMB-frame redshift", "heliocentric": "its heliocentric redshift"}[dist["velocity_frame"]]
            result.evidence.append(
                f"host distance {dist['angular_diameter_mpc']:.3g} Mpc (Planck 2018) from {velocity}, cz = "
                f"{C_KMS * dist['redshift_cmb']:.0f} km/s: uncertain by ~{100.0 * dist['fractional_uncertainty']:.0f}% "
                f"(peculiar velocities ~{dist['peculiar_velocity_kms']:.0f} km/s"
                + (", the group's velocity dispersion" if dist["peculiar_velocity_kms"] > PECULIAR_VELOCITY_KMS else "")
                + ")")
        else:
            if z is None:
                why = "the redshift is unknown"
            else:
                cz = C_KMS * (cmb_redshift(z, ra, dec) if z > 0 else z)
                why = (f"cz_CMB = {cz:.0f} km/s < {HUBBLE_FLOW_MIN_CZ_KMS:.0f} km/s is dominated by peculiar and solar "
                       "motion")
            if failed:
                lookup = "the Cosmicflows-4 lookup failed"
            elif pgc is None:
                lookup = "no PGC identity for a CF4 lookup"
            elif group is not None:
                lookup = f"no Cosmicflows-4 distance for it or its {group_label}"
            else:
                lookup = "no Cosmicflows-4 distance and no group"
            result.evidence.append(f"no projected offset for {host['name']}: {why} and {lookup}")
        return failed

    async def _resolve_d25_galaxy(self, alert: Alert, gal: dict[str, Any], result: AlertEnrichment
                                  ) -> tuple[dict[str, Any], bool]:
        """NED/SIMBAD identity and redshift of a HyperLEDA galaxy outside the host cone (e.g. M31 for M31N 2008-12a)."""
        reach = _alias_reach(gal)
        base: dict[str, Any] = {
            "catalog": HYPERLEDA, "source_id": gal["source_id"], "ra": gal["ra"], "dec": gal["dec"],
            "separation_arcsec": gal["separation_arcsec"], "object_type": "G", "redshift": None, "redshift_source": None,
            "aliases": list(gal["hyperleda_names"]), "pgc": gal["pgc"], "d_dlr": gal["d_dlr"],
            "d25_semi_major_arcsec": gal["semi_major_arcsec"], "projected_offset_kpc": None,
        }
        if not self.host_catalogs:
            return base, False
        try:
            record = await self._crossmatch(self.host_service, gal["ra"], gal["dec"], self.host_catalogs, reach)
        except Exception as exc:  # noqa: BLE001 - the D25 host stands without its NED/SIMBAD identity
            result.failures.append({"catalog": "host_identity", "search": "host", "status": "failed",
                                    "error_type": exc.__class__.__name__, "message": str(exc)})
            return base, True
        result.failures.extend(dict(f, search="host_identity") for f in record.failures)
        found = group_host_candidates(_galaxies(record), alias_arcsec=reach)
        if not found:
            return base, bool(record.failures)
        # Prefer the entry carrying one of HyperLEDA's names, then the best catalogue name, then the centre.
        grp = min(found, key=lambda g: (not _names_hyperleda_galaxy(g, gal), _name_rank(g)[:-1],
                                        haversine_arcsec(g["ra"], g["dec"], gal["ra"], gal["dec"])))
        sep = haversine_arcsec(alert.ra, alert.dec, grp["ra"], grp["dec"])
        resolved = {**grp, "separation_arcsec": sep, "aliases": [*grp["aliases"], f"{HYPERLEDA}:{gal['source_id']}"],
                    "pgc": gal["pgc"], "d_dlr": gal["d_dlr"], "d25_semi_major_arcsec": gal["semi_major_arcsec"],
                    "projected_offset_kpc": projected_offset_kpc(sep, grp["redshift"])}
        return resolved, bool(record.failures)

    def _gaia_verdict(self, entry: dict[str, Any], enclosing: dict[str, Any] | None,
                      associated: dict[str, Any] | None, in_galaxy: str,
                      counterparts: Sequence[dict[str, Any]]) -> tuple[list[str], list[str]]:
        """(reasons the Gaia DR3 source is a Galactic star, notes on evidence not used).

        First, is the astrometry a star's? It is not used at all when the parallax is < -3 sigma
        (a spurious solution) or when the source looks extragalactic: Gaia's DSC gives
        P(galaxy) + P(quasar) > 0.5, it is a Gaia DR3 galaxy candidate, or a SIMBAD/NED galaxy,
        AGN or QSO entry lies within 1.5" *and* the source is not a well-behaved point source
        (RUWE < 1.4, excess-noise significance <= 2) -- a galaxy nucleus, AGN or cluster (bright
        Seyfert 1 nuclei have DSC P(star) ~ 1 but excess noise; a well-fitted star blended with a
        catalogued galaxy keeps its evidence, e.g. the 9.8 mas/yr pair under ZTF26abxsysn). Then:

        * parallax >= 5 sigma: outside galaxies always; inside a D25 ellipse when >= 10 sigma,
          or G < 19, or RUWE < 1.4;
        * proper motion >= 5 sigma and faster than 750 km/s at the distance of the associated
          host (3.2 mas/yr at the LMC's when unknown), for a well-behaved point source;
        * at the distance of the associated host: the Gaia source *is* a catalogued star (a
          SIMBAD/NED stellar-type entry within 0.5" after the J2000 -> J2016 drift), is a
          well-behaved point source, and has M_G = G - DM < -10 (Humphreys & Davidson 1979).
        """
        reasons: list[str] = []
        notes: list[str] = []
        sid, sep = entry["source_id"], entry["separation_arcsec"] or 0.0
        poe, ruwe, g = entry.get("parallax_over_error"), entry.get("ruwe"), entry.get("g_mag")
        pm, pm_sig = entry.get("pm_masyr"), entry.get("pm_over_error")
        aens, igc, p_ext = entry.get("astrometric_excess_noise_sig"), entry.get("in_galaxy_candidates"),             entry.get("dsc_p_extragalactic")
        not_point: list[str] = []  # the single-star astrometric model does not fit it well
        if ruwe is not None and ruwe >= GAIA_RUWE_MAX:
            not_point.append(f"RUWE {ruwe:.2f}")
        if aens is not None and aens > GAIA_EXCESS_NOISE_SIG_MAX:
            not_point.append(f"excess-noise significance {aens:.1f}")
        spurious: list[str] = []
        if poe is not None and poe <= GAIA_NEGATIVE_PARALLAX_SNR:
            spurious.append(f"parallax {poe:.1f} sigma: negative, the solution is spurious")
        if p_ext is not None and p_ext > GAIA_DSC_EXTRAGALACTIC_MIN:
            spurious.append(f"Gaia DSC P(galaxy) + P(quasar) = {p_ext:.3f}")
        if igc:
            spurious.append("a Gaia DR3 galaxy candidate")
        if not_point and entry.get("ra") is not None and entry.get("dec") is not None:
            for other in counterparts:
                if other.get("catalog") in {"simbad", "ned"} and _is_galaxy(other) and other.get("ra") is not None                         and other.get("dec") is not None:
                    d = haversine_arcsec(entry["ra"], entry["dec"], other["ra"], other["dec"])
                    if d <= GALAXY_COINCIDENCE_ARCSEC:
                        spurious.append(f"{other['catalog'].upper()} {other['source_id']} (type {other['object_type']}) "
                                        f"at {d:.2f}\" with {', '.join(not_point)}")
                        break
        if spurious:
            motion = (f"; its proper motion {pm:.2f} mas/yr ({pm_sig:.0f} sigma) is not a star's"
                      if pm is not None and pm_sig is not None and pm_sig >= PM_SNR_STAR else "")
            notes.append(f"Gaia DR3 {sid} at {sep:.2f}\" looks like a galaxy nucleus, AGN or cluster, not a star ("
                         + "; ".join(spurious) + f"): its parallax and proper motion are not used{motion}")
            return reasons, notes
        if poe is not None and poe >= self.parallax_snr:
            plx = (f"Gaia DR3 {sid} at {sep:.2f}\": parallax {entry['parallax_mas']:.3f} +/- "
                   f"{entry['parallax_error_mas']:.3f} mas ({poe:.1f} sigma, G={g}, RUWE {ruwe})")
            if enclosing is None:
                reasons.append(f"{plx} -> Galactic star")
            elif poe >= PARALLAX_SNR_SECURE:
                reasons.append(f"{plx}: >= {PARALLAX_SNR_SECURE:g} sigma, secure even projected{in_galaxy} -> Galactic star")
            elif g is not None and g < GAIA_BRIGHT_G:
                reasons.append(f"{plx}: G < {GAIA_BRIGHT_G:g}, not a faint crowded-field source -> Galactic star")
            elif ruwe is not None and ruwe < GAIA_RUWE_MAX:
                reasons.append(f"{plx}: well-behaved solution (RUWE < {GAIA_RUWE_MAX}) -> Galactic star")
            else:
                brightness = f"faint (G={g:.2f})" if g is not None else "G unknown"
                notes.append(f"Gaia DR3 {sid}: {poe:.1f} sigma parallax not used: projected{in_galaxy}, {brightness}, "
                             f"RUWE {ruwe} (not < {GAIA_RUWE_MAX}): possibly spurious (Rybizki et al. 2022)")
        distance_kpc = associated.get("distance_mpc") * 1000.0 if associated and associated.get("distance_mpc") else None
        pm_max = (MAX_GALAXY_TRANSVERSE_KMS / (KMS_PER_KPC_MASYR * distance_kpc) if distance_kpc
                  else PM_MAX_UNKNOWN_DISTANCE)
        if pm is not None and pm_sig is not None and pm_sig >= PM_SNR_STAR and pm > pm_max:
            where = (f" at the distance of {associated['name']}" if distance_kpc and associated
                     else f" even at the LMC's distance ({LMC_DISTANCE_KPC:g} kpc; host distance unknown)")
            text = (f"Gaia DR3 {sid}: proper motion {pm:.2f} mas/yr ({pm_sig:.0f} sigma) > {pm_max:.3g} mas/yr, "
                    f"i.e. faster than {MAX_GALAXY_TRANSVERSE_KMS:.0f} km/s{where}")
            if not_point:
                notes.append(f"{text}: not used, not a well-behaved point source ({', '.join(not_point)})")
            else:
                reasons.append(f"{text} -> Galactic star")
        dm = associated.get("distance_modulus") if associated else None
        if dm is not None and g is not None and g - dm < M_G_BRIGHTEST_STAR:
            drift = (pm or 0.0) * (GAIA_DR3_EPOCH_YR - CATALOGUE_EPOCH_YR) / 1000.0
            same = [o for o in counterparts if _is_stellar(o) and not _is_extragalactic_star(o)
                    and o.get("ra") is not None and o.get("dec") is not None and entry.get("ra") is not None
                    and haversine_arcsec(entry["ra"], entry["dec"], o["ra"], o["dec"]) <= SAME_SOURCE_ARCSEC + drift]
            where = f" at the distance modulus {dm:.2f}{in_galaxy or ' of ' + str(associated.get('name'))}"
            if not same:
                if any(_is_stellar(o) for o in counterparts):
                    notes.append(f"Gaia DR3 {sid}: G = {g:.2f} would be M_G = {g - dm:.1f}{where}, but no stellar-type "
                                 "entry is this Gaia source: luminosity not used")
            elif not_point:
                notes.append(f"Gaia DR3 {sid} (= {same[0]['catalog'].upper()} {same[0]['source_id']}): M_G = "
                             f"{g - dm:.1f}{where} not used: not a well-behaved point source ({', '.join(not_point)})")
            else:
                star = same[0]
                reasons.append(f"Gaia DR3 {sid} = {star['catalog'].upper()} {star['source_id']} (type "
                               f"{star['object_type']}), a catalogued star with G = {g:.2f}, i.e. M_G = {g - dm:.1f} < "
                               f"{M_G_BRIGHTEST_STAR:g}{where}: brighter than any star -> Galactic foreground star")
        return reasons, notes

    def _flags(self, alert: Alert, result: AlertEnrichment, answered: set[str], *, d25_answered: bool) -> None:
        """Galactic-star, stellar, variable, AGN and 'new' flags (see the module docstring for the rules)."""
        host = result.host
        enclosing = host if host is not None and host.get("method") == "d25_ellipse" else None
        associated = host if host is not None and host.get("method") in {"d25_ellipse", "dlr_outside_d25"} else None
        z = enclosing.get("redshift") if enclosing else None
        in_galaxy = f" in {enclosing['name']} (d_DLR {enclosing['d_dlr']:.2f}, z={z})" if enclosing else ""

        galactic: list[str] = []
        stellar: list[str] = []
        extragalactic_star: list[str] = []
        variable: list[str] = []
        agn: list[str] = []
        service_gaia_ids: set[str] = set()
        for entry in result.counterparts:
            catalog, otype = entry["catalog"], entry["object_type"]
            sep = entry["separation_arcsec"] or 0.0
            if catalog == "gaia_dr3":
                service_gaia_ids.add(str(entry["source_id"]))
                reasons, notes = self._gaia_verdict(entry, enclosing, associated, in_galaxy, result.counterparts)
                galactic.extend(reasons)
                result.evidence.extend(notes)
            if _is_stellar(entry):
                stellar.append(f"{catalog.upper()} {entry['source_id']} (type {otype}) at {sep:.2f}\"")
            if _is_extragalactic_star(entry):
                extragalactic_star.append(f"NED {entry['source_id']} (type {otype}: extragalactic star)")
            if _is_ned_galactic_star(entry):
                galactic.append(f"NED {entry['source_id']} (type {otype}): NED marks it as a Milky Way object -> Galactic star")
            if _is_variable(entry):
                variable.append(f"{catalog.upper()} {entry['source_id']} (type {otype})")
            if _is_agn(entry):
                agn.append(f"{catalog.upper()} {entry['source_id']} (type {otype}) at {sep:.2f}\"")
                if _is_blazar(entry):
                    variable.append(f"{catalog.upper()} {entry['source_id']} (type {otype}: a blazar, variable by definition)")

        # The broker's own cross-matches (Fink: Gaia DR3 and SIMBAD within 1").
        extra = alert.extra or {}
        broker_otype = _text(extra.get("simbad_otype"))
        if broker_otype is not None:
            broker_otype = broker_otype.removeprefix(FINK_SIMBAD_PREFIX)
        broker_galaxy = broker_otype in SIMBAD_GALAXY_TYPES or broker_otype in FINK_LEGACY_GALAXY_LABELS
        plx, plx_err = _float(extra.get("gaia_parallax_mas")), _float(extra.get("gaia_parallax_error_mas"))
        dr3_name = _text(extra.get("gaia_dr3_name"))
        dr3_id = dr3_name.split()[-1] if dr3_name else None
        if plx is not None and plx_err and plx / plx_err >= self.parallax_snr:
            poe = plx / plx_err
            label = f" {dr3_name}" if dr3_name else ""
            text = f"{alert.broker} Gaia DR3 xmatch{label}: parallax {plx:.3f} +/- {plx_err:.3f} mas ({poe:.1f} sigma)"
            if dr3_id is not None and dr3_id in service_gaia_ids:
                result.evidence.append(f"{text}: judged from the Gaia DR3 row of the counterpart search (RUWE, G, pm)")
            elif broker_galaxy:
                result.evidence.append(f"{text} not used: the broker's SIMBAD cross-match is a galaxy/AGN "
                                       f"({broker_otype}), whose Gaia astrometry is spurious")
            elif enclosing is None or poe >= PARALLAX_SNR_SECURE:
                galactic.append(f"{text} -> Galactic star")
            else:
                result.evidence.append(f"{text} not used: projected{in_galaxy}, and the broker gives no RUWE/G to "
                                       f"vet a < {PARALLAX_SNR_SECURE:g} sigma parallax")
        if extra.get("gaia_var_flag") in (1, "1", "VARIABLE"):
            variable.append(f"{alert.broker} Gaia DR3 xmatch (photometric variability flag)")
        if broker_otype in SIMBAD_VARIABLE_TYPES or broker_otype in FINK_LEGACY_VARIABLE_LABELS:
            variable.append(f"{alert.broker} SIMBAD xmatch (type {broker_otype})")
        if broker_otype in SIMBAD_STAR_TYPES or broker_otype in FINK_LEGACY_STAR_LABELS:
            stellar.append(f"{alert.broker} SIMBAD xmatch (type {broker_otype})")
        if broker_otype in SIMBAD_AGN_TYPES or broker_otype in FINK_LEGACY_AGN_LABELS:
            agn.append(f"{alert.broker} SIMBAD xmatch (type {broker_otype})")
            if broker_otype in SIMBAD_BLAZAR_TYPES:
                variable.append(f"{alert.broker} SIMBAD xmatch (type {broker_otype}: a blazar, variable by definition)")
        if broker_galaxy:
            result.evidence.append(
                f"{alert.broker} SIMBAD xmatch type {broker_otype}: coincident with a catalogued galaxy/AGN "
                "(nuclear transient or AGN variability)"
            )

        star: bool | None
        result.evidence.extend(galactic)
        if stellar:
            result.stellar_counterpart = True
            result.evidence.append("stellar-type counterpart: " + "; ".join(stellar))
        elif answered & {"simbad", "ned"}:
            result.stellar_counterpart = False
        # Galaxies near enough (|z| < 0.01, D < ~43 Mpc) for their individual stars to be catalogued.
        local_cone = [g for g in result.host_candidates if g.get("redshift") is not None
                      and abs(g["redshift"]) < LOCAL_VOLUME_MAX_Z and not g.get("transient_named")]
        if galactic:
            star = True
        elif extragalactic_star:
            star = False
            result.evidence.append("NED classifies the counterpart as an extragalactic star: " + "; ".join(extragalactic_star))
        elif stellar:
            if enclosing is not None and z is not None and abs(z) < LOCAL_VOLUME_MAX_Z:
                star = False
                result.evidence.append(f"the stellar-type counterpart lies{in_galaxy}, with no parallax, proper motion "
                                       f"or luminosity marking it as a foreground star: an extragalactic star, not a "
                                       f"Galactic one (|z| < {LOCAL_VOLUME_MAX_Z})")
            elif enclosing is not None and z is not None:
                star = True
                result.evidence.append(f"the stellar-type counterpart is projected{in_galaxy}, too distant for "
                                       "individually catalogued stars: a Galactic foreground star")
            elif enclosing is not None:
                star = None
                result.evidence.append(f"the stellar-type counterpart is projected{in_galaxy} of unknown redshift: "
                                       "Galactic nature not established")
            elif associated is not None and _is_local_host(associated):
                star = None
                result.evidence.append(
                    f"the stellar-type counterpart lies outside the D25 ellipse of {associated['name']} (d_DLR "
                    f"{associated['d_dlr']:.2f}, z={associated.get('redshift')}), a galaxy near enough for its stars to "
                    "be catalogued (e.g. an LBV or SN impostor in its outskirts), with no parallax or proper motion "
                    "marking it as a foreground star: Galactic nature not established")
            elif local_cone:
                near_gal = local_cone[0]
                star = None
                result.evidence.append(
                    f"a galaxy with |z| < {LOCAL_VOLUME_MAX_Z} lies within the host radius ({near_gal['source_id']}, "
                    f"z={near_gal['redshift']}, {near_gal['separation_arcsec']:.1f}\"): without parallax or proper-motion "
                    "evidence the stellar-type counterpart may be one of its stars; Galactic nature not established")
            elif associated is not None and associated.get("redshift") is not None:
                star = True
                result.evidence.append(
                    f"the stellar-type counterpart lies outside the D25 ellipse of {associated['name']} (z="
                    f"{associated['redshift']}), too distant for individually catalogued stars: a Galactic star")
            elif associated is not None:
                star = None
                result.evidence.append(f"the stellar-type counterpart lies near {associated['name']} (d_DLR "
                                       f"{associated['d_dlr']:.2f}) of unknown redshift: Galactic nature not established")
            elif d25_answered:
                star = True
                result.evidence.append("the stellar-type counterpart is not projected on any HyperLEDA galaxy nor near "
                                       "a nearby (|z| < 0.01) galaxy: a Galactic star")
            else:
                star = None
                result.evidence.append("the D25 galaxy search failed: Galactic nature of the stellar counterpart unknown")
        else:
            needed = {"gaia_dr3", "simbad"} & set(self.catalogs)
            star = False if needed and needed <= answered else None
        result.known_star = star

        if variable:
            result.known_variable = True
            where = in_galaxy if (enclosing is not None and star is False) else ""
            result.evidence.append(f"known variable{where}: " + "; ".join(variable))
        else:
            result.known_variable = False if "simbad" in answered else None
        if agn:
            result.known_agn = True
            result.evidence.append("coincident with a catalogued AGN/QSO (AGN variability is a common contaminant of "
                                   "extragalactic alert streams): " + "; ".join(agn))
        else:
            result.known_agn = False if answered & {"simbad", "ned"} else None

        if result.counterparts:
            result.is_new = False
        elif self.catalogs and set(self.catalogs) <= answered:
            result.is_new = True
            result.evidence.append(
                f"no catalogued source within {self.match_radius_arcsec:g}\" in {sorted(answered)}: new source"
            )
        else:
            result.is_new = None


def _is_local_host(host: Mapping[str, Any]) -> bool:
    """A host near enough (|z| < 0.01 or a Cosmicflows-4 distance) for its individual stars to be catalogued."""
    z = host.get("redshift")
    if z is not None and abs(z) < LOCAL_VOLUME_MAX_Z:
        return True
    return host.get("distance_method") in {"cosmicflows4", "cosmicflows4_group"}


def _inside_d25(group: Mapping[str, Any], gal: Mapping[str, Any]) -> bool:
    """True when a catalogued galaxy's position lies inside a HyperLEDA galaxy's D25 ellipse (a part of it)."""
    a, b = gal.get("semi_major_arcsec"), gal.get("semi_minor_arcsec")
    if a is None or b is None:
        return False
    sep = haversine_arcsec(gal["ra"], gal["dec"], group["ra"], group["dec"])
    radius = directional_light_radius(a, b, gal.get("pa_deg"),
                                      position_angle_deg(gal["ra"], gal["dec"], group["ra"], group["dec"]))
    return radius > 0 and sep / radius <= DLR_INSIDE


# ---------------------------------------------------------------------------
# Persistence: the 'alerts' and 'alert_cursors' tables
# ---------------------------------------------------------------------------

_ALERT_COLUMNS = (
    "id", "broker", "object_id", "survey", "ra", "dec", "mjd", "first_mjd", "magpsf", "magpsf_err", "band",
    "classification", "probability", "url", "extra_json", "crossmatch_status", "enrichment_json", "is_new",
    "known_star", "known_variable", "host_name", "host_separation_arcsec", "host_redshift", "first_seen_at",
    "updated_at", "n_updates", "is_negative", "crossmatch_attempts", "last_crossmatch_error", "last_crossmatch_at",
    "known_agn",
)
# Columns added after the first release, created on existing databases by AlertStore.
_ALERT_MIGRATIONS: dict[str, str] = {
    "is_negative": "INTEGER",
    "crossmatch_attempts": "INTEGER NOT NULL DEFAULT 0",
    "last_crossmatch_error": "TEXT",
    "last_crossmatch_at": "TEXT",
    "known_agn": "INTEGER",
}
_BOOL_COLUMNS = ("is_new", "known_star", "known_variable", "is_negative", "known_agn")
# Alert.extra keys that come from the photometry request of one detection (ALeRCE /detections).
_PHOTOMETRY_EXTRA_KEYS = frozenset({"candid", "isdiffpos", "detection_mjd", "n_detections", "n_negative_detections",
                                    "bands", "malformed_fid"})
_STATUS_RANK = {"failed": 0, "partial": 1, "done": 2}


def _bool_or_none(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _int_or_none(value: bool | None) -> int | None:
    return None if value is None else int(value)


@dataclass(slots=True)
class PollWindow:
    """The MJD window of one poll and how it was chosen."""

    since: float
    until: float
    kind: str  # explicit | new | backlog
    key: str
    cursor: dict[str, Any] | None = None


class AlertStore:
    """Alerts table in the metadata database (SQLite by default, PostgreSQL via DATABASE_URL).

    Uses :class:`datasets.MetadataStore`'s connection handling; the ``alerts`` and
    ``alert_cursors`` tables are created (and migrated) here. Rows are keyed by
    ``broker:object_id``. Methods are synchronous (one connection per call); async callers
    run them via ``asyncio.to_thread``.
    """

    def __init__(self, metadata: MetadataStore | None = None, *, database_url: str | None = None,
                 local_dir: str | None = None) -> None:
        if metadata is None:
            url = database_url or os.getenv("ALERTS_DATABASE_URL") or None
            metadata = MetadataStore(url, local_dir=local_dir or os.getenv("DATASET_STORAGE_PATH") or "datasets")
        self.metadata = metadata
        self._initialize()

    @contextlib.contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = self.metadata._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _sql(self, statement: str) -> str:
        return self.metadata._sql(statement)

    def _initialize(self) -> None:
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS alerts ("
                "id TEXT PRIMARY KEY, broker TEXT NOT NULL, object_id TEXT NOT NULL, survey TEXT, "
                "ra DOUBLE PRECISION NOT NULL, dec DOUBLE PRECISION NOT NULL, mjd DOUBLE PRECISION NOT NULL, "
                "first_mjd DOUBLE PRECISION, magpsf DOUBLE PRECISION, magpsf_err DOUBLE PRECISION, band TEXT, "
                "classification TEXT, probability DOUBLE PRECISION, url TEXT, extra_json TEXT, "
                "crossmatch_status TEXT NOT NULL, enrichment_json TEXT, is_new INTEGER, known_star INTEGER, "
                "known_variable INTEGER, host_name TEXT, host_separation_arcsec DOUBLE PRECISION, "
                "host_redshift DOUBLE PRECISION, first_seen_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
                "n_updates INTEGER NOT NULL DEFAULT 0, is_negative INTEGER, "
                "crossmatch_attempts INTEGER NOT NULL DEFAULT 0, last_crossmatch_error TEXT, last_crossmatch_at TEXT, "
                "known_agn INTEGER, UNIQUE (broker, object_id))"
            )
            cur = conn.execute("SELECT * FROM alerts WHERE 1 = 0")
            existing = {d[0].lower() for d in cur.description}
            for column, ddl in _ALERT_MIGRATIONS.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE alerts ADD COLUMN {column} {ddl}")
            conn.execute("CREATE INDEX IF NOT EXISTS alerts_mjd_idx ON alerts (mjd)")
            conn.execute("CREATE INDEX IF NOT EXISTS alerts_broker_mjd_idx ON alerts (broker, mjd)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS alert_cursors ("
                "key TEXT PRIMARY KEY, broker TEXT NOT NULL, options_json TEXT NOT NULL, "
                "resume_mjd DOUBLE PRECISION, backlog_since DOUBLE PRECISION, backlog_until DOUBLE PRECISION, "
                "updated_at TEXT NOT NULL)"
            )

    def _row_dict(self, row: Sequence[Any]) -> dict[str, Any]:
        item = dict(zip(_ALERT_COLUMNS, row, strict=True))
        item["extra"] = json.loads(item.pop("extra_json") or "{}")
        enrichment = item.pop("enrichment_json")
        item["enrichment"] = json.loads(enrichment) if enrichment else None
        for key in _BOOL_COLUMNS:
            item[key] = _bool_or_none(item[key])
        item["crossmatch_attempts"] = int(item["crossmatch_attempts"] or 0)
        return item

    # -- writes ---------------------------------------------------------------

    def _upsert_one(self, conn: Any, alert: Alert, match_radius_arcsec: float) -> tuple[str, str, int]:
        now = _utcnow()
        select = self._sql("SELECT mjd, classification, probability, ra, dec, first_mjd, crossmatch_status, "
                           "crossmatch_attempts, magpsf, magpsf_err, band, is_negative, extra_json FROM alerts WHERE id = ?")
        row = conn.execute(select, (alert.alert_id,)).fetchone()
        if row is None:
            values = (
                alert.alert_id, alert.broker, alert.object_id, alert.survey, alert.ra, alert.dec, alert.mjd,
                alert.first_mjd, alert.magpsf, alert.magpsf_err, alert.band, alert.classification, alert.probability,
                alert.url, json.dumps(alert.extra, default=str), "pending", None, None, None, None, None, None, None,
                now, now, 0, _int_or_none(alert.is_negative), 0, None, None, None,
            )
            cur = conn.execute(
                self._sql(f"INSERT INTO alerts ({', '.join(_ALERT_COLUMNS)}) VALUES ({', '.join('?' * len(_ALERT_COLUMNS))}) "
                          "ON CONFLICT (id) DO NOTHING"),
                values,
            )
            if cur.rowcount == 1:
                return "inserted", "pending", 0
            row = conn.execute(select, (alert.alert_id,)).fetchone()
        (old_mjd, old_class, old_prob, old_ra, old_dec, old_first, status, attempts, old_mag, old_mag_err, old_band,
         old_neg, old_extra) = row
        attempts = int(attempts or 0)
        old_mjd = float(old_mjd)
        first = old_first
        if alert.first_mjd is not None and (old_first is None or alert.first_mjd < float(old_first) - MJD_EPS):
            first = alert.first_mjd
        if alert.mjd < old_mjd - MJD_EPS:
            # An older detection (e.g. a backfill poll) never replaces the stored newer one:
            # its photometry, position, score and candid belong to another epoch.
            if first != old_first:
                conn.execute(self._sql("UPDATE alerts SET first_mjd = ?, updated_at = ?, n_updates = n_updates + 1 "
                                       "WHERE id = ?"), (first, now, alert.alert_id))
                return "updated", str(status), attempts
            return "unchanged", str(status), attempts
        newer = alert.mjd > old_mjd + MJD_EPS
        reclassified = alert.classification != old_class or (
            alert.probability is not None and (old_prob is None or abs(alert.probability - float(old_prob)) > 1e-6)
        )
        # The same detection with photometry the stored row lacks (an earlier /detections request
        # failed) or of another candid (a different packet of the same instant) refreshes the row.
        old_candid = (json.loads(old_extra or "{}") or {}).get("candid")
        new_candid = (alert.extra or {}).get("candid")
        rephotometered = alert.magpsf is not None and (
            old_mag is None or (new_candid is not None and old_candid is not None and str(new_candid) != str(old_candid)))
        if not newer and not reclassified and not rephotometered and first == old_first:
            return "unchanged", str(status), attempts
        if not newer and alert.magpsf is None and old_mag is not None:
            # A re-poll of the same detection whose photometry request failed: keep the stored photometry.
            kept = {k: v for k, v in json.loads(old_extra or "{}").items() if k in _PHOTOMETRY_EXTRA_KEYS}
            alert = replace(alert, magpsf=float(old_mag), magpsf_err=_float(old_mag_err), band=old_band,
                            is_negative=_bool_or_none(old_neg), extra={**(alert.extra or {}), **kept})
        moved = haversine_arcsec(float(old_ra), float(old_dec), alert.ra, alert.dec) > REMATCH_FRACTION * match_radius_arcsec
        conn.execute(
            self._sql(
                "UPDATE alerts SET ra = ?, dec = ?, mjd = ?, first_mjd = ?, magpsf = ?, magpsf_err = ?, band = ?, "
                "classification = ?, probability = ?, url = ?, extra_json = ?, is_negative = ?, updated_at = ?, "
                "n_updates = n_updates + 1"
                + (", crossmatch_status = 'pending', crossmatch_attempts = 0" if moved else "")
                + " WHERE id = ?"
            ),
            (alert.ra, alert.dec, alert.mjd, first, alert.magpsf, alert.magpsf_err, alert.band, alert.classification,
             alert.probability, alert.url, json.dumps(alert.extra, default=str), _int_or_none(alert.is_negative), now,
             alert.alert_id),
        )
        return ("updated", "pending", 0) if moved else ("updated", str(status), attempts)

    def upsert_many(self, alerts: Sequence[Alert], *, match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC
                    ) -> list[tuple[str, str, int]]:
        """Insert or refresh alerts in one transaction; returns (outcome, crossmatch_status,
        crossmatch_attempts) per alert.

        outcome is 'inserted', 'updated' or 'unchanged'. A row is refreshed by a newer
        detection (all per-detection fields) or a changed classification of the same
        detection; an older detection only extends ``first_mjd``. The same detection also
        refreshes a row stored without photometry (its /detections request had failed) or
        with another ``candid``; a re-poll whose photometry request failed keeps the stored
        photometry. A position shift beyond half the match radius re-queues the crossmatch
        (and resets its attempt count). Re-polling a window changes nothing.
        """
        if not alerts:
            return []
        with self._conn() as conn:
            return [self._upsert_one(conn, a, match_radius_arcsec) for a in alerts]

    def upsert(self, alert: Alert, *, match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC) -> str:
        """Insert or refresh one alert; returns 'inserted', 'updated' or 'unchanged' (see :meth:`upsert_many`)."""
        return self.upsert_many([alert], match_radius_arcsec=match_radius_arcsec)[0][0]

    def set_enrichment(self, alert_id: str, enrichment: AlertEnrichment) -> str:
        """Store an enrichment; returns 'stored', 'kept_previous' or 'missing'.

        A failed attempt never replaces an earlier done/partial result, and a partial one
        never replaces a done result of the same position: the earlier result is kept and
        the attempt is recorded in ``last_crossmatch_error``. A row whose position moved
        (status 'pending') takes the new status so that it is retried.
        """
        now = _utcnow()
        with self._conn() as conn:
            row = conn.execute(self._sql("SELECT crossmatch_status, enrichment_json FROM alerts WHERE id = ?"),
                               (alert_id,)).fetchone()
            if row is None:
                return "missing"
            current, previous_json = row
            previous = json.loads(previous_json) if previous_json else None
            prev_status = previous.get("status") if previous else None
            new_rank = _STATUS_RANK.get(enrichment.status, 0)
            prev_rank = _STATUS_RANK.get(prev_status, -1)
            keep = prev_rank > new_rank and (enrichment.status == "failed" or current == "done")
            error = enrichment.exception or enrichment.error
            if keep:
                status = current if current == "done" else enrichment.status
                conn.execute(
                    self._sql("UPDATE alerts SET crossmatch_status = ?, crossmatch_attempts = crossmatch_attempts + 1, "
                              "last_crossmatch_error = ?, last_crossmatch_at = ? WHERE id = ?"),
                    (status, error, now, alert_id),
                )
                return "kept_previous"
            host = enrichment.host or {}
            conn.execute(
                self._sql(
                    "UPDATE alerts SET crossmatch_status = ?, enrichment_json = ?, is_new = ?, known_star = ?, "
                    "known_variable = ?, known_agn = ?, host_name = ?, host_separation_arcsec = ?, host_redshift = ?, "
                    "crossmatch_attempts = crossmatch_attempts + 1, last_crossmatch_error = ?, last_crossmatch_at = ? "
                    "WHERE id = ?"
                ),
                (
                    enrichment.status, json.dumps(enrichment.as_dict(), default=str),
                    _int_or_none(enrichment.is_new), _int_or_none(enrichment.known_star),
                    _int_or_none(enrichment.known_variable), _int_or_none(enrichment.known_agn), host.get("name"),
                    host.get("separation_arcsec"),
                    host.get("redshift"), None if enrichment.status == "done" else error, now, alert_id,
                ),
            )
            return "stored"

    # -- reads ----------------------------------------------------------------

    def crossmatch_status(self, alert_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(self._sql("SELECT crossmatch_status FROM alerts WHERE id = ?"), (alert_id,)).fetchone()
        return row[0] if row else None

    def get(self, alert_id: str) -> dict[str, Any] | None:
        """Row by ``broker:object_id``, or by bare object id (most recent broker entry)."""
        with self._conn() as conn:
            row = conn.execute(self._sql(f"SELECT {', '.join(_ALERT_COLUMNS)} FROM alerts WHERE id = ?"), (alert_id,)).fetchone()
            if row is None:
                row = conn.execute(
                    self._sql(f"SELECT {', '.join(_ALERT_COLUMNS)} FROM alerts WHERE object_id = ? "
                              "ORDER BY mjd DESC, id LIMIT 1"),
                    (alert_id,),
                ).fetchone()
        return self._row_dict(row) if row else None

    def get_many(self, alert_ids: Sequence[str]) -> list[dict[str, Any]]:
        """Rows for ``broker:object_id`` ids (one query per 500 ids), in the given order; missing ids are skipped."""
        ids = list(dict.fromkeys(alert_ids))
        if not ids:
            return []
        rows: dict[str, dict[str, Any]] = {}
        with self._conn() as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start:start + 500]
                found = conn.execute(
                    self._sql(f"SELECT {', '.join(_ALERT_COLUMNS)} FROM alerts WHERE id IN ({', '.join('?' * len(chunk))})"),
                    chunk,
                ).fetchall()
                rows.update({r[0]: self._row_dict(r) for r in found})
        return [rows[i] for i in ids if i in rows]

    def get_alert(self, alert_id: str) -> Alert | None:
        row = self.get(alert_id)
        return Alert.from_dict(row) if row else None

    def incomplete(self, *, broker: str | None = None, limit: int = 10, exclude: Sequence[str] = (),
                   max_attempts: int = MAX_CROSSMATCH_ATTEMPTS) -> list[Alert]:
        """Alerts whose crossmatch is pending/partial/failed and was tried fewer than ``max_attempts`` times."""
        if limit <= 0:
            return []
        clauses = ["crossmatch_status IN ('pending', 'partial', 'failed')", "crossmatch_attempts < ?"]
        params: list[Any] = [int(max_attempts)]
        if broker:
            clauses.append("broker = ?")
            params.append(broker)
        skip = list(dict.fromkeys(exclude))
        if skip:
            clauses.append(f"id NOT IN ({', '.join('?' * len(skip))})")
            params.extend(skip)
        params.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(
                self._sql(f"SELECT {', '.join(_ALERT_COLUMNS)} FROM alerts WHERE {' AND '.join(clauses)} "
                          "ORDER BY mjd DESC, id LIMIT ?"), params,
            ).fetchall()
        return [Alert.from_dict(self._row_dict(r)) for r in rows]

    def list(
        self,
        *,
        since_mjd: float | None = None,
        until_mjd: float | None = None,
        limit: int = 100,
        broker: str | None = None,
        classification: str | None = None,
        only_new: bool | None = None,
        crossmatch_status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Alerts newest first, filtered."""
        clauses: list[str] = []
        params: list[Any] = []
        for column, op, value in (
            ("mjd", ">=", since_mjd), ("mjd", "<=", until_mjd), ("broker", "=", broker),
            ("classification", "=", classification), ("crossmatch_status", "=", crossmatch_status),
        ):
            if value is not None:
                clauses.append(f"{column} {op} ?")
                params.append(value)
        if only_new is not None:
            clauses.append("is_new = ?")
            params.append(int(only_new))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        with self._conn() as conn:
            rows = conn.execute(
                self._sql(f"SELECT {', '.join(_ALERT_COLUMNS)} FROM alerts{where} ORDER BY mjd DESC, id LIMIT ?"), params
            ).fetchall()
        return [self._row_dict(r) for r in rows]

    def latest_mjd(self, broker: str) -> float | None:
        with self._conn() as conn:
            row = conn.execute(self._sql("SELECT MAX(mjd) FROM alerts WHERE broker = ?"), (broker,)).fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def count(self, broker: str | None = None) -> int:
        with self._conn() as conn:
            if broker:
                row = conn.execute(self._sql("SELECT COUNT(*) FROM alerts WHERE broker = ?"), (broker,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()
        return int(row[0])

    # -- polling cursors ------------------------------------------------------

    def get_cursor(self, key: str) -> dict[str, Any] | None:
        """Cursor of one polling stream: resume_mjd and a pending backlog window (or None)."""
        with self._conn() as conn:
            row = conn.execute(
                self._sql("SELECT key, broker, options_json, resume_mjd, backlog_since, backlog_until, updated_at "
                          "FROM alert_cursors WHERE key = ?"), (key,),
            ).fetchone()
        if row is None:
            return None
        keys = ("key", "broker", "options_json", "resume_mjd", "backlog_since", "backlog_until", "updated_at")
        cursor = dict(zip(keys, row, strict=True))
        cursor["options"] = json.loads(cursor.pop("options_json") or "{}")
        return cursor

    def set_cursor(self, key: str, broker: str, options: Mapping[str, Any], *, resume_mjd: float | None,
                   backlog: tuple[float, float] | None) -> None:
        since, until = backlog if backlog is not None else (None, None)
        with self._conn() as conn:
            conn.execute(
                self._sql(
                    "INSERT INTO alert_cursors (key, broker, options_json, resume_mjd, backlog_since, backlog_until, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (key) DO UPDATE SET resume_mjd = excluded.resume_mjd, "
                    "backlog_since = excluded.backlog_since, backlog_until = excluded.backlog_until, "
                    "updated_at = excluded.updated_at"
                ),
                (key, broker, json.dumps(dict(options), sort_keys=True), resume_mjd, since, until, _utcnow()),
            )


# ---------------------------------------------------------------------------
# Alert Service: poll -> dedupe/persist -> auto-crossmatch; watch loop
# ---------------------------------------------------------------------------


def cursor_key(broker: str, options: Mapping[str, Any]) -> str:
    """Identity of a polling stream: the broker and its (normalized) filter options."""
    return f"{broker}|{json.dumps(dict(options), sort_keys=True)}"


class AlertService:
    """Poll brokers, persist alerts idempotently, and auto-crossmatch new ones."""

    def __init__(
        self,
        store: AlertStore,
        client: httpx.AsyncClient,
        enricher: AlertEnricher | None = None,
        *,
        concurrency: int = 3,
        overlap_days: float = 1.0,
        lookback_days: float = 1.0,
        clock: Callable[[], float] = now_mjd,
    ) -> None:
        self.store = store
        self.client = client
        self.enricher = enricher
        self.concurrency = max(1, int(concurrency))
        self.overlap_days = overlap_days
        self.lookback_days = lookback_days
        self.clock = clock
        # Alert id -> the task enriching it now (a poll, a background task or a re-crossmatch
        # request): an alert is never enriched twice at once.
        self._running: dict[str, asyncio.Future[tuple[AlertEnrichment, str]]] = {}

    def plan_window(self, broker: str, options: Mapping[str, Any], since_mjd: float | None,
                    until_mjd: float | None) -> PollWindow:
        """The poll window: explicit ``since``; else a pending backlog; else the stream cursor - overlap.

        Without a cursor (first poll) the window starts at the latest stored alert of the
        broker minus the overlap, or ``lookback_days`` before now. The overlap re-covers
        alerts that reach the broker late. Only fully default windows (no since/until)
        read the backlog and advance the cursor.
        """
        key = cursor_key(broker, options)
        until = float(until_mjd) if until_mjd is not None else self.clock()
        if since_mjd is not None:
            window = PollWindow(float(since_mjd), until, "explicit", key)
        else:
            cursor = self.store.get_cursor(key)
            if until_mjd is None and cursor and cursor.get("backlog_since") is not None:
                window = PollWindow(float(cursor["backlog_since"]), float(cursor["backlog_until"]), "backlog", key, cursor)
            else:
                if cursor and cursor.get("resume_mjd") is not None:
                    since = float(cursor["resume_mjd"]) - self.overlap_days
                else:
                    latest = self.store.latest_mjd(broker)
                    since = latest - self.overlap_days if latest is not None else until - self.lookback_days
                since = min(since, until - 1e-3)
                window = PollWindow(since, until, "new" if until_mjd is None else "explicit", key, cursor)
        if not (math.isfinite(window.since) and math.isfinite(window.until)) or window.since >= window.until:
            raise ValueError(f"since_mjd ({window.since}) must be earlier than until_mjd ({window.until})")
        return window

    def default_window(self, broker: str, since_mjd: float | None, until_mjd: float | None,
                       options: Mapping[str, Any] | None = None) -> tuple[float, float]:
        """[since, until] of the next poll (see :meth:`plan_window`)."""
        window = self.plan_window(broker, normalize_options(broker, options), since_mjd, until_mjd)
        return window.since, window.until

    def _advance_cursor(self, window: PollWindow, broker: str, options: Mapping[str, Any],
                        fetched: FetchResult, result: PollResult) -> None:
        """Record what a default-window poll covered: the resume point and any unfetched backlog.

        A truncated 'new' window leaves a backlog only for the part not covered before
        (from the previous resume point, or the whole window on a first poll): the overlap
        re-covered for late alerts is best-effort, otherwise a busy stream would re-walk the
        same overlap after every poll.
        """
        cursor = window.cursor or {}
        previous = cursor.get("resume_mjd")
        if window.kind == "new":
            resume = window.until
            floor = window.since if previous is None else max(window.since, float(previous))
        else:
            resume = previous
            floor = window.since
        backlog: tuple[float, float] | None = None
        if fetched.truncated:
            boundary = fetched.boundary_mjd if fetched.boundary_mjd is not None else window.until
            if window.kind == "backlog" and boundary >= window.until - MJD_EPS:
                # More than `limit` objects at the backlog's newest instant: step past that second.
                boundary = window.until - 1.0 / 86400.0
                result.warnings.append(f"more than {result.fetched} alerts at MJD {window.until:.6f}: the rest of that "
                                       "second is skipped; raise the limit to ingest them")
            if boundary > floor:
                backlog = (floor, boundary)
            else:
                result.warnings.append(f"the re-covered overlap before MJD {floor:.6f} was truncated: late alerts older "
                                       f"than MJD {boundary:.6f} may be missed; raise the limit")
        self.store.set_cursor(window.key, broker, options, resume_mjd=resume, backlog=backlog)
        result.backlog = {"since_mjd": backlog[0], "until_mjd": backlog[1]} if backlog else None

    @property
    def in_flight(self) -> frozenset[str]:
        """Ids of the alerts being enriched right now."""
        return frozenset(self._running)

    def _claim(self, alert: Alert, semaphore: asyncio.Semaphore | None = None
               ) -> asyncio.Future[tuple[AlertEnrichment, str]]:
        """Start enriching ``alert`` (under ``semaphore``) and register the task until it ends."""
        async def run() -> tuple[AlertEnrichment, str]:
            if semaphore is None:
                return await self._enrich_and_store(alert)
            async with semaphore:
                return await self._enrich_and_store(alert)

        task = asyncio.ensure_future(run())
        alert_id = alert.alert_id
        self._running[alert_id] = task

        def release(done: asyncio.Future[Any]) -> None:
            if self._running.get(alert_id) is done:
                del self._running[alert_id]
            if not done.cancelled() and done.exception() is not None:  # retrieved: no "never retrieved" warning
                logger.error("alert crossmatch task of %s failed: %r", alert_id, done.exception())

        task.add_done_callback(release)
        return task

    async def enrich_alert(self, alert: Alert) -> tuple[AlertEnrichment, str]:
        """Crossmatch one alert and store the result; returns (enrichment, 'stored' | 'kept_previous' | 'missing').

        When the alert is already being enriched (by a poll, a background task or another
        request) the running enrichment is awaited and its result returned: the archives are
        queried once and the attempt counter moves once. Cancelling the caller does not cancel
        the shared enrichment. An exception inside the enricher is stored as status 'failed'
        with ``exception`` set, so the alert stays queryable (and never erases an earlier
        complete result).
        """
        if self.enricher is None:
            raise RuntimeError("AlertService has no enricher (crossmatch disabled)")
        running = self._running.get(alert.alert_id)
        task = running if running is not None else self._claim(alert)
        return await asyncio.shield(task)

    async def _enrich_and_store(self, alert: Alert) -> tuple[AlertEnrichment, str]:
        enricher = self.enricher
        if enricher is None:
            raise RuntimeError("AlertService has no enricher (crossmatch disabled)")
        try:
            enrichment = await enricher.enrich(alert)
        except Exception as exc:  # stored as status 'failed'; the alert stays queryable
            logger.exception("alert crossmatch raised for %s", alert.alert_id)
            enrichment = AlertEnrichment(
                status="failed", match_radius_arcsec=enricher.match_radius_arcsec,
                host_radius_arcsec=enricher.host_radius_arcsec, catalogs=list(enricher.catalogs),
                error=f"{exc.__class__.__name__}: {exc}", exception=f"{exc.__class__.__name__}: {exc}",
                crossmatched_at=_utcnow(),
            )
        stored = await asyncio.to_thread(self.store.set_enrichment, alert.alert_id, enrichment)
        return enrichment, stored

    async def crossmatch_alerts(self, alerts: Sequence[Alert], result: PollResult | None = None) -> dict[str, int]:
        """Enrich ``alerts`` (bounded concurrency), skipping any already being enriched by this service.

        Returns counts {'done', 'partial', 'failed', 'skipped_in_flight'} (also added to ``result``).
        """
        if self.enricher is None:
            raise RuntimeError("AlertService has no enricher (crossmatch disabled)")
        semaphore = asyncio.Semaphore(self.concurrency)
        counts = {"done": 0, "partial": 0, "failed": 0, "skipped_in_flight": 0}
        tasks: list[asyncio.Future[tuple[AlertEnrichment, str]]] = []
        for alert in alerts:
            if alert.alert_id in self._running:  # claimed now, so a duplicate in ``alerts`` is skipped too
                counts["skipped_in_flight"] += 1
                continue
            tasks.append(self._claim(alert, semaphore))
        for enrichment, _stored in await asyncio.gather(*(asyncio.shield(t) for t in tasks)):
            counts[enrichment.status if enrichment.status in counts else "failed"] += 1
        if result is not None:
            result.crossmatched += counts["done"]
            result.crossmatch_partial += counts["partial"]
            result.crossmatch_failed += counts["failed"]
        return counts

    async def crossmatch_in_background(self, alerts: Sequence[Alert]) -> None:
        """:meth:`crossmatch_alerts` for a background task: errors are logged, never raised."""
        try:
            counts = await self.crossmatch_alerts(alerts)
            logger.info("alerts background crossmatch: %s", counts)
        except Exception:  # nobody awaits a background task: log instead
            logger.exception("alerts background crossmatch failed")

    async def ingest(
        self,
        broker: str = "alerce",
        *,
        since_mjd: float | None = None,
        until_mjd: float | None = None,
        limit: int = 20,
        crossmatch: bool = True,
        options: dict[str, Any] | None = None,
        retry_limit: int = 10,
    ) -> tuple[PollResult, list[Alert]]:
        """Fetch the newest ``limit`` alerts of a window and upsert them, without crossmatching.

        Returns the PollResult and the stored alerts whose crossmatch is due: fetched alerts
        not yet 'done' that were tried fewer than ``MAX_CROSSMATCH_ATTEMPTS`` times (a moved
        alert starts again at 0), plus (``retry_limit``) other incomplete rows of this broker.
        """
        started = time.perf_counter()
        _check_limit(limit)
        opts = normalize_options(broker, options)
        if crossmatch and self.enricher is None:
            raise RuntimeError("crossmatch requested but the AlertService has no enricher")
        window = await asyncio.to_thread(self.plan_window, broker, opts, since_mjd, until_mjd)
        result = PollResult(broker=broker, since_mjd=window.since, until_mjd=window.until, options=opts, window=window.kind)
        fetched = await fetch_alerts(self.client, broker, since_mjd=window.since, until_mjd=window.until, limit=limit,
                                     options=opts)
        result.fetched = len(fetched.alerts)
        result.warnings.extend(fetched.warnings)
        result.truncated = fetched.truncated
        result.boundary_mjd = fetched.boundary_mjd
        radius = self.enricher.match_radius_arcsec if self.enricher else DEFAULT_MATCH_RADIUS_ARCSEC
        outcomes = await asyncio.to_thread(self.store.upsert_many, fetched.alerts, match_radius_arcsec=radius)
        to_match: list[str] = []
        for alert, (outcome, status, attempts) in zip(fetched.alerts, outcomes, strict=True):
            setattr(result, outcome, getattr(result, outcome) + 1)
            result.alert_ids.append(alert.alert_id)
            if outcome == "inserted":
                result.new_alert_ids.append(alert.alert_id)
            if crossmatch and status != "done":
                if attempts < MAX_CROSSMATCH_ATTEMPTS:
                    to_match.append(alert.alert_id)
                else:
                    result.crossmatch_capped += 1
        if result.crossmatch_capped:
            result.warnings.append(f"{result.crossmatch_capped} fetched alert(s) not re-crossmatched: "
                                   f"{MAX_CROSSMATCH_ATTEMPTS} attempts already made (re-run them explicitly)")
        if window.kind != "explicit":
            await asyncio.to_thread(self._advance_cursor, window, broker, opts, fetched, result)
        if fetched.truncated:
            boundary = f"{fetched.boundary_mjd:.6f}" if fetched.boundary_mjd is not None else "?"
            if window.kind == "explicit":
                follow = f"poll again with until_mjd={boundary} to fetch them"
            elif result.backlog:
                follow = (f"backlog [{result.backlog['since_mjd']:.6f}, {result.backlog['until_mjd']:.6f}] queued "
                          "for the next polls")
            else:
                follow = "already covered by earlier polls"
            result.warnings.insert(0, f"window truncated at limit={limit}: older alerts at or before MJD {boundary} "
                                      f"were not fetched ({follow})")
        due: list[Alert] = []
        if crossmatch and to_match:
            # Crossmatch the stored rows: an older detection never replaced the newer one.
            rows = await asyncio.to_thread(self.store.get_many, to_match)
            due.extend(Alert.from_dict(r) for r in rows)
        if crossmatch and retry_limit > 0:
            retry = await asyncio.to_thread(self.store.incomplete, broker=broker, limit=retry_limit,
                                            exclude=result.alert_ids)
            result.retried = len(retry)
            due.extend(retry)
        result.crossmatch_queued = len(due)
        result.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        return result, due

    async def poll(
        self,
        broker: str = "alerce",
        *,
        since_mjd: float | None = None,
        until_mjd: float | None = None,
        limit: int = 20,
        crossmatch: bool = True,
        options: dict[str, Any] | None = None,
        retry_limit: int = 10,
    ) -> PollResult:
        """:meth:`ingest` a window, then crossmatch the due alerts before returning.

        Without ``since_mjd`` the window comes from the stream cursor (a pending backlog
        first). With ``crossmatch``, up to ``retry_limit`` stored alerts of this broker whose
        crossmatch is pending/partial/failed are retried as well (at most
        ``MAX_CROSSMATCH_ATTEMPTS`` attempts per alert, fetched or swept).
        """
        started = time.perf_counter()
        result, due = await self.ingest(broker, since_mjd=since_mjd, until_mjd=until_mjd, limit=limit,
                                        crossmatch=crossmatch, options=options, retry_limit=retry_limit)
        if due:
            await self.crossmatch_alerts(due, result)
        result.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)
        logger.info("alerts polled: broker=%s window=%s fetched=%d inserted=%d updated=%d crossmatched=%d truncated=%s",
                    broker, result.window, result.fetched, result.inserted, result.updated, result.crossmatched,
                    result.truncated)
        return result

    async def watch(
        self,
        brokers: Sequence[str],
        *,
        interval_seconds: float = 300.0,
        limit: int = 20,
        crossmatch: bool = True,
        options: dict[str, dict[str, Any]] | None = None,
        iterations: int | None = None,
        stop_event: asyncio.Event | None = None,
        on_result: Callable[[PollResult], Awaitable[None] | None] | None = None,
    ) -> list[PollResult]:
        """Poll every broker each ``interval_seconds`` until ``stop_event``, ``iterations`` or cancellation.

        Broker errors (and any unexpected per-broker exception) are recorded in the PollResult
        (``error``, ``since_mjd`` None) and the loop continues; a backlog left by a truncated
        window is polled before new alerts.
        Each alert is committed as soon as it is processed, so cancelling (Ctrl+C,
        task.cancel()) loses nothing already fetched; CancelledError is re-raised.
        ``on_result`` may be a coroutine function (awaited), so it can do its I/O off the loop.
        """
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        _check_limit(limit)
        if not brokers:
            raise ValueError("at least one broker is required")
        for name in brokers:
            normalize_options(name, (options or {}).get(name))
        stop = stop_event or asyncio.Event()
        results: list[PollResult] = []
        cycle = 0
        try:
            while not stop.is_set():
                for name in brokers:
                    if stop.is_set():
                        break
                    try:
                        res = await self.poll(name, limit=limit, crossmatch=crossmatch, options=(options or {}).get(name))
                    except BrokerError as exc:
                        res = PollResult(broker=name, since_mjd=None, until_mjd=self.clock(), error=str(exc))
                        logger.warning("alerts watch: poll of %s failed: %s", name, exc)
                    except Exception as exc:  # one broker's unexpected failure must not end the watch
                        # (options were validated before the loop; CancelledError is not an Exception).
                        res = PollResult(broker=name, since_mjd=None, until_mjd=self.clock(),
                                         error=f"unexpected {exc.__class__.__name__}: {exc}")
                        logger.exception("alerts watch: poll of %s raised", name)
                    results.append(res)
                    if on_result is not None:
                        pending = on_result(res)
                        if inspect.isawaitable(pending):
                            await pending
                cycle += 1
                if iterations is not None and cycle >= iterations:
                    break
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except asyncio.CancelledError:
            logger.info("alerts watch cancelled after %d poll(s)", len(results))
            raise
        return results


def build_alert_service(
    client: httpx.AsyncClient,
    *,
    service: CrossmatchService | None = None,
    store: AlertStore | None = None,
    crossmatch: bool = True,
    match_radius_arcsec: float = DEFAULT_MATCH_RADIUS_ARCSEC,
    host_radius_arcsec: float = DEFAULT_HOST_RADIUS_ARCSEC,
    concurrency: int = 3,
) -> AlertService:
    """AlertService with a CrossmatchService (``main.build_service`` when none is given)."""
    enricher = None
    if crossmatch:
        if service is None:
            from main import build_service  # lazy: main imports this module for its CLI

            service = build_service(client=client)
        enricher = AlertEnricher(service, match_radius_arcsec=match_radius_arcsec, host_radius_arcsec=host_radius_arcsec)
    return AlertService(store or AlertStore(), client, enricher, concurrency=concurrency)


# ---------------------------------------------------------------------------
# REST API Router (/api/v1/alerts)
# ---------------------------------------------------------------------------


class AlertOut(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    broker: str
    survey: str | None = None
    object_id: str
    ra: float
    dec: float
    mjd: float
    first_mjd: float | None = None
    magpsf: float | None = None
    magpsf_err: float | None = None
    is_negative: bool | None = None
    band: str | None = None
    classification: str | None = None
    probability: float | None = None
    url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    crossmatch_status: str
    enrichment: dict[str, Any] | None = None
    is_new: bool | None = None
    known_star: bool | None = None
    known_variable: bool | None = None
    known_agn: bool | None = None
    host_name: str | None = None
    host_separation_arcsec: float | None = None
    host_redshift: float | None = None
    first_seen_at: str
    updated_at: str
    n_updates: int = 0
    crossmatch_attempts: int = 0
    last_crossmatch_error: str | None = None
    last_crossmatch_at: str | None = None


class AlertListResponse(BaseModel):
    count: int
    alerts: list[AlertOut]


# A synchronous (crossmatch_mode='wait') poll enriches its alerts inside the request (~10-15 s
# per alert with 3 at a time); larger polls must use the background mode.
MAX_WAIT_CROSSMATCH = 25


class PollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broker: BrokerName = "alerce"
    since_mjd: float | None = Field(None, ge=40000.0, le=100000.0, description="Window start (UTC MJD); default: cursor")
    until_mjd: float | None = Field(None, ge=40000.0, le=100000.0, description="Window end (UTC MJD); default now")
    limit: int = Field(20, ge=1, le=MAX_POLL_LIMIT, description="Max alerts stored by this poll")
    crossmatch: bool = True
    class_name: str | None = Field(None, description="ALeRCE class, Fink/ZTF class, or Fink/LSST tag")
    classifier: str | None = Field(None, description="ALeRCE classifier (alerce only)")
    mjd_field: Literal["firstmjd", "lastmjd"] | None = Field(None, description="ALeRCE window column (alerce only)")
    crossmatch_mode: Literal["background", "wait"] = Field(
        "background", description="background: return after ingest and crossmatch in a background task (follow "
        f"crossmatch_status); wait: crossmatch before answering (limit <= {MAX_WAIT_CROSSMATCH})")

    def options(self) -> dict[str, Any]:
        opts = {"class_name": self.class_name}
        if self.broker == "alerce":
            opts.update({"classifier": self.classifier, "mjd_field": self.mjd_field})
        elif self.classifier or self.mjd_field:
            raise ValueError("classifier and mjd_field apply to the alerce broker only")
        return {k: v for k, v in opts.items() if v is not None}


class PollResponse(BaseModel):
    broker: str
    since_mjd: float | None
    until_mjd: float
    options: dict[str, Any]
    window: str
    fetched: int
    inserted: int
    updated: int
    unchanged: int
    crossmatched: int
    crossmatch_partial: int
    crossmatch_failed: int
    retried: int
    crossmatch_queued: int = 0
    crossmatch_deferred: bool = False
    crossmatch_capped: int = 0
    truncated: bool
    boundary_mjd: float | None = None
    backlog: dict[str, float] | None = None
    alert_ids: list[str]
    new_alert_ids: list[str]
    warnings: list[str]
    error: str | None = None
    elapsed_ms: float
    alerts: list[AlertOut]


router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


async def _store_for(request: Request) -> AlertStore:
    """app.state.alert_store, else app.state.alert_service's store, else a store on ALERTS_DATABASE_URL
    (when set) or app.state.metadata."""
    state = request.app.state
    store = getattr(state, "alert_store", None)
    if store is None and getattr(state, "alert_service", None) is not None:
        store = state.alert_service.store
    if store is None:
        url = os.getenv("ALERTS_DATABASE_URL")
        metadata = None if url else getattr(state, "metadata", None)
        store = await asyncio.to_thread(AlertStore, metadata, database_url=url)
        state.alert_store = store
    return store


def _client_for(request: Request) -> httpx.AsyncClient:
    """app.state.client (set by the API lifespan), else one shared client created on first use."""
    state = request.app.state
    client = getattr(state, "client", None)
    if client is not None and not client.is_closed:
        return client
    client = getattr(state, "alerts_client", None)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=120.0, follow_redirects=True)
        state.alerts_client = client
    return client


async def _service_for(request: Request, *, crossmatch: bool = True) -> AlertService:
    """app.state.alert_service (given an enricher when crossmatching and it has none), else one built
    from app.state and cached there (so background crossmatches share one in-flight set)."""
    state = request.app.state
    ready: AlertService | None = getattr(state, "alert_service", None)
    if ready is not None and (ready.enricher is not None or not crossmatch):
        return ready
    cache: dict[bool, AlertService] = getattr(state, "alerts_built_services", None) or {}
    cached = cache.get(crossmatch)
    store = ready.store if ready is not None else await _store_for(request)
    if cached is not None and not cached.client.is_closed and cached.store is store:
        return cached
    client = ready.client if ready is not None else _client_for(request)
    service = getattr(state, "service", None) if crossmatch else None
    built = build_alert_service(client, service=service, store=store, crossmatch=crossmatch)
    if ready is not None:
        built.concurrency, built.overlap_days, built.lookback_days, built.clock = (
            ready.concurrency, ready.overlap_days, ready.lookback_days, ready.clock)
    cache[crossmatch] = built
    state.alerts_built_services = cache
    return built


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    request: Request,
    since_mjd: float | None = Query(None, ge=40000.0, le=100000.0),
    until_mjd: float | None = Query(None, ge=40000.0, le=100000.0),
    limit: int = Query(50, ge=1, le=1000),
    broker: BrokerName | None = None,
    classification: str | None = None,
    only_new: bool | None = Query(None, description="Only alerts with (true) / without (false) the 'new' flag"),
    crossmatch_status: Literal["pending", "done", "partial", "failed"] | None = None,
) -> dict[str, Any]:
    """Stored alerts, newest first."""
    store = await _store_for(request)
    rows = await asyncio.to_thread(store.list, since_mjd=since_mjd, until_mjd=until_mjd, limit=limit, broker=broker,
                                   classification=classification, only_new=only_new, crossmatch_status=crossmatch_status)
    return {"count": len(rows), "alerts": rows}


@router.get("/brokers", response_model=list[dict[str, Any]])
async def list_brokers() -> list[dict[str, Any]]:
    """Supported brokers, their APIs and default filters."""
    return broker_info()


@router.post("/poll", response_model=PollResponse)
async def poll_alerts(request: Request, background: BackgroundTasks,
                      req: Annotated[PollRequest | None, Body()] = None) -> dict[str, Any]:
    """Poll one broker now (an empty body polls ALeRCE with the defaults) and persist new alerts.

    By default the crossmatch runs in a background task after the answer (``crossmatch_queued``
    alerts; their ``crossmatch_status`` turns from 'pending' to done/partial/failed);
    ``crossmatch_mode='wait'`` crossmatches before answering (``limit`` <= MAX_WAIT_CROSSMATCH).
    """
    req = req or PollRequest()
    try:
        options = req.options()
        if req.crossmatch and req.crossmatch_mode == "wait" and req.limit > MAX_WAIT_CROSSMATCH:
            raise ValueError(f"crossmatch_mode='wait' allows limit <= {MAX_WAIT_CROSSMATCH} (each crossmatch takes "
                             "~10 s); use crossmatch_mode='background' for larger polls")
        svc = await _service_for(request, crossmatch=req.crossmatch)
        if req.crossmatch and req.crossmatch_mode == "background":
            result, due = await svc.ingest(req.broker, since_mjd=req.since_mjd, until_mjd=req.until_mjd,
                                           limit=req.limit, crossmatch=True, options=options)
            result.crossmatch_deferred = True
            if due:
                background.add_task(svc.crossmatch_in_background, due)
        else:
            result = await svc.poll(req.broker, since_mjd=req.since_mjd, until_mjd=req.until_mjd, limit=req.limit,
                                    crossmatch=req.crossmatch, options=options)
        rows = await asyncio.to_thread(svc.store.get_many, result.alert_ids)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BrokerError as exc:
        raise HTTPException(status_code=502, detail=f"Alert broker upstream failure: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=f"Crossmatch unavailable: {exc}") from exc
    return {**result.as_dict(), "alerts": rows}


@router.post("/{alert_id}/crossmatch", response_model=AlertOut)
async def crossmatch_alert(request: Request, alert_id: str) -> dict[str, Any]:
    """(Re-)run the crossmatch enrichment of one stored alert (a failed re-run keeps the earlier result)."""
    store = await _store_for(request)
    alert = await asyncio.to_thread(store.get_alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    try:
        svc = await _service_for(request)
        enrichment, stored = await svc.enrich_alert(alert)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=f"Crossmatch unavailable: {exc}") from exc
    if enrichment.status == "failed":
        kept = "; the previous enrichment was kept" if stored == "kept_previous" else ""
        raise HTTPException(status_code=502, detail=f"Crossmatch failed: {enrichment.error}{kept}")
    row = await asyncio.to_thread(svc.store.get, alert.alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return row


@router.get("/{alert_id}", response_model=AlertOut)
async def get_alert(request: Request, alert_id: str) -> dict[str, Any]:
    """One stored alert by 'broker:object_id' (or bare object id) with its crossmatch summary."""
    store = await _store_for(request)
    row = await asyncio.to_thread(store.get, alert_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
    return row


# ---------------------------------------------------------------------------
# Command Line Interface: `alerts poll|watch|list|show`
# ---------------------------------------------------------------------------


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be a positive number")
    return value


def _nonnegative_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(f"{text!r} must be >= 0")
    return value


def _poll_limit(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if not 1 <= value <= MAX_POLL_LIMIT:
        raise argparse.ArgumentTypeError(f"--limit must be between 1 and {MAX_POLL_LIMIT}")
    return value


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be >= 1")
    return value


# A bad --db (unsupported URL, unreachable path, unopenable database): exit 2 with a message.
_CLI_STORE_ERRORS = (ValueError, OSError, sqlite3.Error)


def _cli_store(args: argparse.Namespace) -> AlertStore:
    """--db, else ALERTS_DATABASE_URL, else the dataset metadata database (as the API)."""
    db = getattr(args, "db", None)
    return AlertStore(MetadataStore(db) if db else None)


def _json(payload: Any, *, indent: int | None = 2) -> str:
    # allow_nan=False: strict JSON (no NaN/Infinity tokens).
    return json.dumps(payload, indent=indent, default=str, allow_nan=False)


def _print_poll(result: PollResult, fmt: str, store: AlertStore, *, stream: bool = False) -> None:
    """Summary lines, or JSON (one indented document; one compact line per poll when ``stream``)."""
    if fmt == "json":
        payload = {**result.as_dict(), "alerts": store.get_many(result.alert_ids)}
        print(_json(payload, indent=None if stream else 2), flush=True)
        return
    if result.error:
        print(f"[{result.broker}] poll failed: {result.error}", flush=True)
        return
    since = f"{result.since_mjd:.4f}" if result.since_mjd is not None else "?"
    print(f"[{result.broker}] {result.window} window MJD {since} -> {result.until_mjd:.4f}: fetched {result.fetched}, "
          f"new {result.inserted}, updated {result.updated}, unchanged {result.unchanged}, "
          f"crossmatched {result.crossmatched}, partial {result.crossmatch_partial}, "
          f"crossmatch failed {result.crossmatch_failed}, retried {result.retried}"
          + (f", {result.crossmatch_capped} at the attempt cap" if result.crossmatch_capped else "")
          + (", TRUNCATED" if result.truncated else ""), flush=True)
    for row in store.get_many(result.alert_ids):
        _print_row(row)
    for warning in result.warnings:
        print(f"  warning: {warning}", flush=True)


def _print_row(row: dict[str, Any]) -> None:
    mag = f"{row['magpsf']:.2f}" if row.get("magpsf") is not None else "  -  "
    if row.get("is_negative"):
        mag = f"{mag}(neg)"  # magnitude of a negative difference flux (fainter than the reference)
    prob = f"{row['probability']:.2f}" if row.get("probability") is not None else " -  "
    flags = ",".join(k for k in ("is_new", "known_star", "known_variable", "known_agn") if row.get(k)) or "-"
    host = f"host={row['host_name']} ({row['host_separation_arcsec']:.1f}\")" if row.get("host_name") else ""
    print(f"  {row['id']:<32} MJD {row['mjd']:.5f} RA {row['ra']:10.6f} Dec {row['dec']:+10.6f} "
          f"{row.get('band') or '-':>2} {mag} {row.get('classification') or '-'} p={prob} "
          f"xm={row['crossmatch_status']} flags={flags} {host}".rstrip())


async def _cli_poll(args: argparse.Namespace) -> int:
    options = {"class_name": args.class_name, "classifier": args.classifier, "mjd_field": args.mjd_field}
    if args.broker != "alerce":
        if args.classifier or args.mjd_field:
            print("Error: --classifier/--mjd-field apply to --broker alerce only.", file=sys.stderr)
            return 2
        options = {"class_name": args.class_name}
    try:
        store = await asyncio.to_thread(_cli_store, args)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            svc = build_alert_service(client, store=store, crossmatch=not args.no_crossmatch,
                                      match_radius_arcsec=args.radius, host_radius_arcsec=args.host_radius)
            result = await svc.poll(args.broker, since_mjd=args.since_mjd, until_mjd=args.until_mjd, limit=args.limit,
                                    crossmatch=not args.no_crossmatch, options=options)
    except _CLI_STORE_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except BrokerError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    _print_poll(result, args.format, store)
    return 0


async def _cli_watch(args: argparse.Namespace) -> int:
    brokers = [b.strip() for b in args.broker.split(",") if b.strip()]
    stop = asyncio.Event()
    try:
        store = await asyncio.to_thread(_cli_store, args)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            svc = build_alert_service(client, store=store, crossmatch=not args.no_crossmatch,
                                      match_radius_arcsec=args.radius, host_radius_arcsec=args.host_radius)
            print(f"Watching {', '.join(brokers)} every {args.interval:g} s (Ctrl+C to stop)...", file=sys.stderr,
                  flush=True)

            async def show(res: PollResult) -> None:  # the database read runs off the event loop
                await asyncio.to_thread(_print_poll, res, args.format, store, stream=True)

            await svc.watch(brokers, interval_seconds=args.interval, limit=args.limit, crossmatch=not args.no_crossmatch,
                            iterations=args.iterations, stop_event=stop, on_result=show)
    except _CLI_STORE_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except asyncio.CancelledError:
        print("Watch cancelled; all processed alerts are saved.", file=sys.stderr, flush=True)
        return 130
    return 0


def _cli_list(args: argparse.Namespace) -> int:
    try:
        rows = _cli_store(args).list(since_mjd=args.since_mjd, limit=args.limit, broker=args.broker)
    except _CLI_STORE_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(_json(rows))
    else:
        print(f"{len(rows)} alert(s)")
        for row in rows:
            _print_row(row)
    return 0


def _cli_show(args: argparse.Namespace) -> int:
    try:
        row = _cli_store(args).get(args.alert_id)
    except _CLI_STORE_ERRORS as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if row is None:
        print(f"Alert {args.alert_id} not found.", file=sys.stderr)
        return 1
    print(_json(row))
    return 0


def cli_handler(args: argparse.Namespace) -> int:
    """Dispatch `alerts <action>`; returns the process exit code (130 when interrupted)."""
    action = getattr(args, "alerts_command", None)
    try:
        if action == "poll":
            return asyncio.run(_cli_poll(args))
        if action == "watch":
            return asyncio.run(_cli_watch(args))
        if action == "list":
            return _cli_list(args)
        if action == "show":
            return _cli_show(args)
    except KeyboardInterrupt:
        print("Interrupted; all processed alerts are saved.", file=sys.stderr)
        return 130
    print("Usage: alerts {poll,watch,list,show} ... (see --help)", file=sys.stderr)
    return 2


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    """Add the `alerts` subcommand (poll | watch | list | show) with handler=cli_handler."""
    parser = subparsers.add_parser("alerts", help="Ingest live transient alerts (ALeRCE, Fink) and crossmatch them")
    actions = parser.add_subparsers(dest="alerts_command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--db", help="Database URL (sqlite:///path or postgresql://...); default ALERTS_DATABASE_URL, "
                                    "else DATABASE_URL, else DATASET_STORAGE_PATH/metadata.sqlite3")
        p.add_argument("--format", choices=["summary", "json"], default="summary",
                       help="summary lines, or JSON (watch: one JSON object per line)")

    def polling(p: argparse.ArgumentParser) -> None:
        p.add_argument("--limit", type=_poll_limit, default=20,
                       help=f"Max alerts stored per broker poll, 1-{MAX_POLL_LIMIT} (default 20; the rest of an "
                            "over-full window is reported and, for default windows, queued as a backlog)")
        p.add_argument("--no-crossmatch", action="store_true", help="Store alerts without crossmatching")
        p.add_argument("--radius", type=_positive_float, default=DEFAULT_MATCH_RADIUS_ARCSEC,
                       help="Counterpart radius, arcsec (default 2)")
        p.add_argument("--host-radius", type=_nonnegative_float, default=DEFAULT_HOST_RADIUS_ARCSEC,
                       help="Host-galaxy cone radius, arcsec (default 60; 0 disables the host search)")

    poll = actions.add_parser("poll", help="Poll one broker once")
    poll.add_argument("--broker", choices=sorted(BROKERS), default="alerce")
    poll.add_argument("--since-mjd", type=float, help="Window start (UTC MJD); default: the stream cursor (or a backlog)")
    poll.add_argument("--until-mjd", type=float, help="Window end (UTC MJD); default now")
    poll.add_argument("--class", dest="class_name", help="ALeRCE class / Fink class / Fink-LSST tag")
    poll.add_argument("--classifier", help="ALeRCE classifier (default stamp_classifier)")
    poll.add_argument("--mjd-field", choices=["firstmjd", "lastmjd"], help="ALeRCE window column (default firstmjd)")
    polling(poll)
    common(poll)

    watch = actions.add_parser("watch", help="Poll brokers repeatedly until interrupted")
    watch.add_argument("--broker", default="alerce", help="Comma-separated brokers: alerce,fink,fink_lsst")
    watch.add_argument("--interval", type=_positive_float, default=300.0, help="Seconds between polls (default 300)")
    watch.add_argument("--iterations", type=_positive_int, help="Stop after N polling cycles")
    polling(watch)
    common(watch)

    lst = actions.add_parser("list", help="List stored alerts")
    lst.add_argument("--since-mjd", type=float)
    lst.add_argument("--broker", choices=sorted(BROKERS))
    lst.add_argument("--limit", type=_positive_int, default=50)
    common(lst)

    show = actions.add_parser("show", help="Show one stored alert with its crossmatch summary")
    show.add_argument("alert_id", help="broker:object_id or object id")
    common(show)

    parser.set_defaults(handler=cli_handler)

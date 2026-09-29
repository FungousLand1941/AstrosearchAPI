"""Natural-language astronomy with Claude: query compilation and cited object explanations.

Two capabilities, each usable from notebooks (pure async core) or over HTTP/CLI:

* :func:`compile_query` turns a request such as ``"quasars near M87 with radio
  emission"`` into a validated :class:`crossmatch.AdvancedQuery` dictionary, a
  human-readable plan and (optionally) one ADQL query for a single TAP catalog of the
  registry. Claude answers through a *strict* tool schema whose catalog/profile enums
  come from the live :class:`models.CatalogRegistry`; every submission is re-validated
  in Python (:class:`crossmatch.QueryValidator`, ADQL guards, CDS Sesame name
  resolution) and validation errors are fed back to Claude as ``is_error`` tool results
  for at most ``max_retries`` corrections. Object coordinates never come from the model:
  named objects are resolved with CDS Sesame (Wenger et al. 2000, A&AS 143, 9), and
  typed coordinates are submitted as the verbatim text of the request plus a frame,
  then parsed and transformed to ICRS here with astropy ``SkyCoord``; cone ADQL must
  use the ``{TARGET_CIRCLE}`` / ``{TARGET_POINT}`` placeholders, which are filled with
  the validated target (numeric ADQL positions are rejected). Object-type filters are
  expanded into SIMBAD's type hierarchy (``otypedef``) and the NED / SDSS type
  vocabularies; filters the executed query would silently ignore or that would delete
  every row of a selected catalog (types or parallax distances on catalogs without
  them, distance bounds outside cylinder mode, an inner radius outside shell mode,
  crossmatch filters in an all-sky ADQL query) are fed back as errors.

* :func:`explain_object` gathers facts -- SIMBAD basic data, object types, magnitudes,
  distances and bibliography through the SIMBAD TAP service (tables ``basic``,
  ``otypedef``, ``alltypes``, ``ident``, ``flux``, ``mesDistance``, ``ref``, ``has_ref``;
  column names verified against ``TAP_SCHEMA.columns``), an optional multi-wavelength
  crossmatch summary, optional NASA ADS citation ranking (``ADS_API_TOKEN``) -- and has
  Claude write a concise explanation that may only use those facts, with inline ``[n]``
  markers that are mapped back to bibcodes and ADS URLs. Markers that point at no
  provided source are removed and reported; numbers in the text that cannot be traced
  to a fact are listed in ``unverified_numbers``.

Derived quantities are computed here, never by the model:

* Hubble-flow distances use astropy's ``Planck18`` cosmology (Planck Collaboration
  2020, A&A 641, A6; 2020A&A...641A...6P). They are given only for extragalactic
  object types with a SIMBAD redshift of quality A-C and z >= 0.01; the heliocentric
  redshift is first moved to the CMB rest frame with the Planck 2018 solar dipole
  (369.82 km/s towards l = 264.021 deg, b = 48.253 deg; Planck Collaboration 2020,
  A&A 641, A1, 2020A&A...641A...1P) using (1 + z_hel) = (1 + z_cmb)(1 + z_sun)
  (Davis et al. 2011, ApJ 741, 67). The luminosity distance is
  D_L = (1 + z_hel) D_M(z_cmb) (Davis et al. 2011; Calcino & Davis 2017, JCAP 01, 038),
  the comoving distance and lookback time use z_cmb. Their errors combine the SIMBAD
  redshift error with an uncorrected peculiar-velocity scatter of 250 km/s in
  quadrature (the value assumed by Brout et al. 2022, ApJ 938, 110, whose floor after
  flow corrections is 240 km/s). When SIMBAD gives no redshift error, half a unit in
  the last stated decimal (``rvz_redshift_prec``; Ton 618: z = 2.2 -> +-0.05) is used and
  labelled as assumed; with neither error nor precision no distance is derived.
  Photometric redshifts are labelled as such.
* "Extragalactic" follows SIMBAD's own object-type hierarchy (``otypedef.path``):
  types under G (galaxies, AGN, QSO, blazars ...), GrG, ClG, SCG, PCG, IG, PaG and PoG,
  plus lensed galaxies/quasars (LeG, LeQ), candidates included. SIMBAD's MAIN type
  decides; the other types count only when the main type is non-specific (Rad, X, ?
  ...), since they gather every type any paper gave the object (Mira, the Helix Nebula
  and 47 Tuc all carry a stray "G").
* Parallax distances (1/parallax, with asymmetric 1-sigma errors) only for
  non-extragalactic objects with parallax quality A-C and parallax/error >= 5
  (inversion is biased at larger fractional errors, cf. Bailer-Jones 2015, PASP 127,
  994). Gaia-based parallaxes (by release bibcode, or a reference title naming the
  release) carry their release's zero point -- (E)DR3 about -0.017 mas (Lindegren et
  al. 2021, A&A 649, A4), DR2 about -0.029 mas (Lindegren et al. 2018, A&A 616, A2) --
  stated with its relative size (not applied), and a note when the formal error is
  below the release's systematic floor ((E)DR3 0.01 mas, Vasiliev & Baumgardt 2021,
  MNRAS 505, 5978; DR2 0.04 mas, Lindegren et al. 2018; DR1 0.3 mas, Gaia
  Collaboration 2016, A&A 595, A2).
* A cosmological redshift is never turned into a Doppler "radial velocity": for SIMBAD
  redshifts (``rvz_type`` z or c) only cz, labelled as a recession-velocity proxy, is
  reported, not SIMBAD's relativistic conversion ``rvz_radvel``. Conversely a galaxy
  velocity (``rvz_type`` v, catalogued as cz) gives z = v/c, not SIMBAD's relativistic
  ``rvz_redshift``, and feeds the same Hubble-flow distances.

Configuration (environment):
Anthropic credentials as resolved by the SDK (``ANTHROPIC_API_KEY``,
``ANTHROPIC_AUTH_TOKEN``, an ``ant auth login`` profile or Workload Identity
Federation) - required for Claude calls; ``ASTROSEARCH_AI_MODEL`` (default
``claude-opus-5``); ``ASTROSEARCH_AI_EFFORT`` (default ``high``);
``ASTROSEARCH_AI_FALLBACKS`` (``default`` | ``off``: server-side refusal fallbacks);
``ASTROSEARCH_AI_MAX_TOKENS`` (default 16000); ``ASTROSEARCH_AI_TIMEOUT`` seconds
(default 300); ``ADS_API_TOKEN`` (optional).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol
from urllib.parse import quote

import anthropic  # imported at module load: the import costs seconds and must not block requests
import astropy.units as u
import httpx
from astropy.coordinates import FK4, FK5, ICRS, Galactic, SkyCoord
from astropy.cosmology import Planck18
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from crossmatch import AdvancedQuery, QueryValidator
from models import (
    COMPUTED_COLUMNS,
    FIELD_ALIASES,
    AstroSearchError,
    CatalogDefinition,
    CatalogQueryError,
    CatalogRegistry,
    InvalidCoordinateError,
    ObjectResolutionError,
    ResolvedObject,
    ResponseParseError,
    Settings,
    Target,
    UnifiedRecord,
    bare_column_name,
    haversine_arcsec,
    parse_json_table,
    parse_votable_table,
    resolved_target,
    validate_target,
    votable_query_status,
)
from providers import SesameResolver, service_error_detail

__all__ = [
    "AIConfigurationError",
    "AIError",
    "AINoFactsError",
    "AINotConfiguredError",
    "AIQueryCompilationError",
    "AIRefusalError",
    "AISettings",
    "AIUpstreamError",
    "CompiledQuery",
    "ExplainResult",
    "ObjectFacts",
    "Source",
    "SourceBook",
    "UpstreamServiceError",
    "aclose_state_clients",
    "anthropic_configured",
    "build_anthropic_client",
    "build_query_tools",
    "check_query_request",
    "classify_simbad_types",
    "compile_query",
    "execute_compiled_query",
    "expand_object_types",
    "explain_object",
    "fetch_ads_most_cited",
    "fetch_simbad_bibliography",
    "gaia_release_of",
    "gather_facts",
    "map_citations",
    "parse_user_coordinates",
    "register_cli",
    "router",
    "simbad_otype_tree",
    "summarize_crossmatch",
    "unverified_numbers",
    "validate_adql",
    "write_explanation",
]

# ---------------------------------------------------------------------------
# Configuration & Errors
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"
DEFAULT_MAX_TOKENS = 16000
# Server-side refusal fallbacks ("default" routing by refusal category), Claude API beta;
# documented for Claude Opus 5 and Claude Fable 5.1 (exact model ids, not prefixes).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_CAPABLE_MODELS = frozenset({"claude-opus-5", "claude-fable-5-1"})
EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh", "max"})
# Adaptive thinking starts with the 4.6 generation of Opus/Sonnet (and every Fable/Mythos
# model); ``xhigh`` effort arrived with Opus 4.7, so 4.6 models accept the other four.
_MODEL_VERSION_RE = re.compile(r"^claude-(opus|sonnet|fable|mythos)-(\d+)(?:-(\d{1,2}))?(?:$|-)")
_EFFORT_WITHOUT_XHIGH = frozenset({"low", "medium", "high", "max"})
SPEED_OF_LIGHT_KMS = 299792.458  # exact (SI definition of the metre)
# Planck 2018 solar dipole (Planck Collaboration 2020, A&A 641, A1): 369.82 km/s towards
# Galactic (l, b) = (264.021, 48.253) deg.
CMB_DIPOLE_KMS = 369.82
CMB_DIPOLE_L_DEG = 264.021
CMB_DIPOLE_B_DEG = 48.253
PLANCK18_DIPOLE_BIBCODE = "2020A&A...641A...1P"
# Uncorrected peculiar-velocity scatter added in quadrature to Hubble-flow redshift errors:
# Brout et al. 2022, ApJ 938, 110 (Pantheon+; "250 km/s uncorrected velocity scatter").
PECULIAR_VELOCITY_KMS = 250.0
PECULIAR_VELOCITY_BIBCODE = "2022ApJ...938..110B"
# Gaia EDR3/DR3 parallax zero point: median about -0.017 mas (quasars), varying with
# magnitude, colour and position (Lindegren et al. 2021, A&A 649, A4).
GAIA_PARALLAX_ZERO_POINT_MAS = -0.017
GAIA_ZERO_POINT_BIBCODE = "2021A&A...649A...4L"
# Bibcodes of Gaia EDR3 / DR3 (papers and VizieR catalogues; titles verified in SIMBAD ``ref``).
GAIA_EDR3_DR3_BIBCODES = frozenset({
    "2020yCat.1350....0G", "2021A&A...649A...1G", "2022yCat.1355....0G", "2023A&A...674A...1G",
})


@dataclass(frozen=True, slots=True)
class GaiaRelease:
    """Parallax systematics of one Gaia data release (all numbers from the cited abstracts/papers).

    A SIMBAD parallax is attributed to a release when its bibcode is one of ``bibcodes``
    (the release papers / VizieR catalogues, which carry 12.8 million SIMBAD parallaxes,
    verified live) or when the reference title names the release (e.g. the Gaia DR2
    HRD paper 2018A&A...616A..10G, source of the Pleiades cluster parallax).
    """

    name: str
    bibcodes: frozenset[str]
    title_re: re.Pattern[str]
    zero_point_mas: float | None
    zero_point_bibcode: str | None
    systematic_floor_mas: float
    systematic_floor_bibcode: str
    systematic_floor_note: str


GAIA_RELEASES: tuple[GaiaRelease, ...] = (
    GaiaRelease(
        "Gaia (E)DR3", GAIA_EDR3_DR3_BIBCODES,
        re.compile(r"\bGaia\W{0,3}\s*(?:Early\s+Data\s+Release\s+3|Data\s+Release\s+3|E?DR3)\b", re.IGNORECASE),
        GAIA_PARALLAX_ZERO_POINT_MAS, GAIA_ZERO_POINT_BIBCODE,
        # Vasiliev & Baumgardt 2021, MNRAS 505, 5978: "a lower limit on the uncertainty of
        # average parallaxes ... at the level 0.01 mas".
        0.01, "2021MNRAS.505.5978V", "lower limit on the uncertainty of (E)DR3 parallaxes from spatially correlated "
                                     "systematics",
    ),
    GaiaRelease(
        "Gaia DR2", frozenset({"2018yCat.1345....0G", "2018A&A...616A...1G"}),
        re.compile(r"\bGaia\W{0,3}\s*(?:Data\s+Release\s+2|DR2)\b", re.IGNORECASE),
        # Lindegren et al. 2018, A&A 616, A2: DR2 parallaxes are "too small by about 0.03 mas"
        # (quasar median -0.029 mas) with "spatial correlations of up to 0.04 mas".
        -0.029, "2018A&A...616A...2L",
        0.04, "2018A&A...616A...2L", "spatially correlated DR2 parallax systematics (up to 0.04 mas)",
    ),
    GaiaRelease(
        "Gaia DR1", frozenset({"2016A&A...595A...2G", "2016A&A...595A...4L"}),
        re.compile(r"\bGaia\W{0,3}\s*(?:Data\s+Release\s+1|DR1)\b", re.IGNORECASE),
        # Gaia Collaboration (Brown et al.) 2016, A&A 595, A2: "A systematic component of
        # ~0.3 mas should be added to the parallax uncertainties."
        None, None,
        0.3, "2016A&A...595A...2G", "systematic component to be added to DR1 (TGAS) parallax errors",
    ),
)
# Roots of SIMBAD's object-type hierarchy (otypedef.path, verified live) that are
# extragalactic, plus lensed galaxy / quasar images (path grv > gLS > LeI > LeG|LeQ).
EXTRAGALACTIC_PATH_ROOTS = frozenset({"G", "GrG", "ClG", "SCG", "PCG", "IG", "PaG", "PoG"})
EXTRAGALACTIC_LENSED_OTYPES = frozenset({"LeG", "LeQ"})
# Non-specific SIMBAD types (unknown, wavelength-based, blends, regions): only for an
# object whose main type is one of these do its other types decide whether it is
# extragalactic. A specific main type (star, cluster, nebula ...) is never overridden:
# SIMBAD's ``otypes`` list gathers every type any paper gave the object, so the Helix
# Nebula, Mira or 47 Tuc carry a stray confirmed "G" (verified live).
GENERIC_PATH_ROOTS = frozenset({"", "?", "Rad", "IR", "X", "UV", "Opt", "gam", "mul", "reg", "ev"})

SIMBAD_TAP = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
ADS_SEARCH_URL = "https://api.adsabs.harvard.edu/v1/search/query"
ADS_ABS_URL = "https://ui.adsabs.harvard.edu/abs/{bibcode}/abstract"
SIMBAD_BIBCODE = "2000A&AS..143....9W"  # Wenger et al. 2000, the SIMBAD database paper
PLANCK18_BIBCODE = "2020A&A...641A...6P"  # Planck 2018 results VI (astropy Planck18 meta)

MAX_CONE_RADIUS_ARCSEC = 1800.0  # compiled cone searches fan out to every catalog: cap at 30'
ADQL_MAX_TOP = 2000
# Placeholders a model-written cone ADQL must use for its position constraint; the server
# substitutes the validated (Sesame / parsed) target, so the ADQL never carries model coordinates.
TARGET_CIRCLE = "{TARGET_CIRCLE}"
TARGET_POINT = "{TARGET_POINT}"
MAX_TOOL_ROUNDS = 8
HUBBLE_FLOW_MIN_Z = 0.01
PARALLAX_MIN_SNR = 5.0
GOOD_QUALITY = frozenset({"A", "B", "C"})
CZ_MAX_REDSHIFT = 0.1  # above this, c*z departs from any velocity by > 5 % and is not reported  # SIMBAD quality codes A (best) .. E (worst)
# The highest known stellar proper motion (Barnard's star, 10.39 arcsec/yr; SIMBAD /
# Gaia DR3): bounds how far a star can have moved from its SIMBAD J2000 position.
MAX_PROPER_MOTION_ARCSEC_YR = 10.4
# Crossmatch facts for /explain: search cone, and the chance-coincidence probability
# above which the nearest catalog row is reported as an ambiguous association.
DEFAULT_CROSSMATCH_RADIUS_ARCSEC = 30.0
MAX_CHANCE_PROBABILITY = 0.01
# Registry "wavelength" labels that name literature compilations, not observing bands.
COMPILATION_WAVELENGTHS = frozenset({"multi", "extragalactic", "exoplanet"})

# ADS bibcodes are 19 characters: YYYY JJJJJ VVVV M PPPP A.
BIBCODE_RE = re.compile(r"(?<![\w.&])(\d{4}[A-Za-z&.][\w.&]{13}[\w.])(?![\w.&])")
# Citation markers: [3], [2, 5], [1; 4], [1, 2, and 3], [1, 2 & 3], ranges [2-4] / [2–4] /
# [2—4], optional "ref"/"refs"/"source"/"#" prefixes and a stray trailing separator ([2,]).
_CITE_PREFIX = r"(?:(?:see|cf|refs?|references?|sources?)\.?\s*){0,2}#?\s*"
# Range separators, for use inside a regex character class: hyphen (escaped), Unicode
# hyphens, figure/en/em dashes and the U+2212 minus sign.
_CITE_DASHES = "\\-\u2010\u2011\u2012\u2013\u2014\u2212"
_CITE_ITEM = r"#?\s*\d+(?:\s*[" + _CITE_DASHES + r"]\s*#?\s*\d+)?"
_CITE_SEP = r"\s*(?:[,;]\s*(?:and\b|&)?|\band\b|&)\s*"
_CITE_RE = re.compile(
    r"\[\s*" + _CITE_PREFIX + r"(" + _CITE_ITEM + r"(?:" + _CITE_SEP + _CITE_ITEM + r")*)\s*[,;]?\s*\]",
    re.IGNORECASE,
)
_BRACKET_DIGITS_RE = re.compile(r"\[[^\[\]\n]*\d[^\[\]\n]*\]")
# Words that may accompany citation numbers inside brackets ([see 2], [cf. 3], [nos. 2, 4]).
_CITATION_WORDS_RE = re.compile(r"\b(?:refs?|references?|sources?|see|cf|e\.g|and|or|nos?)\b\.?", re.IGNORECASE)


def _is_citation_like(marker: str) -> bool:
    """True for bracketed text that can only be a citation: integers, citation words and separators.

    ``[3.6]`` (a band), ``[Fe II]`` or ``[2MASS]`` are not citation-like and stay as text.
    """
    inner = _CITATION_WORDS_RE.sub(" ", marker[1:-1])
    return bool(re.search(r"\d", inner)) and re.fullmatch(r"[\s\d#,;&:()" + _CITE_DASHES + r"]*", inner) is not None


MAX_CITATION_RANGE = 20
_DOI_RE = re.compile(r"\bDOI\s*:?\s*(10\.\d{4,9}/[^\s;,]+)", re.IGNORECASE)


class AIError(AstroSearchError):
    """Base class for AI feature errors."""


class AINotConfiguredError(AIError):
    """No Anthropic credentials are configured (HTTP 503)."""


class AIConfigurationError(AIError):
    """The server's AI settings (ASTROSEARCH_AI_* environment) are invalid (HTTP 500)."""


class AINoFactsError(AIError):
    """Nothing is known about the requested position/object, so nothing is explained (HTTP 404)."""


class AIUpstreamError(AIError):
    """The Anthropic API failed or returned an unusable answer (HTTP 502)."""


class AIRefusalError(AIError):
    """Claude (or its safety classifiers) declined the request (HTTP 422)."""

    def __init__(self, message: str, category: str | None = None) -> None:
        super().__init__(message)
        self.category = category


class AIQueryCompilationError(AIError):
    """Claude could not produce a valid query within the retry budget (HTTP 422)."""

    def __init__(self, message: str, errors: Sequence[str] = (), history: Sequence[Sequence[str]] = ()) -> None:
        super().__init__(message)
        self.errors = list(errors)
        self.history = [list(h) for h in history]


class UpstreamServiceError(AIError):
    """A public astronomy service (SIMBAD TAP, ADS, Sesame transport) failed (HTTP 502)."""


@dataclass(slots=True)
class AISettings:
    """Model and request settings for the Claude calls (see module docstring for env vars)."""

    model: str = DEFAULT_MODEL
    effort: str | None = DEFAULT_EFFORT
    fallbacks: bool = True
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_seconds: float = 300.0

    @classmethod
    def from_env(cls) -> AISettings:
        """Read ASTROSEARCH_AI_* settings; invalid values raise :class:`AIConfigurationError`."""
        effort = os.getenv("ASTROSEARCH_AI_EFFORT", DEFAULT_EFFORT).strip().lower() or None
        if effort is not None and effort not in EFFORT_LEVELS:
            raise AIConfigurationError(f"ASTROSEARCH_AI_EFFORT must be one of {sorted(EFFORT_LEVELS)} (got {effort!r}).")
        try:
            max_tokens = int(os.getenv("ASTROSEARCH_AI_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
            timeout = float(os.getenv("ASTROSEARCH_AI_TIMEOUT", "300"))
        except ValueError as exc:
            raise AIConfigurationError(f"ASTROSEARCH_AI_MAX_TOKENS / ASTROSEARCH_AI_TIMEOUT must be numbers: {exc}") from exc
        if not 1024 <= max_tokens <= 128_000:
            raise AIConfigurationError("ASTROSEARCH_AI_MAX_TOKENS must be between 1024 and 128000.")
        if not (math.isfinite(timeout) and timeout > 0):
            raise AIConfigurationError("ASTROSEARCH_AI_TIMEOUT must be a positive number of seconds.")
        settings = cls(
            model=os.getenv("ASTROSEARCH_AI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            effort=effort,
            fallbacks=os.getenv("ASTROSEARCH_AI_FALLBACKS", "default").strip().lower() not in {"off", "0", "false", "no"},
            max_tokens=max_tokens,
            timeout_seconds=timeout,
        )
        if settings.supports_adaptive_thinking and effort and effort not in settings.effort_levels:
            raise AIConfigurationError(f"{settings.model} does not accept effort {effort!r} "
                                       f"(valid: {sorted(settings.effort_levels)}).")
        return settings

    @property
    def _version(self) -> tuple[str, int, int] | None:
        match = _MODEL_VERSION_RE.match(self.model)
        if not match:
            return None
        return match.group(1), int(match.group(2)), int(match.group(3) or 0)

    @property
    def supports_adaptive_thinking(self) -> bool:
        """``thinking: {type: adaptive}`` plus ``effort``: Opus/Sonnet >= 4.6, every Fable/Mythos.

        Older models (Opus 4.5/4.1, Sonnet 4.5, Haiku, Claude 3) need ``budget_tokens``
        or reject effort, so neither is sent to them (thinking stays off).
        """
        version = self._version
        if version is None:
            return False
        family, major, minor = version
        return family in {"fable", "mythos"} or (major, minor) >= (4, 6)

    @property
    def effort_levels(self) -> frozenset[str]:
        version = self._version
        if version and version[0] in {"opus", "sonnet"} and (version[1], version[2]) == (4, 6):
            return _EFFORT_WITHOUT_XHIGH
        return EFFORT_LEVELS

    @property
    def uses_fallbacks(self) -> bool:
        return self.fallbacks and self.model in FALLBACK_CAPABLE_MODELS


_NO_CREDENTIALS = (
    "Claude features need Anthropic credentials in the server environment: ANTHROPIC_API_KEY (or "
    "ANTHROPIC_AUTH_TOKEN, an `ant auth login` profile, or Workload Identity Federation)."
)


def _has_credentials(client: Any) -> bool:
    return any(getattr(client, attr, None) for attr in ("api_key", "auth_token", "credentials"))


def anthropic_configured() -> bool:
    """True when the Anthropic SDK resolves credentials (env key/token, profile or WIF)."""
    if os.getenv("ANTHROPIC_API_KEY", "").strip() or os.getenv("ANTHROPIC_AUTH_TOKEN", "").strip():
        return True
    try:
        probe = anthropic.Anthropic(max_retries=0)
    except Exception:  # noqa: BLE001 - a broken profile file is "not configured", never a crash
        return False
    try:
        return _has_credentials(probe)
    finally:
        probe.close()


def build_anthropic_client(settings: AISettings | None = None) -> Any:
    """Create an ``anthropic.AsyncAnthropic`` client, or raise AINotConfiguredError.

    Credentials are resolved by the SDK itself (API key, auth token, ``ant`` profile,
    Workload Identity Federation). Blocking (SSL context); call it at startup or via
    ``asyncio.to_thread`` from async code.
    """
    active = settings or AISettings.from_env()
    if not anthropic_configured():
        raise AINotConfiguredError(_NO_CREDENTIALS)
    try:
        return anthropic.AsyncAnthropic(timeout=active.timeout_seconds, max_retries=2)
    except Exception as exc:
        raise AINotConfiguredError(f"{_NO_CREDENTIALS} ({exc})") from exc


class CrossmatchLike(Protocol):
    """What this module needs from :class:`crossmatch.CrossmatchService`."""

    async def crossmatch(
        self,
        ra: float | str,
        dec: float | str,
        *,
        radius_arcsec: float | None = ...,
        epoch: float | None = ...,
        query: AdvancedQuery | None = ...,
        pm_ra_masyr: float | None = ...,
        pm_dec_masyr: float | None = ...,
    ) -> UnifiedRecord: ...


async def _create_message(client: Any, settings: AISettings, **params: Any) -> Any:
    """One Messages API call with adaptive thinking, effort, fallbacks and typed errors.

    Uses ``client.beta.messages.create`` so the server-side ``fallbacks="default"``
    refusal routing (beta ``server-side-fallback-2026-07-01``) can be enabled.
    Adaptive thinking and effort are sent only to models that accept them.
    """
    request: dict[str, Any] = {"model": settings.model, "max_tokens": settings.max_tokens, **params}
    output_config = dict(request.pop("output_config", None) or {})
    if settings.supports_adaptive_thinking:
        request["thinking"] = {"type": "adaptive"}
        if settings.effort:
            output_config["effort"] = settings.effort
    if output_config:
        request["output_config"] = output_config
    if settings.uses_fallbacks:
        request["betas"] = [FALLBACK_BETA]
        request["fallbacks"] = "default"
    try:
        response = await client.beta.messages.create(**request)
    except anthropic.AuthenticationError as exc:
        raise AIUpstreamError("Anthropic API rejected the configured credentials (HTTP 401).") from exc
    except anthropic.PermissionDeniedError as exc:
        raise AIUpstreamError(f"Anthropic API denied the request (HTTP 403): {exc.message}") from exc
    except anthropic.NotFoundError as exc:
        raise AIUpstreamError(f"Anthropic model or endpoint not found ({settings.model}): {exc.message}") from exc
    except anthropic.RateLimitError as exc:
        raise AIUpstreamError("Anthropic API rate limit reached (HTTP 429); retry later.") from exc
    except anthropic.BadRequestError as exc:
        raise AIUpstreamError(f"Anthropic API rejected the request (HTTP 400): {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise AIUpstreamError(f"Anthropic API error (HTTP {exc.status_code}): {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise AIUpstreamError(f"Could not reach the Anthropic API: {exc}") from exc
    except anthropic.CredentialsError as exc:
        # Credentials resolved at client construction but unreadable at request time
        # (missing WIF identity-token file, deleted/corrupted `ant` profile ...).
        raise AINotConfiguredError(f"{_NO_CREDENTIALS} The configured credentials could not be loaded: {exc}") from exc
    except anthropic.AnthropicError as exc:  # WorkloadIdentityError, response validation, ...
        raise AIUpstreamError(f"Anthropic SDK error ({exc.__class__.__name__}): {exc}") from exc
    except TypeError as exc:  # the SDK raises TypeError when it cannot resolve any credential
        if "authentication method" in str(exc):
            raise AINotConfiguredError(_NO_CREDENTIALS) from exc
        raise
    if getattr(response, "stop_reason", None) == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if details is not None else None
        raise AIRefusalError(
            "Claude declined this request" + (f" (category: {category})." if category else "."), category
        )
    return response


def _blocks(response: Any) -> list[Any]:
    return list(getattr(response, "content", None) or [])


def _text_of(response: Any) -> str:
    return "".join(getattr(b, "text", "") or "" for b in _blocks(response) if getattr(b, "type", None) == "text")


# ---------------------------------------------------------------------------
# Query Compilation: Tool Schema
# ---------------------------------------------------------------------------

_NULLABLE_NUMBER = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

# Extra SIMBAD tables usable in ADQL (names/columns verified via SIMBAD TAP_SCHEMA.columns).
SIMBAD_ADQL_TABLES: dict[str, str] = {
    "basic": "one row per object: oid, main_id, ra, dec (deg), otype, sp_type, morph_type, plx_value (mas), "
             "pmra, pmdec (mas/yr), rvz_redshift, rvz_radvel (km/s), nbref",
    "ident": "identifiers: oidref, id",
    "alltypes": "oidref, otypes (all object types joined with '|')",
    "otypes": "oidref, otype (one row per assigned type)",
    "otypedef": "otype, label, description (object-type dictionary)",
    "flux": "oidref, filter, flux (magnitude), flux_err, system ('V' Vega / 'A' AB), bibcode",
    "mesDistance": "oidref, dist, unit (pc/kpc/Mpc), method, bibcode",
    "has_ref": "oidref, oidbibref (object-to-paper links)",
    "ref": "oidbib, bibcode, title, \"year\", journal, nbobject",
}


def _base_table(catalog: CatalogDefinition) -> str:
    table = (catalog.table or "").strip()
    if table.startswith('"'):
        end = table.find('"', 1)
        return table[: end + 1] if end > 0 else table
    return table.split()[0] if table else ""


def tap_catalogs(registry: CatalogRegistry) -> dict[str, CatalogDefinition]:
    """Enabled registry catalogs reachable through an IVOA TAP endpoint (ADQL targets)."""
    return {n: c for n, c in sorted(registry.enabled_catalogs().items()) if c.provider == "tap" and c.table and c.endpoint}


def registry_context(registry: CatalogRegistry) -> list[dict[str, Any]]:
    """Deterministic (cache-friendly) description of enabled catalogs for the system prompt."""
    tap = tap_catalogs(registry)
    context: list[dict[str, Any]] = []
    for name, cat in sorted(registry.enabled_catalogs().items()):
        entry: dict[str, Any] = {
            "name": name,
            "wavelength": cat.wavelength,
            "profiles": list(cat.profiles),
            "description": cat.description,
        }
        if name in tap:
            entry["adql"] = {
                "table": _base_table(cat),
                "from_clause_used_by_this_service": cat.table,
                "known_columns": [str(c) for c in cat.parameters.get("columns") or []],
            }
            if name == "simbad":
                entry["adql"]["other_tables"] = SIMBAD_ADQL_TABLES
                entry["adql"]["notes"] = "SIMBAD ORDER BY must use unqualified names or SELECT aliases (b.x AS x)."
        context.append(entry)
    return context


def registry_profiles(registry: CatalogRegistry) -> list[str]:
    return sorted({p for cat in registry.enabled_catalogs().values() for p in cat.profiles})


def _catalog_reports(catalog: CatalogDefinition, canonical: str) -> bool:
    """True when a catalog's result columns map to ``canonical`` (e.g. ``object_type``).

    Uses the same column-name aliases as the providers (models.FIELD_ALIASES) plus the
    catalog's explicit ``field_map``.
    """
    if canonical in (catalog.parameters.get("field_map") or {}):
        return True
    for column in catalog.parameters.get("columns") or []:
        if FIELD_ALIASES.get(bare_column_name(str(column)).lower()) == canonical:
            return True
    return False


def typed_catalogs(registry: CatalogRegistry, canonical: str) -> list[str]:
    """Enabled catalogs whose rows carry ``canonical`` (``object_type`` / ``spectral_type``)."""
    return sorted(n for n, c in registry.enabled_catalogs().items() if _catalog_reports(c, canonical))


def selected_catalogs(registry: CatalogRegistry, catalogs: Sequence[str], profiles: Sequence[str]) -> list[str]:
    """Catalogs a query with these ``catalogs``/``profiles`` lists would query (as QueryBuilder does)."""
    enabled = registry.enabled_catalogs()
    names = [c for c in catalogs if c in enabled] if catalogs else sorted(enabled)
    if profiles:
        names = [n for n in names if any(p in enabled[n].profiles for p in profiles)]
    return names


# ---------------------------------------------------------------------------
# Query Compilation: Object-Type Vocabulary
# ---------------------------------------------------------------------------
#
# crossmatch.AdvancedQuery filters rows by exact (case-insensitive) equality of the row's
# own type code, so a requested class must be expanded into every code the typed
# catalogs actually use:
# * SIMBAD rows carry otype codes; a class or code expands to its subtree of SIMBAD's
#   hierarchy (table ``otypedef``, column ``path``, fetched live; candidate types share
#   their parent's path and are included). Planets (Pl) and planetary nebulae (PN) sit
#   under "*" in that hierarchy but are not stars.
# * NED rows carry NED object types (NEDTAP.objdir.prefphytype; list documented at
#   https://ned.ipac.caltech.edu/help/ui/nearposn-list_objecttypes, values checked live:
#   3C 273 / BL Lac / OJ 287 "QSO", M 87 "G", Vega "*", Mira "V*", M 13 "*Cl", Crab "SNR",
#   Helix "Neb"). NED files BL Lacs under QSO and Seyferts under G, so SIMBAD subclasses
#   such as BLL, AGN or Sy1 have no NED equivalent.
# * SDSS rows carry the spectroscopic class STAR, GALAXY or QSO (rows without a spectrum
#   have none).

OBJECT_CLASS_WORDS = ("star", "galaxy", "quasar", "nebula", "star_cluster")
_CLASS_SYNONYMS = {
    "stars": "star", "galaxies": "galaxy", "quasars": "quasar", "qsos": "quasar", "nebulae": "nebula",
    "nebulas": "nebula", "star_clusters": "star_cluster", "cluster_of_stars": "star_cluster",
}
# class -> (SIMBAD subtree roots, SIMBAD subtrees excluded)
_CLASS_SIMBAD: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "star": (("*",), ("PN", "Pl")),
    "galaxy": (("G",), ()),  # SIMBAD's G subtree includes AGN, Seyferts and quasars
    "quasar": (("QSO",), ()),  # QSO, Q?, Bla, Bz?, BLL, BL?
    "nebula": (("ISM", "PN"), ()),
    "star_cluster": (("Cl*",), ()),
}
_CLASS_NED: dict[str, tuple[str, ...]] = {
    "star": ("*", "**", "Blue*", "C*", "exG*", "Flare*", "Nova", "Psr", "Red*", "V*", "WD*", "WR*",
             "!*", "!**", "!Blue*", "!C*", "!Flar*", "!Nova", "!Psr", "!Red*", "!V*", "!WD*", "!WR*"),
    "galaxy": ("G", "QSO"),
    "quasar": ("QSO",),
    "nebula": ("Neb", "!Neb", "HII", "!HII", "PN", "!PN", "RfN", "!RfN", "SNR", "!SNR", "MCld", "!MCld"),
    "star_cluster": ("*Cl", "!*Cl"),
}
_CLASS_SDSS: dict[str, tuple[str, ...]] = {"star": ("STAR",), "galaxy": ("GALAXY", "QSO"), "quasar": ("QSO",)}
# SIMBAD codes that name a whole class, and SIMBAD codes with an exact NED counterpart.
_SIMBAD_CLASS_CODES = {"*": "star", "G": "galaxy", "QSO": "quasar", "Cl*": "star_cluster"}
_SIMBAD_TO_NED: dict[str, tuple[str, ...]] = {
    "PN": ("PN", "!PN"), "HII": ("HII", "!HII"), "SNR": ("SNR", "!SNR"), "WD*": ("WD*", "!WD*"),
    "WR*": ("WR*", "!WR*"), "Psr": ("Psr", "!Psr"), "RNe": ("RfN", "!RfN"), "MoC": ("MCld", "!MCld"),
    "C*": ("C*", "!C*"), "**": ("**", "!**"),
}
_TYPE_VOCABULARY_CATALOGS = frozenset({"simbad", "ned", "sdss"})

# SIMBAD otype -> (path elements, is_candidate), per TAP endpoint, fetched once per process.
_OTYPE_TREES: dict[str, dict[str, tuple[tuple[str, ...], bool]]] = {}
OTYPEDEF_ADQL = "SELECT otype, path, is_candidate FROM otypedef"


async def simbad_otype_tree(
    client: httpx.AsyncClient, endpoint: str = SIMBAD_TAP
) -> dict[str, tuple[tuple[str, ...], bool]]:
    """SIMBAD's object-type hierarchy (``otypedef``): otype -> (path elements, is_candidate).

    Fetched live once per process and endpoint; a failure raises
    :class:`UpstreamServiceError` (never an empty vocabulary).
    """
    cached = _OTYPE_TREES.get(endpoint)
    if cached:
        return cached
    tree: dict[str, tuple[tuple[str, ...], bool]] = {}
    for row in await _tap_rows(client, endpoint, OTYPEDEF_ADQL):
        code = str(row.get("otype") or "").strip()
        if not code:
            continue
        path = tuple(p.strip() for p in str(row.get("path") or "").split(">") if p.strip())
        tree[code] = (path, _truthy(row.get("is_candidate")))
    if len(tree) < 50 or "*" not in tree or "QSO" not in tree:
        raise UpstreamServiceError(f"SIMBAD otypedef returned an incomplete object-type list ({len(tree)} types).")
    _OTYPE_TREES[endpoint] = tree
    return tree


def simbad_subtree(tree: Mapping[str, tuple[tuple[str, ...], bool]], root: str) -> set[str]:
    """``root`` and every SIMBAD type below it (candidates included: they share the path)."""
    return {code for code, (path, _) in tree.items() if root in path} | {root}


def _class_word(value: str) -> str | None:
    key = "_".join(value.strip().casefold().replace("-", " ").split())
    key = _CLASS_SYNONYMS.get(key, key)
    return key if key in OBJECT_CLASS_WORDS else None


def expand_object_types(
    requested: Sequence[str], catalogs: Sequence[str], tree: Mapping[str, tuple[tuple[str, ...], bool]]
) -> tuple[list[str], list[str]]:
    """Expand class words / SIMBAD codes into every type code the selected catalogs use.

    Returns (codes, errors). A value that is neither a class word
    (:data:`OBJECT_CLASS_WORDS`) nor a SIMBAD otype code is an error, as is a SIMBAD code
    with no counterpart in the vocabulary of a selected NED or SDSS catalog (the filter
    would silently remove all of their rows).
    """
    codes: set[str] = set()
    errors: list[str] = []
    selected = set(catalogs)
    by_fold: dict[str, list[str]] = {}
    for code in tree:
        by_fold.setdefault(code.casefold(), []).append(code)
    for value in requested:
        text = value.strip()
        cls = _class_word(text)
        code: str | None = None
        if cls is None:
            if text in tree:
                code = text
            elif len(by_fold.get(text.casefold(), [])) == 1:
                code = by_fold[text.casefold()][0]
            if code is None:
                errors.append(
                    f"object type {value!r} is neither a class word ({', '.join(OBJECT_CLASS_WORDS)}) nor a SIMBAD "
                    "otype code (e.g. QSO, BLL, Sy1, PM*, WD*, GlC); types such as spectral classes belong in "
                    "spectral_types or the plan."
                )
                continue
            cls = _SIMBAD_CLASS_CODES.get(code)
        if cls is not None:
            roots, excluded = _CLASS_SIMBAD[cls]
            simbad_codes = set().union(*(simbad_subtree(tree, r) for r in roots if r in tree))
            for ex in excluded:
                simbad_codes -= simbad_subtree(tree, ex)
            ned_codes: tuple[str, ...] | None = _CLASS_NED[cls]
            sdss_codes: tuple[str, ...] | None = _CLASS_SDSS.get(cls)
        else:
            assert code is not None
            simbad_codes = simbad_subtree(tree, code)
            ned_codes = _SIMBAD_TO_NED.get(code)
            sdss_codes = None
        if "ned" in selected and ned_codes is None:
            errors.append(f"object type {value!r} has no NED equivalent (NED files BL Lacs under QSO and Seyferts under "
                          "G), so every NED row would be removed: drop ned from catalogs, or use a class word "
                          f"({', '.join(OBJECT_CLASS_WORDS)}).")
            continue
        if "sdss" in selected and sdss_codes is None:
            errors.append(f"object type {value!r} has no SDSS equivalent (SDSS classes are STAR, GALAXY and QSO), so "
                          "every SDSS row would be removed: drop sdss from catalogs, or use star, galaxy or quasar.")
            continue
        codes |= simbad_codes
        codes |= set(ned_codes or ())
        codes |= set(sdss_codes or ())
    return sorted(codes), errors


COORDINATE_FRAMES = ("icrs", "fk5", "fk4", "galactic")


def build_query_tools(registry: CatalogRegistry) -> list[dict[str, Any]]:
    """Strict tool definitions (``resolve_object``, ``submit_query``) for the registry.

    Only JSON-Schema features supported by strict tool use are used (no numeric or
    string-length constraints); ranges are enforced by :func:`_validate_submission`.
    Catalog names, profiles and wavelengths come from the registry.
    """
    enabled = registry.enabled_catalogs()
    catalogs = sorted(enabled)
    tap_names = sorted(tap_catalogs(registry))
    by_wavelength: dict[str, list[str]] = {}
    for name in catalogs:
        by_wavelength.setdefault(enabled[name].wavelength, []).append(name)
    wavelength_note = "; ".join(f"{w}: {', '.join(names)}" for w, names in sorted(by_wavelength.items()))
    object_typed = typed_catalogs(registry, "object_type")
    spectral_typed = typed_catalogs(registry, "spectral_type")
    parallax_typed = typed_catalogs(registry, "parallax")
    string_array = {"type": "array", "items": {"type": "string"}}
    submit_properties: dict[str, Any] = {
        "scope": {
            "type": "string",
            "enum": ["cone", "all_sky"],
            "description": "'cone' for a search around one position (needs target); 'all_sky' when the request has "
                           "no position anchor (needs adql).",
        },
        "target": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "name": {**_NULLABLE_STRING, "description": "Object name as written by the user; the server "
                                 "resolves it with CDS Sesame. Never convert names to coordinates yourself."},
                        "coordinates": {
                            "anyOf": [
                                {"type": "null"},
                                {
                                    "type": "object",
                                    "properties": {
                                        "text": {"type": "string", "description": "The coordinates exactly as typed in "
                                                 "the request (copied verbatim, e.g. '12h30m49.42s +12d23m28.0s' or "
                                                 "'l=0.0, b=0.0'); the server parses and converts them. Never compute "
                                                 "or convert coordinates yourself."},
                                        "frame": {"type": "string", "enum": list(COORDINATE_FRAMES),
                                                  "description": "Frame the user's coordinates are in: 'icrs' for "
                                                                 "RA/Dec without an equinox (or J2000), 'fk4' for "
                                                                 "B1950, 'fk5' for other Julian equinoxes, 'galactic' "
                                                                 "for l/b."},
                                        "equinox": {**_NULLABLE_STRING, "description": "Equinox for fk4/fk5 if the "
                                                    "user gave one (e.g. 'B1950', 'J2000'); null for icrs and "
                                                    "galactic."},
                                    },
                                    "required": ["text", "frame", "equinox"],
                                    "additionalProperties": False,
                                },
                            ],
                            "description": "Only when the user typed coordinates; else null.",
                        },
                    },
                    "required": ["name", "coordinates"],
                    "additionalProperties": False,
                },
            ],
            "description": "Search centre (null for all_sky): a name, or coordinates the user typed.",
        },
        "radius_arcsec": {"type": "number", "description": f"Cone radius in arcsec (0 < r <= {MAX_CONE_RADIUS_ARCSEC:g})."},
        "catalogs": {"type": "array", "items": {"type": "string", "enum": catalogs},
                     "description": "Registry catalogs to query; empty = all catalogs of the chosen profiles. "
                                    f"Catalogs by wavelength - {wavelength_note}."},
        "profiles": {"type": "array", "items": {"type": "string", "enum": registry_profiles(registry)},
                     "description": "Catalog profiles; a catalog is queried if it has any listed profile."},
        "object_types": {**string_array, "description": "Keep only catalog rows of these types: the class words "
                         f"{', '.join(OBJECT_CLASS_WORDS)}, or SIMBAD otype codes (e.g. QSO, AGN, BLL, Sy1, PM*). The "
                         "server expands each into SIMBAD's type hierarchy (quasar includes blazars and BL Lacs, star "
                         "includes high-proper-motion stars, candidates included) and the matching NED/SDSS types; a "
                         "SIMBAD subclass without a NED/SDSS equivalent (AGN, BLL, Sy1 ...) cannot be combined with ned "
                         "or sdss. The filter applies to "
                         f"EVERY row of every queried catalog, and only {', '.join(object_typed)} report a type, so rows "
                         "of all other catalogs are removed. Use it only when every selected catalog is one of those; "
                         "to combine a type with detections at other wavelengths (e.g. radio quasars), leave it empty "
                         "and say in the plan that the crossmatch groups are inspected for the type."},
        "spectral_types": {**string_array, "description": "Exact MK spectral types (case-insensitive exact match, e.g. "
                           "'M4.5V'); prefix classes like 'M' do NOT match - leave empty unless exact types are "
                           f"requested. Only {', '.join(spectral_typed)} report spectral types; same restriction as "
                           "object_types."},
        "search_mode": {"type": "string", "enum": ["cone", "shell", "cylinder"],
                        "description": "shell = annulus (min_radius_arcsec..radius_arcsec); cylinder = cone plus parallax "
                                       "distance bounds (needs min_distance_pc and/or max_distance_pc). Cylinder mode "
                                       "removes every row without a parallax, and only "
                                       f"{', '.join(parallax_typed)} report one: select only those catalogs, or "
                                       "express the distance limit with scope all_sky ADQL instead."},
        "min_radius_arcsec": {"type": "number", "description": "Inner radius for search_mode shell (> 0), else 0."},
        "min_distance_pc": {**_NULLABLE_NUMBER, "description": "Lower distance bound (pc) for search_mode cylinder "
                            "only, else null."},
        "max_distance_pc": {**_NULLABLE_NUMBER, "description": "Upper distance bound (pc) for search_mode cylinder "
                            "only, else null."},
        "start_year": {**_NULLABLE_NUMBER, "description": "Keep observations from this year on, or null."},
        "end_year": {**_NULLABLE_NUMBER, "description": "Keep observations up to this year, or null."},
        "max_results": {"anyOf": [{"type": "integer"}, {"type": "null"}], "description": "Per-catalog match cap or null."},
        "min_confidence": {"type": "number", "description": "Minimum positional match confidence 0..1 (default 0.5)."},
        "adql": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "catalog": {"type": "string", "enum": tap_names, "description": "TAP catalog the query runs on."},
                        "query": {"type": "string", "description": f"One ADQL SELECT with TOP n (n <= {ADQL_MAX_TOP}) "
                                  "reading that catalog's table. Never write sky coordinates: for scope cone, "
                                  f"constrain the position with {TARGET_CIRCLE} (and optionally {TARGET_POINT}), "
                                  "which the server replaces with the validated target and radius, e.g. "
                                  f"WHERE CONTAINS(POINT('ICRS', ra, dec), {TARGET_CIRCLE}) = 1; scope all_sky "
                                  "uses no placeholders and no numeric POINT/CIRCLE/BOX/POLYGON."},
                    },
                    "required": ["catalog", "query"],
                    "additionalProperties": False,
                },
            ],
            "description": "Optional single-catalog ADQL query; required when scope is all_sky.",
        },
        "plan": {**string_array, "description": "Ordered, human-readable execution steps (2-6 short sentences)."},
        "explanation": {"type": "string", "description": "How the request was interpreted, including any caveats."},
    }
    return [
        {
            "name": "resolve_object",
            "description": "Resolve an astronomical object name with CDS Sesame (SIMBAD/NED/VizieR). Returns canonical "
                           "name, ICRS position (deg), SIMBAD object type and redshift. Use it to check that names in "
                           "the request are real objects before submitting.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Object name, e.g. 'M87' or '3C 273'."}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "submit_query",
            "description": "Submit the compiled search. The server validates it against the catalog registry and "
                           "returns errors to fix if it is invalid. Call it exactly once per attempt.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": submit_properties,
                "required": list(submit_properties),
                "additionalProperties": False,
            },
        },
    ]


_QUERY_SYSTEM_PROMPT = """You compile natural-language astronomy requests into catalog searches for AstroSearch, \
a multi-wavelength crossmatching service. Answer by calling the submit_query tool; you may first call \
resolve_object to check object names.

How AstroSearch executes a submitted query (scope "cone"): it queries the selected registry catalogs in a cone \
around the target, crossmatches rows by position, then keeps only matches passing the filters \
(object_types, spectral_types, search_mode shell/cylinder bounds, time window, min_confidence, max_results per \
catalog). Wavelength coverage is chosen with catalogs and/or profiles; for example radio emission means the radio \
catalogs, X-ray brightness means the xray catalogs. The object_types/spectral_types filters remove every row that \
does not report a matching type, including all rows of catalogs that report no type at all (radio, X-ray, infrared \
surveys), so never combine them with such catalogs; instead leave them empty and explain in the plan that the \
crossmatch groups are inspected for the type. Likewise search_mode cylinder removes every row without a parallax \
(only {parallax_catalogs} report one); min_radius_arcsec only acts in search_mode shell and the distance bounds only \
in search_mode cylinder.

Rules:
- Use only catalogs and profiles from the registry below; do not invent catalogs.
- Never supply coordinates for a named object: put the name in target.name and leave target.coordinates null. \
Use target.coordinates only when the user typed coordinates: copy them verbatim into coordinates.text and give \
their frame; the server parses and converts them (never convert or compute coordinates yourself).
- Choose a radius that matches the request ("near", "around", "within 5 arcmin"); the maximum is \
{max_radius} arcsec. Keep min_radius_arcsec 0 unless the search is a shell, and min_confidence 0.5 unless asked.
- If the request has no position anchor (an all-sky selection such as nearby stars of a class), set scope \
"all_sky", target null, and give an ADQL query for the single most suitable TAP catalog.
- ADQL: one SELECT with TOP n (n <= {max_top}); read the catalog's own table (joins to related tables of the same \
service are allowed); use columns you are confident exist - the listed known_columns are verified; ADQL, not SQL \
(no LIMIT, no semicolons). Never write sky coordinates into ADQL. For cone requests ADQL is optional; include it \
when one TAP catalog can answer the request directly, and constrain the position only with the placeholders \
{target_circle} (the validated target and radius as an ADQL CIRCLE) and {target_point}, e.g. \
CONTAINS(POINT('ICRS', ra, dec), {target_circle}) = 1.
- plan: 2-6 short steps a scientist can follow. explanation: how you interpreted the request and what the query \
cannot express (for example filters that no catalog column supports).

Catalog registry (JSON):
{registry}"""


def _query_system_prompt(registry: CatalogRegistry) -> str:
    return _QUERY_SYSTEM_PROMPT.format(
        max_radius=f"{MAX_CONE_RADIUS_ARCSEC:g}",
        max_top=ADQL_MAX_TOP,
        parallax_catalogs=", ".join(typed_catalogs(registry, "parallax")),
        target_circle=TARGET_CIRCLE,
        target_point=TARGET_POINT,
        registry=json.dumps(registry_context(registry), sort_keys=True, separators=(",", ":")),
    )


# ---------------------------------------------------------------------------
# Query Compilation: Validation
# ---------------------------------------------------------------------------


class _CoordinatesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    frame: Literal["icrs", "fk5", "fk4", "galactic"]
    equinox: str | None


class _TargetIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None
    coordinates: _CoordinatesIn | None


def _normalize_text(text: str) -> str:
    table = str.maketrans("−–—′″’”“", "---'\"'\"\"")
    return " ".join(str(text).translate(table).split()).casefold()


_COORD_LABEL_RE = re.compile(
    r"\b(?:r\.?a\.?|dec\.?|decl\.?|alpha|delta|glon|glat|lon|lat|l|b)\s*[=:]\s*|\b(?:ra|dec|decl|glon|glat)\b"
    r"|[αδℓ]\s*[=:]?\s*",  # Greek alpha / delta and the script l of Galactic longitude
    re.IGNORECASE,
)
_COORD_NOISE_RE = re.compile(r"\(?\b(?:icrs|fk5|fk4|galactic|gal|equatorial|[JB]\d{4}(?:\.\d+)?|deg|degrees?)\b\)?",
                             re.IGNORECASE)
_EQUINOX_RE = re.compile(r"^[JB]\d{4}(?:\.\d+)?$")
# Hints the typed text itself gives about its frame and equinox (checked against the
# frame/equinox the model submitted, so a mislabelled frame is fed back, never used).
_GALACTIC_HINT_RE = re.compile(r"\b(?:l|b|glon|glat)\s*[=:]|\b(?:glon|glat|galactic)\b|ℓ", re.IGNORECASE)
_EQUATORIAL_HINT_RE = re.compile(
    r"\b(?:r\.?a|dec|decl|alpha|delta|equatorial)\b\.?|[αδ]|\d\s*h\s*\d|\d\s*:\s*\d", re.IGNORECASE
)
_FRAME_WORD_RE = re.compile(r"\b(icrs|fk5|fk4|galactic)\b", re.IGNORECASE)
_EQUINOX_WORD_RE = re.compile(r"(?<![\w.])([JB])(\d{4}(?:\.\d+)?)(?![\w.])", re.IGNORECASE)
# IAU-style compact positions: [J|B]HHMMSS(.s)+-DDMMSS(.s) (e.g. J123049.42+122328.0).
_COMPACT_POSITION_RE = re.compile(
    r"^\s*([JB])?\s*(\d{2})(\d{2})(\d{2}(?:\.\d+)?)\s*([+\-])\s*(\d{2})(\d{2})(\d{2}(?:\.\d+)?)\s*$", re.IGNORECASE
)
_DASHES = str.maketrans("−–", "--")


@dataclass(frozen=True, slots=True)
class CoordinateHints:
    """Frame / equinox information stated by the typed coordinate text itself."""

    kind: str | None  # "galactic", "equatorial" or None (not stated)
    frame: str | None  # an explicit frame word (icrs, fk5, fk4, galactic)
    equinox: str | None  # e.g. "B1950", "J2000" (from a token or a J/B designation prefix)
    compact: bool  # an IAU-style HHMMSS+-DDMMSS position


def coordinate_hints(text: str) -> CoordinateHints:
    """What the typed coordinate text says about its own frame and equinox."""
    raw = str(text).translate(_DASHES)
    compact = _COMPACT_POSITION_RE.match(raw)
    galactic = bool(_GALACTIC_HINT_RE.search(raw))
    equatorial = bool(_EQUATORIAL_HINT_RE.search(raw)) or compact is not None
    frame_word = _FRAME_WORD_RE.search(raw)
    equinox_word = _EQUINOX_WORD_RE.search(raw)
    equinox = None
    if compact and compact.group(1):
        equinox = "B1950" if compact.group(1).upper() == "B" else "J2000"
    elif equinox_word:
        equinox = equinox_word.group(1).upper() + equinox_word.group(2)
    if galactic and equatorial:
        raise ValueError(f"coordinates {text!r} mix Galactic (l/b) and equatorial (RA/Dec, h:m:s) notation.")
    return CoordinateHints(
        kind="galactic" if galactic else "equatorial" if equatorial else None,
        frame=frame_word.group(1).lower() if frame_word else None,
        equinox=equinox,
        compact=compact is not None,
    )


def check_frame_consistency(text: str, frame: str, equinox: str | None) -> list[str]:
    """Problems between the typed text and a submitted frame/equinox (empty = consistent).

    ICRS and Galactic take no equinox; FK4 goes with Besselian (B) and FK5 with Julian
    (J) equinoxes; l/b labels need the Galactic frame, RA/Dec or h:m:s notation an
    equatorial one; an equinox or frame written in the text must match the submission
    (a J2000 position may be given as ICRS, which agrees with FK5 J2000 to ~0.02 arcsec).
    """
    hints = coordinate_hints(text)
    problems: list[str] = []
    eq = equinox.strip().upper() if equinox else None
    if eq and frame in {"icrs", "galactic"}:
        problems.append(f"frame {frame} takes no equinox (got {equinox!r}): use fk4 for Besselian equinoxes such as "
                        "B1950, fk5 for Julian ones such as J2000, or equinox null.")
    if eq and frame == "fk5" and eq.startswith("B"):
        problems.append(f"equinox {equinox!r} is Besselian: B1950 coordinates are FK4 (frame fk4).")
    if eq and frame == "fk4" and eq.startswith("J"):
        problems.append(f"equinox {equinox!r} is Julian: J2000 coordinates are FK5 or ICRS, not FK4.")
    if hints.kind == "galactic" and frame != "galactic":
        problems.append(f"the text {text!r} labels Galactic coordinates (l/b), but frame is {frame!r}; use frame galactic.")
    if hints.kind == "equatorial" and frame == "galactic":
        problems.append(f"the text {text!r} gives equatorial coordinates (RA/Dec), but frame is galactic.")
    if hints.frame and hints.frame != frame and not (hints.frame == "fk5" and frame == "icrs" and not eq):
        problems.append(f"the text {text!r} says {hints.frame.upper()}, but frame is {frame!r}.")
    if hints.equinox:
        text_eq = hints.equinox
        if text_eq.startswith("B") and frame != "fk4":
            problems.append(f"the text {text!r} gives Besselian equinox {text_eq}: use frame fk4 with equinox {text_eq}.")
        elif text_eq.startswith("J") and frame == "fk4":
            problems.append(f"the text {text!r} gives Julian equinox {text_eq}: use frame fk5 (or icrs), not fk4.")
        elif eq and eq != text_eq:
            problems.append(f"the text {text!r} gives equinox {text_eq}, but equinox {equinox!r} was submitted.")
        elif frame in {"fk4", "fk5"} and eq is None and text_eq not in {"B1950", "J2000"}:
            problems.append(f"the text {text!r} gives equinox {text_eq}: submit it as the equinox.")
    return problems


def parse_user_coordinates(text: str, frame: str, equinox: str | None = None) -> Target:
    """Parse coordinates exactly as a user typed them and transform them to ICRS.

    Accepts decimal degrees (``187.7059 12.3911``), sexagesimal RA/Dec with h/m/s,
    d/m/s or colons (RA in hours unless marked with ``d``/``°``), IAU-style compact
    positions (``J123049.42+122328.0``, ``123049+122328``: RA in hours), labels such as
    ``RA=``/``Dec=``/``l=``/``b=`` and the Greek labels alpha/delta/script-l (U+03B1,
    U+03B4, U+2113), in the ICRS, FK5 (default equinox J2000), FK4 (default B1950) or
    Galactic frame. The frame/equinox must agree with what the text itself states
    (:func:`check_frame_consistency`). RA must lie in [0, 360) degrees or [0, 24) hours:
    a negative or out-of-range RA is rejected, never wrapped. A Galactic longitude may be
    given in [0, 360) or in the signed convention (-180, 0), which means l + 360.
    Conversion is done by astropy ``SkyCoord``; raises ValueError with a readable
    message on any problem.
    """
    frame = frame.lower()
    if frame not in COORDINATE_FRAMES:
        raise ValueError(f"coordinate frame must be one of {', '.join(COORDINATE_FRAMES)}.")
    if equinox is not None and not _EQUINOX_RE.match(equinox.strip().upper()):
        raise ValueError(f"equinox {equinox!r} must look like 'J2000' or 'B1950'.")
    problems = check_frame_consistency(text, frame, equinox)
    if problems:
        raise ValueError(" ".join(problems))
    hints = coordinate_hints(text)
    equinox = equinox.strip().upper() if equinox else hints.equinox
    unit: tuple[Any, Any]
    compact = _COMPACT_POSITION_RE.match(str(text).translate(_DASHES))
    negative_first = False
    if compact:
        _, hh, mm, ss, sign, dd, dm, ds = compact.groups()
        cleaned = f"{hh}h{mm}m{ss}s {sign}{dd}d{dm}m{ds}s"
        unit = (u.hourangle, u.deg)
        numbers = [hh, dd]
    else:
        cleaned = str(text).translate(str.maketrans("−–°′″’”", "--dmsms"))
        cleaned = _COORD_NOISE_RE.sub(" ", _COORD_LABEL_RE.sub(" ", cleaned))
        cleaned = " ".join(cleaned.replace(",", " ").replace(";", " ").replace("=", " ").split())
        numbers = re.findall(r"\d+(?:\.\d+)?", cleaned)
        if len(numbers) < 2:
            raise ValueError(f"could not find two coordinate values in {text!r}.")
        markers = re.search(r"[hms:]|\d\s*d", cleaned, re.IGNORECASE)
        if frame == "galactic" or (len(numbers) == 2 and not markers):
            unit = (u.deg, u.deg)
        else:
            first = cleaned.split()[0]
            unit = (u.deg, u.deg) if re.search(r"\dd", first, re.IGNORECASE) else (u.hourangle, u.deg)
        # The sign of the first coordinate: a negative RA is an error (never wrapped); a
        # negative Galactic longitude (l in (-180, 0), a common convention) is l + 360.
        negative_first = cleaned.startswith("-")
        if negative_first and frame != "galactic":
            raise ValueError(f"RA -{numbers[0]} in {text!r} is negative: RA must lie in [0, 360) degrees or "
                             "[0, 24) hours (nothing is wrapped).")
    first_value = -float(numbers[0]) if negative_first else float(numbers[0])
    if unit[0] is u.deg and frame == "galactic" and -180.0 < first_value < 0.0:
        pass  # l in (-180, 0) is the common signed convention; astropy maps it to l + 360
    elif unit[0] is u.deg and not 0.0 <= first_value < 360.0:
        hint = ("; a compact designation such as J1230+1223 is a truncated name, not a position"
                if "." not in numbers[0] and len(numbers[0]) >= 4 else "")
        raise ValueError(f"longitude/RA {numbers[0]} in {text!r} is outside [0, 360) degrees{hint}.")
    if unit[0] is u.hourangle and not 0.0 <= first_value < 24.0:
        raise ValueError(f"RA {numbers[0]}h in {text!r} is outside [0, 24) hours.")
    frames = {
        "icrs": ICRS(),
        "fk5": FK5(equinox=equinox or "J2000"),
        "fk4": FK4(equinox=equinox or "B1950"),
        "galactic": Galactic(),
    }
    try:
        coord = SkyCoord(cleaned, frame=frames[frame], unit=unit)
        icrs = coord.transform_to(ICRS())
    except (ValueError, TypeError, u.UnitsError) as exc:
        raise ValueError(f"could not parse coordinates {text!r} ({frame}): {exc}") from exc
    return validate_target(float(icrs.ra.deg) % 360.0, float(icrs.dec.deg))


class _AdqlIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    catalog: str
    query: str


class _SubmissionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scope: Literal["cone", "all_sky"]
    target: _TargetIn | None
    radius_arcsec: float
    catalogs: list[str]
    profiles: list[str]
    object_types: list[str]
    spectral_types: list[str]
    search_mode: Literal["cone", "shell", "cylinder"]
    min_radius_arcsec: float
    min_distance_pc: float | None
    max_distance_pc: float | None
    start_year: float | None
    end_year: float | None
    max_results: int | None
    min_confidence: float
    adql: _AdqlIn | None
    plan: list[str]
    explanation: str


class _SubmissionInvalid(Exception):
    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


_ADQL_FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|GRANT|REVOKE|TRUNCATE|MERGE|UPLOAD)\b", re.IGNORECASE)
_ADQL_TOP = re.compile(r"^\s*SELECT\s+(?:ALL\s+|DISTINCT\s+)?TOP\s+(\d+)\b", re.IGNORECASE)


def validate_adql(catalog_name: str, query: str, registry: CatalogRegistry) -> list[str]:
    """Static guards for a model-written ADQL query; returns problems (empty = acceptable).

    Checks: the catalog is an enabled TAP catalog of the registry; a single read-only
    SELECT with ``TOP n`` (n <= ADQL_MAX_TOP); the catalog's own table is read in the
    FROM clause; quotes and parentheses balance. Column existence is checked only by
    the remote verification (``verify_adql``), which asks the service itself.
    """
    tap = tap_catalogs(registry)
    if catalog_name not in tap:
        return [f"ADQL catalog '{catalog_name}' is not a TAP catalog of the registry (valid: {', '.join(tap)})."]
    errors: list[str] = []
    text = query.strip().rstrip(";").strip()
    if ";" in text:
        errors.append("ADQL must be a single statement (no ';').")
    if not re.match(r"^\s*SELECT\b", text, re.IGNORECASE):
        errors.append("ADQL must start with SELECT.")
    if _ADQL_FORBIDDEN.search(re.sub(r"'[^']*'", "''", text)):
        errors.append("ADQL must be read-only (no data-modification keywords).")
    top = _ADQL_TOP.match(text)
    if not top:
        errors.append(f"ADQL must bound its result with SELECT TOP n (n <= {ADQL_MAX_TOP}); ADQL has no LIMIT.")
    elif not 0 < int(top.group(1)) <= ADQL_MAX_TOP:
        errors.append(f"TOP must be between 1 and {ADQL_MAX_TOP}.")
    if re.search(r"\bLIMIT\s+\d+", text, re.IGNORECASE):
        errors.append("ADQL has no LIMIT clause; use SELECT TOP n.")
    base = _base_table(tap[catalog_name])
    from_part = re.split(r"\bFROM\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(from_part) < 2:
        errors.append("ADQL needs a FROM clause.")
    else:
        pattern = re.escape(base) if base.startswith('"') else r"(?<![\w.])" + re.escape(base) + r"(?![\w])"
        if not re.search(pattern, from_part[1], re.IGNORECASE):
            errors.append(f"ADQL for catalog '{catalog_name}' must read its table {base}.")
    if text.count("'") % 2:
        errors.append("ADQL has an unbalanced single quote.")
    if text.count("(") != text.count(")"):
        errors.append("ADQL has unbalanced parentheses.")
    return errors


# A geometry function whose first argument (after an optional frame string) is a number:
# a sky position written by the model.
_ADQL_NUMERIC_GEOMETRY_RE = re.compile(
    r"\b(POINT|CIRCLE|BOX|POLYGON)\s*\(\s*(?:'[^']*'\s*,\s*)?[-+]?\s*(?:\d|\.\d)", re.IGNORECASE
)
_ADQL_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_]+\}")


def adql_position_problems(query: str, scope: str) -> list[str]:
    """Position rules for model-written ADQL (empty list = acceptable).

    Sky coordinates never come from the model: numeric POINT/CIRCLE/BOX/POLYGON
    arguments are rejected; a cone query must constrain its position with
    :data:`TARGET_CIRCLE` (replaced by :func:`substitute_adql_target` with the validated
    target and radius); an all-sky query uses no placeholders.
    """
    problems: list[str] = []
    numeric = _ADQL_NUMERIC_GEOMETRY_RE.search(query)
    if numeric:
        problems.append(f"ADQL must not contain numeric sky positions ({numeric.group(1).upper()}(...) with numbers): "
                        f"use {TARGET_CIRCLE} / {TARGET_POINT}, which the server fills with the validated target.")
    unknown = sorted({p for p in _ADQL_PLACEHOLDER_RE.findall(query)} - {TARGET_CIRCLE, TARGET_POINT})
    if unknown:
        problems.append(f"Unknown ADQL placeholder(s) {', '.join(unknown)}; only {TARGET_CIRCLE} and {TARGET_POINT} "
                        "exist.")
    if scope == "cone" and TARGET_CIRCLE not in query:
        problems.append(f"Cone ADQL must constrain the position with CONTAINS(POINT('ICRS', <ra column>, <dec column>), "
                        f"{TARGET_CIRCLE}) = 1; the server fills in the validated target and radius.")
    if scope == "all_sky" and (TARGET_CIRCLE in query or TARGET_POINT in query):
        problems.append(f"Scope all_sky has no target: {TARGET_CIRCLE}/{TARGET_POINT} cannot be used (use scope cone "
                        "for a search around a position).")
    return problems


def substitute_adql_target(query: str, target: Target, radius_arcsec: float) -> str:
    """Replace the position placeholders with the validated ICRS target and cone radius."""
    point = f"POINT('ICRS', {target.ra:.9f}, {target.dec:.9f})"
    circle = f"CIRCLE('ICRS', {target.ra:.9f}, {target.dec:.9f}, {radius_arcsec / 3600.0:.10f})"
    return query.replace(TARGET_CIRCLE, circle).replace(TARGET_POINT, point)


async def verify_adql_remote(catalog: CatalogDefinition, query: str, client: httpx.AsyncClient) -> str | None:
    """Ask the TAP service to run the query with MAXREC=1; returns its error message or None.

    Network failures raise :class:`UpstreamServiceError` (the query itself may be fine).
    """
    form = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": query.strip().rstrip(";"), "MAXREC": "1"}
    try:
        response = await client.post(str(catalog.endpoint), data=form, timeout=60.0)
    except httpx.HTTPError as exc:
        raise UpstreamServiceError(f"TAP service {catalog.name} unreachable: {exc}") from exc
    if response.status_code >= 500:
        raise UpstreamServiceError(f"TAP service {catalog.name} returned HTTP {response.status_code}.")
    if response.status_code >= 400:
        return f"{catalog.name} TAP rejected the ADQL (HTTP {response.status_code}): {service_error_detail(response)}"
    status, message = votable_query_status(response.content[:200_000])
    if status == "ERROR":
        return f"{catalog.name} TAP rejected the ADQL: {message or 'no message'}"
    # Only a real VOTable result counts as verification: QUERY_STATUS OK/OVERFLOW, or a
    # results RESOURCE. A maintenance page, an empty body or JSON verifies nothing.
    head = response.content[:200_000]
    is_votable = b"<VOTABLE" in head.upper()
    has_results = re.search(rb"<RESOURCE[^>]*type\s*=\s*[\"']results[\"']", head, re.IGNORECASE) is not None
    if not is_votable or not (status in {"OK", "OVERFLOW"} or has_results):
        kind = response.headers.get("content-type", "unknown content type")
        raise UpstreamServiceError(
            f"TAP service {catalog.name} returned HTTP {response.status_code} without a VOTable result ({kind}); "
            "the ADQL could not be verified."
        )
    return None


@dataclass(slots=True)
class CompiledQuery:
    """Validated result of :func:`compile_query`."""

    text: str
    scope: str
    advanced_query: dict[str, Any] | None
    plan: list[str]
    explanation: str
    adql: str | None
    adql_catalog: str | None
    adql_endpoint: str | None
    resolved_objects: list[dict[str, Any]]
    attempts: int
    validation_history: list[list[str]]
    model: str
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolved_summary(resolved: ResolvedObject) -> dict[str, Any]:
    return {
        "query": resolved.query,
        "canonical_name": resolved.canonical_name,
        "ra_deg": resolved.ra_deg,
        "dec_deg": resolved.dec_deg,
        "object_type": resolved.object_type,
        "redshift": resolved.redshift,
        "resolver": resolved.resolver,
        "resolver_name": (resolved.resolver_metadata or {}).get("resolver_name"),
    }


def _is_sesame_service_failure(exc: ObjectResolutionError) -> bool:
    """Transport / HTTP / unparsable-reply failures of Sesame (not 'name unknown').

    providers.SesameResolver reports an unknown name as "Sesame response could not be
    parsed: No coordinates found for object ..." (verified live), so that phrase is a
    user error, while any other parse failure (e.g. an HTML error page) is upstream.
    """
    text = str(exc)
    if text.startswith("Sesame request failed"):
        return True
    return text.startswith("Sesame response could not be parsed") and "No coordinates found" not in text


class _NameCache:
    """Per-compilation Sesame lookups (one request per distinct name).

    Only answers are cached ("resolved" or "unknown name"); a Sesame outage raises
    :class:`UpstreamServiceError` (HTTP 502) instead of being fed back to Claude, so an
    outage can never steer the model towards substitute coordinates.
    """

    def __init__(self, resolver: SesameResolver) -> None:
        self.resolver = resolver
        self._results: dict[str, ResolvedObject | ObjectResolutionError] = {}

    async def resolve(self, name: str) -> ResolvedObject:
        key = " ".join(name.split()).casefold()
        if key not in self._results:
            try:
                self._results[key] = await self.resolver.resolve(name)
            except ObjectResolutionError as exc:
                if _is_sesame_service_failure(exc):
                    raise UpstreamServiceError(f"CDS Sesame name resolution is unavailable: {exc}") from exc
                self._results[key] = exc
        result = self._results[key]
        if isinstance(result, ObjectResolutionError):
            raise result
        return result

    def resolved(self) -> list[ResolvedObject]:
        return [r for r in self._results.values() if isinstance(r, ResolvedObject)]


_REQUEST_BESSELIAN_RE = re.compile(r"(?<![\w.])(?:B1950(?:\.0)?|FK4)(?![\w.])", re.IGNORECASE)
_REQUEST_GALACTIC_RE = re.compile(r"\bgalactic\s+(?:coordinates?|coords?|longitude|latitude|l\b|lon\b)|\b(?:glon|glat)\b",
                                  re.IGNORECASE)


_CLAUSE_BREAK_RE = re.compile(r"[,;()\[\]]|\.\s|\bi\.e\.|\be\.g\.")


def coordinate_context(request: str, coord_text: str) -> str:
    """The words of the request that qualify the copied coordinate text.

    That is the text before it back to the previous clause break (comma, semicolon,
    bracket, sentence end, "i.e."), the text after it up to the next break, and a
    parenthetical that directly follows it when that parenthetical holds no numbers of its
    own (``... +02d19m43s (B1950)``); a parenthetical with its own numbers describes
    another position (``266.41684 -29.00781 (Sgr A*, l=359.944, b=-0.046)``).
    """
    norm_request, norm_coords = _normalize_text(request), _normalize_text(coord_text)
    idx = norm_request.find(norm_coords)
    if idx < 0 or not norm_coords:
        return ""
    before = norm_request[:idx]
    breaks = list(_CLAUSE_BREAK_RE.finditer(before))
    before = before[breaks[-1].end():] if breaks else before
    after = norm_request[idx + len(norm_coords):]
    context_after = ""
    paren = re.match(r"\s*\(([^()]*)\)", after)
    if paren:
        inner = _EQUINOX_WORD_RE.sub(" ", paren.group(1))
        if not re.search(r"\d", inner):
            context_after = paren.group(1)
        after = after[paren.end():]
    stop = _CLAUSE_BREAK_RE.search(after)
    context_after += " " + (after[:stop.start()] if stop else after)
    return f"{before} {context_after}"


def _request_frame_problems(request: str, coord_text: str, frame: str) -> tuple[list[str], list[str]]:
    """Frame statements of the request next to the copied coordinate text: (problems, warnings).

    Only words qualifying the copied text (:func:`coordinate_context`) can demand a frame,
    and only when the text states none itself: RA/Dec labels, h/m/s notation, an equinox
    or a frame word in the text always win. B1950 / FK4 / Galactic mentions elsewhere in
    the request (e.g. "the quasar with B1950 designation 1226+023") only produce a warning.
    """
    problems: list[str] = []
    warnings: list[str] = []
    hints = coordinate_hints(coord_text)
    context = coordinate_context(request, coord_text)
    text_states_equinox = bool(hints.equinox or hints.frame)
    besselian_near = bool(_REQUEST_BESSELIAN_RE.search(context))
    galactic_near = bool(_REQUEST_GALACTIC_RE.search(context))
    if besselian_near and frame != "fk4" and not text_states_equinox and hints.kind != "galactic":
        problems.append("the request calls these coordinates B1950/FK4: use frame fk4 (equinox B1950 unless stated).")
    elif _REQUEST_BESSELIAN_RE.search(request) and frame != "fk4":
        warnings.append(f"The request mentions B1950/FK4 elsewhere; the coordinates {coord_text!r} were read as "
                        f"{frame.upper()}.")
    if galactic_near and frame != "galactic" and hints.kind is None and not text_states_equinox:
        problems.append("the request calls these coordinates Galactic: use frame galactic.")
    elif _REQUEST_GALACTIC_RE.search(request) and frame != "galactic":
        warnings.append(f"The request mentions Galactic coordinates elsewhere; the coordinates {coord_text!r} were "
                        f"read as {frame.upper()}.")
    return problems, warnings


async def _validate_submission(
    raw: Any,
    *,
    text: str,
    registry: CatalogRegistry,
    names: _NameCache,
    settings: AISettings,
    http_client: httpx.AsyncClient,
    verify_adql: bool,
) -> tuple[dict[str, Any] | None, _SubmissionIn, list[str], list[dict[str, Any]], str | None]:
    """Validate one submit_query input; returns (advanced_query, input, warnings, resolved, adql).

    ``adql`` is the submitted ADQL with the position placeholders replaced by the
    validated target (cone scope), ready to run.
    """
    try:
        sub = _SubmissionIn.model_validate(raw)
    except ValidationError as exc:
        raise _SubmissionInvalid(
            [f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}" for err in exc.errors()]
        ) from exc

    errors: list[str] = []
    warnings: list[str] = []
    enabled = registry.enabled_catalogs()
    for name in sub.catalogs:
        if name not in enabled:
            errors.append(f"Unknown catalog: {name} (valid: {', '.join(sorted(enabled))}).")
    valid_profiles = set(registry_profiles(registry))
    for profile in sub.profiles:
        if profile not in valid_profiles:
            errors.append(f"Unknown profile: {profile} (valid: {', '.join(sorted(valid_profiles))}).")
    if not [p for p in sub.plan if p.strip()]:
        errors.append("plan must contain at least one step.")
    if not sub.explanation.strip():
        errors.append("explanation must not be empty.")
    if any(not t.strip() for t in sub.object_types + sub.spectral_types):
        errors.append("object_types and spectral_types must not contain empty strings.")

    if sub.adql is not None:
        errors.extend(validate_adql(sub.adql.catalog, sub.adql.query, registry))
        errors.extend(adql_position_problems(sub.adql.query, sub.scope))

    resolved_list: list[dict[str, Any]] = []
    advanced: dict[str, Any] | None = None
    adql_final = sub.adql.query.strip().rstrip(";").strip() if sub.adql is not None else None
    if sub.scope == "all_sky":
        if sub.target is not None:
            errors.append("scope all_sky must have target null (use scope cone for a search around an object).")
        if sub.adql is None:
            errors.append("scope all_sky requires an adql query for one TAP catalog.")
        # Scope all_sky runs only the ADQL: crossmatch filters would be silently ignored.
        ignored = [label for label, unset in (
            ("object_types", not sub.object_types), ("spectral_types", not sub.spectral_types),
            ("search_mode", sub.search_mode == "cone"), ("min_radius_arcsec", sub.min_radius_arcsec == 0),
            ("min_distance_pc", sub.min_distance_pc is None), ("max_distance_pc", sub.max_distance_pc is None),
            ("start_year", sub.start_year is None), ("end_year", sub.end_year is None),
            ("max_results", sub.max_results is None),
        ) if not unset]
        if ignored:
            errors.append(f"scope all_sky executes only the ADQL, so {', '.join(ignored)} would be ignored: express "
                          "these constraints in the ADQL WHERE clause and reset them (empty lists, null, 0, "
                          "search_mode cone).")
    else:
        target: Target | None = None
        target_source = None
        typed: Target | None = None
        coords = sub.target.coordinates if sub.target is not None else None
        if coords is not None:
            if _normalize_text(coords.text) not in _normalize_text(text) or not coords.text.strip():
                errors.append(f"target.coordinates.text {coords.text!r} does not appear in the request; copy "
                              "coordinates verbatim from the request, and only if the user typed coordinates.")
            else:
                try:
                    request_problems, request_warnings = _request_frame_problems(text, coords.text, coords.frame)
                    warnings.extend(request_warnings)
                    if request_problems:
                        errors.extend(f"target.coordinates: {p}" for p in request_problems)
                    else:
                        typed = parse_user_coordinates(coords.text, coords.frame, coords.equinox)
                except (ValueError, InvalidCoordinateError) as exc:
                    errors.append(f"target.coordinates: {exc}")
        name = (sub.target.name or "").strip() if sub.target is not None else ""
        if sub.target is None or (not name and coords is None):
            errors.append("scope cone requires target.name, or target.coordinates copied from the request.")
        elif name:
            try:
                resolved = await names.resolve(name)
            except ObjectResolutionError as exc:
                errors.append(f"target.name '{name}' could not be resolved by CDS Sesame ({exc}); use a standard "
                              "designation of the object the user named (e.g. 'M 87', 'NGC 4151', '3C 273'). Give "
                              "target.coordinates only if the user typed coordinates.")
            else:
                target = resolved_target(resolved)
                target_source = "sesame"
                resolved_list.append(_resolved_summary(resolved))
                if typed is not None:
                    sep = haversine_arcsec(target.ra, target.dec, typed.ra, typed.dec)
                    if sep > max(sub.radius_arcsec, 60.0):
                        errors.append(
                            f"the typed coordinates {coords.text!r} ({coords.frame}) are {sep:.0f} arcsec from Sesame's "  # type: ignore[union-attr]
                            f"position of '{name}'; give the name only, or only the coordinates."
                        )
        elif typed is not None:
            target = typed
            target_source = "user_coordinates"
        if not math.isfinite(sub.radius_arcsec) or not 0 < sub.radius_arcsec <= MAX_CONE_RADIUS_ARCSEC:
            errors.append(f"radius_arcsec must be in (0, {MAX_CONE_RADIUS_ARCSEC:g}].")
        # Mode-specific fields: crossmatch.AdvancedQuery applies min_radius only in shell
        # mode and distance bounds only in cylinder mode, so elsewhere they would be ignored.
        if sub.search_mode != "cylinder" and (sub.min_distance_pc is not None or sub.max_distance_pc is not None):
            errors.append(f"min_distance_pc/max_distance_pc only act in search_mode cylinder (got {sub.search_mode!r}): "
                          "use search_mode cylinder, or set both to null.")
        if sub.search_mode != "shell" and sub.min_radius_arcsec != 0:
            errors.append(f"min_radius_arcsec only acts in search_mode shell (got {sub.search_mode!r}): use search_mode "
                          "shell, or set it to 0.")
        if sub.search_mode == "shell" and not sub.min_radius_arcsec > 0:
            errors.append("search_mode shell needs min_radius_arcsec > 0 (the inner radius of the annulus).")
        selected = selected_catalogs(registry, sub.catalogs, sub.profiles)
        type_filters: list[tuple[str, list[str], str, str]] = [
            ("object_types", sub.object_types, "object_type", "the type"),
            ("spectral_types", sub.spectral_types, "spectral_type", "the type"),
        ]
        if sub.search_mode == "cylinder":
            type_filters.append(("search_mode cylinder", ["distance bounds"], "parallax", "the distance"))
        untyped_labels: set[str] = set()
        for label, values, canonical, what in type_filters:
            if not values:
                continue
            untyped = [c for c in selected if not _catalog_reports(enabled[c], canonical)]
            if untyped:
                untyped_labels.add(label)
                errors.append(
                    f"{label} {values} would remove every row of {', '.join(untyped)}: those catalogs report no "
                    f"{canonical.replace('_', ' ')}, and the filter applies to every catalog row. Remove it (and say "
                    f"in the plan that the crossmatch groups are inspected for {what}, or use scope all_sky ADQL), "
                    f"or select only {', '.join(typed_catalogs(registry, canonical))}."
                )
        object_types = list(sub.object_types)
        if sub.object_types and "object_types" not in untyped_labels:
            tree = await simbad_otype_tree(http_client)
            object_types, type_errors = expand_object_types(sub.object_types, selected, tree)
            errors.extend(type_errors)
        if target is not None and not errors:
            time_period = None
            if sub.start_year is not None or sub.end_year is not None:
                time_period = {"start_year": sub.start_year, "end_year": sub.end_year}
            payload = {
                "target": target.as_dict(),
                "radius_arcsec": sub.radius_arcsec,
                "profiles": sub.profiles,
                "object_types": object_types,
                "spectral_types": sub.spectral_types,
                "min_confidence": sub.min_confidence,
                "max_results": sub.max_results,
                "min_radius_arcsec": sub.min_radius_arcsec,
                "search_mode": sub.search_mode,
                "min_distance_pc": sub.min_distance_pc,
                "max_distance_pc": sub.max_distance_pc,
                "time_period": time_period,
                "catalogs": sub.catalogs,
                "use_resolved_name": target_source == "sesame",
                "resolved_name": resolved_list[0]["canonical_name"] if resolved_list else None,
                "metadata": {
                    "compiled_by": "astrosearch.ai",
                    "model": settings.model,
                    "request_text": text,
                    "target_source": target_source,
                    "object_types_requested": list(sub.object_types),
                    "user_coordinates": ({"text": coords.text, "frame": coords.frame, "equinox": coords.equinox}
                                         if coords is not None else None),
                },
            }
            try:
                query = AdvancedQuery.from_dict(payload)
                QueryValidator.validate(query, registry)
            except (ValueError, TypeError, InvalidCoordinateError) as exc:
                errors.append(f"AdvancedQuery validation failed: {exc}")
            else:
                if query.catalogs and query.profiles:
                    selected = [c for c in query.catalogs if any(p in enabled[c].profiles for p in query.profiles)]
                    if not selected:
                        errors.append("No listed catalog carries any listed profile, so nothing would be queried; "
                                      "drop profiles or pick matching catalogs.")
                advanced = query.to_dict()
                if adql_final is not None:
                    adql_final = substitute_adql_target(adql_final, target, sub.radius_arcsec)

    if not errors and sub.adql is not None and adql_final is not None and verify_adql:
        catalog = tap_catalogs(registry)[sub.adql.catalog]
        try:
            problem = await verify_adql_remote(catalog, adql_final, http_client)
        except UpstreamServiceError as exc:
            warnings.append(f"ADQL could not be verified remotely: {exc}")
        else:
            if problem:
                errors.append(problem)
    if errors:
        raise _SubmissionInvalid(errors)
    return advanced, sub, warnings, resolved_list, adql_final


# ---------------------------------------------------------------------------
# Query Compilation: Claude Tool Loop
# ---------------------------------------------------------------------------


def _tool_result(tool_use_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        result["is_error"] = True
    return result


def check_query_request(text: str, max_retries: int = 2) -> str:
    """Validate a compile request before any credentials or network are used; returns the
    whitespace-normalised text (ValueError on bad input)."""
    request_text = " ".join(str(text).split())
    if len(request_text) < 3:
        raise ValueError("text must contain a request of at least 3 characters.")
    if len(request_text) > 2000:
        raise ValueError("text must be at most 2000 characters.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be an integer between 0 and 2.")
    return request_text


async def compile_query(
    text: str,
    *,
    anthropic_client: Any,
    registry: CatalogRegistry,
    http_client: httpx.AsyncClient,
    settings: AISettings | None = None,
    max_retries: int = 2,
    verify_adql: bool = False,
    resolver: SesameResolver | None = None,
) -> CompiledQuery:
    """Compile a natural-language request into a validated AdvancedQuery (+ plan, ADQL).

    Claude answers through strict tools; each ``submit_query`` input is validated and
    rejected submissions are returned as ``is_error`` tool results. ``max_retries`` (0-2)
    bounds the number of corrections after the first attempt. Raises
    :class:`AIQueryCompilationError` when no valid query results.
    """
    request_text = check_query_request(text, max_retries)
    active = settings or AISettings.from_env()
    names = _NameCache(resolver or SesameResolver(http_client))
    tools = build_query_tools(registry)
    system = [{"type": "text", "text": _query_system_prompt(registry), "cache_control": {"type": "ephemeral"}}]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Compile this request into a search:\n<request>\n{request_text}\n</request>"}
    ]
    failures = 0
    attempts = 0
    history: list[list[str]] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = await _create_message(
            anthropic_client, active, system=system, tools=tools, tool_choice={"type": "auto"}, messages=messages
        )
        tool_uses = [b for b in _blocks(response) if getattr(b, "type", None) == "tool_use"]
        messages.append({"role": "assistant", "content": _blocks(response)})
        if not tool_uses:
            if getattr(response, "stop_reason", None) == "max_tokens":
                raise AIUpstreamError("Claude's answer was truncated (max_tokens) before a query was submitted.")
            failures += 1
            history.append(["No submit_query call was made."])
            if failures > max_retries:
                raise AIQueryCompilationError("Claude did not submit a query.", history[-1], history)
            messages.append({"role": "user", "content": "Call the submit_query tool with the compiled search now."})
            continue

        results: list[dict[str, Any]] = []
        accepted: CompiledQuery | None = None
        for block in tool_uses:
            tool_input = block.input if isinstance(block.input, Mapping) else {}
            if block.name == "resolve_object":
                name = str(tool_input.get("name", "")).strip()
                try:
                    resolved = await names.resolve(name)
                except ObjectResolutionError as exc:
                    results.append(_tool_result(block.id, f"Could not resolve '{name}': {exc}", is_error=True))
                else:
                    results.append(_tool_result(block.id, json.dumps(_resolved_summary(resolved))))
            elif block.name == "submit_query":
                attempts += 1
                try:
                    advanced, sub, warnings, resolved_list, adql_final = await _validate_submission(
                        dict(tool_input), text=request_text, registry=registry, names=names, settings=active,
                        http_client=http_client, verify_adql=verify_adql,
                    )
                except _SubmissionInvalid as exc:
                    failures += 1
                    history.append(exc.errors)
                    results.append(_tool_result(
                        block.id,
                        "Validation failed:\n- " + "\n- ".join(exc.errors) + "\nFix these problems and call submit_query again.",
                        is_error=True,
                    ))
                    continue
                adql_catalog = sub.adql.catalog if sub.adql else None
                accepted = CompiledQuery(
                    text=request_text,
                    scope=sub.scope,
                    advanced_query=advanced,
                    plan=[p.strip() for p in sub.plan if p.strip()],
                    explanation=sub.explanation.strip(),
                    adql=adql_final,
                    adql_catalog=adql_catalog,
                    adql_endpoint=tap_catalogs(registry)[adql_catalog].endpoint if adql_catalog else None,
                    resolved_objects=resolved_list or [_resolved_summary(r) for r in names.resolved()],
                    attempts=attempts,
                    validation_history=history,
                    model=str(getattr(response, "model", None) or active.model),
                    warnings=warnings,
                )
                results.append(_tool_result(block.id, "Accepted."))
            else:
                results.append(_tool_result(block.id, f"Unknown tool '{block.name}'.", is_error=True))
        if accepted is not None:
            return accepted
        if failures > max_retries:
            last = history[-1] if history else []
            raise AIQueryCompilationError(
                f"Claude could not produce a valid query after {attempts} attempt(s): " + "; ".join(last), last, history
            )
        messages.append({"role": "user", "content": results})
    raise AIQueryCompilationError("Query compilation did not converge.", history[-1] if history else [], history)


def _check_row_limit(row_limit: Any) -> int:
    if isinstance(row_limit, bool) or not isinstance(row_limit, int) or not 1 <= row_limit <= ADQL_MAX_TOP:
        raise ValueError(f"row limit must be an integer between 1 and {ADQL_MAX_TOP} (got {row_limit!r}).")
    return row_limit


async def execute_compiled_query(
    compiled: CompiledQuery,
    *,
    service: CrossmatchLike | None = None,
    http_client: httpx.AsyncClient | None = None,
    registry: CatalogRegistry | None = None,
    row_limit: int = 200,
) -> dict[str, Any]:
    """Run a compiled query: the crossmatch for cone scope, else its ADQL (MAXREC-bounded).

    Returns ``{"kind": "crossmatch", "record": {...}}`` or ``{"kind": "adql",
    "catalog", "endpoint", "columns", "rows", "truncated"}``. ValueError when row_limit is not
    in [1, ADQL_MAX_TOP].
    """
    _check_row_limit(row_limit)
    if compiled.advanced_query is not None:
        if service is None:
            raise ValueError("A crossmatch service is required to run a cone query.")
        query = AdvancedQuery.from_dict(compiled.advanced_query)
        record = await service.crossmatch(query.target.ra, query.target.dec, query=query)
        return {"kind": "crossmatch", "record": record.as_dict()}
    if not compiled.adql or not compiled.adql_catalog or http_client is None:
        raise ValueError("Nothing to execute: no cone query and no ADQL (or no HTTP client).")
    reg = registry or CatalogRegistry(Settings().catalog_registry_path)
    catalog = tap_catalogs(reg).get(compiled.adql_catalog)
    if catalog is None:
        raise ValueError(f"ADQL catalog {compiled.adql_catalog} is not a TAP catalog of the registry.")
    form = {"REQUEST": "doQuery", "LANG": "ADQL", "QUERY": compiled.adql, "MAXREC": str(row_limit)}
    try:
        response = await http_client.post(str(catalog.endpoint), data=form, timeout=180.0)
    except httpx.HTTPError as exc:
        raise UpstreamServiceError(f"TAP service {catalog.name} unreachable: {exc}") from exc
    if response.status_code >= 400:
        raise UpstreamServiceError(
            f"TAP service {catalog.name} returned HTTP {response.status_code}: {service_error_detail(response)}"
        )
    try:
        table = parse_votable_table(response.content)
    except (CatalogQueryError, ResponseParseError) as exc:
        raise UpstreamServiceError(f"TAP service {catalog.name}: {exc}") from exc
    return {
        "kind": "adql",
        "catalog": catalog.name,
        "endpoint": catalog.endpoint,
        "columns": [c.as_dict() for c in table.columns],
        "rows": table.rows[:row_limit],
        "truncated": table.truncated,
    }


# ---------------------------------------------------------------------------
# Explanation: Sources & Citations
# ---------------------------------------------------------------------------


def ads_url(bibcode: str) -> str:
    return ADS_ABS_URL.format(bibcode=quote(bibcode, safe=""))


def bibcode_year(bibcode: str | None) -> int | None:
    if bibcode and bibcode[:4].isdigit():
        return int(bibcode[:4])
    return None


@dataclass(slots=True)
class Source:
    """A citable source: a paper (bibcode), a catalog, or a database."""

    n: int
    kind: str
    bibcode: str | None = None
    title: str | None = None
    year: int | None = None
    url: str | None = None
    label: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceBook:
    """Numbered, de-duplicated list of sources; facts reference them by ``n``."""

    def __init__(self) -> None:
        self.sources: list[Source] = []
        self._index: dict[str, Source] = {}

    def add(self, *, kind: str, bibcode: str | None = None, title: str | None = None, year: int | None = None,
            label: str | None = None, url: str | None = None) -> int:
        bibcode = (bibcode or "").strip() or None
        key = bibcode or f"label:{label or title}"
        existing = self._index.get(key)
        if existing is not None:
            existing.title = existing.title or title
            existing.year = existing.year or year
            existing.label = existing.label or label
            return existing.n
        source = Source(
            n=len(self.sources) + 1,
            kind=kind,
            bibcode=bibcode,
            title=title,
            year=year if year is not None else bibcode_year(bibcode),
            url=ads_url(bibcode) if bibcode else url,
            label=label,
        )
        self.sources.append(source)
        self._index[key] = source
        return source.n

    def get(self, n: int) -> Source | None:
        return self.sources[n - 1] if 0 < n <= len(self.sources) else None

    def bibcodes_missing_titles(self) -> list[str]:
        return [s.bibcode for s in self.sources if s.bibcode and not s.title]

    def fill(self, details: Mapping[str, Mapping[str, Any]]) -> None:
        for source in self.sources:
            info = details.get(source.bibcode or "")
            if info:
                source.title = source.title or info.get("title")
                source.year = source.year or info.get("year")


def _expand_citation_group(group: str) -> tuple[list[int], list[str]]:
    """Numbers of one marker group (``2, 5``, ``2-4``, ``1; 3 and 6``); invalid ranges -> problems."""
    numbers: list[int] = []
    problems: list[str] = []
    for part in re.split(_CITE_SEP, group.strip(), flags=re.IGNORECASE):
        part = part.replace("#", "").strip()
        if not part:
            continue
        bounds = [int(x) for x in re.split(r"\s*[" + _CITE_DASHES + r"]\s*", part)]
        if len(bounds) == 1:
            numbers.append(bounds[0])
        elif bounds[0] <= bounds[1] and bounds[1] - bounds[0] < MAX_CITATION_RANGE:
            numbers.extend(range(bounds[0], bounds[1] + 1))
        else:
            problems.append(part)
    return numbers, problems


def map_citations(summary: str, sources: Sequence[Source]) -> tuple[str, list[Source], list[str]]:
    """Renumber citation markers by first appearance and drop unknown ones.

    Markers may be single (``[3]``), lists (``[2, 5]``, ``[1; 4]``, ``[1, 2, and 3]``,
    ``[1, 2 & 3]``), ranges (``[2-4]``, ``[2–4]``, ``[2—4]``, expanded before
    renumbering) and may carry a ``ref``/``source``/``#`` prefix (``[ref 2]``,
    ``[#2]``) or a stray trailing separator (``[2,]``). A group keeps its valid members;
    an empty group is removed. Bracketed text that can only be a citation but is not
    understood (integers with citation words/separators) is removed and reported, so
    no stale, un-renumbered number ever sits next to renumbered ones; other bracketed
    text with digits (``[3.6]``, a band) is kept and reported.
    Returns (text, cited sources in new order with renumbered ``n``, warnings).
    """
    by_n = {s.n: s for s in sources}
    order: dict[int, int] = {}
    dropped: list[int] = []
    bad_ranges: list[str] = []

    def replace(match: re.Match[str]) -> str:
        kept: list[int] = []
        numbers, problems = _expand_citation_group(match.group(1))
        bad_ranges.extend(problems)
        for number in numbers:
            if number not in by_n:
                dropped.append(number)
                continue
            if number not in order:
                order[number] = len(order) + 1
            if order[number] not in kept:
                kept.append(order[number])
        return "[" + ", ".join(str(k) for k in kept) + "]" if kept else ""

    unparsed: list[str] = []

    def replace_or_strip(match: re.Match[str]) -> str:
        if _CITE_RE.fullmatch(match.group(0)):
            return replace(_CITE_RE.fullmatch(match.group(0)))  # type: ignore[arg-type]
        if _is_citation_like(match.group(0)):
            unparsed.append(match.group(0))
            return ""
        return match.group(0)

    # One pass over every bracketed group with digits: understood markers are renumbered,
    # citation-like leftovers removed, anything else (e.g. "[3.6]") kept.
    text = _BRACKET_DIGITS_RE.sub(replace_or_strip, summary)
    text = re.sub(r"[ \t]+([.,;:])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    cited = [Source(**{**by_n[old].as_dict(), "n": new}) for old, new in sorted(order.items(), key=lambda kv: kv[1])]
    warnings = [f"Removed citation [{n}]: it matches no provided source." for n in sorted(set(dropped))]
    warnings += [f"Removed citation range [{r}]: not a valid ascending range." for r in bad_ranges]
    warnings += [f"Removed unparseable citation marker {m}: it could not be mapped to a source." for m in unparsed]
    output_markers = {m.group(0) for m in _CITE_RE.finditer(text)}
    leftovers = [m.group(0) for m in _BRACKET_DIGITS_RE.finditer(text) if m.group(0) not in output_markers]
    warnings += [f"Unrecognised bracketed text {lo} was left in place; it is not a mapped citation." for lo in leftovers]
    if not cited:
        warnings.append("The explanation cites no sources.")
    return text, cited, warnings


_NUMBER_RE = re.compile(r"(?<![A-Za-z\d.\[])[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\d])")
# A number directly followed by one of these is a value with a unit ("0.99c", "5GHz",
# "12kpc") and is checked; followed by other letters it is part of a designation
# ("2MASS", "3C", "4XMM") and is skipped.
_UNIT_SUFFIX_RE = re.compile(
    r"(?:GHz|MHz|kHz|THz|Hz|keV|MeV|GeV|TeV|eV|Gpc|Mpc|kpc|pc|AU|au|ly|Gyr|Myr|kyr|yr|mas|arcsec|arcmin|deg|mag|"
    r"mJy|µJy|uJy|Jy|km|K|c|x|%)(?![A-Za-z])"
)
# Bookkeeping and naming fields of the facts payload: their digits are not facts (source
# numbers, database ids, bibcodes, titles, designations). Designations and titles are
# removed from the text before it is checked instead.
_NON_FACT_KEYS = frozenset({
    "n", "ref", "simbad_oid", "oid", "bibcode", "doi", "journal", "title", "label", "kind", "identifiers", "main_id",
    "canonical_name", "source_id", "name", "query", "resolver", "catalog", "selection", "error_refs", "zero_point_ref",
    "nearest_row_source_id", "otype_path",
})
_TEXT_NAME_KEYS = frozenset({"title", "identifiers", "main_id", "canonical_name", "source_id", "name", "catalog",
                             "label", "bibcode", "nearest_row_source_id"})


def _collect_numbers(value: Any, out: list[float], key: str | None = None) -> None:
    if key in _NON_FACT_KEYS or isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            out.append(float(value))
    elif isinstance(value, str):
        for token in re.findall(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value):
            out.append(float(token))
    elif isinstance(value, Mapping):
        for item_key, item in value.items():
            _collect_numbers(item, out, str(item_key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_numbers(item, out, key)


def _collect_names(value: Any, out: set[str], key: str | None = None) -> None:
    """Designations, titles and catalog names found in the facts (to blank out of the text)."""
    if isinstance(value, str):
        if key in _TEXT_NAME_KEYS and len(value.strip()) >= 2:
            out.add(" ".join(value.split()))
    elif isinstance(value, Mapping):
        for item_key, item in value.items():
            _collect_names(item, out, str(item_key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_names(item, out, key)


def _blank_names(text: str, names: Iterable[str]) -> str:
    for name in sorted(names, key=len, reverse=True):
        pattern = r"\s+".join(re.escape(part) for part in name.split())
        text = re.sub(r"(?<![\w])" + pattern + r"(?![\w])", " ", text, flags=re.IGNORECASE)
    return text


def _matches(x: float, token: str, v: float) -> bool:
    """True when ``token`` (value ``x``) is ``v`` written with the token's precision.

    Signs are ignored (hyphenated ranges read as negatives). Precision is the number of
    decimals; an integer ending in zeros (``780``) may also be ``v`` rounded to its
    significant figures; exponent notation must agree to 0.5 %.
    """
    x, v = abs(x), abs(v)
    lowered = token.lower()
    if "e" in lowered:
        return v > 0 and abs(x - v) <= 5e-3 * v
    decimals = len(lowered.split(".", 1)[1]) if "." in lowered else 0
    if abs(x - round(v, decimals)) <= 0.5 * 10 ** (-decimals - 6):
        return True
    digits = lowered.lstrip("+-−")
    if decimals == 0 and x > 0 and v >= 1 and digits.endswith("0"):
        significant = len(digits.rstrip("0"))
        magnitude = math.floor(math.log10(v))
        return abs(round(v, significant - 1 - magnitude) - x) < 1e-9
    return False


def unverified_numbers(summary: str, facts: Any) -> list[str]:
    """Numbers in ``summary`` that do not equal (after rounding) any fact value in ``facts``.

    Only fact values count: source numbers (``n``/``ref``), database ids, bibcodes,
    titles and identifiers are excluded from the pool; instead, designations and titles
    that appear in the text (e.g. "3C 273", a quoted paper title) are blanked out
    before checking, as are citation markers. Numbers with a unit suffix ("0.99c",
    "5GHz") are checked; digits that start a designation ("2MASS") are not. A
    heuristic, reported as a warning.
    """
    pool: list[float] = []
    _collect_numbers(facts, pool)
    names: set[str] = set()
    _collect_names(facts, names)
    text = _blank_names(_CITE_RE.sub(" ", summary), names)
    missing: list[str] = []
    for match in _NUMBER_RE.finditer(text):
        rest = text[match.end():]
        if rest[:1].isalpha() and not _UNIT_SUFFIX_RE.match(rest):
            continue  # designation prefix such as 2MASS / 3C / 4XMM
        if rest[:1] == "." and rest[1:2].isdigit():
            continue
        token = match.group(0).replace("−", "-")
        try:
            x = float(token.replace(",", ""))
        except ValueError:
            continue
        if not any(_matches(x, token.replace(",", ""), v) for v in pool):
            missing.append(match.group(0))
    return sorted(set(missing), key=missing.index)


# ---------------------------------------------------------------------------
# Explanation: Fact Gathering (SIMBAD TAP, crossmatch, ADS)
# ---------------------------------------------------------------------------


async def _tap_rows(client: httpx.AsyncClient, endpoint: str, adql: str, *, timeout: float = 60.0) -> list[dict[str, Any]]:
    """Run a synchronous ADQL query (JSON output) and return named rows."""
    form = {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": adql}
    try:
        response = await client.post(endpoint, data=form, timeout=timeout)
    except httpx.HTTPError as exc:
        raise UpstreamServiceError(f"SIMBAD TAP unreachable: {exc!r}") from exc
    if response.status_code >= 400:
        raise UpstreamServiceError(f"SIMBAD TAP returned HTTP {response.status_code}: {service_error_detail(response)}")
    try:
        return parse_json_table(response.content).rows
    except (ResponseParseError, ValueError) as exc:
        raise UpstreamServiceError(f"SIMBAD TAP response could not be parsed: {exc}") from exc


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


async def _gather_or_cancel(*aws: Awaitable[Any]) -> list[Any]:
    """Like ``asyncio.gather`` but the first failure cancels (and awaits) its siblings.

    Used for the fan-out of SIMBAD TAP queries: during an outage no orphaned requests
    keep running against the service (or against a client that is being closed).
    """
    tasks = [asyncio.ensure_future(a) for a in aws]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
        return [task.result() for task in tasks]
    finally:
        pending = [t for t in tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


# SIMBAD has_ref.ref_flag bits (verified against live records, e.g. 3C 273 / Proxima Cen:
# papers naming the object in their title carry bit 1, in their abstract bit 2; bit 128
# marks a mention in the body text only). ADQL 2.0 has no bitwise operators, so
# "title or abstract" is MOD(ref_flag, 4) > 0.
_IN_TITLE_OR_ABSTRACT = "MOD(h.ref_flag, 4) > 0"


def _ref_query(oid: int, limit: int, order: str, where: str = "") -> str:
    return (
        f'SELECT TOP {int(limit)} r.bibcode AS bibcode, r.title AS title, r."year" AS pub_year, r.journal AS journal, '
        f"r.nbobject AS nbobject, r.doi AS doi, h.ref_flag AS ref_flag, h.obj_freq AS obj_freq "
        f"FROM ref AS r JOIN has_ref AS h ON h.oidbibref = r.oidbib "
        f"WHERE h.oidref = {int(oid)}{where} ORDER BY {order}"
    )


def _ref_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    flag = _num(row.get("ref_flag"))
    freq = _num(row.get("obj_freq"))
    return {
        "bibcode": str(row.get("bibcode") or "").strip(),
        "title": row.get("title"),
        "year": int(row["pub_year"]) if row.get("pub_year") is not None else None,
        "journal": (str(row["journal"]).strip() if row.get("journal") else None),
        "nbobject": row.get("nbobject"),
        "doi": row.get("doi"),
        "object_in_title": bool(int(flag) & 1) if flag is not None else None,
        "object_in_abstract": bool(int(flag) & 2) if flag is not None else None,
        "simbad_obj_freq": int(freq) if freq is not None else None,
    }


async def fetch_simbad_bibliography(
    client: httpx.AsyncClient, oid: int, *, limit: int = 8, endpoint: str = SIMBAD_TAP
) -> dict[str, list[dict[str, Any]]]:
    """SIMBAD references for one object (tables ``ref`` + ``has_ref``).

    SIMBAD holds no citation counts, so three reproducible selections are returned, all
    restricted to papers that are ABOUT the object rather than passing mentions:

    * ``recent`` - newest papers naming the object in their title or abstract
      (``has_ref.ref_flag`` bits 1/2);
    * ``foundational`` - earliest papers studying at most 5 objects that name it in title
      or abstract (or predate SIMBAD's ref_flag classification, flag null);
    * ``focused`` - papers naming it in title or abstract, ranked by SIMBAD's per-paper
      object frequency ``has_ref.obj_freq`` (how often the object occurs in the paper),
      then newest first, excluding papers already listed as recent or foundational.

    Ties are broken by bibcode so results are deterministic. The three queries run
    concurrently; the first failure cancels the others.
    """
    about = f" AND {_IN_TITLE_OR_ABSTRACT}"
    focused, recent, foundational = await _gather_or_cancel(
        _tap_rows(client, endpoint, _ref_query(
            oid, 3 * limit, "obj_freq DESC, pub_year DESC, bibcode ASC", about + " AND h.obj_freq IS NOT NULL")),
        _tap_rows(client, endpoint, _ref_query(oid, limit, "pub_year DESC, bibcode ASC", about)),
        _tap_rows(client, endpoint, _ref_query(
            oid, limit, "pub_year ASC, nbobject ASC, bibcode ASC",
            f" AND r.nbobject <= 5 AND (h.ref_flag IS NULL OR {_IN_TITLE_OR_ABSTRACT})")),
    )
    recent_entries = [_ref_entry(r) for r in recent]
    foundational_entries = [_ref_entry(r) for r in foundational]
    listed = {e["bibcode"] for e in recent_entries + foundational_entries}
    return {
        "focused": [e for e in (_ref_entry(r) for r in focused) if e["bibcode"] not in listed][:limit],
        "recent": recent_entries,
        "foundational": foundational_entries,
    }


async def fetch_ads_most_cited(
    client: httpx.AsyncClient, token: str, object_name: str, *, rows: int = 5
) -> list[dict[str, Any]]:
    """Most-cited ADS papers about an object (``object:"name"``, ``sort=citation_count desc``).

    ADS search API: GET /v1/search/query with ``q``, ``fl``, ``rows``, ``sort`` and a
    Bearer token (adsabs-dev-api documentation).
    """
    params = {
        "q": 'object:"' + object_name.replace('"', " ") + '"',
        "fl": "bibcode,title,year,citation_count",
        "rows": str(int(rows)),
        "sort": "citation_count desc",
    }
    try:
        response = await client.get(ADS_SEARCH_URL, params=params, headers={"Authorization": f"Bearer {token}"}, timeout=30.0)
    except httpx.HTTPError as exc:
        raise UpstreamServiceError(f"NASA ADS unreachable: {exc!r}") from exc
    if response.status_code >= 400:
        raise UpstreamServiceError(f"NASA ADS returned HTTP {response.status_code}: {service_error_detail(response)}")
    try:
        docs = response.json()["response"]["docs"]
    except (ValueError, KeyError, TypeError) as exc:
        raise UpstreamServiceError(f"NASA ADS response could not be parsed: {exc}") from exc
    out = []
    for doc in docs:
        title = doc.get("title")
        out.append({
            "bibcode": doc.get("bibcode"),
            "title": title[0] if isinstance(title, list) and title else title,
            "year": int(doc["year"]) if str(doc.get("year", "")).isdigit() else None,
            "citation_count": doc.get("citation_count"),
        })
    return [d for d in out if d["bibcode"]]


@dataclass(slots=True)
class ObjectFacts:
    """Everything Claude may use to explain an object; facts cite ``sources`` by ``ref``."""

    query: dict[str, Any]
    object: dict[str, Any]
    identity: dict[str, Any] = field(default_factory=dict)
    measurements: list[dict[str, Any]] = field(default_factory=list)
    photometry: list[dict[str, Any]] = field(default_factory=list)
    distances: list[dict[str, Any]] = field(default_factory=list)
    derived: list[dict[str, Any]] = field(default_factory=list)
    crossmatch: dict[str, Any] | None = None
    bibliography: dict[str, Any] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["sources"] = [s.as_dict() for s in self.sources]
        return data

    def prompt_payload(self) -> dict[str, Any]:
        """Facts as sent to Claude (no URLs; sources as ``{n, bibcode, title, year}``)."""
        data = self.as_dict()
        data.pop("warnings", None)
        data["sources"] = [
            {"n": s.n, "kind": s.kind, "bibcode": s.bibcode, "title": s.title, "year": s.year, "label": s.label}
            for s in self.sources
        ]
        return data


_FLUX_SYSTEMS = {"V": "Vega", "A": "AB"}
# Columns never reported as measurements: positions, match/cone distances computed by
# the service (models.COMPUTED_COLUMNS), times/epochs, flags, keys and row ids.
_SKIP_FIELD = re.compile(r"(^|_)(ra|dec|raj2000|dej2000|ramean|decmean|glon|glat)($|_)|^(ra|dec)(mean)?(err)?$|"
                         r"corr|mjd|(^|_)jd|jdate|epoch|time|flag|flg|qual|key|pixel|^objid$|^srcid$|^objname$",
                         re.IGNORECASE)
_ERROR_FIELD = re.compile(r"err|^e_|unc|_lo$|_hi$|sig(?!nif)", re.IGNORECASE)


def _error_base_names(key: str) -> set[str]:
    """Value-column names an error column may belong to (flux_20_cm_error -> flux_20_cm)."""
    bases = {
        re.sub(r"^e_", "", key),
        re.sub(r"(?i)_?(error|err)$", "", key),
        re.sub(r"(?i)_(lo|hi)$", "", key),
        re.sub(r"(?i)unc$", "", key),
        re.sub(r"(?i)sig(com)?", "", key, count=1),
    }
    bases.discard(key)
    return {b for b in bases if b}


def _catalog_fields(data: Mapping[str, Any], max_fields: int) -> dict[str, Any]:
    """Measured values of a catalog row plus their paired errors, exactly as catalogued."""
    usable = {
        str(k): v for k, v in data.items()
        if isinstance(v, (int, float, str)) and not isinstance(v, bool) and v != ""
        and not (isinstance(v, float) and not math.isfinite(v))
        and str(k).lower() not in COMPUTED_COLUMNS and not _SKIP_FIELD.search(str(k))
    }
    values = [k for k in usable if not _ERROR_FIELD.search(k)][:max_fields]
    fields = {k: usable[k] for k in values}
    for key, value in usable.items():
        if key not in fields and _ERROR_FIELD.search(key) and _error_base_names(key) & set(values):
            fields[key] = value
    return fields


def coverage_status(coverage: str | None, dec_deg: float) -> tuple[str, str | None]:
    """Classify a position against a registry coverage string.

    Returns ``("covered" | "outside" | "unknown", caveat)``: ``all-sky ...`` is covered;
    ``Dec > X`` (a hard declination limit, e.g. NVSS/VLASS Dec > -40, Pan-STARRS 3pi
    Dec > -30) is covered or outside; any text after ``;`` is a caveat that makes the
    absence of a source inconclusive; every other footprint description (survey areas,
    pointed observations, target lists) is ``unknown`` because it cannot be evaluated
    for a position here.
    """
    text = (coverage or "").strip()
    main, _, caveat = text.partition(";")
    caveat_text = caveat.strip() or None
    if re.match(r"^all-sky\b", main.strip(), re.IGNORECASE):
        return ("unknown" if caveat_text else "covered"), caveat_text
    match = re.match(r"^Dec\s*>\s*([+\-−]?\d+(?:\.\d+)?)\s*$", main.strip(), re.IGNORECASE)
    if match:
        if dec_deg <= float(match.group(1).replace("−", "-")):
            return "outside", caveat_text
        return ("unknown" if caveat_text else "covered"), caveat_text
    return "unknown", caveat_text


def summarize_crossmatch(
    record: UnifiedRecord | Mapping[str, Any],
    book: SourceBook,
    *,
    registry: CatalogRegistry | None = None,
    association_radius_arcsec: float = 5.0,
    max_fields: int = 10,
) -> dict[str, Any]:
    """Classify each catalog of a crossmatch for the explanation, honestly.

    For each catalog with rows in the search cone (radius R), the nearest row at
    separation s with 1-sigma positional error sigma is a candidate counterpart only when
    s <= max(association_radius_arcsec, 3 sigma). Its chance-coincidence probability is
    P = 1 - exp(-rho pi s^2) with the local background density rho = (N - 1) / (pi R^2)
    measured from the OTHER rows in the cone (Poisson statistics). With no other row the
    density is not measured, so P is reported as null (unknown), never as 0.

    * ``detections``: candidates with P <= MAX_CHANCE_PROBABILITY, or with an unmeasured
      P (positional agreement only; stated in a note);
    * ``ambiguous``: candidates with P > MAX_CHANCE_PROBABILITY (the nearest catalogued
      source may be unrelated, e.g. field stars in the Galactic Centre);
    * a nearest row beyond the association radius is never associated and never called
      "unlikely to be a chance coincidence"; the catalog is classified by footprint
      (:func:`coverage_status`) with ``nearest_row_arcsec``:
      ``no_source_within_radius`` only where the coverage includes the position;
      ``outside_coverage`` and ``coverage_unknown`` are NOT non-detections.

    ``wavelengths_detected`` lists observing bands only; detections in literature
    compilations (registry wavelength ``multi``/``extragalactic``/``exoplanet``: SIMBAD,
    NED, the Exoplanet Archive) are listed in ``listed_in`` instead. Field values are
    exactly as catalogued, with their paired errors; the catalog records carry no units,
    so none are given (``field_units``).
    """
    data = record.as_dict() if isinstance(record, UnifiedRecord) else dict(record)
    reg = registry or CatalogRegistry(Settings().catalog_registry_path)
    enabled = reg.enabled_catalogs()
    results = data.get("catalog_results") or {}
    provenance = data.get("provenance") or {}
    target = data.get("target") or {}
    dec = float(target.get("dec") or 0.0)
    search_radius = float(provenance.get("query_radius_arcsec") or association_radius_arcsec)
    rows_by_catalog: dict[str, list[dict[str, Any]]] = {}
    for sources in (data.get("counterparts") or {}).values():
        for src in sources:
            rows_by_catalog.setdefault(src["catalog"], []).append(src)

    detections: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    no_source: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for name, info in sorted(results.items()):
        status = info.get("status")
        coverage = enabled[name].coverage if name in enabled else None
        if status == "failed":
            failures.append({"catalog": name, "error_type": info.get("error_type")})
            continue
        rows = sorted(rows_by_catalog.get(name, []), key=lambda s: float(s.get("separation_arcsec") or 0.0))
        # row_count counts the rows inside the requested search radius (not the provider's
        # possibly widened cone), so densities use the search radius.
        radius = search_radius
        nearest_unassociated: float | None = None
        nearest_source_id: Any = None
        if rows:
            src = rows[0]
            sep = float(src.get("separation_arcsec") or 0.0)
            sigma = src.get("positional_error_arcsec")
            sigma = float(sigma) if sigma is not None and math.isfinite(float(sigma)) else None
            assoc = max(association_radius_arcsec, 3.0 * sigma) if sigma else association_radius_arcsec
            if sep <= assoc:
                n_rows = int(info.get("row_count") or len(rows))
                background = max(n_rows - 1, 0)
                p_chance: float | None = None
                if background:
                    p_chance = float(f"{1.0 - math.exp(-background / (math.pi * radius ** 2) * math.pi * sep ** 2):.2g}")
                competitors = sum(1 for other in rows[1:] if float(other.get("separation_arcsec") or 0.0) <= assoc)
                citation = info.get("citation") or (src.get("metadata") or {}).get("citation")
                ref = None
                if citation:
                    codes = BIBCODE_RE.findall(str(citation))
                    doi = _DOI_RE.search(str(citation))
                    ref = book.add(kind="catalog", bibcode=codes[0] if codes else None, title=str(citation),
                                   label=f"{name} catalog", url=f"https://doi.org/{doi.group(1)}" if doi else None)
                wavelength = (src.get("metadata") or {}).get("wavelength") or (
                    enabled[name].wavelength if name in enabled else None)
                entry: dict[str, Any] = {
                    "catalog": name,
                    "wavelength": wavelength,
                    "source_id": src.get("source_id"),
                    "separation_arcsec": round(sep, 3),
                    "positional_error_arcsec": round(sigma, 3) if sigma is not None else None,
                    "association_radius_arcsec": round(assoc, 2),
                    "rows_in_search_cone": n_rows,
                    "chance_coincidence_probability": p_chance,
                    "other_rows_within_association_radius": competitors,
                    "fields": _catalog_fields(src.get("data") or {}, max_fields),
                    "ref": ref,
                }
                if str(wavelength) in COMPILATION_WAVELENGTHS:
                    entry["compilation"] = True
                if p_chance is None:
                    entry["note"] = (f"the only catalog row within the {radius:g} arcsec search cone: the local source "
                                     "density is not measured, so the chance-coincidence probability is unknown; the "
                                     "association rests on positional agreement alone")
                elif info.get("truncated"):
                    entry["note"] = "row count truncated; the chance-coincidence probability is a lower bound"
                if p_chance is None or p_chance <= MAX_CHANCE_PROBABILITY:
                    detections.append(entry)
                else:
                    entry["reason"] = (f"chance-coincidence probability {p_chance} > {MAX_CHANCE_PROBABILITY}: the "
                                       "nearest catalogued source may be unrelated")
                    ambiguous.append(entry)
                continue
            nearest_unassociated = round(sep, 2)
            nearest_source_id = src.get("source_id")
        state, caveat = coverage_status(coverage, dec)
        item: dict[str, Any] = {"catalog": name, "coverage": coverage}
        if nearest_unassociated is not None:
            item["nearest_row_arcsec"] = nearest_unassociated
            item["nearest_row_source_id"] = nearest_source_id
            item["nearest_row_note"] = ("beyond the association radius: not associated with the object (it may be an "
                                        "unrelated neighbour or a displaced centroid of a blended/extended source)")
        if state == "covered":
            item["no_source_within_arcsec"] = round(association_radius_arcsec if rows else radius, 2)
            no_source.append(item)
        elif state == "outside":
            outside.append(item)
        else:
            if caveat:
                item["caveat"] = caveat
            unknown.append(item)
    return {
        "search_radius_arcsec": search_radius,
        "association_radius_arcsec": association_radius_arcsec,
        "catalogs_queried": data.get("catalogs_queried"),
        "detected_in": len(detections),
        "wavelengths_detected": sorted({str(d["wavelength"]) for d in detections
                                        if d["wavelength"] and str(d["wavelength"]) not in COMPILATION_WAVELENGTHS}),
        "listed_in": sorted(d["catalog"] for d in detections if d.get("compilation")),
        "field_units": "not provided by the catalog records; values are exactly as catalogued",
        "detections": detections,
        "ambiguous": ambiguous,
        "no_source_within_radius": no_source,
        "outside_coverage": outside,
        "coverage_unknown": unknown,
        "failures": failures,
    }


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _measure(quantity: str, value: Any, *, unit: str | None = None, error: Any = None, ref: int | None = None,
             note: str | None = None, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"quantity": quantity, "value": value, "unit": unit, "ref": ref}
    if _num(error) is not None:
        entry["error"] = _num(error)
    entry.update({k: v for k, v in extra.items() if v is not None})
    if note:
        entry["note"] = note
    return entry


def _round_with_error(value: float, error: float | None) -> float:
    """Round to the error's second significant figure (4 significant figures without an error)."""
    if error is not None and error > 0:
        decimals = max(0, 1 - math.floor(math.log10(error)))
    else:
        decimals = max(0, 3 - math.floor(math.log10(abs(value)))) if value else 0
    return round(value, decimals)


def _path_root(path: Any) -> str | None:
    """First element of a SIMBAD otypedef path ('G > AGN > QSO' -> 'G'); None when unknown."""
    if path is None:
        return None
    return str(path).split(">")[0].strip()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def simbad_type_is_extragalactic(otype: str | None, path: str | None) -> bool:
    """True when a SIMBAD object type is extragalactic in SIMBAD's own hierarchy.

    Uses the ``otypedef.path`` root (G, GrG, ClG, SCG, PCG, IG, PaG, PoG; candidates such
    as AG?, Q?, Bz?, BL?, C?G, Gr?, SC?, PCG? share their parent's path) plus the lensed
    galaxy/quasar images LeG and LeQ. Without a path (type unknown to otypedef) the
    type is not treated as extragalactic, except the lensed images.
    """
    code = str(otype or "").strip()
    if code in EXTRAGALACTIC_LENSED_OTYPES:
        return True
    return _path_root(path) in EXTRAGALACTIC_PATH_ROOTS


def classify_simbad_types(
    otype: str | None, otype_path: str | None, types: Mapping[str, Mapping[str, Any]]
) -> tuple[bool, bool]:
    """(extragalactic, any_extragalactic) of a SIMBAD object from its main and other types.

    SIMBAD's main object type decides. Only when the main type is non-specific
    (:data:`GENERIC_PATH_ROOTS`: unknown, a wavelength class such as Rad or X, a blend, a
    region) do the other types (table ``otypes``) count: a confirmed extragalactic class
    makes the object extragalactic (cosmology applies, parallax distance withheld); a
    candidate one only withholds the parallax distance. A specific stellar, cluster or
    nebula main type is never overridden by the other types, which gather every type any
    paper gave the object: Mira (Mi*), the Helix Nebula (PN) and 47 Tuc (GlC) carry a
    stray confirmed "G", Cyg X-1 (HXB) a candidate "BL?" (verified live).
    """
    main = simbad_type_is_extragalactic(otype, otype_path)
    generic_main = otype_path is not None and _path_root(otype_path) in GENERIC_PATH_ROOTS
    if not generic_main:
        return main, main
    confirmed = any(simbad_type_is_extragalactic(t, info.get("path")) and not _truthy(info.get("is_candidate"))
                    for t, info in types.items())
    candidate = any(simbad_type_is_extragalactic(t, info.get("path")) for t, info in types.items())
    extragalactic = main or confirmed
    return extragalactic, extragalactic or candidate


def cmb_frame_redshift(z_helio: float, ra_deg: float, dec_deg: float) -> float:
    """Heliocentric -> CMB-frame redshift with the Planck 2018 solar dipole.

    (1 + z_hel) = (1 + z_cmb)(1 + z_sun) with z_sun = -v_sun cos(theta) / c, theta the
    angle between the object and the dipole apex (Davis et al. 2011, ApJ 741, 67;
    dipole 369.82 km/s towards l = 264.021, b = 48.253 deg, Planck Collaboration 2020,
    A&A 641, A1). Objects towards the apex are blueshifted by the solar motion, so their
    CMB-frame redshift is larger than the heliocentric one.
    """
    position = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    apex = SkyCoord(l=CMB_DIPOLE_L_DEG * u.deg, b=CMB_DIPOLE_B_DEG * u.deg, frame="galactic")
    cos_theta = math.cos(float(position.separation(apex).rad))
    z_sun = -CMB_DIPOLE_KMS * cos_theta / SPEED_OF_LIGHT_KMS
    return (1.0 + z_helio) / (1.0 + z_sun) - 1.0


def hubble_flow_redshift_error(z_err: float | None) -> float:
    """Redshift uncertainty of a Hubble-flow distance: the measurement error and an
    uncorrected peculiar-velocity scatter (PECULIAR_VELOCITY_KMS / c) in quadrature."""
    sigma_pec = PECULIAR_VELOCITY_KMS / SPEED_OF_LIGHT_KMS
    return math.hypot(z_err or 0.0, sigma_pec)


def _hubble_flow_distances(
    z_cmb: float, z_hel: float, z_err: float | None
) -> list[tuple[str, float, float | None, float | None, str]]:
    """(quantity, value, plus_error, minus_error, unit) of the Hubble-flow distances.

    D_L = (1 + z_hel) D_M(z_cmb) (Davis et al. 2011, ApJ 741, 67; Calcino & Davis 2017,
    JCAP 01, 038) -- not (1 + z_cmb) D_M(z_cmb) as astropy's luminosity_distance(z_cmb)
    would give; comoving distance and lookback time are evaluated at z_cmb. Errors
    propagate :func:`hubble_flow_redshift_error` (redshift error plus peculiar velocity)
    by evaluating the functions at z_cmb +- sigma; values are rounded to that error.
    """
    sigma = hubble_flow_redshift_error(z_err)
    low_z = max(z_cmb - sigma, 1e-6)
    functions: tuple[tuple[str, Callable[[float], float], str], ...] = (
        ("luminosity_distance", lambda z: (1.0 + z_hel) * float(Planck18.comoving_transverse_distance(z).value), "Mpc"),
        ("comoving_distance", lambda z: float(Planck18.comoving_distance(z).value), "Mpc"),
        ("lookback_time", lambda z: float(Planck18.lookback_time(z).value), "Gyr"),
    )
    out = []
    for quantity, func, unit in functions:
        value = func(z_cmb)
        plus, minus = func(z_cmb + sigma) - value, value - func(low_z)
        scale = max(plus, minus)
        out.append((quantity, _round_with_error(value, scale), _round_with_error(plus, scale),
                    _round_with_error(minus, scale), unit))
    return out


async def _simbad_lookup(
    client: httpx.AsyncClient, ra: float, dec: float, radius_arcsec: float, endpoint: str, epoch: float | None = None,
) -> dict[str, Any] | None:
    """Nearest SIMBAD object to a position, allowing for proper motion when ``epoch`` is given.

    SIMBAD ``basic`` positions are ICRS at epoch J2000. Without an epoch (or at J2000)
    objects within ``radius_arcsec`` are compared as catalogued. With another epoch,
    every object within ``radius + 10.4"/yr * |epoch - 2000|`` (10.4"/yr bounds the
    largest known proper motion, Barnard's star) is moved linearly by its SIMBAD proper
    motion to the epoch before comparing. The nearest object within ``radius_arcsec``
    wins; objects within 0.05" of the nearest (a star and its planets, a galaxy and its
    nucleus) are ranked by SIMBAD reference count, so the primary object is chosen.
    Returns ``{oid, ra, dec, pmra, pmdec, separation_arcsec}`` or None.
    """
    dt = 0.0 if epoch is None else float(epoch) - 2000.0
    radius = min(float(radius_arcsec), 60.0)
    search = min(radius + MAX_PROPER_MOTION_ARCSEC_YR * abs(dt), 900.0)
    point = f"POINT('ICRS', {ra:.9f}, {dec:.9f})"
    adql = (
        f"SELECT TOP {50 if dt == 0 else 2000} b.oid AS oid, b.ra AS ra, b.dec AS dec, b.pmra AS pmra, "
        f"b.pmdec AS pmdec, b.nbref AS nbref, DISTANCE(POINT('ICRS', b.ra, b.dec), {point}) AS dist_deg FROM basic AS b "
        f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra:.9f}, {dec:.9f}, {search / 3600.0:.10f})) = 1 "
        "ORDER BY dist_deg ASC, oid ASC"
    )
    rows = await _tap_rows(client, endpoint, adql)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        r_ra, r_dec = _num(row.get("ra")), _num(row.get("dec"))
        if row.get("oid") is None or r_ra is None or r_dec is None:
            continue
        pmra, pmdec = _num(row.get("pmra")), _num(row.get("pmdec"))
        if dt and pmra is not None and pmdec is not None:
            r_dec_t = r_dec + pmdec * dt / 3.6e6
            r_ra_t = r_ra + pmra * dt / 3.6e6 / max(math.cos(math.radians(r_dec)), 1e-6)
        else:
            r_ra_t, r_dec_t = r_ra, r_dec
        sep = haversine_arcsec(ra, dec, r_ra_t % 360.0, r_dec_t)
        if sep <= radius:
            candidates.append({"oid": int(row["oid"]), "ra": r_ra, "dec": r_dec, "pmra": pmra, "pmdec": pmdec,
                               "separation_arcsec": round(sep, 3), "nbref": int(_num(row.get("nbref")) or 0)})
    if not candidates:
        return None
    nearest = min(c["separation_arcsec"] for c in candidates)
    tied = [c for c in candidates if c["separation_arcsec"] <= nearest + 0.05]
    best = max(tied, key=lambda c: (c["nbref"], -c["separation_arcsec"], -c["oid"]))
    best.pop("nbref")
    return best


async def _cancel(task: asyncio.Future[Any] | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):  # the SIMBAD error is what gets reported
            await task


async def gather_facts(
    *,
    http_client: httpx.AsyncClient,
    name: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    epoch: float | None = None,
    service: CrossmatchLike | None = None,
    include_crossmatch: bool = True,
    radius_arcsec: float = 5.0,
    crossmatch_radius_arcsec: float = DEFAULT_CROSSMATCH_RADIUS_ARCSEC,
    max_references: int = 8,
    ads_token: str | None = None,
    resolver: SesameResolver | None = None,
    simbad_endpoint: str = SIMBAD_TAP,
    registry: CatalogRegistry | None = None,
) -> ObjectFacts:
    """Collect cited facts about an object from Sesame, SIMBAD TAP, a crossmatch and ADS.

    Give ``name`` or ``ra``/``dec``. A name is resolved with CDS Sesame; the SIMBAD object
    is the one Sesame's SIMBAD answer names (its oid). When Sesame answers from another
    resolver (e.g. NED only), the name itself is looked up among SIMBAD identifiers
    (table ``ident``); only if SIMBAD knows no such identifier is the nearest SIMBAD object
    within ``radius_arcsec`` of the resolver position used, with a warning and an
    identification note in the facts (e.g. NED's "M 1" position is the Crab Pulsar's, not
    SIMBAD's nebula centre). With ``ra``/``dec`` the nearest SIMBAD object within
    ``radius_arcsec`` is used; with ``epoch`` (the Julian year of the coordinates, e.g.
    2016.0 for Gaia DR3) SIMBAD proper motions are applied before choosing. The
    crossmatch searches ``crossmatch_radius_arcsec`` and associates rows within
    ``radius_arcsec`` (or 3 sigma of their positional error); see
    :func:`summarize_crossmatch`. ``max_references`` (1-25) bounds each bibliography list.
    """
    if (name is None) == (ra is None or dec is None):
        raise ValueError("Give either name or both ra and dec.")
    if not 0 < radius_arcsec <= 60:
        raise ValueError("radius_arcsec must be in (0, 60].")
    if not radius_arcsec <= crossmatch_radius_arcsec <= 120:
        raise ValueError("crossmatch_radius_arcsec must be in [radius_arcsec, 120].")
    if epoch is not None and not 1900.0 <= float(epoch) <= 2100.0:
        raise ValueError("epoch must be a Julian year between 1900 and 2100.")
    if isinstance(max_references, bool) or not isinstance(max_references, int) or not 1 <= max_references <= 25:
        raise ValueError("max_references must be an integer between 1 and 25.")
    book = SourceBook()
    simbad_ref = book.add(kind="database", bibcode=SIMBAD_BIBCODE, label="SIMBAD database")
    warnings: list[str] = []
    oid: int | None = None
    target: Target
    if name is not None:
        resolved = await (resolver or SesameResolver(http_client)).resolve(name)
        target = resolved_target(resolved)
        resolver_name = str((resolved.resolver_metadata or {}).get("resolver_name") or resolved.resolver or "")
        raw_oid = ((resolved.resolver_metadata or {}).get("raw_fields") or {}).get("oid")
        if raw_oid and "simbad" in resolver_name.lower():
            oid = int(str(raw_oid[0]).strip())
        query_info: dict[str, Any] = {"name": name, "resolver": resolved.resolver, "canonical_name": resolved.canonical_name}
        if oid is None:
            oid, target = await _identify_name_in_simbad(
                http_client, name, resolved, target, resolver_name, radius_arcsec, simbad_endpoint, query_info, warnings,
            )
    else:
        target = validate_target(ra, dec, epoch=epoch)  # type: ignore[arg-type]
        query_info = {"ra_deg": target.ra, "dec_deg": target.dec, "radius_arcsec": radius_arcsec}
        if epoch is not None:
            query_info["epoch"] = float(epoch)
        match = await _simbad_lookup(http_client, target.ra, target.dec, radius_arcsec, simbad_endpoint, epoch)
        if match is not None:
            oid = match["oid"]
            query_info["simbad_match_separation_arcsec"] = match["separation_arcsec"]
            if epoch is not None and match["pmra"] is not None and match["pmdec"] is not None:
                # Crossmatch from the SIMBAD J2000 position with its proper motion.
                target = validate_target(match["ra"], match["dec"], epoch=2000.0,
                                         pm_ra_masyr=match["pmra"], pm_dec_masyr=match["pmdec"])
        else:
            warnings.append(f"No SIMBAD object within {radius_arcsec:g} arcsec of the position"
                            + (f" at epoch {float(epoch):g}." if epoch is not None else "."))

    facts = ObjectFacts(query=query_info, object={"ra_deg": target.ra, "dec_deg": target.dec, "simbad_oid": oid})

    crossmatch_task: asyncio.Future[Any] | None = None
    if include_crossmatch and service is not None:
        crossmatch_task = asyncio.ensure_future(service.crossmatch(
            target.ra, target.dec, radius_arcsec=crossmatch_radius_arcsec, epoch=target.epoch,
            pm_ra_masyr=target.pm_ra_masyr, pm_dec_masyr=target.pm_dec_masyr,
        ))
    try:
        if oid is not None:
            await _add_simbad_facts(facts, http_client, oid, book, simbad_ref, warnings, max_references=max_references,
                                    ads_token=ads_token, endpoint=simbad_endpoint)
    except BaseException:
        await _cancel(crossmatch_task)  # never leave the catalog fan-out running after a failure
        raise

    if crossmatch_task is not None:
        try:
            record = await crossmatch_task
        except (AstroSearchError, ValueError, httpx.HTTPError, TimeoutError) as exc:  # SIMBAD facts still stand
            warnings.append(f"Crossmatch unavailable: {exc.__class__.__name__}: {exc}")
        else:
            facts.crossmatch = summarize_crossmatch(
                record, book, registry=registry or getattr(service, "registry", None),
                association_radius_arcsec=radius_arcsec,
            )
    facts.sources = list(book.sources)
    facts.warnings = warnings
    return facts


async def _simbad_oids_for_names(
    client: httpx.AsyncClient, names: Sequence[str], endpoint: str
) -> list[dict[str, Any]]:
    """SIMBAD objects carrying any of ``names`` as an identifier (``ident.id``, which SIMBAD
    compares in its normalised form: 'M 1', 'M1' and 'MESSIER 001' all match 'M   1')."""
    unique = list(dict.fromkeys(" ".join(n.split()) for n in names if n and n.strip()))
    if not unique:
        return []
    adql = (
        "SELECT DISTINCT b.oid AS oid, b.main_id AS main_id, b.ra AS ra, b.dec AS dec, b.pmra AS pmra, "
        "b.pmdec AS pmdec, b.otype AS otype, od.path AS otype_path FROM ident AS i JOIN basic AS b ON b.oid = i.oidref "
        "LEFT OUTER JOIN otypedef AS od ON od.otype = b.otype WHERE i.id IN (" + ", ".join(_sql_str(n) for n in unique) + ")"
    )
    return [r for r in await _tap_rows(client, endpoint, adql) if r.get("oid") is not None]


async def _identify_name_in_simbad(
    client: httpx.AsyncClient,
    name: str,
    resolved: ResolvedObject,
    target: Target,
    resolver_name: str,
    radius_arcsec: float,
    endpoint: str,
    query_info: dict[str, Any],
    warnings: list[str],
) -> tuple[int | None, Target]:
    """SIMBAD oid (and search target) for a name Sesame did not answer from SIMBAD."""
    source = resolver_name or resolved.resolver or "the name resolver"
    matches = await _simbad_oids_for_names(client, [name, resolved.canonical_name], endpoint)
    if matches:
        def offset(row: Mapping[str, Any]) -> float:
            r_ra, r_dec = _num(row.get("ra")), _num(row.get("dec"))
            return haversine_arcsec(target.ra, target.dec, r_ra, r_dec) if r_ra is not None and r_dec is not None else 1e9

        best = min(matches, key=lambda r: (offset(r), int(r["oid"])))
        query_info["identification"] = {
            "method": "SIMBAD identifier",
            "note": f"Sesame answered from {source} without a SIMBAD identifier; SIMBAD lists the name as an identifier "
                    f"of {' '.join(str(best.get('main_id') or '').split())}. Its SIMBAD position is used.",
        }
        if len(matches) > 1:
            warnings.append(f"Several SIMBAD objects carry the name {name!r}; the one nearest to the {source} position "
                            "was used.")
        ra_b, dec_b = _num(best.get("ra")), _num(best.get("dec"))
        if ra_b is not None and dec_b is not None:
            pmra, pmdec = _num(best.get("pmra")), _num(best.get("pmdec"))
            if simbad_type_is_extragalactic(str(best.get("otype") or ""), best.get("otype_path")):
                target = validate_target(ra_b, dec_b, epoch=2000.0, pm_ra_masyr=0.0, pm_dec_masyr=0.0)
            elif pmra is not None and pmdec is not None:
                target = validate_target(ra_b, dec_b, epoch=2000.0, pm_ra_masyr=pmra, pm_dec_masyr=pmdec)
            else:
                target = validate_target(ra_b, dec_b)
        return int(best["oid"]), target
    match = await _simbad_lookup(client, target.ra, target.dec, radius_arcsec, endpoint)
    if match is None:
        warnings.append(f"SIMBAD has no identifier {name!r} and no object within {radius_arcsec:g} arcsec of the "
                        f"{source} position.")
        return None, target
    note = (f"SIMBAD lists no identifier {name!r}; this is the nearest SIMBAD object, "
            f"{match['separation_arcsec']} arcsec from the {source} position, and may be a DIFFERENT object than the "
            "one named (e.g. a star or pulsar inside a nebula).")
    query_info["identification"] = {"method": "nearest SIMBAD object to the resolver position", "note": note}
    query_info["simbad_match_separation_arcsec"] = match["separation_arcsec"]
    warnings.append(note)
    return match["oid"], target


@dataclass(frozen=True, slots=True)
class RedshiftInput:
    """A heliocentric redshift that may feed Hubble-flow distances.

    ``error`` is SIMBAD's measurement error, or -- when SIMBAD gives none -- half a unit
    in the last stated digit (``error_inferred``); None when neither is known.
    ``decimals`` is SIMBAD's stated precision (``rvz_redshift_prec``) of a z / cz record.
    """

    value: float
    error: float | None
    error_inferred: bool
    decimals: int | None
    source: str  # "redshift", "cz" or "cz/c of the SIMBAD velocity"


def _int(value: Any) -> int | None:
    number = _num(value)
    return int(number) if number is not None and number == int(number) else None


def _half_last_digit(decimals: int | None) -> float | None:
    """Half a unit in the last stated decimal (SIMBAD precision codes are small integers)."""
    if decimals is None or not 0 <= decimals <= 12:
        return None
    return 0.5 * 10.0 ** -decimals


def _sig2(value: float) -> float:
    return float(f"{value:.2g}")


def gaia_release_of(bibcode: str | None, title: str | None) -> tuple[GaiaRelease | None, bool]:
    """(release, is_release_catalogue) of a parallax reference: by release bibcode (raw
    catalogue values) or else by a title naming the release (a paper based on it)."""
    code = (bibcode or "").strip()
    for release in GAIA_RELEASES:
        if code in release.bibcodes:
            return release, True
    for release in GAIA_RELEASES:
        if title and release.title_re.search(title):
            return release, False
    return None, False


async def _fill_titles(book: SourceBook, http_client: httpx.AsyncClient, endpoint: str) -> None:
    missing = book.bibcodes_missing_titles()
    for start in range(0, len(missing), 40):
        chunk = missing[start:start + 40]
        rows = await _tap_rows(
            http_client, endpoint,
            'SELECT bibcode, title, "year" AS pub_year FROM ref WHERE bibcode IN (' + ", ".join(_sql_str(c) for c in chunk) + ")",
        )
        book.fill({str(r.get("bibcode") or "").strip(): {"title": r.get("title"), "year": r.get("pub_year")} for r in rows})


async def _add_simbad_facts(
    facts: ObjectFacts,
    http_client: httpx.AsyncClient,
    oid: int,
    book: SourceBook,
    simbad_ref: int,
    warnings: list[str],
    *,
    max_references: int,
    ads_token: str | None,
    endpoint: str,
) -> None:
    """SIMBAD basic data, types, identifiers, photometry, distances, bibliography and derived values."""
    basic_q = (
        "SELECT b.oid AS oid, b.main_id AS main_id, b.otype AS otype, od.description AS otype_description, "
        "od.path AS otype_path, "
        "b.ra AS ra, b.dec AS dec, b.coo_bibcode AS coo_bibcode, b.plx_value AS plx_value, b.plx_err AS plx_err, "
        "b.plx_qual AS plx_qual, b.plx_bibcode AS plx_bibcode, b.pmra AS pmra, b.pmdec AS pmdec, "
        "b.pm_bibcode AS pm_bibcode, b.rvz_type AS rvz_type, b.rvz_nature AS rvz_nature, b.rvz_qual AS rvz_qual, "
        "b.rvz_radvel AS rvz_radvel, b.rvz_redshift AS rvz_redshift, b.rvz_err AS rvz_err, "
        "b.rvz_radvel_prec AS rvz_radvel_prec, b.rvz_redshift_prec AS rvz_redshift_prec, "
        "b.rvz_bibcode AS rvz_bibcode, b.sp_type AS sp_type, b.sp_bibcode AS sp_bibcode, b.morph_type AS morph_type, "
        "b.morph_bibcode AS morph_bibcode, b.galdim_majaxis AS galdim_majaxis, b.galdim_minaxis AS galdim_minaxis, "
        "b.galdim_bibcode AS galdim_bibcode, b.nbref AS nbref FROM basic AS b "
        f"LEFT OUTER JOIN otypedef AS od ON od.otype = b.otype WHERE b.oid = {oid}"
    )
    basic_rows, types_rows, ident_rows, flux_rows, dist_rows, biblio = await _gather_or_cancel(
        _tap_rows(http_client, endpoint, basic_q),
        _tap_rows(http_client, endpoint,
                  "SELECT DISTINCT o.otype AS otype, d.path AS path, d.is_candidate AS is_candidate FROM otypes AS o "
                  f"LEFT OUTER JOIN otypedef AS d ON d.otype = o.otype WHERE o.oidref = {oid}"),
        _tap_rows(http_client, endpoint, f"SELECT TOP 25 id FROM ident WHERE oidref = {oid} ORDER BY id ASC"),
        _tap_rows(http_client, endpoint,
                  f"SELECT filter, flux, flux_err, system, bibcode FROM flux WHERE oidref = {oid} ORDER BY filter ASC"),
        _tap_rows(http_client, endpoint,
                  f"SELECT TOP 10 dist, plus_err, minus_err, unit, method, bibcode FROM mesDistance "
                  f"WHERE oidref = {oid} ORDER BY bibcode DESC, dist ASC"),
        fetch_simbad_bibliography(http_client, oid, limit=max_references, endpoint=endpoint),
    )
    if not basic_rows:
        raise UpstreamServiceError(f"SIMBAD returned no basic data for oid {oid}.")
    b = basic_rows[0]
    main_id = " ".join(str(b.get("main_id") or "").split())
    facts.object.update({"main_id": main_id, "ra_deg": _num(b.get("ra")), "dec_deg": _num(b.get("dec"))})
    otype = str(b.get("otype") or "").strip()
    type_info = {str(r.get("otype") or "").strip(): r for r in types_rows if str(r.get("otype") or "").strip()}
    all_types = sorted(type_info, key=str.casefold)
    otype_path = b.get("otype_path")
    facts.identity = {
        "main_id": main_id,
        "otype": otype or None,
        "otype_description": b.get("otype_description"),
        "otype_path": otype_path,
        "all_otypes": all_types,
        "identifiers": [" ".join(str(r.get("id") or "").split()) for r in ident_rows if r.get("id")],
        "simbad_reference_count": b.get("nbref"),
        "ref": simbad_ref,
    }
    extragalactic, any_extragalactic = classify_simbad_types(otype, otype_path, type_info)

    def ref_for(bibcode_key: str) -> int:
        """The value's own reference, or SIMBAD itself when SIMBAD gives none (every fact is citable)."""
        code = str(b.get(bibcode_key) or "").strip()
        return book.add(kind="measurement", bibcode=code) if code else simbad_ref

    m = facts.measurements
    m.append(_measure("position_icrs_j2000", {"ra_deg": _num(b.get("ra")), "dec_deg": _num(b.get("dec"))},
                      unit="deg", ref=ref_for("coo_bibcode")))
    redshift = _num(b.get("rvz_redshift"))
    radvel = _num(b.get("rvz_radvel"))
    rvz_type = str(b.get("rvz_type") or "").strip().lower()
    nature = str(b.get("rvz_nature") or "").strip().lower()
    rvz_qual = str(b.get("rvz_qual") or "").strip().upper() or None
    rvz_err = _num(b.get("rvz_err"))
    z_decimals = _int(b.get("rvz_redshift_prec"))
    v_decimals = _int(b.get("rvz_radvel_prec"))
    photometric = nature.startswith("p")
    nature_label = "photometric" if photometric else "spectroscopic" if nature.startswith("s") else None
    rvz_ref = ref_for("rvz_bibcode")
    quality_note = f"; SIMBAD quality {rvz_qual} (A best .. E worst)" if rvz_qual else ""
    z_input: RedshiftInput | None = None
    if redshift is not None and rvz_type in {"z", "c"}:
        # For rvz_type z and c SIMBAD's rvz_err is in redshift units (verified live: 3C 273
        # z 0.15756751 +- 0.0005; M 81 cz-type, error 7.0e-05); rvz_redshift_prec is the
        # number of decimals SIMBAD states (Ton 618: z = 2.2, precision 1, no error).
        measured = "cz" if rvz_type == "c" else "redshift"
        inferred = _half_last_digit(z_decimals) if rvz_err is None else None
        if rvz_err is not None:
            error_note = ""
        elif inferred is not None:
            error_note = (f"; SIMBAD gives no error: the value is stated to {z_decimals} decimal(s), i.e. known to "
                          f"about +-{inferred:g} at best")
        else:
            error_note = "; SIMBAD gives neither an error nor a precision"
        m.append(_measure(
            "photometric_redshift" if photometric else "redshift", redshift, error=rvz_err, ref=rvz_ref,
            quality=rvz_qual, nature=nature_label, decimals=z_decimals if rvz_err is None else None,
            note=(f"measured as {measured}" + ("; a photometric estimate, far less precise than a spectroscopic "
                                                "redshift" if photometric else "") + quality_note + error_note),
        ))
        if extragalactic and abs(redshift) < CZ_MAX_REDSHIFT:
            # cz is a useful proxy only at low redshift; SIMBAD's rvz_radvel (a special-
            # relativistic Doppler conversion of z) is deliberately not reported.
            m.append(_measure(
                "cz", round(SPEED_OF_LIGHT_KMS * redshift, 1), unit="km/s",
                error=round(SPEED_OF_LIGHT_KMS * rvz_err, 1) if rvz_err is not None else None, ref=rvz_ref,
                note="c times the redshift: a recession-velocity proxy, not a measured Doppler velocity; a "
                     "cosmological redshift is not a velocity through space",
            ))
        z_input = RedshiftInput(redshift, rvz_err if rvz_err is not None else inferred, rvz_err is None and
                                inferred is not None, z_decimals, measured)
    elif radvel is not None:
        m.append(_measure("radial_velocity", radvel, unit="km/s", error=rvz_err, ref=rvz_ref, quality=rvz_qual,
                          note="measured as a velocity" + quality_note))
        if extragalactic:
            # Catalogued galaxy velocities are cz (optical convention), so z = v / c. SIMBAD's
            # own rvz_redshift for these records is the relativistic Doppler conversion
            # sqrt((1 + v/c) / (1 - v/c)) - 1 (verified live: NGC 6086, v = 9581 km/s,
            # z = 0.03248618 instead of v/c = 0.031959), which is not used.
            inferred_v = _half_last_digit(v_decimals) if rvz_err is None else None
            v_err = rvz_err if rvz_err is not None else inferred_v
            z_cz = radvel / SPEED_OF_LIGHT_KMS
            z_err = v_err / SPEED_OF_LIGHT_KMS if v_err is not None else None
            m.append(_measure(
                "redshift", _round_with_error(z_cz, z_err), ref=rvz_ref, quality=rvz_qual, nature=nature_label,
                error=_sig2(z_err) if z_err is not None and rvz_err is not None else None,
                note="cz/c: the SIMBAD velocity divided by c (catalogued galaxy velocities are cz, the optical "
                     "convention); not a relativistic Doppler conversion" + quality_note,
            ))
            z_input = RedshiftInput(z_cz, z_err, rvz_err is None and z_err is not None, None,
                                    "cz/c of the SIMBAD velocity")
    plx, plx_err = _num(b.get("plx_value")), _num(b.get("plx_err"))
    plx_qual = str(b.get("plx_qual") or "").strip().upper() or None
    if plx is not None:
        plx_note = None
        if extragalactic:
            plx_note = "parallax of an extragalactic object: consistent with zero or spurious, not a distance"
        elif any_extragalactic:
            plx_note = ("SIMBAD's main type is non-specific and its other types include candidate extragalactic "
                        "classes: the parallax may be spurious and is not used as a distance")
        m.append(_measure("parallax", plx, unit="mas", error=plx_err, ref=ref_for("plx_bibcode"), quality=plx_qual,
                          note=plx_note))
    if _num(b.get("pmra")) is not None and _num(b.get("pmdec")) is not None:
        m.append(_measure("proper_motion", {"pmra_cosdec": _num(b.get("pmra")), "pmdec": _num(b.get("pmdec"))},
                          unit="mas/yr", ref=ref_for("pm_bibcode")))
    sp_type = str(b.get("sp_type") or "").strip()
    if re.search(r"[A-Za-z0-9]", sp_type):
        m.append(_measure("spectral_type", sp_type, ref=ref_for("sp_bibcode")))
    morph = str(b.get("morph_type") or "").strip()
    if re.search(r"[A-Za-z0-9]", morph):  # SIMBAD placeholders such as '?...' are not a morphology
        m.append(_measure("morphological_type", morph, ref=ref_for("morph_bibcode")))
    if _num(b.get("galdim_majaxis")) is not None:
        m.append(_measure("angular_size", {"major_axis": _num(b.get("galdim_majaxis")), "minor_axis": _num(b.get("galdim_minaxis"))},
                          unit="arcmin", ref=ref_for("galdim_bibcode")))

    for row in flux_rows:
        value = _num(row.get("flux"))
        if value is None:
            continue
        code = str(row.get("bibcode") or "").strip()
        facts.photometry.append({
            "band": str(row.get("filter") or "").strip(),
            "magnitude": value,
            "error": _num(row.get("flux_err")),
            "system": _FLUX_SYSTEMS.get(str(row.get("system") or "").strip(), row.get("system")),
            "ref": book.add(kind="measurement", bibcode=code) if code else simbad_ref,
        })
    for row in dist_rows:
        value = _num(row.get("dist"))
        if value is None:
            continue
        code = str(row.get("bibcode") or "").strip()
        facts.distances.append({
            "value": value,
            "unit": str(row.get("unit") or "").strip() or None,
            # SIMBAD stores minus_err as a negative number; every error here is a magnitude.
            "plus_error": abs(_num(row.get("plus_err"))) if _num(row.get("plus_err")) is not None else None,
            "minus_error": abs(_num(row.get("minus_err"))) if _num(row.get("minus_err")) is not None else None,
            "method": str(row.get("method") or "").strip() or None,
            "ref": book.add(kind="measurement", bibcode=code) if code else simbad_ref,
        })

    biblio_out: dict[str, Any] = {"simbad_reference_count": b.get("nbref"), "ref": simbad_ref}
    for key, entries in biblio.items():
        for entry in entries:
            entry["ref"] = book.add(kind="reference", bibcode=entry["bibcode"], title=entry.get("title"), year=entry.get("year"))
        biblio_out[key] = entries
    biblio_out["selection"] = (
        "SIMBAD has no citation counts. Only papers naming the object in their title or abstract are listed: "
        "'recent' = the newest such papers; 'foundational' = the earliest papers studying <= 5 objects; 'focused' = "
        "other papers in which SIMBAD records the most occurrences of the object (simbad_obj_freq)."
    )
    facts.bibliography = biblio_out

    if ads_token and main_id:
        try:
            cited = await fetch_ads_most_cited(http_client, ads_token, main_id, rows=min(max_references, 10))
        except UpstreamServiceError as exc:
            warnings.append(f"NASA ADS ranking unavailable: {exc}")
        else:
            for entry in cited:
                entry["ref"] = book.add(kind="reference", bibcode=entry["bibcode"], title=entry.get("title"), year=entry.get("year"))
            facts.bibliography["most_cited_ads"] = cited

    # Titles first: the parallax reference's title identifies Gaia-based parallaxes.
    await _fill_titles(book, http_client, endpoint)
    plx_bibcode = str(b.get("plx_bibcode") or "").strip()
    plx_title = next((s.title for s in book.sources if plx_bibcode and s.bibcode == plx_bibcode), None)
    _add_derived(facts, b, book, warnings, extragalactic=extragalactic, any_extragalactic=any_extragalactic,
                 z_input=z_input, rvz_qual=rvz_qual, photometric=photometric, plx=plx, plx_err=plx_err,
                 plx_qual=plx_qual, plx_ref=ref_for("plx_bibcode") if plx is not None else None, plx_title=plx_title)
    await _fill_titles(book, http_client, endpoint)  # sources added by the derived values


def _add_derived(
    facts: ObjectFacts,
    b: Mapping[str, Any],
    book: SourceBook,
    warnings: list[str],
    *,
    extragalactic: bool,
    any_extragalactic: bool,
    z_input: RedshiftInput | None,
    rvz_qual: str | None,
    photometric: bool,
    plx: float | None,
    plx_err: float | None,
    plx_qual: str | None,
    plx_ref: int | None,
    plx_title: str | None = None,
) -> None:
    """Hubble-flow and parallax distances, only where they are physically meaningful."""
    otype = facts.identity.get("otype")
    if z_input is not None:
        ra, dec = _num(b.get("ra")), _num(b.get("dec"))
        if not extragalactic:
            warnings.append(f"No cosmological distances derived: SIMBAD object type {otype!r} is not one of the "
                            f"galaxy/AGN classes, so its redshift {z_input.value} is not treated as a Hubble-flow "
                            "redshift.")
        elif rvz_qual not in GOOD_QUALITY:
            warnings.append(f"No cosmological distances derived: SIMBAD redshift quality is {rvz_qual or 'unknown'} "
                            "(only A-C are used).")
        elif z_input.error is None:
            warnings.append(f"No cosmological distances derived: SIMBAD gives neither an error nor a precision for the "
                            f"{z_input.source} {z_input.value}, so no uncertainty can be stated.")
        elif ra is not None and dec is not None:
            z_hel, z_err = z_input.value, z_input.error
            z_cmb = cmb_frame_redshift(z_hel, ra, dec)
            if z_cmb >= HUBBLE_FLOW_MIN_Z:
                planck_ref = book.add(kind="cosmology", bibcode=PLANCK18_BIBCODE, label="Planck18 cosmology")
                dipole_ref = book.add(kind="cosmology", bibcode=PLANCK18_DIPOLE_BIBCODE, label="Planck 2018 CMB dipole")
                pec_ref = book.add(kind="cosmology", bibcode=PECULIAR_VELOCITY_BIBCODE,
                                   label="peculiar-velocity scatter (Pantheon+)")
                if z_input.error_inferred:
                    err_text = (f"SIMBAD gives no error for this {z_input.source}; a measurement error of "
                                f"{_sig2(z_err):g} (half a unit in its last stated digit) is assumed")
                    err_label = "the assumed measurement error (SIMBAD gives none)"
                else:
                    err_text = f"the error {_sig2(z_err):g} is the SIMBAD measurement error only"
                    err_label = "the SIMBAD redshift error"
                digits = 6 if z_input.decimals is None else max(0, min(z_input.decimals, 6))
                facts.derived.append(_measure(
                    "redshift_cmb_frame", round(z_cmb, digits), error=_sig2(z_err), ref=dipole_ref,
                    error_inferred=True if z_input.error_inferred else None,
                    note=f"heliocentric SIMBAD {'photometric ' if photometric else ''}{z_input.source} "
                         f"{_round_with_error(z_hel, z_err)} moved to the CMB rest frame with the Planck 2018 solar "
                         f"dipole (369.82 km/s); {err_text}",
                ))
                sigma_z = hubble_flow_redshift_error(z_err)
                note = (f"Hubble-flow value from the CMB-frame redshift with astropy Planck18 (flat LCDM); the luminosity "
                        f"distance is (1 + heliocentric z) times the transverse comoving distance at the CMB-frame "
                        f"redshift. The uncertainty combines {err_label} with a peculiar-velocity scatter "
                        f"of {PECULIAR_VELOCITY_KMS:g} km/s (see error_refs) in quadrature (total redshift uncertainty "
                        f"{sigma_z:.2g}); the object's actual peculiar velocity is unknown."
                        + (" Based on a PHOTOMETRIC redshift." if photometric else ""))
                for quantity, value, plus, minus, unit in _hubble_flow_distances(z_cmb, z_hel, z_err):
                    facts.derived.append(_measure(quantity, value, unit=unit, ref=planck_ref, plus_error=plus,
                                                  minus_error=minus, note=note, error_refs=[pec_ref]))
    if plx is not None and plx > 0 and plx_err and plx / plx_err >= PARALLAX_MIN_SNR:
        if any_extragalactic:
            warnings.append(f"Parallax distance withheld: SIMBAD type {otype!r} (other types {facts.identity.get('all_otypes')}) "
                            f"is extragalactic, so the parallax {plx} mas is not a distance indicator.")
        elif plx_qual not in GOOD_QUALITY:
            warnings.append(f"Parallax distance withheld: SIMBAD parallax quality is {plx_qual or 'unknown'}.")
        else:
            value = 1000.0 / plx
            plus = 1000.0 / (plx - plx_err) - value  # plx / plx_err >= 5, so plx - plx_err > 0
            minus = value - 1000.0 / (plx + plx_err)
            scale = max(plus, minus)
            note = (f"1/parallax with the 1-sigma range 1/(parallax -+ error), given because parallax/error = "
                    f"{plx / plx_err:.1f} >= {PARALLAX_MIN_SNR:g}")
            extra: dict[str, Any] = {}
            release, is_catalogue = gaia_release_of(str(b.get("plx_bibcode") or ""), plx_title)
            if release is not None:
                extra["gaia_release"] = release.name
                basis = "catalogue value" if is_catalogue else "per its reference title, based on"
                if release.zero_point_mas is not None and release.zero_point_bibcode:
                    zp_ref = book.add(kind="measurement", bibcode=release.zero_point_bibcode,
                                      label=f"{release.name} parallax zero point")
                    zp_fraction = abs(release.zero_point_mas) / plx
                    if is_catalogue:
                        note += (f". {release.name} {basis}: its zero-point offset (median about "
                                 f"{release.zero_point_mas} mas, depending on magnitude, colour and position; see "
                                 f"zero_point_ref) is NOT applied; it is about {100 * zp_fraction:.2g}% of this parallax")
                        if zp_fraction > plx_err / plx:
                            note += ", larger than the formal parallax error, so the distance is biased by more than its error"
                    else:
                        note += (f". A parallax {basis} {release.name}; whether the {release.name} zero-point offset "
                                 f"(median about {release.zero_point_mas} mas, see zero_point_ref; about "
                                 f"{100 * zp_fraction:.2g}% of this parallax) was corrected is not recorded here")
                    extra["zero_point_ref"] = zp_ref
                elif not is_catalogue:
                    note += f". A parallax {basis} {release.name}"
                if plx_err < release.systematic_floor_mas:
                    sys_ref = book.add(kind="measurement", bibcode=release.systematic_floor_bibcode,
                                       label=f"{release.name} parallax systematics")
                    note += (f". Its formal error ({plx_err:g} mas) is below the {release.systematic_floor_mas:g} mas "
                             f"{release.systematic_floor_note} (see systematics_ref), so the distance errors "
                             "understate the true uncertainty")
                    extra["systematics_ref"] = sys_ref
            facts.derived.append(_measure(
                "parallax_distance", _round_with_error(value, scale), unit="pc", ref=plx_ref,
                plus_error=_round_with_error(plus, scale), minus_error=_round_with_error(minus, scale),
                note=note, **extra,
            ))


# ---------------------------------------------------------------------------
# Explanation: Claude Writer
# ---------------------------------------------------------------------------

_EXPLAIN_SYSTEM_PROMPT = """You write short, factual explanations of astronomical objects for a research \
database. You receive a JSON object of facts; each fact carries "ref", the number of an entry in "sources".

Rules:
- Use ONLY the provided facts. Do not add knowledge from memory: no extra numbers, dates, discoverers, physical \
interpretation or history that the JSON does not state. A reference title may be used only to say what that \
paper studies.
- Put a citation marker [n] after every sentence that states a fact, where n is the "ref" of each fact used; \
several markers may be combined as [2, 5]. Only cite numbers that appear in "sources".
- Copy numbers as given (rounding is fine) together with their stated units and errors. Do not convert units or \
compute new quantities; derived values are already provided in "derived", with their notes. Respect every "note": \
a photometric redshift must be called photometric, "cz" is not a velocity through space, and a withheld quantity \
must not be supplied.
- Crossmatch "fields" have no units (see "field_units"): quote them only with their column name and never attach a \
unit to them.
- Crossmatch classes: only "detections" may be called detections ("listed_in" are literature compilations, not \
observations). For "ambiguous" say that the nearest catalogued source may be unrelated. For \
"no_source_within_radius" say only that the catalog has no source within the given radius; a "nearest_row_arcsec" \
row is not associated with the object. "outside_coverage" and "coverage_unknown" are NOT non-detections: never \
describe them as undetected or faint. A null "chance_coincidence_probability" means unknown, not zero.
- If "query" has an "identification" note, state it: the described SIMBAD object may not be the one named.
- Some facts carry extra source numbers ("error_refs", "zero_point_ref") for their notes; cite them when using \
those notes.
- If sources disagree (for example different object types in SIMBAD and NED), say so rather than choosing.
- Write 120-220 words of plain prose (no headings, no lists, no markdown)."""

_EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string", "description": "The explanation with inline [n] citation markers."}},
    "required": ["summary"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class ExplainResult:
    """Result of :func:`explain_object`."""

    object: dict[str, Any]
    summary: str | None
    citations: list[Source]
    facts: ObjectFacts
    model: str | None
    warnings: list[str] = field(default_factory=list)
    unverified_numbers: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "object": self.object,
            "summary": self.summary,
            "citations": [c.as_dict() for c in self.citations],
            "facts": self.facts.as_dict(),
            "model": self.model,
            "warnings": self.warnings,
            "unverified_numbers": self.unverified_numbers,
        }


async def write_explanation(
    facts: ObjectFacts, *, anthropic_client: Any, settings: AISettings | None = None
) -> tuple[str, list[Source], list[str], list[str], str]:
    """Have Claude write the explanation; returns (summary, citations, warnings, unverified, model)."""
    active = settings or AISettings.from_env()
    payload = facts.prompt_payload()
    response = await _create_message(
        anthropic_client,
        active,
        system=_EXPLAIN_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": "Facts (JSON):\n" + json.dumps(payload, ensure_ascii=False, default=str) + "\n\nWrite the explanation.",
        }],
        output_config={"format": {"type": "json_schema", "schema": _EXPLAIN_SCHEMA}},
    )
    if getattr(response, "stop_reason", None) == "max_tokens":
        raise AIUpstreamError("Claude's explanation was truncated (max_tokens).")
    text = _text_of(response).strip()
    try:
        summary = str(json.loads(text)["summary"]).strip()
    except (ValueError, KeyError, TypeError) as exc:
        raise AIUpstreamError(f"Claude returned an explanation that is not the expected JSON: {text[:200]!r}") from exc
    if not summary:
        raise AIUpstreamError("Claude returned an empty explanation.")
    mapped, cited, warnings = map_citations(summary, facts.sources)
    return mapped, cited, warnings, unverified_numbers(mapped, payload), str(getattr(response, "model", None) or active.model)


async def explain_object(
    *,
    http_client: httpx.AsyncClient,
    name: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    epoch: float | None = None,
    anthropic_client: Any | None = None,
    service: CrossmatchLike | None = None,
    include_crossmatch: bool = True,
    radius_arcsec: float = 5.0,
    crossmatch_radius_arcsec: float = DEFAULT_CROSSMATCH_RADIUS_ARCSEC,
    max_references: int = 8,
    facts_only: bool = False,
    settings: AISettings | None = None,
    ads_token: str | None = None,
    resolver: SesameResolver | None = None,
    registry: CatalogRegistry | None = None,
) -> ExplainResult:
    """Gather facts (see :func:`gather_facts`) and, unless ``facts_only``, a cited summary.

    ``anthropic_client`` defaults to :func:`build_anthropic_client` (raises
    AINotConfiguredError before any network work when no credentials are configured).
    Raises :class:`AINoFactsError` (and makes no Claude call) when there is no SIMBAD
    object and no crossmatch detection at the position.
    """
    active = settings or AISettings.from_env()
    client = None
    if not facts_only:
        client = anthropic_client if anthropic_client is not None else build_anthropic_client(active)
    facts = await gather_facts(
        http_client=http_client, name=name, ra=ra, dec=dec, epoch=epoch, service=service,
        include_crossmatch=include_crossmatch, radius_arcsec=radius_arcsec,
        crossmatch_radius_arcsec=max(crossmatch_radius_arcsec, radius_arcsec), max_references=max_references,
        ads_token=ads_token if ads_token is not None else (os.getenv("ADS_API_TOKEN") or None), resolver=resolver,
        registry=registry,
    )
    obj = {**facts.object, "query": facts.query}
    if facts_only:
        return ExplainResult(object=obj, summary=None, citations=[], facts=facts, model=None, warnings=list(facts.warnings))
    if facts.object.get("simbad_oid") is None and not (facts.crossmatch or {}).get("detected_in"):
        raise AINoFactsError(
            "Nothing is catalogued at this position (no SIMBAD object within the radius and no crossmatch "
            "detection), so there is nothing to explain."
        )
    summary, cited, warnings, unverified, model = await write_explanation(facts, anthropic_client=client, settings=active)
    return ExplainResult(object=obj, summary=summary, citations=cited, facts=facts, model=model,
                         warnings=list(facts.warnings) + warnings, unverified_numbers=unverified)


# ---------------------------------------------------------------------------
# FastAPI Router (/api/v1/ai)
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1/ai", tags=["ai"])


class AIQueryRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000, description="Natural-language search request.")
    max_retries: int = Field(2, ge=0, le=2, description="Validation corrections allowed after the first attempt.")
    verify_adql: bool = Field(False, description="Let the TAP service check the ADQL (runs it with MAXREC=1).")


class AIQueryResponse(BaseModel):
    advanced_query: dict[str, Any] | None = Field(
        description="AdvancedQuery.to_dict() for scope 'cone'; null for scope 'all_sky' (no position: see adql)."
    )
    plan: list[str]
    explanation: str
    adql: str | None
    adql_catalog: str | None
    adql_endpoint: str | None
    scope: str
    resolved_objects: list[dict[str, Any]]
    attempts: int
    validation_history: list[list[str]]
    warnings: list[str]
    model: str


class AIExplainRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=200)
    ra: float | None = Field(None, ge=0.0, lt=360.0)
    dec: float | None = Field(None, ge=-90.0, le=90.0)
    epoch: float | None = Field(None, ge=1900.0, le=2100.0,
                                description="Julian year of ra/dec (e.g. 2016.0 for Gaia DR3); SIMBAD proper motions "
                                            "are applied. Default: J2000 positions.")
    radius_arcsec: float = Field(5.0, gt=0.0, le=60.0, description="SIMBAD identification / association radius.")
    crossmatch_radius_arcsec: float = Field(DEFAULT_CROSSMATCH_RADIUS_ARCSEC, gt=0.0, le=120.0,
                                            description="Catalog search cone for the crossmatch facts.")
    include_crossmatch: bool = True
    max_references: int = Field(8, ge=1, le=25)
    facts_only: bool = False

    @model_validator(mode="after")
    def _one_target(self) -> AIExplainRequest:
        has_coords = self.ra is not None and self.dec is not None
        if (self.name is None) == (not has_coords) or ((self.ra is None) != (self.dec is None)):
            raise ValueError("Give either name or both ra and dec.")
        if self.epoch is not None and self.name is not None:
            raise ValueError("epoch applies to ra/dec only.")
        return self


class CitationOut(BaseModel):
    n: int
    kind: str
    bibcode: str | None
    title: str | None
    year: int | None
    url: str | None
    label: str | None


class AIExplainResponse(BaseModel):
    object: dict[str, Any]
    summary: str | None
    citations: list[CitationOut]
    facts: dict[str, Any]
    model: str | None
    warnings: list[str]
    unverified_numbers: list[str]


async def _state_anthropic(state: Any) -> Any:
    """The app's Anthropic client (``app.state.anthropic``), built once off the event loop.

    Concurrent first requests share one build: a lock stored on ``state`` (created
    without an intervening await, so creating it is race-free) serialises the build and
    the client is re-checked after acquiring it, so no client is built twice and leaked.
    """
    client = getattr(state, "anthropic", None)
    if client is not None:
        return client
    lock = getattr(state, "anthropic_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        state.anthropic_lock = lock
    async with lock:
        client = getattr(state, "anthropic", None)
        if client is None:
            client = await asyncio.to_thread(build_anthropic_client)
            state.anthropic = client
    return client


async def aclose_state_clients(state: Any) -> None:
    """Close the Anthropic client cached on ``app.state`` (call from the app lifespan shutdown)."""
    client = getattr(state, "anthropic", None)
    if client is not None and hasattr(client, "close"):
        await client.close()
    state.anthropic = None


def _state_registry(state: Any) -> CatalogRegistry:
    """The configured registry: app.state.registry, the service's, or CATALOG_REGISTRY_PATH."""
    registry = getattr(state, "registry", None)
    if registry is None:
        registry = getattr(getattr(state, "service", None), "registry", None)
    if registry is None:
        registry = CatalogRegistry(Settings().catalog_registry_path)
        state.registry = registry
    return registry


@asynccontextmanager
async def _state_http_client(state: Any) -> AsyncIterator[httpx.AsyncClient]:
    client = getattr(state, "client", None)
    if client is not None:
        yield client
        return
    # Fallback only (integration provides app.state.client): build off the event loop,
    # because creating the SSL context blocks for seconds on some systems.
    own = await asyncio.to_thread(httpx.AsyncClient, timeout=60.0, follow_redirects=True)
    async with own:
        yield own


def _http_error(exc: Exception) -> HTTPException:
    """Map module errors to HTTP: 503 no credentials, 500 misconfiguration, 404 nothing
    catalogued, 422 bad/unanswerable input, 502 upstream (Anthropic, SIMBAD, Sesame, TAP)."""
    if isinstance(exc, AINotConfiguredError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, AIConfigurationError):
        return HTTPException(status_code=500, detail=f"AI feature misconfigured on the server: {exc}")
    if isinstance(exc, AINoFactsError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, AIQueryCompilationError):
        return HTTPException(status_code=422, detail={"message": str(exc), "errors": exc.errors, "history": exc.history})
    if isinstance(exc, AIRefusalError):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (AIUpstreamError, UpstreamServiceError)):
        return HTTPException(status_code=502, detail=str(exc))
    if isinstance(exc, ObjectResolutionError):
        if _is_sesame_service_failure(exc):
            return HTTPException(status_code=502, detail=str(exc))
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (ValueError, InvalidCoordinateError)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, (CatalogQueryError, ResponseParseError)):
        return HTTPException(status_code=502, detail=f"Upstream failure: {exc}")
    return HTTPException(status_code=502, detail=f"Upstream failure: {exc.__class__.__name__}: {exc}")


_HANDLED = (AstroSearchError, ValueError, httpx.HTTPError)


@router.post("/query", response_model=AIQueryResponse)
async def ai_query_endpoint(req: AIQueryRequest, request: Request) -> dict[str, Any]:
    """Compile natural language into a validated AdvancedQuery, a plan and optional ADQL."""
    state = request.app.state
    try:
        check_query_request(req.text, req.max_retries)  # bad input is 422 even without credentials
        settings = AISettings.from_env()
        client = await _state_anthropic(state)
        async with _state_http_client(state) as http_client:
            compiled = await compile_query(
                req.text, anthropic_client=client, registry=_state_registry(state), http_client=http_client,
                max_retries=req.max_retries, verify_adql=req.verify_adql, settings=settings,
            )
    except _HANDLED as exc:
        raise _http_error(exc) from exc
    return compiled.as_dict()


@router.post("/explain", response_model=AIExplainResponse)
async def ai_explain_endpoint(req: AIExplainRequest, request: Request) -> dict[str, Any]:
    """Explain an object from gathered facts with inline [n] citations mapped to bibcodes."""
    state = request.app.state
    try:
        settings = AISettings.from_env()
        client = None if req.facts_only else await _state_anthropic(state)
        async with _state_http_client(state) as http_client:
            service = getattr(state, "service", None)
            if service is None and req.include_crossmatch:
                from main import build_service

                service = build_service(client=http_client)
            result = await explain_object(
                http_client=http_client, name=req.name, ra=req.ra, dec=req.dec, epoch=req.epoch,
                anthropic_client=client, service=service, include_crossmatch=req.include_crossmatch,
                radius_arcsec=req.radius_arcsec, crossmatch_radius_arcsec=req.crossmatch_radius_arcsec,
                max_references=req.max_references, facts_only=req.facts_only, settings=settings,
                registry=_state_registry(state),
            )
    except _HANDLED as exc:
        raise _http_error(exc) from exc
    return result.as_dict()


# ---------------------------------------------------------------------------
# Command Line Interface: ask / explain
# ---------------------------------------------------------------------------

# Factories are module attributes so tests (and embedding applications) can replace them.
anthropic_factory: Callable[[], Any] = build_anthropic_client


def _http_client_factory() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=60.0, follow_redirects=True)


def _service_factory(client: httpx.AsyncClient) -> CrossmatchLike:
    from main import build_service

    return build_service(client=client)


async def _aclose(client: Any) -> None:
    close = getattr(client, "close", None)
    if close is not None:
        result = close()
        if asyncio.iscoroutine(result):
            await result


def _print_compiled(compiled: CompiledQuery) -> None:
    print(f"Scope: {compiled.scope}   (model {compiled.model}, {compiled.attempts} attempt(s))")
    for obj in compiled.resolved_objects:
        print(f"Resolved: {obj['query']} -> {obj['canonical_name']} (RA {obj['ra_deg']:.6f}, Dec {obj['dec_deg']:.6f}, "
              f"{obj.get('object_type')})")
    print("\nPlan:")
    for i, step in enumerate(compiled.plan, 1):
        print(f"  {i}. {step}")
    print(f"\nInterpretation: {compiled.explanation}")
    if compiled.advanced_query:
        q = compiled.advanced_query
        print(f"\nAdvancedQuery: RA {q['target']['ra']:.6f} Dec {q['target']['dec']:.6f} radius {q['radius_arcsec']}\" "
              f"catalogs={q['catalogs']} profiles={q['profiles']} object_types={q['object_types']} mode={q['search_mode']}")
    if compiled.adql:
        print(f"\nADQL ({compiled.adql_catalog}):\n  {compiled.adql}")
    for warning in compiled.warnings:
        print(f"Warning: {warning}")


async def _run_ask(args: argparse.Namespace) -> int:
    check_query_request(args.question, args.max_retries)  # bad input exits 2 before credentials are needed
    _check_row_limit(args.limit)
    client = anthropic_factory()
    try:
        async with _http_client_factory() as http_client:
            # Compile against the registry the executing service uses (CATALOG_REGISTRY_PATH).
            service = _service_factory(http_client)
            registry = getattr(service, "registry", None) or CatalogRegistry(Settings().catalog_registry_path)
            compiled = await compile_query(
                args.question, anthropic_client=client, registry=registry, http_client=http_client,
                max_retries=args.max_retries, verify_adql=args.verify_adql,
            )
            output: dict[str, Any] = {"compiled": compiled.as_dict()}
            if args.run:
                output["result"] = await execute_compiled_query(
                    compiled, service=service, http_client=http_client, registry=registry, row_limit=args.limit,
                )
    finally:
        await _aclose(client)
    if args.json:
        print(json.dumps(output, indent=2, default=str))
        return 0
    _print_compiled(compiled)
    result = output.get("result")
    if result and result["kind"] == "crossmatch":
        record = result["record"]
        print(f"\nCrossmatch: {record['catalogs_queried']} catalogs, {len(record['crossmatch_groups'])} object group(s)")
        for wave, sources in record["counterparts"].items():
            print(f"  [{wave}] {len(sources)} match(es)")
    elif result:
        print(f"\n{len(result['rows'])} row(s) from {result['catalog']}{' (truncated)' if result['truncated'] else ''}")
        for row in result["rows"][:20]:
            print("  " + ", ".join(f"{k}={v}" for k, v in row.items()))
    return 0


def _check_explain_args(args: argparse.Namespace) -> None:
    has_coords = args.ra is not None and args.dec is not None
    if (args.name is None) == (not has_coords) or ((args.ra is None) != (args.dec is None)):
        raise ValueError("Give either --name or both --ra and --dec.")
    if args.ra is not None and not 0.0 <= args.ra < 360.0:
        raise ValueError("--ra must be within [0, 360) degrees.")
    if args.dec is not None and not -90.0 <= args.dec <= 90.0:
        raise ValueError("--dec must be within [-90, 90] degrees.")
    if args.epoch is not None and args.name is not None:
        raise ValueError("--epoch applies to --ra/--dec only.")


async def _run_explain(args: argparse.Namespace) -> int:
    _check_explain_args(args)
    client = None if args.facts_only else anthropic_factory()
    try:
        async with _http_client_factory() as http_client:
            service = _service_factory(http_client) if args.crossmatch else None
            result = await explain_object(
                http_client=http_client, name=args.name, ra=args.ra, dec=args.dec, epoch=args.epoch,
                anthropic_client=client, service=service, include_crossmatch=args.crossmatch, radius_arcsec=args.radius,
                crossmatch_radius_arcsec=args.crossmatch_radius, max_references=args.max_references,
                facts_only=args.facts_only, registry=getattr(service, "registry", None),
            )
    finally:
        if client is not None:
            await _aclose(client)
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0
    obj = result.object
    print(f"{obj.get('main_id') or obj.get('query')}  (RA {obj.get('ra_deg')}, Dec {obj.get('dec_deg')})")
    if result.summary:
        print("\n" + result.summary + "\n")
        for c in result.citations:
            print(f"[{c.n}] {c.bibcode or c.label}  {c.title or ''} {c.url or ''}".rstrip())
    else:
        print(json.dumps(result.facts.as_dict(), indent=2, default=str))
    for warning in result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if result.unverified_numbers:
        print(f"Warning: numbers not traced to facts: {', '.join(result.unverified_numbers)}", file=sys.stderr)
    return 0


def _cli(runner: Callable[[argparse.Namespace], Any]) -> Callable[[argparse.Namespace], int]:
    """Wrap an async runner: exit 0 ok, 1 error, 2 bad input, 3 no credentials; never a traceback."""

    def handler(args: argparse.Namespace) -> int:
        for stream in (sys.stdout, sys.stderr):
            try:  # paper titles contain Greek letters etc.; never crash on a legacy console codepage
                stream.reconfigure(errors="replace")  # type: ignore[union-attr]
            except (AttributeError, ValueError):
                pass
        try:
            return int(asyncio.run(runner(args)))
        except AINotConfiguredError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 3
        except (ValueError, InvalidCoordinateError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        except ObjectResolutionError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1 if _is_sesame_service_failure(exc) else 2  # an unknown name is bad input
        except (AstroSearchError, httpx.HTTPError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            if isinstance(exc, AIQueryCompilationError):
                for err in exc.errors:
                    print(f"  - {err}", file=sys.stderr)
            return 1

    return handler


def _row_limit_arg(value: str) -> int:
    """argparse type for --limit: an integer in [1, ADQL_MAX_TOP] (argparse exits 2 otherwise)."""
    try:
        return _check_row_limit(int(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--limit must be an integer between 1 and {ADQL_MAX_TOP}") from exc


def register_cli(subparsers: Any) -> None:
    """Add the ``ask`` and ``explain`` subcommands (handler via ``set_defaults(handler=...)``)."""
    ask = subparsers.add_parser("ask", help="Compile a natural-language request into a validated query (Claude)")
    ask.add_argument("question", help="e.g. \"quasars near M87 with radio emission\"")
    ask.add_argument("--run", action="store_true", help="Execute the compiled query (crossmatch or ADQL)")
    ask.add_argument("--limit", type=_row_limit_arg, default=200,
                     help=f"Row limit for ADQL execution, 1-{ADQL_MAX_TOP} (default 200)")
    ask.add_argument("--verify-adql", action="store_true", help="Let the TAP service check the ADQL (MAXREC=1)")
    ask.add_argument("--max-retries", type=int, choices=[0, 1, 2], default=2, help="Validation corrections allowed")
    ask.add_argument("--json", action="store_true", help="Print JSON")
    ask.set_defaults(handler=_cli(_run_ask))

    explain = subparsers.add_parser("explain", help="Explain an object from SIMBAD/crossmatch facts with citations (Claude)")
    explain.add_argument("--name", help="Object name (resolved with CDS Sesame)")
    explain.add_argument("--ra", type=float, help="Right ascension (deg, ICRS)")
    explain.add_argument("--dec", type=float, help="Declination (deg, ICRS)")
    explain.add_argument("--epoch", type=float, help="Julian year of --ra/--dec (e.g. 2016.0 for Gaia DR3); default J2000")
    explain.add_argument("--radius", type=float, default=5.0, help="SIMBAD identification / association radius in arcsec (default 5)")
    explain.add_argument("--crossmatch-radius", type=float, default=DEFAULT_CROSSMATCH_RADIUS_ARCSEC,
                         help=f"Catalog search cone in arcsec (default {DEFAULT_CROSSMATCH_RADIUS_ARCSEC:g})")
    explain.add_argument("--no-crossmatch", dest="crossmatch", action="store_false", help="Skip the catalog crossmatch")
    explain.add_argument("--max-references", type=int, default=8, help="References per bibliography list, 1-25 (default 8)")
    explain.add_argument("--facts-only", action="store_true", help="Gather facts without calling Claude")
    explain.add_argument("--json", action="store_true", help="Print JSON")
    explain.set_defaults(handler=_cli(_run_explain))


def _iter_schema_objects(schema: Any) -> Iterable[Mapping[str, Any]]:
    """Yield every object node of a JSON schema (used by tests to check strictness)."""
    if isinstance(schema, Mapping):
        if schema.get("type") == "object":
            yield schema
        for value in schema.values():
            yield from _iter_schema_objects(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _iter_schema_objects(item)

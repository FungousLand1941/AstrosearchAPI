"""Query building, spatial geometry, proper-motion epoch propagation, and crossmatching engine.

Associations between the target and catalogue rows, and between rows of different
catalogues, are Bayesian (Budavari & Szalay 2008, ApJ 679, 301; NWAY, Salvato et al. 2018,
MNRAS 473, 4937): see :mod:`astrometry`. Every row is first brought to the target's epoch
(its own proper motion, else the target's motion and parallax) with its positional
covariance grown by the proper-motion uncertainty over the epoch difference; coordinates
without an epoch are matched over J2000 / J2016. A match's ``confidence`` is the posterior
probability that the row is the target's counterpart, with the prior probability of a
counterpart set by the target's class (a star's radio / X-ray counterparts are rare and
distance dependent) and the target's positional error including the rounding of the
coordinates as typed. The row of an identity catalogue (SIMBAD, NED) that is the target --
the resolved name's, or the one at exactly the searched position -- gets identity prior
odds. An extended target (a cluster, nebula or remnant) gets the scatter of its catalogued
centres on its position, its identity listed by other compilations as identities, and the
point-source prior on compact rows (stars, galaxies, compact sources inside it). Rows of one
catalogue that are one object listed several times (a compilation's duplicate entries, a
fast mover detected at several epochs) share one association. Crowded stellar systems get
their local density measured (a density probe started as soon as the catalogue answers).
:meth:`CrossmatchService.crossmatch_stream` yields results catalogue by catalogue.
"""

from __future__ import annotations

import asyncio
import math
import re
import statistics
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from time import monotonic
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from astrometry import (
    ASTROMETRIC_FLOOR_ARCSEC,
    COINCIDENT_ARCSEC,
    DEFAULT_PRIOR_COMPLETENESS,
    DEFAULT_TARGET_PM_SIGMA_MASYR,
    DEFAULT_TARGET_SIGMA_ARCSEC,
    DENSITY_MAP_SCALING,
    DUPLICATE_MAX_CHI2,
    EXTENDED_EMISSION_MIN_ARCSEC,
    EXTENDED_POINT_COMPLETENESS,
    IDENTITY_PRIOR_LN_ODDS,
    LINK_CHI2,
    MAX_SIGMA_ARCSEC,
    TARGET_CLASSES,
    UNDATED_EPOCH_SPREAD_YR,
    UNDATED_TARGET_EPOCHS,
    AssociationConfig,
    AssociationResult,
    Detection,
    SkyDensity,
    _along_track,
    _moved,
    _row_span,
    associate,
    completeness_priors,
    cone_area_deg2,
    crowded_region,
    detection_covariance,
    emission_extent_arcsec,
    estimate_density_deg2,
    local_prior_density,
    placement_chi2_matrix,
    pm_sigma_masyr,
    source_detection,
    target_chi2,
)
from models import (
    TARGET_PM_METHODS,
    CatalogDefinition,
    CatalogFailure,
    CatalogRegistry,
    CatalogSource,
    CatalogUnavailableError,
    InvalidCoordinateError,
    Match,
    QueryPlan,
    QueryTimeoutError,
    Target,
    UnifiedRecord,
    _to_float,
    epoch_separation_arcsec,
    haversine_arcsec,
    is_extragalactic_type,
    propagate_radec,
    source_position_at,
    tangent_offset_arcsec,
    validate_target,
)
from providers import CatalogProvider, QueryResult, classify_sources

# A proper motion is adopted from a matched catalog row only when that row lies this
# close to the target (after propagation), so an unrelated field star is not used.
PM_ADOPTION_MAX_ARCSEC = 2.0
# |z| above this (cz ~ 900 km/s, beyond any Galactic star's radial velocity) marks the
# target's identity row as extragalactic, so its catalog proper motion is not adopted.
EXTRAGALACTIC_MIN_REDSHIFT = 0.003
# A Gaia-like row whose parallax is below this significance and whose proper motion is
# smaller than PM_NOISE_MASYR is not used for adoption: such a motion is consistent with
# a distant or extragalactic source, and ignoring it changes positions by < 0.02"/yr.
PM_ADOPTION_MIN_PARALLAX_SNR = 3.0
PM_NOISE_MASYR = 20.0
# Two candidate motions describe the same object when they differ by less than
# max(PM_AGREE_MASYR, PM_AGREE_FRACTION x |pm|) (catalogs differ by a few mas/yr).
PM_AGREE_MASYR = 10.0
PM_AGREE_FRACTION = 0.1
# A candidate with a DIFFERENT motion closer than 2 x (nearest separation) + this margin
# makes the adoption ambiguous (crowded field, e.g. the S-stars around Sgr A*).
PM_AMBIGUITY_MARGIN_ARCSEC = 0.5
# A parallax is adopted with the motion when it is at least this significant.
PARALLAX_ADOPTION_MIN_SNR = 5.0
# A parallax published without its error (SIMBAD's plx_value; the query does not fetch
# plx_err) is adopted only from this value: SIMBAD takes parallaxes from Gaia and Hipparcos,
# whose errors stay below ~2 mas for the stars they list, so 10 mas is >= 5 sigma (Vega's
# 130 mas is adopted; a quasar's 0.011 mas of Gaia noise is not).
PARALLAX_WITHOUT_ERROR_MIN_MAS = 10.0
# Rows whose parallax could not be removed get the target parallax as extra uncertainty
# when it is at least this large (smaller parallaxes are within catalog errors).
PARALLAX_INFLATION_MIN_MAS = 50.0
# SIMBAD / NED rows that are not point-like identities: extended objects (clouds, regions,
# clusters and groups of stars or galaxies, whose catalogued position is a nominal centre)
# and SIMBAD positions of quality E (>= 10", possibly rounded to whole degrees). Unless the
# row is the target's own identity (see _point_identities), it can only 'contain' the
# target: its position gets this 1-sigma scatter (30'), so a star a few arcsec from a
# cloud's nominal centre is not the cloud.
EXTENDED_IDENTITY_SIGMA_ARCSEC = 1800.0
EXTENDED_OTYPES: frozenset[str] = frozenset({
    # SIMBAD condensed object types (lower case): interstellar matter and regions
    "hvc", "moc", "cld", "gne", "dne", "rne", "hii", "snr", "sfr", "reg", "ism", "bub", "cor", "sh", "cgb",
    "glb", "mgr", "st*", "as*", "cl*", "glc", "opc", "poc", "flt", "pog",
    # groups and clusters of galaxies, voids
    "clg", "grg", "cgg", "scg", "pcg", "void", "vid", "pag", "ig",
    # candidates (current SIMBAD codes and older ones)
    "sfr?", "cl?", "gl?", "as?", "c?g", "gr?", "sc?", "pcg?", "sr?", "cl*?", "glc?", "as*?", "cld?", "clg?",
    "grg?", "scg?",
    # NED types (a leading '!' -- NED's mark of Galactic objects -- is ignored)
    "*ass", "*cl", "gclstr", "ggroup", "gpair", "gtrpl", "qgroup", "neb", "rfn", "mcld", "pofg",
})
# The identity row coincides with the target (it is the object searched) when its offset
# satisfies chi2 = sep^2 / (target_sigma^2 + row_sigma^2) <= this (99% for 2 d.o.f.).
IDENTITY_COINCIDENCE_CHI2 = -2.0 * math.log(0.01)
# The catalogued centre of an extended object is uncertain beyond its formal error: M 13's
# centre is 0.8" apart in SIMBAD and NED, Cas A's 15". The centre sigma of an extended
# target is at least this, at least the error of its identity rows (SIMBAD quality D: 5",
# E: 60") and at least half the largest offset of an identity row from the target (the
# observed spread of the catalogued centres; see _point_identities). It is added in
# quadrature to the target position and to the identity rows in the association, so a
# precise point source at the nominal centre (a star in a nebula, the brightest galaxy of
# a cluster) is not preferred to the object by its positional precision alone.
EXTENDED_CENTRE_SIGMA_ARCSEC = 1.0
# Families of extended object types: an identity row of the target's family at its centre
# is the same object listed by another catalogue (NED's GClstr for SIMBAD's ClG); rows of
# another family are objects inside or near it (the globular clusters of a cluster galaxy,
# the star cluster in a nebula).
EXTENDED_FAMILIES: dict[str, frozenset[str]] = {
    "galaxy_group": frozenset({"clg", "grg", "cgg", "scg", "pcg", "c?g", "gr?", "sc?", "pcg?", "clg?", "grg?", "scg?",
                               "pag", "ig", "void", "vid", "gclstr", "ggroup", "gpair", "gtrpl", "qgroup"}),
    "star_cluster": frozenset({"glc", "opc", "cl*", "as*", "st*", "mgr", "gl?", "cl?", "as?", "cl*?", "glc?", "as*?",
                               "*cl", "*ass"}),
    "supernova_remnant": frozenset({"snr", "sr?"}),
    "nebula": frozenset({"hii", "gne", "rne", "neb", "rfn", "sfr", "sfr?"}),
    "cloud": frozenset({"hvc", "moc", "cld", "dne", "glb", "cor", "cgb", "flt", "poc", "mcld", "cld?", "ism", "bub",
                        "sh", "reg"}),
    "part_of_galaxy": frozenset({"pog", "pofg"}),
}
# A SIMBAD / NED row within this distance of the target position IS the target: the
# coordinates searched are that row's catalogued position (a name resolved by Sesame and
# searched by its coordinates -- POST /api/v1/search, main.search_object -- or coordinates
# copied from SIMBAD). Its prior odds get IDENTITY_PRIOR_LN_ODDS, as a resolved name's row.
# Sesame repeats SIMBAD's stored coordinates (M 42: 83.8201, -5.3876 in both), so they
# coincide to rounding (1e-8 deg = 0.04 mas).
IMPLICIT_IDENTITY_ARCSEC = 1e-3
# SIMBAD / NED types of entries that only record a detection at some wavelength (a radio,
# infrared, X-ray source; no type at all), not a classified object: such an entry coinciding
# within its errors with exactly one classified entry of the same compilation is a second
# listing of that object (SIMBAD's 'NVSS J203225+405728' 0.1" from 'V* V1521 Cyg' = Cyg X-3; NED's
# WISEA entry of a star it also lists by name).
DETECTION_ONLY_OTYPES: frozenset[str] = frozenset({
    "rad", "cm", "mm", "smm", "hi", "rb", "ir", "fir", "mir", "nir", "blu", "uv", "x", "gam", "gb", "ev", "?",
    "radios", "irs", "xrays", "uvs", "uves", "viss", "gammas", "smms", "mms", "other",
})
# A fast mover detected by one survey at several epochs (5XMM lists Proxima Cen three times
# along its track): rows of one catalogue placed with the target's motion that coincide at
# the target's epoch (chi2 <= DUPLICATE_MAX_CHI2 under every epoch hypothesis) while the
# target's motion over their epoch difference explains at least this fraction of their
# measured separation are one object listed at several epochs.
LISTING_MOTION_FRACTION = 0.5
# A detection-only entry is merged with the entry it coincides with only when both positions
# are at least this precise (1-sigma, arcsec): a coarse entry (SIMBAD's X-ray knot of the
# 3C 273 jet, quality D, 4.3" from the quasar) is consistent with several objects and may be
# a distinct component.
DETECTION_LISTING_MAX_SIGMA_ARCSEC = 1.0
# The most probable association must have p_any above this before its members' types
# decide the target's class (star / extragalactic) for the completeness priors.
CLASS_EVIDENCE_MIN_P_ANY = 0.5
# Stellar evidence: a parallax at least this significant (Gaia DR3 counterpart, resolver).
CLASS_PARALLAX_MIN_SNR = 5.0
# SIMBAD / NED types of radio / X-ray / gamma-ray emitters (compact binaries, pulsars,
# novae, generic radio / IR / X-ray sources), and of stars whose emission the stellar
# calibration -- random Gaia stars, mostly quiet dwarfs -- does not describe: Wolf-Rayet,
# luminous blue variables and blue supergiants, symbiotic, Be and emission-line stars,
# planetary nebulae, young (T Tauri, Herbig Ae/Be, YSO) and chromospherically active
# stars (RS CVn, BY Dra, eruptive / flare stars), OH/IR stars and masers. Such a target
# keeps the default priors ('unknown' class).
EMITTER_OTYPES: frozenset[str] = frozenset({
    "psr", "psr?", "n*", "n*?", "bh", "bh?", "xb*", "xb?", "lxb", "lx?", "lxb?", "hxb", "hx?", "hxb?", "ulx", "ux?",
    "cv*", "cv?", "no*", "no?", "sn*", "sn?", "gb", "grb", "gam", "x", "ux", "rad", "cm", "mr", "mm", "smm",
    "mas", "hi", "rb", "ir", "fir", "mir", "nir", "uv", "ev", "grv", "gwe", "lev", "pn", "pn?", "wr*", "wr?",
    "s*b", "s?b", "sy*", "sy?", "be*", "be?", "em*", "ma*", "ma?", "oh*", "oh?", "y*o", "y*?", "tt*", "tt?", "ae*",
    "ae?", "or*", "rs*", "rs?", "by*", "by?", "er*", "er?", "fl*",
    # NED
    "radios", "xrays", "gammas", "irs", "uvs", "uves", "emls", "emobj", "nova", "sn", "flare*",
})
# Types of ordinary stars, for which the stellar counterpart fractions were measured
# (``astrometry.STELLAR_COUNTERPART_FRACTION``: random Gaia DR3 stars).
ORDINARY_STELLAR_OTYPES: frozenset[str] = frozenset({
    "*", "star", "**", "**?", "sb*", "sb?", "eb*", "eb?", "el*", "el?", "v*", "v*?", "var", "pm*", "hpm*", "hv*",
    "ms*", "ms?", "ev*", "ev?", "rg*", "rb?", "hb*", "hb?", "ab*", "ab?", "c*", "c*?", "s*", "s*?", "lp*", "lp?",
    "mi*", "mi?", "sg*", "sg?", "s*r", "s?r", "s*y", "s?y", "ce*", "ce?", "cc*", "rr*", "rr?", "wv*", "wv?", "rv*",
    "rv?", "hs*", "hs?", "wd*", "wd?", "bd*", "bd?", "lm*", "lm?", "bs*", "bs?", "sx*", "ds*", "gd*", "bc*", "bc?",
    "pe*", "pe?", "a2*", "a2?", "ro*", "ro?", "pu*", "pu?", "ir*", "rc*", "rc?", "pa*", "pa?", "pl", "pl?",
    # NED
    "blue*", "red*", "exg*",
})
# Spectral types of hot massive stars (O stars, Wolf-Rayet WN / WC / WO), which are X-ray
# (wind shocks) and radio (free-free) emitters: treated as EMITTER_OTYPES.
HOT_MASSIVE_SPTYPE = re.compile(r"^\s*(?:O\d|O[CN]\d|W[NCOR])", re.IGNORECASE)
# Orbital motion of a binary component makes a linear proper motion wrong over decades
# (Kruger 60, P = 45 yr, a = 2.4": the components' motions differ by ~480 mas/yr). A
# resolver target that is a component (a name ending in a component letter, or a
# double-star type) gets at least this proper-motion uncertainty (per axis): 0.5" over the
# 16 yr between SIMBAD's J2000 and Gaia's J2016. It cannot cover the fastest orbits
# without making neighbouring components indistinguishable (Kruger 60 A's Gaia row keeps
# P = 0.98, B's 2-parameter row 1.3" farther gets 0.01; 100 mas/yr would give 0.57 / 0.38).
BINARY_COMPONENT_PM_SIGMA_MASYR = 30.0
BINARY_OTYPES: frozenset[str] = frozenset({"**", "sb*", "eb*", "el*", "**?", "sb?", "eb?"})
# IAU constellation abbreviations (and their SIMBAD genitive forms end in these): a star
# name ending in 'CrA', 'CrB', 'PsA', 'TrA', 'CMa', ... ends in a capital letter that is
# not a component letter.
CONSTELLATION_ABBREVIATIONS: frozenset[str] = frozenset({
    "And", "Ant", "Aps", "Aqr", "Aql", "Ara", "Ari", "Aur", "Boo", "Cae", "Cam", "Cnc", "CVn", "CMa", "CMi", "Cap",
    "Car", "Cas", "Cen", "Cep", "Cet", "Cha", "Cir", "Col", "Com", "CrA", "CrB", "Crv", "Crt", "Cru", "Cyg", "Del",
    "Dor", "Dra", "Equ", "Eri", "For", "Gem", "Gru", "Her", "Hor", "Hya", "Hyi", "Ind", "Lac", "Leo", "LMi", "Lep",
    "Lib", "Lup", "Lyn", "Lyr", "Men", "Mic", "Mon", "Mus", "Nor", "Oct", "Oph", "Ori", "Pav", "Peg", "Per", "Phe",
    "Pic", "Psc", "PsA", "Pup", "Pyx", "Ret", "Sge", "Sgr", "Sco", "Scl", "Sct", "Ser", "Sex", "Tau", "Tel", "Tri",
    "TrA", "Tuc", "UMa", "UMi", "Vel", "Vir", "Vol", "Vul",
})
# Prefixes of year-letter designations of transients (IAU supernova and transient names,
# novae): 'SN 1987A' is not component A of a star 'SN 1987'.
TRANSIENT_PREFIXES: frozenset[str] = frozenset({"SN", "SNR", "AT", "NOVA", "GRB", "FRB", "TDE", "PSN"})
# Position error assumed for a name resolved by neither SIMBAD nor NED (Sesame's VizieR
# fallback: a catalogue position without errors or epoch). Catalogues of named objects quote
# positions to 0.1-1" at their own epochs; the object's motion since is unknown and is
# warned about (resolved_search_target).
RESOLVER_UNDATED_SIGMA_ARCSEC = 1.0
# Position error of a galaxy's resolved centre when the resolver gives none. SIMBAD takes the
# centres of nearby galaxies from the 2MASS extended-source catalogue (Sesame refPos
# 2006AJ....131.1163S) without an error; the centre of a galaxy arcminutes across differs
# between catalogues by about an arcsecond (NED's 'NGC 4565' 1.2", 'Messier 101' 0.8",
# 'NGC 7318a' 0.7" from SIMBAD's). Point-like extragalactic types (QSO, BL Lac), and objects
# that merely have a redshift (supernovae, novae, X-ray sources), keep the resolver's
# (default) precision (is_galaxy_centre_type).
GALAXY_CENTRE_SIGMA_ARCSEC = 1.0
POINTLIKE_EXTRAGALACTIC_OTYPES: frozenset[str] = frozenset({
    "qso", "qso?", "q?", "bla", "bla?", "bll", "bll?", "bz?", "bl?", "lev?", "gle", "gls", "le?", "ls?", "li?",
    # lensed images / quasars, broad-absorption-line QSOs, and absorbers listed at a QSO's position
    "lei", "leq", "bal", "lya", "dla", "mal", "lls", "als", "q_lens", "qsolens",
})


def is_galaxy_centre_type(object_type: Any) -> bool:
    """True for the object types whose resolved position is a galaxy's centre (and so gets
    GALAXY_CENTRE_SIGMA_ARCSEC when the resolver gives no error): extragalactic types that are
    neither point-like (QSO, BL Lac, lenses) nor groups / clusters of galaxies. Objects that
    merely have a redshift (supernovae, novae, X-ray sources of other galaxies) are not."""
    key = _otype_key(object_type)
    return is_extragalactic_type(key) and key not in POINTLIKE_EXTRAGALACTIC_OTYPES and key not in EXTENDED_OTYPES


# Star clusters of the same compilation closer than this to each other cannot be Galactic
# clusters (arcminutes across): they are extragalactic clusters -- point-like at the distances
# of other galaxies, such as M87's globular clusters ([JPB2009], 1.9" apart) or the young
# star clusters of Stephan's Quintet (NED [FGD2015], 1-4" apart) -- and are matched as compact
# sources, never as an extended object that contains the target.
COMPACT_CLUSTER_NEIGHBOUR_ARCSEC = 10.0
# Largest target position uncertainty accepted (per axis, arcsec).
MAX_TARGET_SIGMA_ARCSEC = 3600.0
# Largest plausible proper motion of a row (mas/yr; Barnard's star moves 10,400 mas/yr):
# larger or non-finite catalogue motions are measurement errors and are left out.
MAX_ROW_PM_MASYR = 20_000.0
# Local density probe: a survey catalogue whose density follows the stars and galaxies
# (astrometry.DENSITY_MAP_SCALING) and whose small cone holds far more field rows than the
# 1.8-deg density map predicts (a globular cluster core, a galaxy) is queried again over a
# cone of this radius to measure the local density: 5 rows in 3" where the map expects
# 0.15 is a small-scale overdensity the map cannot resolve (M 13's core: 2.9e6 per deg^2
# vs a map value of 2.1e4).
DENSITY_PROBE_RADIUS_ARCSEC = 30.0
# Probe when P(N >= field rows | Poisson(DENSITY_PROBE_SLACK x expected)) is below this
# and at least DENSITY_PROBE_MIN_ROWS field rows were found (the slack covers the map's
# 0.1-0.3 dex scatter).
DENSITY_PROBE_P_VALUE = 1e-3
DENSITY_PROBE_SLACK = 3.0
DENSITY_PROBE_MIN_ROWS = 3
POISSON_EXCESS = "poisson excess"
# Time allowed for one density probe (seconds; at most the catalogue's own limit, and no
# fallback archive): a probe that does not answer in time leaves the conservative
# small-cone estimate (see catalog_densities) instead of holding the search back.
DENSITY_PROBE_TIMEOUT_SECONDS = 20.0
# Survey catalogues probed. Not SIMBAD: its overdensities around famous objects are
# literature entries of the object itself (a quasar's radio components and knots).
DENSITY_PROBE_EXCLUDED = frozenset({"simbad"})
DENSITY_PROBE_CATALOGS = frozenset(set(DENSITY_MAP_SCALING) - DENSITY_PROBE_EXCLUDED)

# ---------------------------------------------------------------------------
# Advanced Query Representation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdvancedQuery:
    """Extended query specification with astrophysical filters, geometric constraints, and limits."""

    target: Target
    radius_arcsec: float = 3.0
    profiles: list[str] | None = field(default_factory=list)
    object_types: list[str] | None = field(default_factory=list)
    spectral_types: list[str] | None = field(default_factory=list)
    morphology: list[str] | None = field(default_factory=list)
    count_threshold: int = 5
    min_confidence: float = 0.5
    max_results: int | None = None
    min_radius_arcsec: float = 0.0
    search_mode: str = "cone"
    spatial_constraints: dict[str, Any] = field(default_factory=dict)
    proper_motion: bool = True
    adaptive_radius: bool = False
    min_distance_pc: float | None = None
    max_distance_pc: float | None = None
    time_period: dict[str, Any] | None = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    catalogs: list[str] | None = field(default_factory=list)
    use_resolved_name: bool = False
    resolved_name: str | None = None
    export_format: str = "parquet"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AdvancedQuery:
        """Construct an AdvancedQuery from a JSON dictionary payload."""
        target_data = data.get("target") or {}
        ra = target_data.get("ra", data.get("ra"))
        dec = target_data.get("dec", data.get("dec"))
        if ra is None or dec is None:
            raise ValueError("Coordinates (ra and dec) are required")

        pm_ra = target_data.get("pm_ra_masyr", data.get("pm_ra_masyr"))
        pm_dec = target_data.get("pm_dec_masyr", data.get("pm_dec_masyr"))
        target = validate_target(
            ra, dec, epoch=target_data.get("epoch", data.get("epoch")), pm_ra_masyr=pm_ra, pm_dec_masyr=pm_dec,
            parallax_mas=target_data.get("parallax_mas", data.get("parallax_mas")),
        )
        return cls(
            target=target,
            radius_arcsec=float(data.get("radius_arcsec", 3.0)),
            profiles=data.get("profiles") or None,
            object_types=data.get("object_types") or None,
            spectral_types=data.get("spectral_types") or None,
            morphology=data.get("morphology") or None,
            count_threshold=int(data.get("count_threshold", 5)),
            min_confidence=float(data.get("min_confidence", 0.5)),
            max_results=int(data["max_results"]) if data.get("max_results") is not None else None,
            min_radius_arcsec=float(data.get("min_radius_arcsec", 0.0)),
            search_mode=str(data.get("search_mode", "cone")),
            spatial_constraints=data.get("spatial_constraints") or {},
            proper_motion=bool(data.get("proper_motion", True)),
            adaptive_radius=bool(data.get("adaptive_radius", False)),
            min_distance_pc=float(data["min_distance_pc"]) if data.get("min_distance_pc") is not None else None,
            max_distance_pc=float(data["max_distance_pc"]) if data.get("max_distance_pc") is not None else None,
            time_period=data.get("time_period") or None,
            filters=data.get("filters") or {},
            catalogs=data.get("catalogs") or None,
            use_resolved_name=bool(data.get("use_resolved_name", False)),
            resolved_name=data.get("resolved_name"),
            export_format=str(data.get("export_format", "parquet")),
            metadata=data.get("metadata") or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert query into JSON-serializable representation."""
        return {
            "target": self.target.as_dict(),
            "radius_arcsec": self.radius_arcsec,
            "profiles": self.profiles,
            "object_types": self.object_types,
            "spectral_types": self.spectral_types,
            "morphology": self.morphology,
            "count_threshold": self.count_threshold,
            "min_confidence": self.min_confidence,
            "max_results": self.max_results,
            "min_radius_arcsec": self.min_radius_arcsec,
            "search_mode": self.search_mode,
            "spatial_constraints": self.spatial_constraints,
            "proper_motion": self.proper_motion,
            "adaptive_radius": self.adaptive_radius,
            "min_distance_pc": self.min_distance_pc,
            "max_distance_pc": self.max_distance_pc,
            "time_period": self.time_period,
            "filters": self.filters,
            "catalogs": self.catalogs,
            "use_resolved_name": self.use_resolved_name,
            "resolved_name": self.resolved_name,
            "export_format": self.export_format,
            "metadata": self.metadata,
        }

    def apply_filters(self, source: dict[str, Any]) -> bool:
        """Apply astrophysical, geometric, distance, and temporal constraints to a detection."""
        physical = source.get("physical") or source.get("metadata", {}).get("physical", {})

        # Object type filtering
        if self.object_types:
            obj_type = physical.get("object_type") or source.get("data", {}).get("object_type")
            if not obj_type or self._canonical_type(obj_type) not in {self._canonical_type(x) for x in self.object_types}:
                return False

        # Spectral type filtering
        if self.spectral_types:
            sp_type = physical.get("spectral_type") or source.get("data", {}).get("sp_type")
            if not sp_type or str(sp_type).casefold() not in {x.casefold() for x in self.spectral_types}:
                return False

        # Morphology filtering
        if self.morphology:
            morph = physical.get("morphology") or source.get("data", {}).get("morphology")
            if not morph or str(morph).casefold() not in {x.casefold() for x in self.morphology}:
                return False

        # Shell search min radius
        separation = source.get("separation_arcsec")
        if self.search_mode == "shell" and separation is not None and separation < self.min_radius_arcsec:
            return False

        # 3D Cylinder distance bounds via parallax inversion
        if self.search_mode == "cylinder":
            parallax = physical.get("parallax") or source.get("data", {}).get("parallax")
            dist = source.get("data", {}).get("distance_pc")
            try:
                distance_pc = float(dist) if dist is not None else 1000.0 / float(parallax)
            except (TypeError, ValueError, ZeroDivisionError):
                return False
            if distance_pc <= 0:
                return False
            if self.min_distance_pc is not None and distance_pc < self.min_distance_pc:
                return False
            if self.max_distance_pc is not None and distance_pc > self.max_distance_pc:
                return False

        # Spatial radius zones
        zones = self.spatial_constraints.get("radius_zones", [])
        if zones and separation is not None and not any(
            float(zone.get("min_arcsec", 0)) <= separation <= float(zone["max_arcsec"]) for zone in zones
        ):
            return False

        # Time period observation windows
        if self.time_period:
            observed = physical.get("observation_date") or source.get("data", {}).get("observation_date") or source.get("epoch")
            obs_year = self._year(observed)
            if obs_year is None:
                return False
            start = self._year(self.time_period.get("start_year", self.time_period.get("start")))
            end = self._year(self.time_period.get("end_year", self.time_period.get("end")))
            if start is not None and obs_year < start:
                return False
            if end is not None and obs_year > end:
                return False

        # Ray-casting exclusion polygons
        for polygon in self.spatial_constraints.get("exclude_polygons", []):
            if self._inside_polygon(float(source["ra"]), float(source["dec"]), polygon):
                return False

        return True

    @staticmethod
    def _canonical_type(value: Any) -> str:
        name = str(value).strip().casefold()
        return {
            "*": "star", "star": "star", "g": "galaxy", "galaxy": "galaxy",
            "qso": "quasar", "quasar": "quasar", "neb": "nebula",
            "nebula": "nebula", "cl*": "star_cluster", "star cluster": "star_cluster",
            "star_cluster": "star_cluster",
        }.get(name, name)

    @staticmethod
    def _year(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return float(value.year)
        try:
            return float(value)
        except (ValueError, TypeError):
            try:
                return float(datetime.fromisoformat(str(value)).year)
            except ValueError:
                return None

    @staticmethod
    def _inside_polygon(ra: float, dec: float, polygon: list[list[float]]) -> bool:
        """Ray-casting point-in-polygon containment handling 0/360 RA boundary crossing."""
        vertices = [(((float(x) - ra + 180) % 360) - 180, float(y)) for x, y in polygon]
        inside = False
        prev = vertices[-1]
        for curr in vertices:
            if (curr[1] > dec) != (prev[1] > dec):
                crossing = curr[0] + (dec - curr[1]) * (prev[0] - curr[0]) / (prev[1] - curr[1])
                if crossing > 0:
                    inside = not inside
            prev = curr
        return inside


# ---------------------------------------------------------------------------
# Query Validator & Builder
# ---------------------------------------------------------------------------


class QueryValidator:
    """Validates advanced search query constraints."""

    @staticmethod
    def validate(query: AdvancedQuery, registry: CatalogRegistry | None = None) -> bool:
        """Perform comprehensive constraint validation on an AdvancedQuery."""
        if not math.isfinite(query.radius_arcsec) or query.radius_arcsec <= 0:
            raise ValueError("radius_arcsec must be positive")
        if query.count_threshold <= 0:
            raise ValueError("count_threshold must be positive")
        if not math.isfinite(query.min_confidence) or not 0 <= query.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if query.max_results is not None and query.max_results <= 0:
            raise ValueError("max_results must be positive")
        if query.search_mode not in {"cone", "shell", "cylinder"}:
            raise ValueError("search_mode must be cone, shell, or cylinder")
        if query.search_mode == "cylinder" and query.min_distance_pc is None and query.max_distance_pc is None:
            raise ValueError("cylinder searches require a distance bound")
        for dist in (query.min_distance_pc, query.max_distance_pc):
            if dist is not None and (not math.isfinite(dist) or dist <= 0):
                raise ValueError("distance bounds must be positive finite parsecs")
        if query.min_distance_pc is not None and query.max_distance_pc is not None and query.min_distance_pc > query.max_distance_pc:
            raise ValueError("min_distance_pc must not exceed max_distance_pc")
        if not math.isfinite(query.min_radius_arcsec) or query.min_radius_arcsec < 0 or query.min_radius_arcsec >= query.radius_arcsec:
            raise ValueError("min_radius_arcsec must be nonnegative and smaller than radius_arcsec")

        start = query._year((query.time_period or {}).get("start_year", (query.time_period or {}).get("start")))
        end = query._year((query.time_period or {}).get("end_year", (query.time_period or {}).get("end")))
        for k, v in (query.time_period or {}).items():
            if k in {"start_year", "end_year", "start", "end"} and v is not None and query._year(v) is None:
                raise ValueError(f"Invalid time_period {k}")
        if query.time_period and (start is None and end is None or start is not None and end is not None and start > end):
            raise ValueError("Invalid time_period")

        for poly in query.spatial_constraints.get("exclude_polygons", []):
            if not isinstance(poly, list) or len(poly) < 3 or any(not isinstance(pt, list) or len(pt) != 2 for pt in poly):
                raise ValueError("Each exclusion polygon needs at least three [ra, dec] vertices")
            if any(not all(math.isfinite(float(coord)) for coord in pt) or not -90 <= float(pt[1]) <= 90 for pt in poly):
                raise ValueError("Invalid exclusion polygon coordinate")

        for zone in query.spatial_constraints.get("radius_zones", []):
            try:
                inner = float(zone.get("min_arcsec", 0))
                outer = float(zone["max_arcsec"])
            except (TypeError, ValueError, KeyError, AttributeError) as exc:
                raise ValueError("Invalid radius zone") from exc
            if not math.isfinite(inner) or not math.isfinite(outer) or inner < 0 or outer > query.radius_arcsec or outer <= inner:
                raise ValueError("radius zones must lie within radius_arcsec")

        if query.catalogs or query.profiles:
            active_registry = registry or CatalogRegistry()
            for name in query.catalogs or []:
                if name not in active_registry.enabled_catalogs():
                    raise ValueError(f"Unknown catalog: {name}")
            for profile in query.profiles or []:
                validate_profile(profile, active_registry)
            check_catalogs_in_profiles(active_registry, query.catalogs, query.profiles)
        return True


def check_catalogs_in_profiles(registry: CatalogRegistry | None, catalogs: Sequence[str] | None,
                               profiles: Sequence[str] | str | None) -> None:
    """ValueError when ``catalogs`` names a catalogue outside every one of ``profiles``.

    Catalogues are intersected with the profile, so such a catalogue would silently not be
    queried and the search would 'succeed' with nothing from it; it is an input error instead
    (HTTP 422 / CLI exit 2). The one implementation of this rule: :class:`QueryValidator`
    (``POST /api/v1/search``, AI-compiled queries), :meth:`CrossmatchService.prepare` (the
    library's ``crossmatch`` / ``crossmatch_stream``) and ``main.check_catalogs_in_profile``
    (checked before a name is resolved) all call it. A catalogue with no profiles is planned for
    every profile; an unknown name is not judged here (it has its own error)."""
    if isinstance(profiles, str):
        profiles = [profiles]
    profiles = [p for p in profiles or [] if p]
    if not catalogs or not profiles or registry is None:
        return
    enabled = registry.enabled_catalogs()
    outside = [name for name in catalogs if name in enabled and enabled[name].profiles
               and not any(p in enabled[name].profiles for p in profiles)]
    if outside:
        raise ValueError(
            f"Catalog(s) {', '.join(outside)} are not in profile '{', '.join(profiles)}' and would not be queried: "
            "catalogs are intersected with the profile. Omit the profile (or use one that includes them) to query "
            "these catalogs.")


def known_profiles(registry: CatalogRegistry) -> set[str]:
    """Profiles declared by at least one enabled catalog."""
    return {p for catalog in registry.enabled_catalogs().values() for p in catalog.profiles}


def validate_profile(profile: str | None, registry: CatalogRegistry) -> None:
    """Raise ValueError for a profile no enabled catalog declares (a typo would otherwise
    plan zero catalogs and return an empty but 'successful' result)."""
    if profile is None:
        return
    enabled = registry.enabled_catalogs().values()
    if any(not catalog.profiles for catalog in enabled):
        return  # a catalog without profiles is planned for every profile
    known = known_profiles(registry)
    if profile not in known:
        raise ValueError(f"Unknown profile '{profile}'; known profiles: {', '.join(sorted(known))}")


class QueryBuilder:
    """Builds catalog-specific query plans from an AdvancedQuery."""

    def __init__(self, registry: CatalogRegistry) -> None:
        self.registry = registry

    def build(self, query: AdvancedQuery) -> list[QueryPlan]:
        """Construct execution plans for all enabled catalogs matching the query profiles."""
        QueryValidator.validate(query, self.registry)
        plans = []
        for name, catalog in self.registry.enabled_catalogs().items():
            if query.catalogs and name not in query.catalogs:
                continue
            # A catalogue without profiles is planned for every profile (as QueryPlanner.plan and
            # validate_profile treat it), so the validator's intersection check and the plan agree.
            if query.profiles and catalog.profiles and not any(p in catalog.profiles for p in query.profiles):
                continue
            plans.append(
                QueryPlan(
                    catalog=name,
                    provider=catalog.provider,
                    endpoint=catalog.endpoint,
                    parameters={
                        "catalog": catalog.catalog,
                        "table": catalog.table,
                        **catalog.parameters,
                        "object_types": query.object_types,
                        "spectral_types": query.spectral_types,
                        "count_threshold": query.count_threshold,
                        "time_period": query.time_period,
                    },
                    radius_arcsec=query.radius_arcsec,
                    wavelength=catalog.wavelength,
                )
            )
        return plans


# ---------------------------------------------------------------------------
# Astrometric Geometry & Probabilistic Matching
# ---------------------------------------------------------------------------


def _source_coordinate(source: CatalogSource, epoch: float | None = None,
                       fallback_pm: tuple[float, float] | None = None) -> SkyCoord:
    """Catalog source coordinate at ``epoch`` (own proper motion, else ``fallback_pm``)."""
    ra, dec, _ = source_position_at(source, epoch, fallback_pm)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _icrs_target(target: Target) -> Target:
    """Targets are matched in ICRS; other frames are converted once up front."""
    if str(target.frame).lower() == "icrs":
        return target
    coord = SkyCoord(ra=target.ra * u.deg, dec=target.dec * u.deg, frame=target.frame).icrs
    return replace(target, ra=float(coord.ra.deg) % 360.0, dec=float(coord.dec.deg), frame="icrs")


def angular_separation_arcsec(target: Target, source: CatalogSource | Target) -> float:
    """Angular separation in arcseconds; catalog sources are brought to the target epoch.

    A source moves with its own proper motion; one without (2MASS, AllWISE, ...) moves
    with the target's proper motion when that is known.
    """
    target = _icrs_target(target)
    if isinstance(source, CatalogSource):
        return epoch_separation_arcsec(target, source)[0]
    other = _icrs_target(source)
    return haversine_arcsec(target.ra, target.dec, other.ra, other.dec)


def match_score(
    separation_arcsec: float,
    *,
    positional_error_arcsec: float | None = None,
    target_uncertainty_arcsec: float | None = None,
) -> float:
    """Gaussian positional score exp(-sep^2 / 2 sigma^2) of one row against the target.

    A relative likelihood, not a probability: the crossmatch service reports the
    Bayesian posterior of :mod:`astrometry` as a match's ``confidence``; this score is
    kept for :func:`match_target` (quick in-radius ranking) and backward compatibility.
    """
    if separation_arcsec < 0:
        return 0.0
    if positional_error_arcsec is not None or target_uncertainty_arcsec is not None:
        source_sigma = min(max(_to_float(positional_error_arcsec) or 0.0, 0.1), MAX_SIGMA_ARCSEC)
        target_sigma = min(max(_to_float(target_uncertainty_arcsec) or 0.0, 0.0), MAX_SIGMA_ARCSEC)
        sigma = math.hypot(source_sigma, target_sigma)
        likelihood = math.exp(-0.5 * (separation_arcsec / sigma) ** 2)
        return round(max(0.0, min(1.0, likelihood)), 6)
    scale = separation_arcsec / 3.0
    return round(max(0.0, 1.0 - min(scale, 10.0) / 10.0), 6)


def parallax_uncertainty_arcsec(target: Target, method: str) -> float | None:
    """Extra target uncertainty (arcsec) for a row placed with the target's motion whose
    annual parallax could not be removed (unknown row epoch, or a multi-epoch mean such
    as AllWISE/PS1): up to the parallax itself. None when nothing is added."""
    if not target.parallax_mas or method not in TARGET_PM_METHODS or method == "target_pm_parallax":
        return None
    if target.parallax_mas < PARALLAX_INFLATION_MIN_MAS:
        return None
    return target.parallax_mas / 1000.0


def match_target(target: Target, sources: list[CatalogSource], radius_arcsec: float) -> list[Match]:
    """Filter sources by radius and rank by separation.

    Rows compared through the target's own motion without a parallax correction get the
    target's parallax as an extra (target-side) uncertainty in their confidence.
    """
    icrs = _icrs_target(target)
    matches = []
    for source in sources:
        sep, method = epoch_separation_arcsec(icrs, source)
        if sep <= radius_arcsec:
            matches.append(
                Match(
                    source.catalog,
                    source,
                    sep,
                    match_score(sep, positional_error_arcsec=source.positional_error_arcsec,
                                target_uncertainty_arcsec=parallax_uncertainty_arcsec(icrs, method)),
                )
            )
    return sorted(matches, key=lambda m: m.separation_arcsec)


def _source_dict(match: Match) -> dict[str, Any]:
    """Serialize a Match object into a comprehensive counterpart dictionary."""
    source = match.source
    return {
        "catalog": source.catalog,
        "source_id": source.source_id,
        "ra": source.ra,
        "dec": source.dec,
        "separation_arcsec": match.separation_arcsec,
        "confidence": match.confidence,
        "metadata": source.metadata,
        "data": source.data,
        "provenance": source.provenance,
        "positional_error_arcsec": source.positional_error_arcsec,
        "epoch": source.epoch,
        "epoch_range": list(source.epoch_range) if source.epoch_range else None,
        "proper_motion_ra_masyr": source.proper_motion_ra_masyr,
        "proper_motion_dec_masyr": source.proper_motion_dec_masyr,
        "physical": source.metadata.get("physical", {}),
        "links": {name: url for name, url in source.metadata.get("links", {}).items() if url},
    }


def _row_priority(source: CatalogSource) -> int:
    """Planet rows (SIMBAD 'Pl', 'Pl?') share their host's coordinates: the host represents them."""
    return 1 if _is_planet(source) else 0


def _row_rank(source: CatalogSource) -> float:
    """Tie-break among duplicate listings of one source (lower first): the row measured
    more often (Pan-STARRS nDetections) represents the others."""
    detections = _to_float((source.data or {}).get("nDetections"))
    return -detections if detections is not None else 0.0


IDENTITY_CATALOGS = ("simbad", "ned")


def _otype_key(value: Any) -> str:
    """A SIMBAD / NED object type as a lower-case key (NED's leading '!' removed)."""
    return str(value or "").strip().lower().lstrip("!")


def row_object_type(source: CatalogSource) -> str:
    """The row's object type key (``physical.object_type``, else SIMBAD ``otype`` / NED ``prefphytype``)."""
    data = source.data or {}
    physical = source.metadata.get("physical") or {}
    return _otype_key(physical.get("object_type") or data.get("otype") or data.get("prefphytype"))


def identity_kind(object_type: Any, spectral_type: Any = None) -> str | None:
    """What a SIMBAD / NED object type says about the target, for its class and priors:
    'extended' (EXTENDED_OTYPES), 'extragalactic', 'emitter' (EMITTER_OTYPES, or an ordinary
    stellar type with an O / Wolf-Rayet spectral type), 'star' (ORDINARY_STELLAR_OTYPES), or
    None when it says none of these."""
    name = _otype_key(object_type)
    if not name:
        return None
    if name in EXTENDED_OTYPES:
        return "extended"
    if is_extragalactic_type(name):
        return "extragalactic"
    if name in EMITTER_OTYPES:
        return "emitter"
    if name in ORDINARY_STELLAR_OTYPES:
        if spectral_type and HOT_MASSIVE_SPTYPE.match(str(spectral_type)):
            return "emitter"
        return "star"
    return None


def is_extended_identity(source: CatalogSource) -> bool:
    """True for a SIMBAD or NED row that is not a point-like identity (see EXTENDED_OTYPES):
    an extended object type, or a SIMBAD coordinate of quality E (>= 10", open-ended). A star
    cluster marked compact (:func:`mark_compact_clusters`: an extragalactic cluster) is not."""
    if source.catalog not in IDENTITY_CATALOGS:
        return False
    if (source.metadata or {}).get("compact_cluster"):
        return False
    quality = str((source.data or {}).get("coo_qual") or "").strip().upper() if source.catalog == "simbad" else ""
    return row_object_type(source) in EXTENDED_OTYPES or quality == "E"


def _normalised_name(value: Any) -> str:
    return "".join(str(value or "").split()).casefold()


# SIMBAD's prefixes of main identifiers ('V* RR Lyr', 'NAME Virgo Cluster', '* alf Lyr').
_ID_PREFIX = re.compile(r"^(?:v\*|\*\*|\*|name|em\*)\s+", re.IGNORECASE)
# Zero padding of numbers (NED's 'MESSIER 013', 'NGC 0224', '3C 048', 'HR 0936', 'MRK 0421'):
# leading zeros of every digit run, except after a decimal point ('1.05' keeps its zero).
_PADDED_NUMBER = re.compile(r"(?<![\d.])0+(?=\d)")
# Bayer letters: SIMBAD's three-letter abbreviations ('* alf Ori', '* mu. Cep', '* ksi UMa')
# and the Greek names NED spells out ('alpha Ori', 'mu Cep', 'xi UMa'), with an optional
# superscript number ('alf01 Lib', 'alpha1 Lib').
GREEK_LETTERS: dict[str, str] = {
    "alf": "alpha", "alp": "alpha", "alpha": "alpha", "bet": "beta", "beta": "beta", "gam": "gamma",
    "gamma": "gamma", "del": "delta", "delta": "delta", "eps": "epsilon", "epsilon": "epsilon", "zet": "zeta",
    "zeta": "zeta", "eta": "eta", "tet": "theta", "the": "theta", "theta": "theta", "iot": "iota", "iota": "iota",
    "kap": "kappa", "kappa": "kappa", "lam": "lambda", "lambda": "lambda", "mu": "mu", "nu": "nu", "ksi": "xi",
    "xi": "xi", "omi": "omicron", "omicron": "omicron", "pi": "pi", "rho": "rho", "sig": "sigma", "sigma": "sigma",
    "tau": "tau", "ups": "upsilon", "upsilon": "upsilon", "phi": "phi", "chi": "chi", "psi": "psi", "ome": "omega",
    "omega": "omega",
}
_BAYER_TOKEN = re.compile(r"^([a-z]+)\.?(\d*)$")
_CONSTELLATION_KEYS = frozenset(c.casefold() for c in CONSTELLATION_ABBREVIATIONS)


def _bayer_tokens(tokens: list[str]) -> list[str]:
    """Tokens with a Bayer letter before a constellation abbreviation spelt as its Greek name
    ('alf Ori' -> 'alpha Ori', 'mu. Cep' -> 'mu Cep', 'alf01 Lib' -> 'alpha1 Lib')."""
    out = list(tokens)
    for k in range(len(tokens) - 1):
        if tokens[k + 1].casefold() not in _CONSTELLATION_KEYS:
            continue
        match = _BAYER_TOKEN.match(tokens[k].casefold())
        if match and match.group(1) in GREEK_LETTERS:
            digits = match.group(2).lstrip("0")
            out[k] = GREEK_LETTERS[match.group(1)] + digits
    return out


def identifier_key(value: Any) -> str:
    """An object name reduced for comparison across compilations: SIMBAD prefixes and blanks
    removed, case folded, 'Messier' written 'M', Bayer letters written as Greek names, and
    leading zeros of every number dropped ('M 101' = 'Messier 101' = 'M101', 'V* RR Lyr' =
    'RR Lyr', 'NGC 7318A' = 'NGC 7318a', '* alf Ori' = 'alpha Ori', '3C 48' = '3C 048',
    'HR 936' = 'HR 0936', 'Mrk 421' = 'MRK 0421')."""
    text = _ID_PREFIX.sub("", " ".join(str(value or "").split()))
    key = "".join(_bayer_tokens(text.split())).casefold()
    key = re.sub(r"^messier", "m", key)
    return _PADDED_NUMBER.sub("", key)


def _designation_family(source_id: Any) -> str:
    """The catalogue-of-origin part of a designation: its first word ('[LHL2013] 611' ->
    '[lhl2013]', '2CXO J161702.4-225834' -> '2cxo', 'MESSIER 080' -> 'messier')."""
    words = _ID_PREFIX.sub("", " ".join(str(source_id or "").split())).split()
    return words[0].casefold() if len(words) > 1 else ""


def _distinct_clusters(a: CatalogSource, b: CatalogSource, separation_arcsec: float) -> bool:
    """True when two star-cluster rows of one compilation are two clusters, not one cluster
    listed twice. Rows at distinct positions (chi2 above IDENTITY_COINCIDENCE_CHI2 with both
    errors) are two; so are rows of one catalogue of origin under different numbers
    ('[LHL2013] 611' and '[LHL2013] 613', 1.1" apart with 0.5" errors). Rows coincident within
    their errors under designations of different origins are one cluster listed twice (NED's
    'MESSIER 080' and '2CXO J161702.4-225834', both '*Cl', 0.01" apart: the Galactic globular
    cluster M 80)."""
    sigma2 = _row_sigma(a) ** 2 + _row_sigma(b) ** 2 + 2.0 * ASTROMETRIC_FLOOR_ARCSEC**2
    if separation_arcsec * separation_arcsec / sigma2 > IDENTITY_COINCIDENCE_CHI2:
        return True
    family = _designation_family(a.source_id)
    return bool(family) and family == _designation_family(b.source_id) \
        and identifier_key(a.source_id) != identifier_key(b.source_id)


def mark_compact_clusters(matches: list[Match]) -> set[int]:
    """Mark (``metadata['compact_cluster']``) the star-cluster rows of SIMBAD / NED that are
    extragalactic clusters, and return their indices: two precisely placed star clusters of one
    compilation within COMPACT_CLUSTER_NEIGHBOUR_ARCSEC of each other (at distinct positions:
    rows coincident within their errors are one cluster listed twice), or one with a
    redshift beyond EXTRAGALACTIC_MIN_REDSHIFT. Coarse positions (SIMBAD quality D/E, errors
    above DETECTION_LISTING_MAX_SIGMA_ARCSEC) are left alone: the Trapezium and OCSN 244 in
    M 42 are extended Galactic clusters listed at nominal centres."""
    candidates: dict[str, list[int]] = {}
    marked: set[int] = set()
    for i, m in enumerate(matches):
        if m.catalog not in IDENTITY_CATALOGS or extended_family(row_object_type(m.source)) != "star_cluster":
            continue
        data = m.source.data or {}
        if str(data.get("coo_qual") or "").strip().upper() in {"D", "E"}:
            continue
        if _row_sigma(m.source) > DETECTION_LISTING_MAX_SIGMA_ARCSEC:
            continue
        redshift = _to_float(data.get("rvz_redshift") if m.catalog == "simbad" else data.get("z"))
        if redshift is not None and abs(redshift) >= EXTRAGALACTIC_MIN_REDSHIFT:
            marked.add(i)
        candidates.setdefault(m.catalog, []).append(i)
    for rows in candidates.values():
        for a, i in enumerate(rows):
            for j in rows[a + 1:]:
                si, sj = matches[i].source, matches[j].source
                sep = haversine_arcsec(si.ra, si.dec, sj.ra, sj.dec)
                if sep > COMPACT_CLUSTER_NEIGHBOUR_ARCSEC:
                    continue
                if not _distinct_clusters(si, sj, sep):
                    continue
                marked.update((i, j))
    for i in marked:
        matches[i].source.metadata["compact_cluster"] = True
    return marked


def resolver_identity_catalog(resolved: dict[str, Any] | None) -> str | None:
    """The registry catalogue the resolver took its answer from ('simbad', 'ned'), or None."""
    if not resolved:
        return None
    # Sesame names the resolver that answered ('Sc=Simbad (CDS, via client/server)', 'N=NED').
    text = " ".join(str(v or "") for v in (resolved.get("resolver"),
                                           (resolved.get("resolver_metadata") or {}).get("resolver_name"))).lower()
    if "simbad" in text:
        return "simbad"
    if "ned" in text:
        return "ned"
    return None


def _named_identity_rows(matches: list[Match], resolved: dict[str, Any] | None) -> set[int]:
    """Indices of the rows that are the resolved name: the resolver catalogue's row whose
    identifier is the resolved name (SIMBAD's 'M 13' for a name Sesame answered from SIMBAD
    as 'M 13'), and the other compilation's row listed under that name or the name searched
    (:func:`identifier_key`: NED's 'NGC 4565', 'Messier 101', 'RR Lyr')."""
    catalog = resolver_identity_catalog(resolved)
    name = _normalised_name((resolved or {}).get("canonical_name"))
    if catalog is None or not name:
        return set()
    keys = {identifier_key((resolved or {}).get("canonical_name")), identifier_key((resolved or {}).get("query"))}
    keys.discard("")
    rows = {i for i, m in enumerate(matches) if m.catalog == catalog and _normalised_name(m.source.source_id) == name}
    rows |= {i for i, m in enumerate(matches)
             if m.catalog in IDENTITY_CATALOGS and m.catalog != catalog and identifier_key(m.source.source_id) in keys}
    return rows


def _row_sigma(source: CatalogSource) -> float:
    """The row's 1-sigma position error (0 when missing or not finite; at most
    astrometry.MAX_SIGMA_ARCSEC, so its square never overflows)."""
    value = _to_float(source.positional_error_arcsec)
    return min(float(value), MAX_SIGMA_ARCSEC) if value is not None and value > 0 else 0.0


def _coincidence_chi2(match: Match, target_sigma: float) -> float:
    """Offset of a row from the target in units of the target's and the row's errors (a
    SIMBAD position of quality E, possibly rounded to degrees, counts with no error)."""
    source = match.source
    quality_e = source.catalog == "simbad" and str((source.data or {}).get("coo_qual") or "").strip().upper() == "E"
    sigma2 = target_sigma * target_sigma + (0.0 if quality_e else _row_sigma(source) ** 2) + ASTROMETRIC_FLOOR_ARCSEC**2
    if is_extended_identity(source):
        sigma2 += EXTENDED_CENTRE_SIGMA_ARCSEC**2
    return float(match.separation_arcsec) ** 2 / sigma2


def _implicit_identity_rows(matches: list[Match]) -> set[int]:
    """Indices of SIMBAD / NED rows at the target position (within IMPLICIT_IDENTITY_ARCSEC):
    the searched coordinates are their catalogued position (planets listed at their host's
    coordinates excepted)."""
    return {i for i, m in enumerate(matches) if m.catalog in IDENTITY_CATALOGS and not _is_planet(m.source)
            and float(m.separation_arcsec) <= IMPLICIT_IDENTITY_ARCSEC}


def extended_family(object_type: Any) -> str | None:
    """The family of an extended object type (EXTENDED_FAMILIES), or None."""
    key = _otype_key(object_type)
    return next((name for name, members in EXTENDED_FAMILIES.items() if key in members), None)


def _point_identities(matches: list[Match], target_sigma: float,
                      named: set[int]) -> tuple[set[int], int | None, float]:
    """Extended identity rows that ARE the target, the index of the row that shows the
    target is an extended object (or None), and the centre sigma of the target (arcsec).

    An extended row is the target's identity when it is the named identity (or at the
    target position, :func:`_implicit_identity_rows`), or when it coincides with the target
    (``_coincidence_chi2 <= IDENTITY_COINCIDENCE_CHI2``) and no other row -- of any
    catalogue -- coincides better: the target was taken from that object's catalogued
    centre (M 13 by name, or its SIMBAD coordinates), not from a star near it (a star's
    Gaia position near a cluster centre is matched best by the star).

    The same object listed again -- by the other identity catalogue, or twice by one --
    is an identity too when it is of the same family (EXTENDED_FAMILIES: a GClstr for a
    ClG, never a star cluster inside a cluster galaxy) and lies within the centre scatter:
    chi2 = sep^2 / (target^2 + row^2 + 2 sigma_c^2) <= IDENTITY_COINCIDENCE_CHI2, with the
    centre sigma sigma_c at least EXTENDED_CENTRE_SIGMA_ARCSEC and the decider's own error.
    The returned centre sigma also covers every identity's own error and half its offset
    from the target (the observed spread of the catalogued centres: Cas A's SIMBAD and NED
    centres are 15" apart)."""
    point: set[int] = {i for i in named if is_extended_identity(matches[i].source)}
    decider: int | None = min(point) if point else None
    if not matches:
        return point, decider, EXTENDED_CENTRE_SIGMA_ARCSEC
    if decider is None:
        chi2 = [_coincidence_chi2(m, target_sigma) for m in matches]
        best = min(range(len(matches)), key=lambda i: (chi2[i], i))
        if chi2[best] <= IDENTITY_COINCIDENCE_CHI2 and is_extended_identity(matches[best].source):
            point.add(best)
            decider = best
    if decider is None:
        return point, None, EXTENDED_CENTRE_SIGMA_ARCSEC
    source = matches[decider].source
    family = extended_family(row_object_type(source))
    otype = row_object_type(source)
    sigma_c = max(EXTENDED_CENTRE_SIGMA_ARCSEC, _row_sigma(source))
    for i, m in enumerate(matches):
        if i in point or not is_extended_identity(m.source):
            continue
        kind = row_object_type(m.source)
        if not ((family is not None and extended_family(kind) == family) or (family is None and kind == otype)):
            continue
        sigma2 = (target_sigma**2 + _row_sigma(m.source) ** 2 + ASTROMETRIC_FLOOR_ARCSEC**2 + 2.0 * sigma_c**2)
        if float(m.separation_arcsec) ** 2 / sigma2 <= IDENTITY_COINCIDENCE_CHI2:
            point.add(i)
    for i in point:
        sigma_c = max(sigma_c, _row_sigma(matches[i].source), 0.5 * float(matches[i].separation_arcsec))
    return point, decider, min(sigma_c, MAX_TARGET_SIGMA_ARCSEC)


def _compact_for_extended_target(source: CatalogSource) -> bool:
    """A row that cannot be an extended target itself: a SIMBAD / NED entry that is not an
    extended object (a star, a galaxy, a detection) or a radio / X-ray detection smaller
    than astrometry.EXTENDED_EMISSION_MIN_ARCSEC. Rows of catalogues without sizes (and
    optical / infrared ones, whose class prior already is EXTENDED_POINT_COMPLETENESS) are not."""
    if source.catalog in IDENTITY_CATALOGS:
        return not is_extended_identity(source)
    extent = emission_extent_arcsec(source)
    return extent is not None and extent < EXTENDED_EMISSION_MIN_ARCSEC


def _extended_row_odds(matches: list[Match], identity: set[int], config: AssociationConfig) -> dict[int, float]:
    """ln prior-odds factors of the compact rows of an extended target: each gets the odds of
    EXTENDED_POINT_COMPLETENESS instead of its catalogue's (a star inside a nebula, the
    galaxies and compact X-ray sources of a cluster are not the object)."""
    point_odds = math.log(EXTENDED_POINT_COMPLETENESS) - math.log1p(-EXTENDED_POINT_COMPLETENESS)
    out: dict[int, float] = {}
    for i, m in enumerate(matches):
        if i in identity or not _compact_for_extended_target(m.source):
            continue
        c = config.completeness_of(m.catalog)
        if c <= EXTENDED_POINT_COMPLETENESS:
            continue
        out[i] = point_odds - (math.log(c) - math.log1p(-c))
    return out


def _is_detection_only(source: CatalogSource) -> bool:
    return source.catalog in IDENTITY_CATALOGS and (row_object_type(source) in DETECTION_ONLY_OTYPES
                                                    or not row_object_type(source))


def _target_motion_rows(infos: list[dict[str, Any]]) -> set[int]:
    return {i for i, info in enumerate(infos)
            if info.get("propagation") in TARGET_PM_METHODS or info.get("propagation") == "target_pm_undated"}


def _listings(matches: list[Match], detections: list[Detection], infos: list[dict[str, Any]], target: Target,
              identity: set[int], target_sigma_arcsec: float = DEFAULT_TARGET_SIGMA_ARCSEC) -> dict[int, Any]:
    """``Detection.listing`` keys of rows that are one object listed several times by one
    catalogue (astrometry collapses them onto one representative):

    * the target's extended identity rows of one catalogue (NED lists Cas A twice);
    * a SIMBAD / NED detection-only entry (DETECTION_ONLY_OTYPES) coinciding within its
      errors (chi2 <= DUPLICATE_MAX_CHI2) with exactly one classified (not detection-only)
      non-extended entry of its compilation -- or, coinciding with none, with exactly one
      other detection entry --, both positions known to DETECTION_LISTING_MAX_SIGMA_ARCSEC;
    * rows of one catalogue placed with the target's motion far from where they were
      measured (astrometry._moved) that, under some epoch hypothesis, both lie at the target
      (chi2 <= DUPLICATE_MAX_CHI2 with ``target_sigma_arcsec``) and at each other, when the
      target's motion over their epoch difference explains at least LISTING_MOTION_FRACTION
      of their measured separation (a fast mover detected at several epochs: 5XMM's
      Proxima Cen and 61 Cyg A, NED's Barnard's star)."""
    parent = list(range(len(matches)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    origin = (target.ra, target.dec)
    moving = _target_motion_rows(infos)
    pm = target.proper_motion
    speed = math.hypot(*pm) / 1000.0 if pm else 0.0
    by_catalog: dict[str, list[int]] = {}
    for i, m in enumerate(matches):
        by_catalog.setdefault(m.catalog, []).append(i)
    for catalog, rows in by_catalog.items():
        ids = sorted(i for i in rows if i in identity)
        for i in ids[1:]:
            union(ids[0], i)
        if catalog in IDENTITY_CATALOGS:
            compact = [i for i in rows if not is_extended_identity(matches[i].source)
                       and _row_sigma(matches[i].source) <= DETECTION_LISTING_MAX_SIGMA_ARCSEC]
            if len(compact) >= 2 and any(_is_detection_only(matches[i].source) for i in compact):
                chi2 = placement_chi2_matrix([detections[i] for i in compact], origin)
                close = chi2 <= DUPLICATE_MAX_CHI2
                np.fill_diagonal(close, False)
                detection_only = np.array([_is_detection_only(matches[i].source) for i in compact])
                for a, i in enumerate(compact):
                    if not detection_only[a]:
                        continue
                    partners = np.flatnonzero(close[a])
                    classified = [p for p in partners.tolist() if not detection_only[p]]
                    # The one classified entry it coincides with (NED's 'V1521 Cyg' for its
                    # WISEA and SSTSL2 entries); else the one other detection entry.
                    if len(classified) == 1:
                        union(i, compact[classified[0]])
                    elif not classified and len(partners) == 1:
                        union(i, compact[int(partners[0])])
        movers = [i for i in rows if i in moving and _moved(detections[i])] if speed > 0 else []
        if len(movers) < 2:
            continue
        subset = [detections[i] for i in movers]
        n_hyp = max(len(d.epoch_placements or ()) for d in subset) or 1
        together = np.zeros((len(movers), len(movers)), dtype=bool)
        for h in range(n_hyp):
            # Under some epoch hypothesis both rows are placed at the target (each is a
            # credible counterpart) and at each other: field rows of different epochs moved
            # with a fast motion also meet each other by chance, away from the target.
            at_target = target_chi2(subset, origin, target_sigma_arcsec, hypothesis=h) <= DUPLICATE_MAX_CHI2
            pair = placement_chi2_matrix(subset, origin, role="target", hypothesis=h) <= DUPLICATE_MAX_CHI2
            together |= pair & at_target[:, None] & at_target[None, :]
        for a, b in zip(*np.nonzero(np.triu(together, 1)), strict=True):
            i, j = movers[int(a)], movers[int(b)]
            lo_i, hi_i = _row_span(matches[i].source)
            lo_j, hi_j = _row_span(matches[j].source)
            gap = max(abs(hi_j - lo_i), abs(hi_i - lo_j))
            separation = haversine_arcsec(detections[i].ra, detections[i].dec, detections[j].ra, detections[j].dec)
            if speed * gap >= LISTING_MOTION_FRACTION * separation:
                union(i, j)
    roots = [find(i) for i in range(len(matches))]
    sizes: dict[int, int] = {}
    for root in roots:
        sizes[root] = sizes.get(root, 0) + 1
    return {i: ("listing", root) for i, root in enumerate(roots) if sizes[root] > 1}


def _match_detections(
    matches: list[Match],
    target: Target,
    *,
    target_pm_sigma_masyr: float | None = None,
    point_identities: set[int] | None = None,
    identity_rows: set[int] | None = None,
    centre_sigma_arcsec: float = EXTENDED_CENTRE_SIGMA_ARCSEC,
    row_ln_odds: dict[int, float] | None = None,
) -> tuple[list[Detection], list[dict[str, Any]]]:
    """Detections of the matched rows at the common epoch (see :func:`astrometry.source_detection`).

    Extragalactic rows (galaxy/QSO types, redshift >= EXTRAGALACTIC_MIN_REDSHIFT) are not moved.
    Rows compared through the target's own motion without a parallax correction get the
    target's parallax as an extra uncertainty (:func:`parallax_uncertainty_arcsec`).
    Extended SIMBAD / NED objects (:func:`is_extended_identity`) get
    ``EXTENDED_IDENTITY_SIGMA_ARCSEC`` of positional scatter unless they are the target's own
    identity (``point_identities``: ``centre_sigma_arcsec``, the scatter of the object's
    catalogued centres). Rows in ``identity_rows`` (the resolved name's row) get
    ``astrometry.IDENTITY_PRIOR_LN_ODDS`` on their prior odds, and rows in ``row_ln_odds``
    (the compact rows of an extended target, :func:`_extended_row_odds`) that factor.
    """
    pm_sigma = DEFAULT_TARGET_PM_SIGMA_MASYR if target_pm_sigma_masyr is None else float(target_pm_sigma_masyr)
    point_identities = point_identities or set()
    identity_rows = identity_rows or set()
    row_ln_odds = row_ln_odds or {}
    detections: list[Detection] = []
    infos: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        method = source_position_at(match.source, target.epoch, target.proper_motion, (target.ra, target.dec),
                                    target.parallax_mas)[2]
        own_pm = match.source.proper_motion_ra_masyr is not None and match.source.proper_motion_dec_masyr is not None
        if target.epoch is None and target.proper_motion is not None and not own_pm:
            # Undated: a row without its own motion is placed with the target's motion
            # (astrometry "target_pm_undated"); its annual parallax is not removed.
            method = "target_pm"
        det, info = source_detection(match.source, target, target_pm_sigma_masyr=pm_sigma,
                                     extra_sigma_arcsec=parallax_uncertainty_arcsec(target, method),
                                     priority=_row_priority(match.source), label=idx,
                                     extragalactic=_is_extragalactic_row(match.source),
                                     rank=_row_rank(match.source))
        if is_extended_identity(match.source):
            info["extended_identity"] = True
            if idx in point_identities:
                info["target_identity"] = True
                var = float(centre_sigma_arcsec) ** 2
            else:
                var = EXTENDED_IDENTITY_SIGMA_ARCSEC**2
            det.cov = (det.cov[0] + var, det.cov[1], det.cov[2] + var)
            if det.target_cov is not None:
                det.target_cov = (det.target_cov[0] + var, det.target_cov[1], det.target_cov[2] + var)
            if det.epoch_placements:
                det.epoch_placements = tuple((ra, dec, (c[0] + var, c[1], c[2] + var))
                                             for ra, dec, c in det.epoch_placements)
            info["covariance_shape"] = f"{info.get('covariance_shape')}+{'centre' if idx in point_identities else 'extended'}"
            info["sigma_arcsec"] = math.sqrt(float(info.get("sigma_arcsec") or 0.0) ** 2 + var)
        if idx in identity_rows:
            det.prior_ln_odds = IDENTITY_PRIOR_LN_ODDS
            info["resolved_identity"] = True
        elif idx in point_identities:
            # The extended object listed at the target's centre by another compilation (or
            # found there by coordinates), of the object's own family: identified by type and
            # position as the resolver identifies by name -- the catalogue's density of all its
            # objects (NED: ~1e5 per deg^2 in a cluster) is not the chance of an unrelated
            # galaxy cluster or supernova remnant at the centre.
            det.prior_ln_odds = IDENTITY_PRIOR_LN_ODDS
        if idx in row_ln_odds:
            det.prior_ln_odds += float(row_ln_odds[idx])
            info["compact_in_extended_target"] = True
        detections.append(det)
        infos.append(info)
    return detections, infos


def _target_rows(sources: list[CatalogSource], target: Target, target_sigma_arcsec: float) -> int:
    """Rows of one catalogue explained by the target itself (not field sources): the row
    nearest the target when it is consistent with it (chi2 <= LINK_CHI2), plus the rows
    listed at exactly its position (a star and its planets)."""
    best: tuple[float, CatalogSource] | None = None
    for src in sources:
        sep = src.metadata.get("epoch_separation_arcsec")
        if sep is None:
            sep = epoch_separation_arcsec(target, src)[0]
        row_sigma = _row_sigma(src)
        sigma2 = row_sigma * row_sigma + target_sigma_arcsec * target_sigma_arcsec
        if sigma2 > 0 and sep * sep / sigma2 <= LINK_CHI2 and (best is None or sep < best[0]):
            best = (float(sep), src)
    if best is None:
        return 0
    ref = best[1]
    return sum(1 for s in sources if haversine_arcsec(s.ra, s.dec, ref.ra, ref.dec) <= COINCIDENT_ARCSEC)


def catalog_sky_density(catalog: CatalogDefinition | None) -> SkyDensity | None:
    """A catalogue's published all-sky density from its definition, when it declares one:
    ``parameters["sky_density"] = {"sources": N, "area_deg2": A, "reference": "..."}`` or
    ``parameters["sky_density_per_deg2"]`` (e.g. set when a VizieR table is registered
    from its row count). None: the built-in table (``astrometry.CATALOG_SKY_DENSITY``)
    or the generic prior applies."""
    if catalog is None:
        return None
    params = catalog.parameters or {}
    spec = params.get("sky_density")
    if isinstance(spec, dict):
        sources, area = _to_float(spec.get("sources")), _to_float(spec.get("area_deg2"))
        if sources is not None and sources > 0 and area is not None and area > 0:
            return SkyDensity(int(sources), float(area), str(spec.get("reference") or f"{catalog.name} definition"))
    per_deg2 = _to_float(params.get("sky_density_per_deg2"))
    if per_deg2 is not None and per_deg2 > 0:
        # Expressed over the whole sky; 'sources' is only used through the ratio.
        return SkyDensity(max(1, round(per_deg2 * 41252.96)), 41252.96,
                          str(params.get("sky_density_reference") or f"{catalog.name} definition"))
    return None


def catalog_densities(
    successes: list[tuple[str, list[CatalogSource]]],
    target: Target,
    radius_arcsec: float,
    target_sigma_arcsec: float,
    *,
    registry: CatalogRegistry | None = None,
    unprobed: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Field-source density (deg^-2) of every queried catalogue around the target.

    ``unprobed`` names catalogues whose local density should have been measured by a
    density probe (``{catalog: reason}``, see :func:`_density_probe_reason`) but was not
    (not run -- e.g. a batch finalising fetched cones -- or failed): this is recorded in
    the details' ``probe``, and when the cone itself shows a significant excess over the
    map ('poisson excess') the estimate is at least the cone's own (field rows / area, no
    map prior; conservative: a higher density lowers the posteriors).

    Uses every row fetched from the archive (in-radius, beyond max_rows and epoch pad)
    over the cone actually queried; a cone the archive truncated (rows are returned
    nearest-first) covers only the area inside its farthest returned row. When the
    catalogue was probed over a wider cone (``meta["density_probe"]``, see
    :meth:`CrossmatchService.probe_densities`) the probe's rows and cone are used. The prior mean
    is the catalogue's density at the target position (density map, published all-sky
    mean, a density declared in its definition -- :func:`catalog_sky_density` -- or the
    generic value). See :func:`astrometry.estimate_density_deg2` for the Gamma-Poisson estimate.
    """
    densities: dict[str, float] = {}
    info: dict[str, dict[str, Any]] = {}
    for name, sources in successes:
        meta = getattr(sources, "meta", {}) or {}
        probe = meta.get("density_probe") if isinstance(meta.get("density_probe"), dict) else None
        if probe is not None and probe.get("status") == "success":
            # The local density measured over the (wider) probe cone.
            rows = list(probe.get("rows") or [])
            radius = float(probe.get("query_radius_arcsec") or probe.get("radius_arcsec") or radius_arcsec)
            centre = _pair(probe.get("cone_center")) or (target.ra, target.dec)
            truncated = bool(probe.get("archive_truncated"))
        else:
            rows = list(sources) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or [])
            radius = float(meta.get("query_radius_arcsec") or radius_arcsec)
            centre = _pair(meta.get("cone_center")) or (target.ra, target.dec)
            truncated = bool(meta.get("archive_truncated"))
        if truncated and rows:
            radius = min(radius, max(haversine_arcsec(centre[0], centre[1], s.ra, s.dec) for s in rows))
        definition = registry.catalogs.get(name) if registry is not None else None
        density, details = estimate_density_deg2(
            len(rows), cone_area_deg2(max(radius, 1e-3)), catalog=name,
            n_target_rows=_target_rows(rows, target, target_sigma_arcsec),
            position=(target.ra, target.dec), sky_density=catalog_sky_density(definition),
        )
        details["radius_arcsec"] = radius
        details["truncated"] = truncated
        if probe is not None:
            details["probe"] = {k: v for k, v in probe.items() if k != "rows"}
        if unprobed and name in unprobed and (probe is None or probe.get("status") != "success"):
            if probe is None:
                details["probe"] = {"status": "not_run", "reason": unprobed[name],
                                    "radius_arcsec": DENSITY_PROBE_RADIUS_ARCSEC}
            area = cone_area_deg2(max(radius, 1e-3))
            cone_density = details["field_rows"] / area if area > 0 else 0.0
            # Only a significant excess measured by the cone itself replaces the map prior
            # (one or two rows in a few arcsec measure nothing: their ML density can be 3x off).
            if unprobed[name] == POISSON_EXCESS and cone_density > density:
                density = cone_density
                details["method"] = f"{details['method']}+cone_only"
                details["density_per_deg2"] = density
            details["unmeasured_crowding"] = unprobed[name]
        densities[name] = density
        info[name] = details
    return densities, info


def associate_matches(
    matches: list[Match],
    target: Target,
    *,
    densities: dict[str, float] | None = None,
    config: AssociationConfig | None = None,
    radius_arcsec: float | None = None,
    target_pm_sigma_masyr: float | None = None,
    point_identities: set[int] | None = None,
    identity_rows: set[int] | None = None,
    centre_sigma_arcsec: float = EXTENDED_CENTRE_SIGMA_ARCSEC,
    row_ln_odds: dict[int, float] | None = None,
) -> tuple[AssociationResult, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Bayesian association of ``matches`` with the target and with each other.

    Returns (association, per-match propagation info, density provenance). Densities
    default to an estimate from the matches themselves over the cone of
    ``radius_arcsec`` (the largest match separation when not given). ``point_identities``,
    ``identity_rows``, ``centre_sigma_arcsec`` and ``row_ln_odds`` are passed to
    :func:`_match_detections`; rows that are one object listed several times by one
    catalogue (:func:`_listings`) are collapsed onto one representative.
    """
    cfg = config or AssociationConfig()
    density_info: dict[str, dict[str, Any]] = {}
    if densities is None:
        radius = radius_arcsec or max([m.separation_arcsec for m in matches] + [1.0])
        by_catalog: dict[str, list[CatalogSource]] = {}
        for m in matches:
            by_catalog.setdefault(m.catalog, []).append(m.source)
        densities = {}
        for name, rows in by_catalog.items():
            densities[name], density_info[name] = estimate_density_deg2(
                len(rows), cone_area_deg2(radius), catalog=name,
                n_target_rows=_target_rows(rows, target, cfg.target_sigma_arcsec), position=(target.ra, target.dec))
    else:
        densities = dict(densities)
        for m in matches:  # a catalogue without an estimate (e.g. rows injected by a caller)
            if m.catalog not in densities:
                densities[m.catalog], density_info[m.catalog] = estimate_density_deg2(
                    sum(1 for x in matches if x.catalog == m.catalog),
                    cone_area_deg2(radius_arcsec or max(x.separation_arcsec for x in matches) or 1.0), catalog=m.catalog,
                    position=(target.ra, target.dec))
    detections, infos = _match_detections(matches, target, target_pm_sigma_masyr=target_pm_sigma_masyr,
                                          point_identities=point_identities, identity_rows=identity_rows,
                                          centre_sigma_arcsec=centre_sigma_arcsec, row_ln_odds=row_ln_odds)
    identity = set(point_identities or ()) | {i for i in (identity_rows or ())
                                              if is_extended_identity(matches[i].source)}
    sigma_t = max(cfg.target_sigma_axes) if cfg.target_sigma_axes else cfg.target_sigma_arcsec
    for i, key in _listings(matches, detections, infos, target, identity, sigma_t).items():
        detections[i].listing = key
        infos[i]["listing"] = True
    # Rows beyond the searched radius were not fetched: rows near its edge keep only the
    # observed part of their partners' region in the correlated-field model
    # (astrometry._observed_fractions).
    cone = (target.ra, target.dec, float(radius_arcsec)) if radius_arcsec else None
    result = associate(detections, densities, target=(target.ra, target.dec), config=cfg, cone=cone)
    return result, infos, density_info


def _probability(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _groups_from_association(
    matches: list[Match],
    result: AssociationResult,
    infos: list[dict[str, Any]],
    keep: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Serialize association groups (only members whose index is in ``keep``, if given).

    Output per group (the former keys first): ``group_id``, ``catalogs``,
    ``wavelengths``, ``members`` (counterpart dicts plus ``match_probability`` -- the
    posterior that the row belongs to this object --, ``target_probability``,
    ``coincident_with`` and the epoch-propagated ``position_at_epoch``), then
    ``contains_target``, ``match_flag`` ('best' / 'secondary' / None),
    ``match_probability``, ``p_any``, ``p_i``, ``log10_bayes_factor``, ``log10_prior``
    and ``alternatives``. The target group comes first, then objects by distance.
    """
    def ident(i: int) -> dict[str, str]:
        return {"catalog": matches[i].catalog, "source_id": matches[i].source.source_id}

    def distance(group: Any) -> float:
        return min(matches[i].separation_arcsec for i in group.members)

    ordered = sorted(result.groups, key=lambda g: (not g.contains_target, distance(g)))
    output: list[dict[str, Any]] = []
    for group in ordered:
        kept = [i for i in group.members if keep is None or i in keep]
        if not kept:
            continue
        members = []
        for i in kept:
            member = _source_dict(matches[i])
            info = infos[i]
            rep = group.coincident_with.get(i)
            member.update({
                "match_probability": _probability(group.member_probability.get(i)),
                "target_probability": _probability(float(result.target_probability[i])),
                "coincident_with": matches[rep].source.source_id if rep is not None else None,
                "position_at_epoch": {"epoch": info.get("epoch"), "propagation": info.get("propagation"),
                                      "field_propagation": info.get("field_propagation"),
                                      "epoch_window": info.get("epoch_window"),
                                      "sigma_arcsec": info.get("sigma_arcsec"),
                                      "pm_growth_arcsec": info.get("pm_growth_arcsec"),
                                      "covariance_shape": info.get("covariance_shape")},
                # An extended object (cloud, cluster, ...) or a coarse position: it may
                # contain the target but is not a point-like counterpart.
                "extended_identity": bool(info.get("extended_identity", False)),
                # The row is the target by identity: the resolved name's row, or an extended
                # object whose catalogued centre is the target.
                "target_identity": bool(info.get("target_identity", False) or info.get("resolved_identity", False)),
            })
            members.append(member)
        output.append({
            "group_id": f"object-{len(output) + 1}",
            "catalogs": sorted({matches[i].catalog for i in kept}),
            "wavelengths": sorted({str(matches[i].source.metadata.get("wavelength", "unknown")) for i in kept}),
            "members": members,
            "contains_target": group.contains_target,
            "match_flag": group.match_flag,
            "match_probability": _probability(group.match_probability),
            "exact_probability": _probability(group.exact_probability),
            "p_any": _probability(group.p_any),
            "p_i": _probability(group.p_i),
            "log10_bayes_factor": round(group.log10_bayes_factor, 4),
            "log10_prior": None if group.log10_prior is None else round(group.log10_prior, 4),
            "alternatives": [
                {"members": [ident(i) for i in alt["members"]], "p_i": _probability(alt["p_i"]),
                 "match_probability": _probability(alt["match_probability"]),
                 "log10_bayes_factor": round(alt["log10_bayes_factor"], 4)}
                for alt in group.alternatives
            ],
        })
    return output


def _group_matches(
    matches: list[Match],
    target: Target,
    radius_arcsec: float,
    *,
    association: tuple[AssociationResult, list[dict[str, Any]]] | None = None,
    config: AssociationConfig | None = None,
    keep: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Group multi-catalog detections into physical objects (Bayesian N-way association).

    Replaces the former O(n^2) single-linkage (disjoint-set) grouping: candidate pairs
    come from cKDTree range searches and the groups are the most probable partition in
    which each catalogue contributes at most one source per object (see
    :mod:`astrometry`). Member ``confidence`` values are those of the Match objects (the
    crossmatch service sets them to the target-association posterior).
    """
    if not matches:
        return []
    if association is None:
        result, infos, _ = associate_matches(matches, _icrs_target(target), config=config, radius_arcsec=radius_arcsec)
    else:
        result, infos = association
    return _groups_from_association(matches, result, infos, keep)


# ---------------------------------------------------------------------------
# Query Planner & Concurrent Executor
# ---------------------------------------------------------------------------


class QueryPlanner:
    """Creates default catalog query plans based on profile selections."""

    def __init__(self, registry: CatalogRegistry) -> None:
        self.registry = registry

    def plan(self, radius_arcsec: float, profile: str | None = None) -> list[QueryPlan]:
        validate_profile(profile, self.registry)
        return [
            QueryPlan(
                catalog=name,
                provider=catalog.provider,
                endpoint=catalog.endpoint,
                parameters={"catalog": catalog.catalog, "table": catalog.table, **catalog.parameters},
                radius_arcsec=radius_arcsec,
                wavelength=catalog.wavelength,
            )
            for name, catalog in self.registry.enabled_catalogs().items()
            if profile is None or not catalog.profiles or profile in catalog.profiles
        ]


class QueryExecutor:
    """Executes catalog queries concurrently with timeouts, fallbacks, and error isolation.

    When a registry is supplied the full CatalogDefinition (epoch, pos_error,
    citation, timeouts, ...) is used; otherwise a minimal one is rebuilt from the plan.
    A catalog may declare ``parameters["fallback"]`` (provider/endpoint/catalog/
    parameters overrides) that is tried when the primary archive is unavailable.
    """

    # When the primary fails, the fallback gets the rest of the catalog's time budget but
    # at least min(limit, this) seconds, so one catalog takes at most limit + that.
    FALLBACK_MIN_SECONDS = 20.0

    def __init__(
        self,
        providers: dict[str, CatalogProvider],
        *,
        timeout: float = 30.0,
        registry: CatalogRegistry | None = None,
        timeout_cap: float | None = None,
    ) -> None:
        self.providers = providers
        self.timeout = timeout
        self.registry = registry
        # Upper bound on every catalog's own timeout_seconds (Settings.catalog_timeout_cap_seconds).
        self.timeout_cap = timeout_cap

    def catalog_limit(self, catalog: CatalogDefinition) -> float:
        """Time allowed for one catalog query: its own timeout (else the default), capped."""
        limit = float(catalog.timeout_seconds or self.timeout)
        if self.timeout_cap is not None:
            limit = min(limit, float(self.timeout_cap))
        return limit

    def definition_for(self, plan: QueryPlan) -> CatalogDefinition:
        """Resolve the CatalogDefinition used to execute ``plan``."""
        base = self.registry.catalogs.get(plan.catalog) if self.registry is not None else None
        if base is not None:
            return replace(
                base,
                provider=plan.provider or base.provider,
                endpoint=plan.endpoint or base.endpoint,
                table=plan.parameters.get("table") or base.table,
                catalog=plan.parameters.get("catalog") or base.catalog,
                parameters={**base.parameters, **plan.parameters},
            )
        return CatalogDefinition(
            name=plan.catalog,
            provider=plan.provider,
            wavelength=plan.wavelength,
            endpoint=plan.endpoint,
            table=plan.parameters.get("table"),
            catalog=plan.parameters.get("catalog"),
            parameters=dict(plan.parameters),
        )

    async def _run_one(
        self, catalog: CatalogDefinition, target: Target, radius_arcsec: float, limit: float | None = None
    ) -> list[CatalogSource]:
        provider = self.providers.get(catalog.provider)
        if provider is None:
            raise CatalogUnavailableError(f"No provider configured for {catalog.provider}")
        limit = self.catalog_limit(catalog) if limit is None else limit
        # Providers size their own request budget from timeout_seconds: hand them the
        # (capped) limit so their timeout path runs before the executor's.
        catalog = replace(catalog, timeout_seconds=limit)
        try:
            return await asyncio.wait_for(provider.query(catalog, target, radius_arcsec), timeout=limit)
        except TimeoutError as exc:
            raise QueryTimeoutError(f"Catalog {catalog.name} timed out after {limit:g}s") from exc

    async def _run_plan(self, plan: QueryPlan, target: Target) -> tuple[str, list[CatalogSource]]:
        catalog = self.definition_for(plan)
        started = monotonic()
        fallback_used: dict[str, Any] | None = None
        try:
            try:
                sources = await self._run_one(catalog, target, plan.radius_arcsec)
            except (CatalogUnavailableError, QueryTimeoutError) as primary_error:
                fallback = catalog.parameters.get("fallback")
                if not isinstance(fallback, dict):
                    raise
                fallback_def = replace(
                    catalog,
                    provider=str(fallback.get("provider", catalog.provider)),
                    endpoint=fallback.get("endpoint", catalog.endpoint),
                    catalog=fallback.get("catalog", catalog.catalog),
                    table=fallback.get("table", catalog.table),
                    parameters={**catalog.parameters, **(fallback.get("parameters") or {}), "fallback": None},
                )
                fallback_used = {
                    "provider": fallback_def.provider,
                    "endpoint": fallback_def.endpoint,
                    "reason": f"{primary_error.__class__.__name__}: {primary_error}",
                }
                limit = self.catalog_limit(catalog)
                remaining = limit - (monotonic() - started)
                budget = max(remaining, min(limit, self.FALLBACK_MIN_SECONDS))
                try:
                    sources = await self._run_one(fallback_def, target, plan.radius_arcsec, budget)
                except Exception as fallback_error:
                    raise _combined_failure(primary_error, fallback_error, fallback_used) from fallback_error
        except Exception as exc:
            exc.elapsed_ms = round((monotonic() - started) * 1000.0, 1)  # type: ignore[attr-defined]
            raise
        meta = dict(getattr(sources, "meta", {}) or {})
        meta["status"] = "success" if sources else "empty"
        meta["row_count"] = len(sources)
        meta["elapsed_ms"] = round((monotonic() - started) * 1000.0, 1)
        if fallback_used:
            meta["fallback"] = fallback_used
        return plan.catalog, QueryResult(list(sources), meta)

    async def execute(
        self, plans: list[QueryPlan], target: Target
    ) -> tuple[list[tuple[str, list[CatalogSource]]], list[CatalogFailure]]:
        gathered = await asyncio.gather(*(self._run_plan(p, target) for p in plans), return_exceptions=True)
        successes: list[tuple[str, list[CatalogSource]]] = []
        failures: list[CatalogFailure] = []

        for plan, item in zip(plans, gathered):
            if isinstance(item, BaseException):
                failures.append(self.failure_for(plan, item))
            else:
                successes.append(item)
        return successes, failures

    @staticmethod
    def failure_for(plan: QueryPlan, error: BaseException) -> CatalogFailure:
        """The CatalogFailure recorded for a plan whose query raised ``error``."""
        return CatalogFailure(
            plan.catalog,
            error_type=error.__class__.__name__,
            message=str(error),
            elapsed_ms=getattr(error, "elapsed_ms", None),
            fallback=getattr(error, "fallback", None),
        )


def _combined_failure(primary: BaseException, fallback: BaseException, fallback_used: dict[str, Any]) -> Exception:
    """The error reported when both the primary archive and its fallback failed.

    Keeps the fallback's error class (the final outcome) but names both causes, so the
    root cause (usually the primary outage) is never hidden behind the fallback's error.
    """
    message = (
        f"primary {primary.__class__.__name__}: {primary}; "
        f"fallback {fallback.__class__.__name__}: {fallback}"
    )
    try:
        combined = type(fallback)(message)
    except Exception:
        combined = CatalogUnavailableError(message)
    if not isinstance(combined, Exception):  # e.g. a BaseException subclass
        combined = CatalogUnavailableError(message)
    combined.fallback = {**fallback_used, "error": f"{fallback.__class__.__name__}: {fallback}"}  # type: ignore[attr-defined]
    return combined


# ---------------------------------------------------------------------------
# Crossmatch Service
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SearchContext:
    """Validated inputs of one crossmatch: target model, catalogue plans and options."""

    target: Target
    plans: list[QueryPlan]
    search_radius: float
    query: AdvancedQuery | None
    profile: str | None
    pm_source: str | None
    target_sigma_arcsec: float
    target_pm_sigma_masyr: float | None
    # Where target_sigma_arcsec came from: input, resolver, coordinate_precision or default.
    target_sigma_source: str = "default"
    # Prior completeness override (scalar: every catalogue; {catalog: c}: those catalogues,
    # the others by the target class); None: by target class.
    completeness: float | dict[str, float] | None = None
    # Target class override ('unknown', 'star', 'extragalactic', 'extended'); None: inferred.
    target_class: str | None = None
    notes: list[str] = field(default_factory=list)
    # The name resolver's answer (ResolvedObject.as_dict()) when the target was a name: its
    # object type, spectral type and parallax decide the class, and its identity row is the
    # target by definition.
    resolved: dict[str, Any] | None = None
    # Per-axis (east, north) target sigmas when the rounding of the coordinates as typed
    # differs between the axes (sexagesimal RA in seconds of time); None: circular.
    target_sigma_axes: tuple[float, float] | None = None
    # Warnings for the record's provenance known before the search (e.g. the name resolver's).
    warnings: list[str] = field(default_factory=list)


class CrossmatchService:
    """Orchestrates catalog querying, filtering, Bayesian association, and UnifiedRecord assembly.

    ``association_config`` sets the association parameters (target uncertainty, prior
    completeness, ...; see :class:`astrometry.AssociationConfig`). ``max_concurrency``
    bounds the number of targets :meth:`crossmatch_many` runs at once. The association is
    CPU-bound: the async entry points run it in a worker thread (``asyncio.to_thread``) so
    the event loop keeps serving other requests.
    """

    def __init__(
        self,
        registry: CatalogRegistry,
        providers: dict[str, CatalogProvider],
        *,
        radius_arcsec: float = 3.0,
        timeout: float = 30.0,
        timeout_cap: float | None = None,
        association_config: AssociationConfig | None = None,
        max_concurrency: int = 4,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.radius_arcsec = radius_arcsec
        self.planner = QueryPlanner(registry)
        self.executor = QueryExecutor(providers, timeout=timeout, registry=registry, timeout_cap=timeout_cap)
        self.association_config = association_config or AssociationConfig()
        if int(max_concurrency) < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.max_concurrency = int(max_concurrency)

    # -- inputs -------------------------------------------------------------------------

    def prepare(
        self,
        ra: float | str,
        dec: float | str,
        *,
        radius_arcsec: float | None = None,
        epoch: float | None = None,
        profile: str | None = None,
        query: AdvancedQuery | None = None,
        pm_ra_masyr: float | None = None,
        pm_dec_masyr: float | None = None,
        pm_source: str | None = None,
        parallax_mas: float | None = None,
        catalogs: list[str] | None = None,
        target_uncertainty_arcsec: float | None = None,
        target_pm_error_masyr: float | None = None,
        completeness: float | dict[str, float] | None = None,
        target_class: str | None = None,
        target_uncertainty_source: str | None = None,
        resolved_object: Any = None,
    ) -> SearchContext:
        """Validate the inputs of a crossmatch and plan its catalogue queries.

        ``ra``/``dec`` are numbers (exact: the digits of a float say nothing about how the
        position was typed) or text as typed -- decimal degrees ('187.278') or sexagesimal
        ('12 29 07', '12:29:06.7', '12h29m07s'; RA in hours unless marked with 'd' / a
        degree sign). Without ``target_uncertainty_arcsec`` the target's positional sigma is
        the association default (0.1") combined in quadrature with the rounding of text
        coordinates (:func:`coordinate_sigma_arcsec`: '187.278' has a 0.001 deg quantum,
        sigma 1.0" per axis; '12 29 07' a 1 s one) or of numbers that carry the signature of
        a sexagesimal conversion (a 1 s / 1" grid in a long decimal expansion).
        ``completeness`` (a scalar for every catalogue, or {catalog: c} for some of them,
        the others by class) and ``target_class`` override the prior completeness of the
        target's counterparts (default: by the target class inferred from its identity and
        counterparts; see :func:`astrometry.completeness_prior`). ``resolved_object`` (a
        :class:`models.ResolvedObject` or its ``as_dict()``) is the name resolver's answer
        for a named target.
        """
        try:
            ra_value, dec_value, rounding = target_coordinate_rounding(ra, dec)
        except OverflowError as exc:
            raise ValueError(f"the coordinates as given ({ra!r}, {dec!r}) have no usable precision ({exc})") from exc
        quantum = (math.sqrt((rounding[0] ** 2 + rounding[1] ** 2) / 2.0)
                   if all(math.isfinite(v) for v in rounding) else math.inf)
        target = validate_target(ra_value, dec_value, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                                 parallax_mas=parallax_mas)
        if catalogs is not None and not [c for c in catalogs if str(c).strip()]:
            raise ValueError("catalogs must name at least one catalogue (omit it to query all of them)")
        if query is not None:
            QueryValidator.validate(query, self.registry)
            pm_source = (query.metadata or {}).get("pm_source") or pm_source
            meta = query.metadata or {}
            if target_uncertainty_arcsec is None and meta.get("target_uncertainty_arcsec") is not None:
                target_uncertainty_arcsec = meta.get("target_uncertainty_arcsec")
            if completeness is None and meta.get("completeness") is not None:
                completeness = meta.get("completeness")
            if target_class is None and meta.get("target_class") is not None:
                target_class = meta.get("target_class")
            if resolved_object is None and meta.get("resolved_object") is not None:
                resolved_object = meta.get("resolved_object")
            use_pm = query.proper_motion
            target = validate_target(
                query.target.ra,
                query.target.dec,
                epoch=query.target.epoch if use_pm else None,
                pm_ra_masyr=query.target.pm_ra_masyr if use_pm else None,
                pm_dec_masyr=query.target.pm_dec_masyr if use_pm else None,
                parallax_mas=query.target.parallax_mas if use_pm else None,
            )
        else:
            validate_profile(profile, self.registry)
        target = _icrs_target(target)

        try:
            search_radius = (
                query.radius_arcsec if query else self.radius_arcsec if radius_arcsec is None else float(radius_arcsec)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("radius_arcsec must be a finite number greater than zero.") from exc
        if not math.isfinite(search_radius) or search_radius <= 0:
            raise ValueError("radius_arcsec must be a finite number greater than zero.")

        notes: list[str] = []
        axes: tuple[float, float] | None = None
        if target_uncertainty_arcsec is not None:
            sigma = _positive(target_uncertainty_arcsec, "target_uncertainty_arcsec")
            if sigma > MAX_TARGET_SIGMA_ARCSEC:
                raise ValueError(f"target_uncertainty_arcsec must be at most {MAX_TARGET_SIGMA_ARCSEC:g} arcsec "
                                 "(a position known to worse than a degree cannot be crossmatched)")
            sigma_source = target_uncertainty_source or "input"
        else:
            if not math.isfinite(quantum) or quantum > MAX_TARGET_SIGMA_ARCSEC:
                raise ValueError(f"the coordinates as given ({ra!r}, {dec!r}) are too coarse to locate a target: their "
                                 f"rounding is worth more than {MAX_TARGET_SIGMA_ARCSEC:g} arcsec per axis")
            floor = float(self.association_config.target_sigma_arcsec)
            # Rounding below 10% of the floor changes sigma by < 0.5%: the floor is used.
            sigma_source = "coordinate_precision" if quantum > 0.1 * floor else "default"
            sigma = math.hypot(floor, quantum) if sigma_source == "coordinate_precision" else floor
            if sigma_source == "coordinate_precision":
                axes = (math.hypot(floor, rounding[0]), math.hypot(floor, rounding[1]))
                notes.append(
                    f"Target position uncertainty {axes[0]:.3g} x {axes[1]:.3g} arcsec (east x north, 1 sigma) from the "
                    f"precision of the coordinates as given ({ra!r}, {dec!r}); supply target_uncertainty_arcsec to "
                    "override.")
        pm_sigma = None if target_pm_error_masyr is None else _positive(target_pm_error_masyr, "target_pm_error_masyr",
                                                                         allow_zero=True)
        completeness = _completeness_override(completeness)
        if target_class is not None and target_class not in TARGET_CLASSES:
            raise ValueError(f"target_class must be one of {', '.join(TARGET_CLASSES)}")

        plans = QueryBuilder(self.registry).build(query) if query else self.planner.plan(search_radius, profile=profile)
        if catalogs:
            enabled = self.registry.enabled_catalogs()
            unknown = [c for c in catalogs if c not in enabled]
            if unknown:
                raise ValueError(f"Unknown catalog(s): {', '.join(unknown)}; known: {', '.join(sorted(enabled))}")
            check_catalogs_in_profiles(self.registry, catalogs, (query.profiles if query else None) or profile)
            plans = [p for p in plans if p.catalog in set(catalogs)]
        return SearchContext(target, plans, search_radius, query, profile, pm_source, sigma, pm_sigma,
                             sigma_source, completeness, target_class, notes, _resolved_dict(resolved_object), axes)

    # -- one target ------------------------------------------------------------------------

    async def crossmatch(
        self,
        ra: float | str,
        dec: float | str,
        *,
        radius_arcsec: float | None = None,
        epoch: float | None = None,
        profile: str | None = None,
        query: AdvancedQuery | None = None,
        pm_ra_masyr: float | None = None,
        pm_dec_masyr: float | None = None,
        pm_source: str | None = None,
        parallax_mas: float | None = None,
        catalogs: list[str] | None = None,
        target_uncertainty_arcsec: float | None = None,
        target_pm_error_masyr: float | None = None,
        completeness: float | dict[str, float] | None = None,
        target_class: str | None = None,
        resolved_object: Any = None,
        target_uncertainty_source: str | None = None,
    ) -> UnifiedRecord:
        """Execute full crossmatch pipeline for given coordinates or AdvancedQuery.

        ``ra``/``dec``: numbers (exact) or the text as typed (its rounding sets the target
        uncertainty; decimal or sexagesimal -- see :meth:`prepare`).

        ``parallax_mas`` (the target's parallax; with a query, ``query.target.parallax_mas``)
        removes the annual parallax from single-epoch positions of the target (2MASS, SDSS,
        ...); when not given, a significant parallax (known error, >= 5 sigma) is adopted
        with the proper motion from a non-extragalactic row.

        ``epoch`` is the Julian year of (ra, dec); with it, every catalog cone follows
        the target to that catalog's epoch (using ``pm_ra_masyr``/``pm_dec_masyr`` when
        given, else widened by the largest plausible proper motion) and rows are
        compared after propagation. Without it positions are compared as given, the
        target's epoch being unknown within J2000-J2016 (``astrometry.UNDATED_TARGET_EPOCHS``):
        rows with their own proper motion are matched along that part of their track, and
        rows without one move with the motion of the target's identity row when one is
        found at the target (a warning says when that motion could carry the target's rows
        at other epochs outside the cone, which is not widened without an epoch).
        ``pm_source`` records where a given proper motion came from ("input" by
        default, "resolver" for a name-resolver motion; with a query it is read from
        ``query.metadata["pm_source"]``).

        ``catalogs`` restricts the search to these registry catalogues.
        ``target_uncertainty_arcsec`` is the 1-sigma per-axis uncertainty of the target
        position (default: 0.1" combined with the rounding of the coordinates as given,
        see :meth:`prepare`) and ``target_pm_error_masyr`` that of a given proper motion
        (default 1 mas/yr). ``completeness`` / ``target_class`` override the prior
        probability of a counterpart per catalogue (default: by the target's class --
        star, extragalactic, extended or unknown -- from the resolver's identity, the
        identity rows and the counterparts). ``resolved_object`` is the name resolver's
        answer for a named target; ``target_uncertainty_source`` says where a given
        ``target_uncertainty_arcsec`` came from (default "input"; "resolver" for a name
        resolver's error, as :func:`resolved_search_target` reports it). With a query, ``query.metadata`` may carry
        ``target_uncertainty_arcsec``, ``completeness``, ``target_class`` and ``resolved_object``.

        Every match's ``confidence`` is the posterior probability that the row is the
        target's counterpart (NWAY-style, :mod:`astrometry`); ``crossmatch_groups`` are
        the most probable partition of the matches into physical objects with their
        association probabilities; ``provenance["association"]`` records the densities,
        priors, the target class and the target's ``p_any``.

        Catalog statistics: ``row_count``/``status`` count the rows inside the radius
        (the nearest ``max_rows``); with an AdvancedQuery, ``sources`` holds only the rows
        that pass its confidence/type filters and ``returned_count`` is their number.
        """
        ctx = self.prepare(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile, query=query,
                           pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, pm_source=pm_source,
                           parallax_mas=parallax_mas, catalogs=catalogs,
                           target_uncertainty_arcsec=target_uncertainty_arcsec,
                           target_pm_error_masyr=target_pm_error_masyr,
                           completeness=completeness, target_class=target_class, resolved_object=resolved_object,
                           target_uncertainty_source=target_uncertainty_source)
        successes, failures = await self._execute(ctx)
        return await asyncio.to_thread(self.finalize, ctx, successes, failures)

    async def _execute(self, ctx: SearchContext) -> tuple[list[tuple[str, list[CatalogSource]]], list[CatalogFailure]]:
        """Run every catalogue query of ``ctx`` concurrently; a catalogue that needs a density
        probe starts it as soon as its own result arrives (:meth:`probe_density`)."""
        async def run(plan: QueryPlan) -> tuple[str, list[CatalogSource]]:
            result = await self.executor._run_plan(plan, ctx.target)
            await self.probe_density(ctx, plan, result[1])
            return result

        gathered = await asyncio.gather(*(run(p) for p in ctx.plans), return_exceptions=True)
        successes: list[tuple[str, list[CatalogSource]]] = []
        failures: list[CatalogFailure] = []
        for plan, item in zip(ctx.plans, gathered, strict=True):
            if isinstance(item, BaseException):
                if not isinstance(item, Exception):
                    raise item
                failures.append(self.executor.failure_for(plan, item))
            else:
                successes.append(item)
        return successes, failures

    async def probe_density(self, ctx: SearchContext, plan: QueryPlan, sources: Any) -> None:
        """Measure one catalogue's local density when it must be (:func:`_density_probe_reason`:
        a crowded stellar system, or a small cone far denser than the density map predicts):
        the catalogue is queried again over a DENSITY_PROBE_RADIUS_ARCSEC cone, within
        DENSITY_PROBE_TIMEOUT_SECONDS (at most the catalogue's own limit) and without a
        fallback archive, and the rows are attached to its result as
        ``meta["density_probe"]`` with the reason and the elapsed time (used by
        :func:`catalog_densities`). A failed probe is recorded there too; the density then
        falls back to the conservative small-cone estimate."""
        meta = getattr(sources, "meta", None)
        if meta is None or "density_probe" in meta:
            return
        reason = _density_probe_reason(plan.catalog, sources, ctx, self.registry)
        if reason is None:
            return
        probe_plan = replace(plan, radius_arcsec=DENSITY_PROBE_RADIUS_ARCSEC)
        catalog = self.executor.definition_for(probe_plan)
        budget = min(self.executor.catalog_limit(catalog), DENSITY_PROBE_TIMEOUT_SECONDS)
        started = monotonic()
        base = {"radius_arcsec": probe_plan.radius_arcsec, "reason": reason, "timeout_seconds": budget}
        try:
            rows = await self.executor._run_one(catalog, ctx.target, probe_plan.radius_arcsec, budget)
        except Exception as exc:  # noqa: BLE001 - recorded; the search goes on without the probe
            meta["density_probe"] = {**base, "status": "failed", "error": f"{exc.__class__.__name__}: {exc}",
                                     "elapsed_ms": round((monotonic() - started) * 1000.0, 1)}
            return
        probe_meta = getattr(rows, "meta", {}) or {}
        meta["density_probe"] = {
            **base, "status": "success", "elapsed_ms": round((monotonic() - started) * 1000.0, 1),
            "rows": list(rows) + list(probe_meta.get("excess_sources") or []) + list(probe_meta.get("pad_sources") or []),
            "query_radius_arcsec": probe_meta.get("query_radius_arcsec") or probe_plan.radius_arcsec,
            "cone_center": probe_meta.get("cone_center"),
            "archive_truncated": bool(probe_meta.get("archive_truncated")),
        }

    async def probe_densities(self, ctx: SearchContext, successes: list[tuple[str, list[CatalogSource]]]) -> None:
        """:meth:`probe_density` for every catalogue result of ``ctx`` not probed yet,
        concurrently (for callers that fetched the cones themselves, e.g. a batch)."""
        plan_of = {p.catalog: p for p in ctx.plans}
        await asyncio.gather(*(self.probe_density(ctx, plan_of[name], sources) for name, sources in successes
                               if name in plan_of))

    def finalize(
        self,
        ctx: SearchContext,
        successes: list[tuple[str, list[CatalogSource]]],
        failures: list[CatalogFailure],
    ) -> UnifiedRecord:
        """Assemble the UnifiedRecord from the catalogue results of ``ctx``'s plans (CPU-bound:
        the async entry points call it in a worker thread)."""
        target, plans, search_radius, query, profile, pm_source = (
            ctx.target, ctx.plans, ctx.search_radius, ctx.query, ctx.profile, ctx.pm_source)
        resolved = ctx.resolved
        resolver_extragalactic = bool(((resolved or {}).get("resolver_metadata") or {}).get("extragalactic"))

        # Target proper motion: given, or adopted from a matched catalog row (e.g. Gaia)
        # so rows without their own proper motion (2MASS, AllWISE, ...) can be checked.
        pm_origin: dict[str, Any] | None = None
        parallax_origin: dict[str, Any] | None = None
        if target.parallax_mas is not None:
            parallax_origin = {"parallax_mas": target.parallax_mas,
                               "source": "resolver" if pm_source == "resolver" else "input"}
        adoption_warnings: list[str] = []
        if not plans:
            adoption_warnings.append(
                "No catalogs were queried: no enabled catalog matches the requested profile/catalog selection."
            )
        pm_sigma = ctx.target_pm_sigma_masyr
        if target.proper_motion is not None:
            pm_origin = {"source": pm_source or "input"}
            if resolver_extragalactic and pm_source == "resolver":
                pm_origin["extragalactic"] = True
            # An extragalactic target (the resolver said so) has no parallax to adopt.
            if target.parallax_mas is None and not pm_origin.get("extragalactic"):
                found = _adopt_parallax(target, successes, search_radius)
                if found is not None:
                    target, parallax_origin = found
        else:
            # Dated: follow the target to every catalogue's epoch. Undated: rows without
            # their own motion move with the identity's motion over the undated window.
            adopted = _adopt_proper_motion(target, successes, search_radius, warnings=adoption_warnings,
                                           target_sigma_arcsec=ctx.target_sigma_arcsec)
            if adopted is not None:
                target, pm_origin = adopted
                if pm_origin.get("parallax") is not None:
                    parallax_origin = pm_origin["parallax"]
                if pm_sigma is None:
                    pm_sigma = _adopted_pm_sigma(pm_origin, successes)
        if pm_sigma is None:
            pm_sigma = DEFAULT_TARGET_PM_SIGMA_MASYR
        if target.epoch is None and target.proper_motion is not None:
            window = max(UNDATED_TARGET_EPOCHS) - min(UNDATED_TARGET_EPOCHS)
            drift = math.hypot(*target.proper_motion) / 1000.0 * window
            if drift >= 0.5 * search_radius:
                origin = pm_origin or {}
                adoption_warnings.append(
                    f"Undated target: its proper motion ({target.pm_ra_masyr:.1f}, {target.pm_dec_masyr:.1f}) mas/yr"
                    + (f", from {origin.get('catalog')} {origin.get('source_id')}," if origin.get("source") == "adopted"
                       else ",")
                    + f" moves it {drift:.1f} arcsec between J{min(UNDATED_TARGET_EPOCHS):g} and "
                    f"J{max(UNDATED_TARGET_EPOCHS):g}, but without an epoch the cones were searched at the given position"
                    f" only: rows of catalogues measured at other epochs (e.g. Gaia DR3 at J2016.0) may lie outside the "
                    f"{search_radius:g} arcsec cone. Give the coordinates' epoch (e.g. epoch=2000 for SIMBAD "
                    "coordinates) to follow the target to every catalogue's epoch.")

        # Catalogues whose local density should have been measured by a density probe but
        # was not (a caller that fetched the cones itself, e.g. a batch; a failed probe).
        unprobed: dict[str, str] = {}
        for name, sources in successes:
            probe = (getattr(sources, "meta", {}) or {}).get("density_probe")
            if isinstance(probe, dict) and probe.get("status") == "success":
                continue
            reason = _density_probe_reason(name, sources, ctx, self.registry)
            if reason is not None:
                unprobed[name] = reason
                status = probe.get("status") if isinstance(probe, dict) else "not run"
                adoption_warnings.append(
                    f"{name}: {reason} -- the local source density was not measured over "
                    f"{DENSITY_PROBE_RADIUS_ARCSEC:g} arcsec (density probe {status}), and the density map cannot "
                    "resolve it: "
                    + (f"the {search_radius:g} arcsec cone's own density is used (conservative)."
                       if reason == POISSON_EXCESS else "posteriors may be overconfident."))

        # Final in-cone / pad split with the final target model over EVERY fetched row
        # (in-radius, beyond max_rows, and pad): rows fetched only because of the epoch
        # pad are never counted as results, and none is lost to a provider-level cut.
        # Rows whose astrometry is unusable (non-finite position, error, epoch or motion)
        # are left out with a warning instead of failing the whole association.
        classified: list[tuple[str, QueryResult]] = []
        invalid: dict[str, list[str]] = {}
        for name, sources in successes:
            meta = dict(getattr(sources, "meta", {}) or {})
            combined = list(sources) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or [])
            usable = []
            for src in combined:
                problem = _row_problem(src)
                if problem is None:
                    usable.append(src)
                else:
                    invalid.setdefault(name, []).append(f"{src.source_id}: {problem}")
            max_rows = int(meta.get("max_rows") or max(len(sources), 1))
            split = classify_sources(
                usable, target, search_radius, catalog_name=name, max_rows=max_rows,
                row_limit=int(meta.get("row_limit") or max_rows),
                archive_truncated=bool(meta.get("archive_truncated", meta.get("truncated", False))),
                cone_center=_pair(meta.get("cone_center")), epoch_span=_pair(meta.get("epoch_span")),
            )
            meta["pad_sources"] = split.pad
            meta["excess_sources"] = split.excess
            meta["truncated"] = split.truncated
            if "base_warnings" in meta:
                meta["warnings"] = list(meta["base_warnings"]) + split.warnings
            if name in invalid:
                bad = invalid[name]
                note = (f"{name}: {len(bad)} row(s) with invalid astrometry were left out of the association "
                        f"({'; '.join(bad[:3])}{'; ...' if len(bad) > 3 else ''}).")
                meta["warnings"] = [*list(meta.get("warnings") or []), note]
            classified.append((name, QueryResult(split.inside, meta)))
        successes = classified

        all_sources = [source for _, sources in successes for source in sources]
        all_matches = match_target(target, all_sources, search_radius)
        # Extragalactic star clusters are compact sources, never the extended object at the target.
        mark_compact_clusters(all_matches)

        # Bayesian association over every in-radius row: confidence = posterior that the
        # row is the target's counterpart. The prior completeness depends on the target's
        # class: given, else from the resolver's identity or an extended identity at the
        # target, else from the identity rows / counterparts found with the default priors
        # (a second pass then uses the class priors). A per-catalogue override is laid over
        # the class priors.
        config = replace(self.association_config, target_sigma_arcsec=ctx.target_sigma_arcsec,
                         target_sigma_axes=ctx.target_sigma_axes)
        densities, density_info = catalog_densities(successes, target, search_radius, config.target_sigma_arcsec,
                                                    registry=self.registry, unprobed=unprobed)
        queried = sorted({m.catalog for m in all_matches} | {p.catalog for p in plans})
        # Rows that are the target by identity: the resolved name's row, else a SIMBAD / NED
        # row at exactly the searched position (its catalogued coordinates were searched:
        # POST /search and main.search_object pass the resolver's position without its answer).
        named = _named_identity_rows(all_matches, resolved)
        identity_reason = {i: "resolved name" for i in named}
        if not named:
            named = _implicit_identity_rows(all_matches)
            identity_reason = {i: "the searched position is this row's catalogued position" for i in named}
        point_ids, extended_row, centre_sigma = _point_identities(all_matches, ctx.target_sigma_arcsec, named)
        assoc_kwargs: dict[str, Any] = {"densities": densities, "radius_arcsec": search_radius,
                                        "target_pm_sigma_masyr": pm_sigma, "point_identities": point_ids,
                                        "identity_rows": named, "centre_sigma_arcsec": centre_sigma}
        target_plx = _class_parallax(target, parallax_origin, resolved)
        association: AssociationResult | None = None
        first_pass: tuple[Any, ...] | None = None
        extra_density: dict[str, dict[str, Any]] = {}
        det_infos: list[dict[str, Any]] = []
        override = ctx.completeness
        class_info: dict[str, Any] | None
        if isinstance(override, float):
            config = replace(config, completeness=override)
            class_info = {"class": ctx.target_class or "unknown", "source": "completeness given"}
        else:
            overlay = dict(override or {})
            first_priors = {c: DEFAULT_PRIOR_COMPLETENESS for c in queried} | overlay
            if ctx.target_class is not None:
                class_info = {"class": ctx.target_class, "source": "given", "parallax_mas": target_plx}
            else:
                class_info = _class_before_association(resolved, pm_origin, extended_row, all_matches, target_plx)
                if class_info is None or (class_info["class"] == "star" and class_info.get("parallax_mas") is None):
                    first_config = replace(config, completeness=first_priors)
                    association, det_infos, extra_density = associate_matches(
                        all_matches, target, config=first_config, **assoc_kwargs)
                    first_pass = (first_config, dict(assoc_kwargs))
                    found_class = _class_from_association(all_matches, association, target_parallax=target_plx)
                    if class_info is None:
                        class_info = found_class
                    elif found_class.get("class") == "star" and found_class.get("parallax_mas") is not None:
                        class_info["parallax_mas"] = found_class["parallax_mas"]
                        class_info["parallax_source"] = found_class.get("source")
            wavelengths, fractions = self._catalog_prior_inputs(queried)
            priors = completeness_priors(queried, class_info["class"], class_info.get("parallax_mas"),
                                         wavelengths=wavelengths, fractions=fractions) | overlay
            config = replace(config, completeness=priors)
        # An extended target (a cluster, nebula, remnant): its catalogued centre is uncertain
        # by the centre sigma, which widens the target position; compact rows (stars, galaxies,
        # compact radio / X-ray sources inside it) get the point-source prior.
        extended = class_info["class"] == "extended"
        compact_odds: dict[int, float] = {}
        if extended:
            axes = config.target_sigma_axes
            config = replace(config, target_sigma_arcsec=math.hypot(config.target_sigma_arcsec, centre_sigma),
                             target_sigma_axes=(math.hypot(axes[0], centre_sigma), math.hypot(axes[1], centre_sigma))
                             if axes else None)
            if not isinstance(override, float):
                compact_odds = _extended_row_odds(all_matches, point_ids | named, config)
            assoc_kwargs["row_ln_odds"] = compact_odds
        if association is None or first_pass != (config, assoc_kwargs):
            association, det_infos, extra_density = associate_matches(all_matches, target, config=config,
                                                                      **assoc_kwargs)
        density_info.update(extra_density)
        if not np.all(np.isfinite(association.target_probability)):
            raise RuntimeError("the association produced a non-finite posterior; rows: "
                               + ", ".join(f"{m.catalog} {m.source.source_id}" for m in all_matches[:10]))
        for idx, match in enumerate(all_matches):
            match.confidence = round(float(association.target_probability[idx]), 6)
        index_of = {id(m): i for i, m in enumerate(all_matches)}
        matches = list(all_matches)

        effective_radius = search_radius
        if query and query.adaptive_radius and matches:
            nearest = [m.separation_arcsec for m in matches[:5]]
            effective_radius = min(search_radius, max(1.0, 1.5 * statistics.median(nearest)))
            matches = [m for m in matches if m.separation_arcsec <= effective_radius]

        if query:
            counts: dict[str, int] = {}
            filtered: list[Match] = []
            for match in matches:
                if not match.confidence >= query.min_confidence or not query.apply_filters(_source_dict(match)):
                    continue
                c = counts.get(match.catalog, 0)
                if query.max_results is not None and c >= query.max_results:
                    continue
                counts[match.catalog] = c + 1
                filtered.append(match)
            matches = filtered

        allowed = {(m.catalog, m.source.source_id) for m in matches}
        catalog_results: dict[str, Any] = {}
        catalog_stats: dict[str, dict[str, Any]] = {}
        citations: dict[str, str] = {}
        all_warnings: list[str] = list(ctx.warnings) + list(adoption_warnings)
        for name, sources in successes:
            meta = dict(getattr(sources, "meta", {}) or {})
            kept = [s for s in sources if (s.catalog, s.source_id) in allowed] if query else list(sources)
            pad_sources = list(meta.get("pad_sources") or [])
            excess_count = len(meta.get("excess_sources") or [])
            max_rows = int(meta.get("max_rows") or max(len(pad_sources), 1))
            warnings = list(meta.get("warnings") or [])
            unchecked = [s for s in pad_sources if s.metadata.get("epoch_propagation") == "none"]
            epoch_incomplete = False
            if target.epoch is not None and unchecked:
                if target.proper_motion is None:
                    # These rows might be the target seen at another epoch: we cannot tell.
                    epoch_incomplete = not sources
                    warnings.append(
                        f"{name}: {len(unchecked)} row(s) beyond {search_radius:g} arcsec have no proper motion and the "
                        "target's is unknown, so they could not be epoch-checked; results may be incomplete."
                    )
                elif any(target.proper_motion):
                    # (A stationary target -- pm 0 -- is where it is at every epoch.)
                    undated = sum(1 for s in unchecked if s.epoch is None)
                    if undated:
                        warnings.append(
                            f"{name}: {undated} row(s) beyond {search_radius:g} arcsec have no epoch and were compared "
                            "at their catalog positions."
                        )
            all_warnings.extend(warnings)
            stats = {
                # 'success' = rows inside the requested radius; 'empty' = valid query, none inside.
                "status": "success" if len(sources) else "empty",
                "row_count": len(sources),
                # Rows returned in 'sources' (after AdvancedQuery confidence/type filters).
                "returned_count": len(kept),
                "matched_count": sum(1 for m in matches if m.catalog == name),
                "elapsed_ms": meta.get("elapsed_ms"),
                "truncated": bool(meta.get("truncated", False)),
                "fallback": meta.get("fallback"),
                "raw_row_count": meta.get("raw_row_count", len(sources) + len(pad_sources)),
                "dropped_rows": meta.get("dropped_rows", 0),
                # Rows removed by the catalog's exclude_values (VLASS 'Redundant' duplicates).
                "filtered_rows": meta.get("filtered_rows", 0),
                "pad_row_count": len(pad_sources),
                # In-radius rows beyond max_rows (checked, but not returned).
                "excess_row_count": excess_count,
                "query_radius_arcsec": meta.get("query_radius_arcsec", search_radius),
                "epoch_incomplete": epoch_incomplete,
                # Field-source density used by the association prior (per deg^2).
                "source_density_deg2": densities.get(name),
                # Rows left out of the association because their astrometry is unusable.
                "invalid_rows": len(invalid.get(name, [])),
                "warnings": warnings,
            }
            catalog_stats[name] = stats
            if meta.get("citation"):
                citations[name] = str(meta["citation"])
            catalog_results[name] = {
                "sources": kept,
                **stats,
                "query": meta.get("query"),
                "endpoint": meta.get("endpoint"),
                "citation": meta.get("citation"),
                "acknowledgement": meta.get("acknowledgement"),
            }
            if not query:
                # Rows fetched only because the cone was widened for proper motion (all were
                # epoch-checked above; only the nearest max_rows are returned).
                catalog_results[name]["pad_sources"] = pad_sources[:max_rows]
                catalog_results[name]["pad_sources_truncated"] = len(pad_sources) > max_rows
        for failure in failures:
            stats = {
                "status": "failed",
                "row_count": 0,
                "returned_count": 0,
                "matched_count": 0,
                "elapsed_ms": failure.elapsed_ms,
                "truncated": False,
                "fallback": failure.fallback,
                "error_type": failure.error_type,
                "message": failure.message,
            }
            catalog_stats[failure.catalog] = stats
            catalog_results[failure.catalog] = {"sources": [], **stats}

        counterparts: dict[str, list[dict[str, Any]]] = {}
        for match in matches:
            wave = str(match.source.metadata.get("wavelength", "unknown"))
            counterparts.setdefault(wave, []).append(_source_dict(match))

        failures_list = [f.as_dict() for f in failures]
        keep = {index_of[id(m)] for m in matches} if len(matches) != len(all_matches) else None
        groups = _groups_from_association(all_matches, association, det_infos, keep) if all_matches else []
        target_group = next((g for g in association.groups if g.contains_target), None)

        provenance = {
            "query_radius_arcsec": search_radius,
            "effective_radius_arcsec": effective_radius,
            "target_epoch": target.epoch,
            "target_proper_motion": (
                {"pm_ra_masyr": target.pm_ra_masyr, "pm_dec_masyr": target.pm_dec_masyr, **(pm_origin or {})}
                if target.proper_motion is not None else None
            ),
            # Parallax used to remove the annual parallax from single-epoch positions.
            "target_parallax": parallax_origin,
            "warnings": all_warnings,
            "profile": profile,
            "advanced_query": query.to_dict() if query else None,
            "catalogs_planned": [p.catalog for p in plans],
            "catalog_stats": catalog_stats,
            "citations": citations,
            "matches": [
                {
                    "catalog": m.catalog,
                    "source_id": m.source.source_id,
                    "separation_arcsec": m.separation_arcsec,
                    "confidence": m.confidence,
                }
                for m in matches
            ],
            "association": {
                "method": ("Bayesian N-way cross-identification (Budavari & Szalay 2008, ApJ 679, 301) with "
                           "NWAY-style target association (Salvato et al. 2018, MNRAS 473, 4937)"),
                "confidence": "posterior probability that the row is the target's counterpart",
                "config": config.as_dict(),
                "target_sigma_arcsec": config.target_sigma_arcsec,
                "target_sigma_axes": list(ctx.target_sigma_axes) if ctx.target_sigma_axes else None,
                "target_sigma_source": ctx.target_sigma_source,
                "target_pm_error_masyr": pm_sigma if target.proper_motion is not None else None,
                # Target class deciding the prior completeness, and the priors used: every
                # queried catalogue's (a scalar override as given), with the override itself.
                "target_class": class_info,
                "completeness_override": dict(override) if isinstance(override, dict) else override,
                # Undated coordinates: the epochs the target position may have.
                "undated_target_epochs": list(UNDATED_TARGET_EPOCHS) if target.epoch is None else None,
                "completeness": (dict(config.completeness) if isinstance(config.completeness, dict)
                                 else config.completeness),
                # Rows that are the target by identity: the resolved name's row, and extended
                # objects whose catalogued centre is the target.
                "identity_rows": [{"catalog": all_matches[i].catalog, "source_id": all_matches[i].source.source_id,
                                   "reason": identity_reason.get(i, "extended object at the target")}
                                  for i in sorted(named | point_ids)],
                # Extended target: the scatter of its catalogued centres (added to the target
                # position and to its identity rows) and the compact rows given the
                # point-source prior (stars, galaxies, compact sources inside it).
                "extended_centre_sigma_arcsec": centre_sigma if extended else None,
                "compact_rows_in_extended_target": len(compact_odds),
                "p_any": _probability(association.p_any),
                "best_match_probability": _probability(target_group.match_probability) if target_group else None,
                "states": association.n_states,
                "exact": association.exact,
                "links": association.n_links,
                "densities": density_info,
                "notes": list(ctx.notes) + list(association.notes),
            },
        }

        return UnifiedRecord(
            target={"ra": target.ra, "dec": target.dec, "frame": target.frame, "epoch": target.epoch,
                    "pm_ra_masyr": target.pm_ra_masyr, "pm_dec_masyr": target.pm_dec_masyr,
                    "parallax_mas": target.parallax_mas},
            catalogs_queried=len(plans),
            catalog_results=catalog_results,
            counterparts=counterparts,
            failures=failures_list,
            provenance=provenance,
            crossmatch_groups=groups,
        )

    def _catalog_prior_inputs(self, catalogs: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
        """Wavelength and declared stellar counterpart table of each catalogue (registry definitions)."""
        wavelengths: dict[str, str] = {}
        fractions: dict[str, Any] = {}
        for name in catalogs:
            definition = self.registry.catalogs.get(name)
            if definition is None:
                continue
            if definition.wavelength:
                wavelengths[name] = str(definition.wavelength)
            declared = (definition.parameters or {}).get("stellar_counterpart_fraction")
            if declared is not None:
                fractions[name] = declared
        return wavelengths, fractions

    # -- streaming -------------------------------------------------------------------------

    async def crossmatch_stream(
        self,
        ra: float | str | None = None,
        dec: float | str | None = None,
        *,
        name: str | None = None,
        resolver: Any = None,
        radius_arcsec: float | None = None,
        epoch: float | None = None,
        profile: str | None = None,
        query: AdvancedQuery | None = None,
        pm_ra_masyr: float | None = None,
        pm_dec_masyr: float | None = None,
        pm_source: str | None = None,
        parallax_mas: float | None = None,
        catalogs: list[str] | None = None,
        target_uncertainty_arcsec: float | None = None,
        target_pm_error_masyr: float | None = None,
        completeness: float | dict[str, float] | None = None,
        target_class: str | None = None,
        serializer: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Crossmatch as an async stream of events, one per catalogue as it completes.

        Same parameters as :meth:`crossmatch`, plus ``name`` (resolved with CDS Sesame --
        ``resolver`` or a :class:`providers.SesameResolver` on the providers' HTTP client --
        which supplies the position, epoch, proper motion, parallax, their errors and the
        object's identity; see :func:`resolved_search_target` for how a requested ``epoch``
        and motion combine with it). ``name`` excludes ``ra``/``dec``.
        Events are dicts ``{"event": kind, "data": {...}}``, in this order:

        * ``start``: target, planned catalogues, resolved object (when ``name``);
        * ``catalog`` (one per catalogue, in completion order): ``catalog``, ``status``
          (success / empty / failed), ``count`` (rows inside the radius at the provider's
          first-pass epoch split), ``elapsed_ms``, ``sources`` (serialised rows) and, on
          failure, ``error_type``/``message``;
        * ``group`` (one per crossmatch group of the final record, target group first);
        * ``done``: ``{"record": UnifiedRecord.as_dict()}``.

        A catalogue whose local density must be measured starts its density probe as soon
        as its own result arrives (:meth:`probe_density`). The final record, its groups and
        their dicts are built in a worker thread. ``serializer`` (a callable taking an
        event's data, e.g. :func:`streaming.sse_payload`) is also run off the event loop:
        each event then carries its text as ``"json"`` too, so a multi-megabyte ``done``
        payload never blocks other requests.

        Closing the generator (e.g. a disconnected client) cancels the catalogue queries
        and probes still running.
        """
        resolved: dict[str, Any] | None = None
        sigma_source: str | None = None
        resolution_notes: list[str] = []
        resolution_warnings: list[str] = []
        if name is not None:
            from providers import SesameResolver

            if ra is not None or dec is not None:
                raise ValueError("Give either an object name or ra/dec, not both.")
            validate_search_inputs(epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                                   parallax_mas=parallax_mas)
            if resolver is None:
                client = next((getattr(p, "client", None) for p in self.providers.values()
                               if getattr(p, "client", None) is not None), None)
                resolver = SesameResolver(client)
            obj = await resolver.resolve(name)
            resolved = obj.as_dict()
            spec = resolved_search_target(obj, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                                          parallax_mas=parallax_mas, target_uncertainty_arcsec=target_uncertainty_arcsec,
                                          target_pm_error_masyr=target_pm_error_masyr)
            ra, dec, epoch = spec["ra"], spec["dec"], spec["epoch"]
            pm_ra_masyr, pm_dec_masyr, parallax_mas = spec["pm_ra_masyr"], spec["pm_dec_masyr"], spec["parallax_mas"]
            pm_source = spec["pm_source"] or pm_source
            target_uncertainty_arcsec, target_pm_error_masyr = spec["target_uncertainty_arcsec"], spec["target_pm_error_masyr"]
            sigma_source = spec["target_uncertainty_source"]
            resolution_notes = spec["notes"]
            resolution_warnings = spec["warnings"]
        if ra is None or dec is None:
            raise ValueError("Provide ra and dec, or an object name.")
        ctx = self.prepare(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile, query=query,
                           pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, pm_source=pm_source,
                           parallax_mas=parallax_mas, catalogs=catalogs,
                           target_uncertainty_arcsec=target_uncertainty_arcsec,
                           target_pm_error_masyr=target_pm_error_masyr, completeness=completeness,
                           target_class=target_class, target_uncertainty_source=sigma_source,
                           resolved_object=resolved)
        ctx.notes.extend(resolution_notes)
        ctx.warnings.extend(resolution_warnings)
        yield {"event": "start", "data": {
            "target": ctx.target.as_dict(), "radius_arcsec": ctx.search_radius,
            "catalogs": [p.catalog for p in ctx.plans], "target_sigma_arcsec": ctx.target_sigma_arcsec,
            "target_pm_error_masyr": ctx.target_pm_sigma_masyr,
            "resolved_object": resolved,
        }}

        started = monotonic()
        tasks: dict[asyncio.Task[Any], QueryPlan] = {
            asyncio.create_task(self.executor._run_plan(plan, ctx.target), name=f"crossmatch:{plan.catalog}"): plan
            for plan in ctx.plans
        }
        # Density probes start as soon as their catalogue has answered (not after the others).
        probes: list[asyncio.Task[Any]] = []
        outcomes: dict[str, tuple[str, list[CatalogSource]] | CatalogFailure] = {}

        async def event(kind: str, data: dict[str, Any]) -> dict[str, Any]:
            out: dict[str, Any] = {"event": kind, "data": data}
            if serializer is not None:
                out["json"] = await asyncio.to_thread(serializer, data)
            return out

        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda t: ctx.plans.index(tasks[t])):
                    plan = tasks[task]
                    error = task.exception()
                    if error is not None:
                        failure = self.executor.failure_for(plan, error)
                        outcomes[plan.catalog] = failure
                        yield await event("catalog", {
                            "catalog": plan.catalog, "wavelength": plan.wavelength, "status": "failed", "count": 0,
                            "elapsed_ms": failure.elapsed_ms, "sources": [], "error_type": failure.error_type,
                            "message": failure.message, "since_start_ms": round((monotonic() - started) * 1000.0, 1),
                        })
                        continue
                    name_, sources = task.result()
                    outcomes[plan.catalog] = (name_, sources)
                    probes.append(asyncio.create_task(self.probe_density(ctx, plan, sources),
                                                      name=f"density-probe:{plan.catalog}"))
                    meta = getattr(sources, "meta", {}) or {}
                    yield await event("catalog", {
                        "catalog": plan.catalog, "wavelength": plan.wavelength,
                        "status": "success" if sources else "empty", "count": len(sources),
                        "elapsed_ms": meta.get("elapsed_ms"), "truncated": bool(meta.get("truncated")),
                        "since_start_ms": round((monotonic() - started) * 1000.0, 1),
                        "sources": [_stream_source_dict(s) for s in sources],
                    })
            if probes:
                await asyncio.gather(*probes)
        finally:
            leftover = [t for t in [*tasks, *probes] if not t.done()]
            for task in leftover:
                task.cancel()
            if leftover:
                await asyncio.gather(*leftover, return_exceptions=True)

        successes = [o for p in ctx.plans if isinstance(o := outcomes.get(p.catalog), tuple)]
        failures = [o for p in ctx.plans if isinstance(o := outcomes.get(p.catalog), CatalogFailure)]

        def build() -> list[dict[str, Any]]:
            # CPU-bound (association, serialisation of every row): off the event loop.
            record = self.finalize(ctx, successes, failures)  # type: ignore[arg-type]
            if resolved is not None:
                record.resolved_object = resolved
                record.provenance["resolver"] = resolved.get("resolver")
            record_dict = record.as_dict()
            out = [{"event": "group", "data": group} for group in record_dict["crossmatch_groups"]]
            out.append({"event": "done", "data": {"record": record_dict,
                                                  "elapsed_ms": round((monotonic() - started) * 1000.0, 1)}})
            if serializer is not None:
                for item in out:
                    item["json"] = serializer(item["data"])
            return out

        for item in await asyncio.to_thread(build):
            yield item

    # -- many targets ----------------------------------------------------------------------

    async def crossmatch_many(
        self,
        targets: list[dict[str, Any]],
        *,
        radius_arcsec: float | None = None,
        epoch: float | None = None,
        profile: str | None = None,
        max_concurrency: int | None = None,
    ) -> list[UnifiedRecord]:
        """Crossmatch several targets concurrently (at most ``max_concurrency`` at once,
        default ``self.max_concurrency``); results are returned in the input order.

        Each target dict takes ``ra``/``dec`` and optionally every per-target option of
        :meth:`crossmatch` (``radius_arcsec``, ``epoch``, ``profile``, ``pm_ra_masyr``,
        ``pm_dec_masyr``, ``pm_source``, ``parallax_mas``, ``catalogs``,
        ``target_uncertainty_arcsec``, ``target_pm_error_masyr``, ``completeness``,
        ``target_class``, ``resolved_object``). Every target is validated before any
        archive is queried, so a bad one raises without starting the others; if a
        crossmatch fails anyway, the others still running are cancelled (and awaited)
        before the error propagates.
        """
        limit = self.max_concurrency if max_concurrency is None else int(max_concurrency)
        if limit < 1:
            raise ValueError("max_concurrency must be at least 1")
        known = {"ra", "dec", *MANY_TARGET_OPTIONS}
        contexts: list[SearchContext] = []
        for index, t in enumerate(targets):
            unknown = sorted(set(t) - known)
            if unknown:
                raise ValueError(f"target {index}: unknown option(s) {', '.join(unknown)}")
            if "ra" not in t or "dec" not in t:
                raise ValueError(f"target {index}: ra and dec are required")
            options = {k: t[k] for k in MANY_TARGET_OPTIONS if k in t}
            options.setdefault("radius_arcsec", radius_arcsec)
            options.setdefault("epoch", epoch)
            options.setdefault("profile", profile)
            try:
                contexts.append(self.prepare(t["ra"], t["dec"], **options))
            except (ValueError, InvalidCoordinateError) as exc:
                raise type(exc)(f"target {index}: {exc}") from exc
        semaphore = asyncio.Semaphore(limit)

        async def one(ctx: SearchContext) -> UnifiedRecord:
            async with semaphore:
                successes, failures = await self._execute(ctx)
                return await asyncio.to_thread(self.finalize, ctx, successes, failures)

        tasks = [asyncio.create_task(one(ctx), name=f"crossmatch_many:{i}") for i, ctx in enumerate(contexts)]
        try:
            return list(await asyncio.gather(*tasks))
        finally:
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)


# Per-target options of CrossmatchService.crossmatch_many (keyword arguments of prepare()).
MANY_TARGET_OPTIONS = ("radius_arcsec", "epoch", "profile", "pm_ra_masyr", "pm_dec_masyr", "pm_source", "parallax_mas",
                       "catalogs", "target_uncertainty_arcsec", "target_pm_error_masyr", "completeness", "target_class",
                       "resolved_object")


def validate_search_inputs(
    *,
    epoch: Any = None,
    pm_ra_masyr: Any = None,
    pm_dec_masyr: Any = None,
    parallax_mas: Any = None,
) -> None:
    """Check a requested epoch / proper motion / parallax as :func:`models.validate_target`
    does (raises ``InvalidCoordinateError``), without a position (e.g. before a name is resolved)."""
    validate_target(0.0, 0.0, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, parallax_mas=parallax_mas)


def is_binary_component(obj: Any) -> bool:
    """A resolved object that is (a component of) a double star: a double / binary type, or
    a name ending in a component letter A-D written after the identifier ('HD 239960A',
    '* 61 Cyg A', 'Wolf 1069 B'). A name ending in a constellation abbreviation ('* alf PsA',
    'V* R CrB', '* alf TrA') or in a single-letter word of the name itself ('Sgr A', a
    two-word name) is not a component."""
    otype = str(getattr(obj, "object_type", None) or "").strip().lower()
    if otype in BINARY_OTYPES:
        return True
    return any(_component_suffix(str(n or "")) for n in (getattr(obj, "canonical_name", None), getattr(obj, "query", None)))


def _component_suffix(name: str) -> bool:
    words = name.strip().split()
    if words and words[0].upper() == "NAME":  # SIMBAD's prefix of proper names ('NAME Sgr A')
        words = words[1:]
    if len(words) < 2:
        return False
    last, previous = words[-1], words[-2]
    if re.fullmatch(r"[A-D]", last):
        # A component letter after a complete identifier: a catalogue number ('Wolf 1069 B',
        # 'GJ 860 A') or a Bayer / Flamsteed designation with its constellation ('* 61 Cyg A',
        # 'alf Cen A'); not a constellation alone ('Sgr A', 'Cas A' are sources, not stars).
        if previous in CONSTELLATION_ABBREVIATIONS:
            return len(words) >= 3
        return bool(re.search(r"[0-9]", previous))
    # 'HD 239960A', 'BD+56 2783A': a letter glued to a catalogue number. Constellation
    # abbreviations ('* alf PsA', 'V* R CrB') end in a capital letter too but are not numbers,
    # and year-letter designations of transients ('SN 1987A', 'SN 1572A' -- Sesame's name of
    # Tycho's SNR --, 'AT 2018cow', 'Nova 1901A') are the year's first event, not a component.
    if previous.upper().rstrip(".") in TRANSIENT_PREFIXES and re.fullmatch(r"(1[0-9]|20)[0-9]{2}[A-Za-z]+", last):
        return False
    return bool(re.fullmatch(r"[0-9]+[A-D]", last))


def resolved_search_target(
    obj: Any,
    *,
    epoch: float | None = None,
    pm_ra_masyr: float | None = None,
    pm_dec_masyr: float | None = None,
    parallax_mas: float | None = None,
    target_uncertainty_arcsec: float | None = None,
    target_pm_error_masyr: float | None = None,
) -> dict[str, Any]:
    """The search target of a resolved name, with a requested epoch and motion applied.

    The resolver position has its own epoch (J2000.0 for SIMBAD) and motion. Without a
    requested ``epoch`` both are used as they are. With one, the position is moved from
    the resolver epoch to it with the requested motion (``pm_ra_masyr``/``pm_dec_masyr``)
    or else the resolver's; the target uncertainty grows by the motion's error over the
    interval. A dated position is never relabelled: a star whose motion is unknown cannot
    be moved, so that request raises ``ValueError``; an undated one (a VizieR answer) is
    used at the requested epoch with a warning. Extragalactic objects do not move. A binary
    component's resolver motion gets at least ``BINARY_COMPONENT_PM_SIGMA_MASYR`` of
    uncertainty (orbital motion). An answer that did not come from SIMBAD or NED (Sesame's
    VizieR fallback) is warned about and, without errors of its own, gets
    ``RESOLVER_UNDATED_SIGMA_ARCSEC``; several answers to the name ('Multiple (2) answers')
    are warned about too. Returns ra, dec, epoch, pm_ra_masyr, pm_dec_masyr, parallax_mas,
    pm_source, target_uncertainty_arcsec (+ its source), target_pm_error_masyr, notes and
    warnings (for the record's provenance).
    """
    from models import resolved_target

    rt = resolved_target(obj)
    notes: list[str] = []
    user_pm = pm_ra_masyr is not None and pm_dec_masyr is not None
    pm = (float(pm_ra_masyr), float(pm_dec_masyr)) if user_pm else rt.proper_motion  # type: ignore[arg-type]
    pm_source = "input" if user_pm else ("resolver" if rt.proper_motion is not None else None)
    pm_error = target_pm_error_masyr
    if pm_error is None and pm_source == "resolver":
        pm_error = _resolver_pm_error(obj)
    if pm_source == "resolver" and rt.proper_motion is not None and any(rt.proper_motion) and is_binary_component(obj):
        floor = BINARY_COMPONENT_PM_SIGMA_MASYR
        if pm_error is None or pm_error < floor:
            notes.append(f"{getattr(obj, 'canonical_name', None) or getattr(obj, 'query', '')} is a binary component: "
                         f"proper-motion uncertainty raised to {floor:g} mas/yr for orbital motion.")
            pm_error = floor
    sigma = target_uncertainty_arcsec
    sigma_source = "input" if sigma is not None else None
    if sigma is None:
        sigma = _resolver_position_error(obj)
        sigma_source = "resolver" if sigma is not None else None
    if sigma is None and resolver_identity_catalog({"resolver": getattr(obj, "resolver", None),
                                                    "resolver_metadata": getattr(obj, "resolver_metadata", None)}):
        otype = _otype_key(getattr(obj, "object_type", None))
        # Galaxy types only: an object that merely has a redshift (a supernova 'SN*', a nova,
        # an X-ray source in another galaxy) is a point-like source, not a galaxy's centre.
        if is_galaxy_centre_type(otype):
            # A galaxy's centre without a published error (2MASS XSC centres): about 1".
            sigma, sigma_source = GALAXY_CENTRE_SIGMA_ARCSEC, "galaxy_centre"
    warnings: list[str] = []
    meta = getattr(obj, "resolver_metadata", None) or {}
    label = getattr(obj, "canonical_name", None) or getattr(obj, "query", None) or "the object"
    answered_by = str(meta.get("resolver_name") or getattr(obj, "resolver", None) or "an unknown resolver")
    if meta.get("resolver_name") and resolver_identity_catalog({"resolver_metadata": meta}) is None:
        # Not SIMBAD / NED (Sesame fell through to VizieR): an undated catalogue position,
        # without errors or motion, of whichever catalogue listed the name.
        retry = meta.get("simbad_retry") or {}
        warnings.append(
            f"{getattr(obj, 'query', None) or label!r} was resolved by {answered_by}, not SIMBAD"
            + (f" (SIMBAD retried: {retry.get('error')})" if retry.get("error") else "")
            + ": its position is an undated catalogue position without errors or motion, so rows of a moving "
              "object may be missed; search by coordinates (with their epoch) or retry later.")
        if sigma is None and rt.epoch is None:
            sigma, sigma_source = RESOLVER_UNDATED_SIGMA_ARCSEC, "resolver_fallback"
    from providers import SesameResolver

    multiple = SesameResolver.multiple_answers(meta)
    if multiple:
        warnings.append(f"Sesame found {multiple} objects for {getattr(obj, 'query', None) or label!r} "
                        f"and returned the first ({label}): check that it is the object meant.")
    ra, dec = rt.ra, rt.dec
    plx = parallax_mas if parallax_mas is not None else rt.parallax_mas
    if epoch is None:
        epoch = rt.epoch
    elif rt.epoch is not None and float(epoch) != float(rt.epoch):
        extragalactic = bool((getattr(obj, "resolver_metadata", None) or {}).get("extragalactic"))
        if pm is not None and any(pm):
            dt = float(epoch) - float(rt.epoch)
            ra, dec = propagate_radec(ra, dec, pm[0], pm[1], float(rt.epoch), float(epoch))
            grow = (pm_error if pm_error is not None else DEFAULT_TARGET_PM_SIGMA_MASYR) * abs(dt) / 1000.0
            if grow > 0:
                sigma = math.hypot(sigma if sigma is not None else 0.0, grow)
                sigma_source = f"{sigma_source or 'default'}+pm_propagation"
            notes.append(f"Resolver position moved from J{float(rt.epoch):g} to the requested epoch J{float(epoch):g} "
                         f"with proper motion ({pm[0]:g}, {pm[1]:g}) mas/yr.")
        elif not extragalactic and pm is None:
            raise ValueError(
                f"{getattr(obj, 'canonical_name', None) or 'the object'} has no known proper motion: its "
                f"J{float(rt.epoch):g} position cannot be moved to epoch {float(epoch):g}. Omit epoch, or give "
                "pm_ra_masyr and pm_dec_masyr.")
    elif rt.epoch is None and epoch is not None:
        warnings.append(f"The resolver position of {label} ({answered_by}) has no epoch: it is used as the "
                        f"position at the requested epoch J{float(epoch):g} without moving it, which is wrong by the "
                        "object's motion over the unknown epoch difference.")
    return {"ra": ra, "dec": dec, "epoch": epoch,
            "pm_ra_masyr": pm[0] if pm is not None else None, "pm_dec_masyr": pm[1] if pm is not None else None,
            "parallax_mas": plx, "pm_source": pm_source, "target_uncertainty_arcsec": sigma,
            "target_uncertainty_source": sigma_source, "target_pm_error_masyr": pm_error, "notes": notes,
            "warnings": warnings}


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{label} must be a finite number {'>= 0' if allow_zero else '> 0'}")
    return number


_DECIMAL_TEXT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_SEXAGESIMAL_SPLIT = re.compile(r"[\s:hmsdHMSD°'\"′″]+")


def _decimal_quantum(value: Any) -> float | None:
    """Last-digit quantum of a decimal number WRITTEN AS TEXT ('187.278' -> 0.001; '150'
    -> 1). None for numbers (a float's repr says nothing about how it was typed: 150.5 may
    be exact), for non-decimal text and for booleans. OverflowError for a quantum beyond
    float range ('0e400')."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not _DECIMAL_TEXT.match(text):
        return None
    mantissa, _, exponent = text.lower().partition("e")
    exp = int(exponent) if exponent else 0
    digits = mantissa.lstrip("+-")
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    return 10.0 ** (exp - decimals)  # OverflowError for '0e400' (a digit worth 10**400 degrees)


def _sexagesimal_parts(text: str) -> tuple[float, list[str]] | None:
    """(sign, [field, ...]) of a sexagesimal string ('12 29 07.2', '-02:03:09', '12h29m'),
    or None when it is not one (a plain decimal, or malformed)."""
    body = text.strip().replace("−", "-")
    if not body or _DECIMAL_TEXT.match(body):
        return None
    sign = -1.0 if body.startswith("-") else 1.0
    body = body.lstrip("+-").strip()
    fields = [f for f in _SEXAGESIMAL_SPLIT.split(body) if f]
    if not 2 <= len(fields) <= 3 or not all(re.fullmatch(r"\d+(?:\.\d*)?", f) for f in fields):
        return None
    if any(float(f) >= 60.0 for f in fields[1:]):
        return None
    return sign, fields


def _parse_sexagesimal(text: str, *, hours: bool) -> tuple[float, float] | None:
    """(degrees, quantum in degrees) of a sexagesimal angle; ``hours`` for RA in hours."""
    parts = _sexagesimal_parts(text)
    if parts is None:
        return None
    sign, fields = parts
    values = [float(f) for f in fields]
    unit = 15.0 if hours else 1.0
    total = 0.0
    for i, v in enumerate(values):  # plain left-to-right sum, as a client computes it (not fsum)
        total = total + v / 60.0**i
    degrees = sign * unit * total
    last = fields[-1]
    decimals = len(last.split(".", 1)[1]) if "." in last else 0
    quantum = unit * 10.0 ** (-decimals) / 60.0 ** (len(fields) - 1)
    return degrees, quantum


def _ra_in_degrees(text: str) -> bool:
    """A sexagesimal RA marked in degrees ('187d16m44s', '187°16′44″') rather than hours."""
    return bool(re.search(r"[dD°]", text)) and not re.search(r"[hH]", text)


# Sexagesimal quanta (degrees) a converted RA / Dec may carry: 1 s, 0.1 s, 0.01 s of time
# and 1", 0.1", 0.01" of arc (coarsest first).
_RA_SEXAGESIMAL_STEPS = tuple(15.0 * s / 3600.0 for s in (1.0, 0.1, 0.01))
_DEC_SEXAGESIMAL_STEPS = tuple(s / 3600.0 for s in (1.0, 0.1, 0.01))


def _repr_quantum(value: float) -> float:
    """Last-digit quantum of a float's shortest repr (1e-14 for 187.27916666666664)."""
    return _decimal_quantum(repr(float(value))) or 0.0


def _on_grid(value: float, step: float) -> bool:
    k = value / step
    return abs(k - round(k)) < 1e-6


def _sexagesimal_signature(ra: float, dec: float) -> tuple[float, float] | None:
    """(RA, Dec quantum in degrees) when floats are the decimal conversion of rounded
    sexagesimal coordinates -- on a 1 s / 0.1 s / 0.01 s (RA) or 1" / 0.1" / 0.01" (Dec)
    grid although their decimal expansion is long (187.27916666666664 = 12h29m07s) -- else
    None. A float with a short expansion (150.5, 2.2) is exact: it lies on every grid by
    construction and carries no such signature."""
    def signature(value: float, steps: tuple[float, ...]) -> float | None:
        if not math.isfinite(value):
            return None
        fine = _repr_quantum(value)
        for step in steps:
            if fine < step / 100.0 and _on_grid(value, step):
                return step
        return None

    def coarsest(value: float, steps: tuple[float, ...]) -> float:
        return next((step for step in steps if _on_grid(value, step)), 0.0)

    q_ra, q_dec = signature(ra, _RA_SEXAGESIMAL_STEPS), signature(dec, _DEC_SEXAGESIMAL_STEPS)
    if q_ra is None and q_dec is None:
        return None
    return (q_ra if q_ra is not None else coarsest(ra, _RA_SEXAGESIMAL_STEPS),
            q_dec if q_dec is not None else coarsest(dec, _DEC_SEXAGESIMAL_STEPS))


def _quanta_axes_arcsec(q_ra: float | None, q_dec: float | None, dec_deg: float) -> tuple[float, float]:
    """Per-axis (east, north) sigma (arcsec) of coordinates rounded to these quanta (degrees):
    a quantum q is uniformly distributed, sigma = q / sqrt(12); RA's is q cos(dec) on the sky."""
    cos_dec = abs(math.cos(math.radians(dec_deg))) if math.isfinite(dec_deg) else 1.0
    s_ra = (q_ra or 0.0) * 3600.0 * cos_dec / math.sqrt(12.0)
    s_dec = (q_dec or 0.0) * 3600.0 / math.sqrt(12.0)
    return s_ra, s_dec


def _quanta_sigma_arcsec(q_ra: float | None, q_dec: float | None, dec_deg: float) -> float:
    """RMS per-axis sigma (arcsec) of coordinates rounded to these quanta (degrees)."""
    s_ra, s_dec = _quanta_axes_arcsec(q_ra, q_dec, dec_deg)
    if not (math.isfinite(s_ra) and math.isfinite(s_dec)):
        return math.inf
    return math.sqrt((s_ra * s_ra + s_dec * s_dec) / 2.0)


def parse_target_coordinates(ra: Any, dec: Any) -> tuple[Any, Any, float]:
    """(ra, dec, rounding sigma in arcsec -- the RMS of the two axes) of target coordinates
    as given (see :func:`target_coordinate_rounding` for the per-axis values).

    Numbers are exact (sigma 0) unless they carry the signature of a sexagesimal
    conversion (:func:`_sexagesimal_signature`). Text is decimal degrees -- its last digit
    is the quantum -- or sexagesimal (RA in hours unless marked in degrees); unparseable
    text is passed on unchanged for :func:`models.validate_target` to reject.
    """
    ra_value, dec_value, (s_ra, s_dec) = target_coordinate_rounding(ra, dec)
    if not (math.isfinite(s_ra) and math.isfinite(s_dec)):
        return ra_value, dec_value, math.inf
    return ra_value, dec_value, math.sqrt((s_ra * s_ra + s_dec * s_dec) / 2.0)


def target_coordinate_rounding(ra: Any, dec: Any) -> tuple[Any, Any, tuple[float, float]]:
    """(ra, dec, (east, north) rounding sigma in arcsec) of target coordinates as given
    (see :func:`parse_target_coordinates`)."""
    if not isinstance(ra, str) and not isinstance(dec, str):
        try:
            ra_f, dec_f = float(ra), float(dec)
        except (TypeError, ValueError):
            return ra, dec, (0.0, 0.0)
        if isinstance(ra, bool) or isinstance(dec, bool):
            return ra, dec, (0.0, 0.0)
        quanta = _sexagesimal_signature(ra_f, dec_f) if isinstance(ra, float) or isinstance(dec, float) else None
        return ra, dec, _quanta_axes_arcsec(*quanta, dec_f) if quanta else (0.0, 0.0)
    ra_value: Any = ra
    dec_value: Any = dec
    q_ra = q_dec = None
    if isinstance(ra, str):
        q_ra = _decimal_quantum(ra)
        if q_ra is not None:
            ra_value = float(ra.strip())
        else:
            parsed = _parse_sexagesimal(ra, hours=not _ra_in_degrees(ra))
            if parsed is not None:
                ra_value, q_ra = parsed
    if isinstance(dec, str):
        q_dec = _decimal_quantum(dec)
        if q_dec is not None:
            dec_value = float(dec.strip())
        else:
            parsed = _parse_sexagesimal(dec, hours=False)
            if parsed is not None:
                dec_value, q_dec = parsed
    try:
        dec_deg = float(dec_value)
    except (TypeError, ValueError):
        dec_deg = 0.0
    return ra_value, dec_value, _quanta_axes_arcsec(q_ra, q_dec, dec_deg)


def coordinate_sigma_arcsec(ra: Any, dec: Any) -> float:
    """Per-axis 1-sigma (arcsec) of the rounding of coordinates as given.

    A coordinate rounded to a quantum q is uniformly distributed over q: sigma = q / sqrt(12)
    (RA: q x cos dec on the sky); the RMS of the two axes is returned. Text: '187.278' has
    q = 0.001 deg (1.04" per axis for 3C 273), '12 29 07' q = 1 s of time. Numbers are exact
    (0), except floats converted from rounded sexagesimal (see :func:`parse_target_coordinates`).
    """
    return parse_target_coordinates(ra, dec)[2]


def _completeness_override(value: Any) -> float | dict[str, float] | None:
    """Validate a user completeness prior: a number in (0, 1) or {catalog: number in (0, 1)}."""
    if value is None:
        return None
    if isinstance(value, dict):
        out: dict[str, float] = {}
        for key, v in value.items():
            c = _to_float(v)
            if c is None or not 0.0 < c < 1.0:
                raise ValueError(f"completeness for {key} must lie strictly between 0 and 1")
            out[str(key)] = c
        return out
    c = _to_float(value)
    if c is None or not 0.0 < c < 1.0:
        raise ValueError("completeness must lie strictly between 0 and 1")
    return float(c)


def _resolved_dict(value: Any) -> dict[str, Any] | None:
    """A resolver answer as a dict (ResolvedObject.as_dict()), or None."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        result = as_dict()
        return result if isinstance(result, dict) else None
    raise ValueError("resolved_object must be a ResolvedObject or its as_dict()")


def _resolver_field(resolved: dict[str, Any] | None, *keys: str) -> float | None:
    raw = ((resolved or {}).get("resolver_metadata") or {}).get("raw_fields") or {}
    return _first_number(raw, *keys)


def _resolver_sptype(resolved: dict[str, Any] | None) -> str | None:
    raw = ((resolved or {}).get("resolver_metadata") or {}).get("raw_fields") or {}
    value = raw.get("sptype") or raw.get("sp_type")
    if isinstance(value, list):
        value = value[0] if value else None
    return str(value) if value else None


def _class_parallax(target: Target, parallax_origin: dict[str, Any] | None, resolved: dict[str, Any] | None) -> float | None:
    """The target parallax usable for its class: given, adopted (significant by
    construction), or the resolver's when its error is unknown or it is significant
    (>= CLASS_PARALLAX_MIN_SNR sigma). None otherwise."""
    plx = target.parallax_mas
    if plx is None or plx <= 0:
        return None
    if (parallax_origin or {}).get("source") == "resolver" and resolved:
        err = _resolver_field(resolved, "plx.e")
        if err is not None and err > 0 and plx / err < CLASS_PARALLAX_MIN_SNR:
            return None
    return float(plx)


def _class_before_association(
    resolved: dict[str, Any] | None,
    pm_origin: dict[str, Any] | None,
    extended_row: int | None,
    matches: list[Match],
    target_parallax: float | None,
) -> dict[str, Any] | None:
    """Target class known before the association, from the resolver's identity (object
    type, spectral type, extragalactic flag), an adopted extragalactic identity, or an
    extended identity row at the target. None: the identity rows / counterparts decide
    (:func:`_class_from_association`). For a resolver 'star' without a significant parallax
    the parallax is left None (the counterparts may supply Gaia's)."""
    if resolved:
        otype = resolved.get("object_type")
        kind = identity_kind(otype, _resolver_sptype(resolved))
        info = {"source": "resolver identity", "object_type": otype, "name": resolved.get("canonical_name")}
        if kind == "extended":
            return {"class": "extended", **info}
        if kind == "extragalactic" or ((resolved.get("resolver_metadata") or {}).get("extragalactic")):
            return {"class": "extragalactic", **info}
        if kind == "emitter":
            return {"class": "unknown", **info, "reason": "radio / X-ray emitter or uncalibrated stellar type"}
        if kind == "star":
            return {"class": "star", **info, "parallax_mas": target_parallax}
    if pm_origin is not None and pm_origin.get("source") == "extragalactic":
        return {"class": "extragalactic", "source": "adopted identity", "catalog": pm_origin.get("catalog"),
                "source_id": pm_origin.get("source_id")}
    if extended_row is not None:
        src = matches[extended_row].source
        return {"class": "extended", "source": "extended identity at the target", "catalog": src.catalog,
                "source_id": src.source_id, "object_type": row_object_type(src) or None}
    return None


def _row_kind(src: CatalogSource) -> str | None:
    """identity_kind of a row (a redshift >= EXTRAGALACTIC_MIN_REDSHIFT, or an extragalactic
    star cluster, :func:`mark_compact_clusters`: extragalactic)."""
    if (src.metadata or {}).get("compact_cluster"):
        return "extragalactic"
    if _is_extragalactic_row(src) and row_object_type(src) not in EXTENDED_OTYPES:
        return "extragalactic"
    return identity_kind(row_object_type(src), (src.data or {}).get("sp_type"))


def _class_from_association(matches: list[Match], association: AssociationResult, *,
                            target_parallax: float | None = None) -> dict[str, Any]:
    """Target class from the most probable counterparts (found with the default priors).

    Identity rows decide first, SIMBAD's before NED's before other catalogues': any row of
    an extended, extragalactic or radio / X-ray emitter type decides ('extended',
    'extragalactic', 'unknown'); then an ordinary stellar type -> 'star' (with Gaia's
    significant parallax, else the row's own significant one, else the target's).
    Without a decisive identity: a Gaia DR3 counterpart with a parallax of at least
    CLASS_PARALLAX_MIN_SNR sigma, else a (significant or given) target parallax -> 'star'.
    Without a counterpart of p_any >= CLASS_EVIDENCE_MIN_P_ANY: 'star' when the target has
    a parallax, else 'unknown'.
    """
    group = next((g for g in association.groups if g.contains_target), None)
    if group is None or (association.p_any or 0.0) < CLASS_EVIDENCE_MIN_P_ANY:
        if target_parallax is not None:
            return {"class": "star", "source": "target parallax", "parallax_mas": target_parallax,
                    "p_any": association.p_any}
        return {"class": "unknown", "source": "no counterpart", "p_any": association.p_any}
    members = [matches[i].source for i in group.members if i not in group.coincident_with]
    tiers = sorted(members, key=lambda s: (IDENTITY_CATALOGS.index(s.catalog) if s.catalog in IDENTITY_CATALOGS
                                           else len(IDENTITY_CATALOGS)))
    kinds = [(src, _row_kind(src)) for src in tiers]
    names = {"extended": "extended", "extragalactic": "extragalactic", "emitter": "unknown"}
    for src, kind in kinds:
        if kind in names:
            info = {"class": names[kind], "source": "identity type", "catalog": src.catalog, "source_id": src.source_id,
                    "object_type": row_object_type(src) or None}
            if kind == "emitter":
                info["reason"] = "radio / X-ray emitter or uncalibrated stellar type"
            return info
    gaia_plx = _gaia_parallax(members)
    for src, kind in kinds:
        if kind == "star":
            own = _significant_parallax(src)
            plx = gaia_plx if gaia_plx is not None else (own if own is not None else target_parallax)
            return {"class": "star", "source": "identity type", "catalog": src.catalog, "source_id": src.source_id,
                    "object_type": row_object_type(src) or None, "parallax_mas": plx}
    if gaia_plx is not None:
        gaia = next(s for s in members if s.catalog == "gaia_dr3")
        return {"class": "star", "source": "gaia parallax", "catalog": "gaia_dr3", "source_id": gaia.source_id,
                "parallax_mas": gaia_plx}
    if target_parallax is not None:
        return {"class": "star", "source": "target parallax", "parallax_mas": target_parallax}
    return {"class": "unknown", "source": "counterparts without a class", "p_any": association.p_any}


def _row_problem(src: CatalogSource) -> str | None:
    """Why a row's astrometry is unusable for the association, or None: a non-finite or
    impossible position, a non-finite or negative positional error, a non-finite epoch or
    epoch range, or a non-finite or implausible (> MAX_ROW_PM_MASYR) proper motion."""
    try:
        ra, dec = float(src.ra), float(src.dec)
    except (TypeError, ValueError):
        return "non-numeric position"
    if not (math.isfinite(ra) and math.isfinite(dec)) or not -90.0 <= dec <= 90.0:
        return "invalid position"
    err = src.positional_error_arcsec
    if err is not None:
        try:
            err = float(err)
        except (TypeError, ValueError):
            return "non-numeric positional error"
        if not math.isfinite(err) or err < 0:
            return "invalid positional error"
    if src.epoch is not None and not (isinstance(src.epoch, (int, float)) and math.isfinite(src.epoch)):
        return "invalid epoch"
    if src.epoch_range is not None and not all(isinstance(v, (int, float)) and math.isfinite(v) for v in src.epoch_range):
        return "invalid epoch range"
    pm = (src.proper_motion_ra_masyr, src.proper_motion_dec_masyr)
    if any(v is not None for v in pm):
        try:
            values = [float(v) for v in pm if v is not None]
        except (TypeError, ValueError):
            return "non-numeric proper motion"
        if not all(math.isfinite(v) for v in values) or math.hypot(*values) > MAX_ROW_PM_MASYR:
            return "implausible proper motion"
    return None


def _poisson_sf(n: int, lam: float) -> float:
    """P(N >= n) for N ~ Poisson(lam)."""
    if n <= 0:
        return 1.0
    if lam <= 0:
        return 0.0
    term = math.exp(-lam)
    below = term
    for k in range(1, n):
        term *= lam / k
        below += term
    return max(0.0, 1.0 - below)


def _density_probe_reason(name: str, sources: Any, ctx: SearchContext, registry: CatalogRegistry | None) -> str | None:
    """Why a star-following catalogue's local density must be measured over a
    DENSITY_PROBE_RADIUS_ARCSEC cone (None: it need not be):

    * 'poisson excess': the small cone holds far more field rows than the density map
      predicts (P(N >= n | DENSITY_PROBE_SLACK x expected) < DENSITY_PROBE_P_VALUE, at least
      DENSITY_PROBE_MIN_ROWS field rows);
    * 'crowded region <name>': else, the target lies in a globular cluster or a dense part
      of a Local Group galaxy (astrometry.crowded_region), whatever the small cone holds --
      a 3" cone in 47 Tuc's core often holds 0-2 Gaia rows although the local density is
      10x the map's.

    Never for a search radius of at least DENSITY_PROBE_RADIUS_ARCSEC, nor for an archive-
    truncated cone (it measured the local density itself)."""
    # Looked up at call time: aliases registered later (batch.py's VizieR views of the
    # registry surveys) follow their survey.
    if name not in DENSITY_MAP_SCALING or name in DENSITY_PROBE_EXCLUDED or ctx.search_radius >= DENSITY_PROBE_RADIUS_ARCSEC:
        return None
    meta = getattr(sources, "meta", {}) or {}
    if meta.get("archive_truncated") or meta.get("truncated"):
        return None  # a truncated cone measured the local density itself
    rows = list(sources) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or [])
    field_rows = len(rows) - _target_rows(rows, ctx.target, ctx.target_sigma_arcsec)
    if field_rows >= DENSITY_PROBE_MIN_ROWS:
        radius = float(meta.get("query_radius_arcsec") or ctx.search_radius)
        definition = registry.catalogs.get(name) if registry is not None else None
        prior, _ = local_prior_density(name, ctx.target.ra, ctx.target.dec, catalog_sky_density(definition))
        expected = DENSITY_PROBE_SLACK * prior * cone_area_deg2(radius)
        if _poisson_sf(field_rows, expected) < DENSITY_PROBE_P_VALUE:
            return POISSON_EXCESS
    region = crowded_region(ctx.target.ra, ctx.target.dec)
    return f"crowded region {region}" if region is not None else None


def _needs_density_probe(name: str, sources: Any, ctx: SearchContext, registry: CatalogRegistry | None) -> bool:
    """True when the catalogue's local density must be measured (see :func:`_density_probe_reason`)."""
    return _density_probe_reason(name, sources, ctx, registry) is not None


def _gaia_parallax(members: list[CatalogSource]) -> float | None:
    for src in members:
        if src.catalog != "gaia_dr3":
            continue
        plx, err = _to_float((src.data or {}).get("parallax")), _to_float((src.data or {}).get("parallax_error"))
        if plx is not None and err is not None and err > 0 and plx / err >= CLASS_PARALLAX_MIN_SNR:
            return plx
    return None


def _adopted_pm_sigma(origin: dict[str, Any], successes: list[tuple[str, list[CatalogSource]]]) -> float | None:
    """Uncertainty of an adopted target motion: 0 for an extragalactic (stationary)
    target, else that of the catalogue row it was taken from."""
    if origin.get("source") == "extragalactic":
        return 0.0
    for name, sources in successes:
        if name != origin.get("catalog"):
            continue
        meta = getattr(sources, "meta", {}) or {}
        for src in list(sources) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or []):
            if src.source_id == origin.get("source_id"):
                return pm_sigma_masyr(src)
    return None


def _resolver_values(obj: Any) -> dict[str, Any]:
    return ((getattr(obj, "resolver_metadata", None) or {}).get("raw_fields") or {})


def _first_number(values: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = values.get(key)
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        number = _to_float(raw)
        if number is not None:
            return number
    return None


def _resolver_position_error(obj: Any) -> float | None:
    """Per-axis 1-sigma position error (arcsec) of a Sesame answer (errRAmas/errDEmas)."""
    values = _resolver_values(obj)
    era, ede = _first_number(values, "errramas"), _first_number(values, "errdemas")
    errs = [e for e in (era, ede) if e is not None and e > 0]
    if not errs:
        return None
    return math.sqrt(sum(e * e for e in errs) / len(errs)) / 1000.0


def _resolver_pm_error(obj: Any) -> float | None:
    """Per-axis proper-motion error (mas/yr) of a Sesame answer (pm.epmra / pm.epmde)."""
    values = _resolver_values(obj)
    errs = [e for e in (_first_number(values, "pm.epmra"), _first_number(values, "pm.epmde")) if e is not None and e >= 0]
    if not errs:
        return None
    return math.sqrt(sum(e * e for e in errs) / len(errs))


def _stream_source_dict(source: CatalogSource) -> dict[str, Any]:
    """Compact serialisation of a catalogue row for a streamed ``catalog`` event."""
    meta = source.metadata or {}
    return {
        "catalog": source.catalog,
        "source_id": source.source_id,
        "ra": source.ra,
        "dec": source.dec,
        "separation_arcsec": meta.get("epoch_separation_arcsec", meta.get("query_separation_arcsec")),
        "epoch_propagation": meta.get("epoch_propagation"),
        "positional_error_arcsec": source.positional_error_arcsec,
        "epoch": source.epoch,
        "epoch_range": list(source.epoch_range) if source.epoch_range else None,
        "proper_motion_ra_masyr": source.proper_motion_ra_masyr,
        "proper_motion_dec_masyr": source.proper_motion_dec_masyr,
        "wavelength": meta.get("wavelength"),
        "physical": meta.get("physical", {}),
        "data": source.data,
    }


def _pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return None


def _is_extragalactic_row(src: CatalogSource) -> bool:
    physical = src.metadata.get("physical") or {}
    if is_extragalactic_type(physical.get("object_type")):
        return True
    redshift = physical.get("redshift")
    try:
        return redshift is not None and abs(float(redshift)) >= EXTRAGALACTIC_MIN_REDSHIFT
    except (TypeError, ValueError):
        return False


def _pm_is_noise(src: CatalogSource) -> bool:
    """A small proper motion of a row whose parallax is insignificant (distant/extragalactic)."""
    try:
        plx = src.data.get("parallax")
        plx_err = src.data.get("parallax_error")
        total = math.hypot(float(src.proper_motion_ra_masyr), float(src.proper_motion_dec_masyr))  # type: ignore[arg-type]
    except (TypeError, ValueError, AttributeError):
        return False
    if plx_err in (None, 0) or plx is None:
        return False
    try:
        snr = float(plx) / float(plx_err)
    except (TypeError, ValueError, ZeroDivisionError):
        return False
    return snr < PM_ADOPTION_MIN_PARALLAX_SNR and total < PM_NOISE_MASYR


def _pm_pair_agree(pa: tuple[float, float], pb: tuple[float, float]) -> bool:
    tolerance = max(PM_AGREE_MASYR, PM_AGREE_FRACTION * max(math.hypot(*pa), math.hypot(*pb)))
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= tolerance


def _motions_agree(a: CatalogSource, b: CatalogSource) -> bool:
    """True when two rows' proper motions describe the same object (within catalog scatter)."""
    pa = (float(a.proper_motion_ra_masyr), float(a.proper_motion_dec_masyr))  # type: ignore[arg-type]
    pb = (float(b.proper_motion_ra_masyr), float(b.proper_motion_dec_masyr))  # type: ignore[arg-type]
    return _pm_pair_agree(pa, pb)


def _adopt_parallax(
    target: Target, successes: list[tuple[str, list[CatalogSource]]], radius_arcsec: float
) -> tuple[Target, dict[str, Any]] | None:
    """Adopt a significant parallax for a target whose proper motion is known but whose
    parallax is not: from the nearest row within min(radius, PM_ADOPTION_MAX_ARCSEC)
    whose own motion agrees with the target's (the same star in Gaia or SIMBAD). Rows of
    extragalactic identities (a quasar's Gaia-noise parallax) and insignificant parallaxes
    (:func:`_significant_parallax`) are never adopted."""
    pm = target.proper_motion
    if pm is None or target.epoch is None:
        return None
    limit = min(radius_arcsec, PM_ADOPTION_MAX_ARCSEC)
    best: tuple[float, str, CatalogSource, float] | None = None
    for name, sources in successes:
        meta = getattr(sources, "meta", {}) or {}
        for src in list(sources) + list(meta.get("excess_sources") or []):
            if src.proper_motion_ra_masyr is None or src.proper_motion_dec_masyr is None or src.epoch is None:
                continue
            if _is_extragalactic_row(src):
                continue
            if not _pm_pair_agree(pm, (float(src.proper_motion_ra_masyr), float(src.proper_motion_dec_masyr))):
                continue
            plx = _significant_parallax(src)
            if plx is None:
                continue
            sep, _ = epoch_separation_arcsec(target, src)
            if sep <= limit and (best is None or sep < best[0]):
                best = (sep, name, src, plx)
    if best is None:
        return None
    sep, name, src, plx = best
    return replace(target, parallax_mas=plx), {"parallax_mas": plx, "source": "adopted", "catalog": name,
                                               "source_id": src.source_id, "separation_arcsec": sep}


def _is_planet(src: CatalogSource) -> bool:
    return str((src.metadata.get("physical") or {}).get("object_type") or "").strip().lower() in {"pl", "pl?"}


def _significant_parallax(src: CatalogSource) -> float | None:
    """The row's parallax (mas) when positive, below 1000 mas and significant: at least
    PARALLAX_ADOPTION_MIN_SNR sigma when its error is known (Gaia's ``parallax_error``).
    A parallax without an error (SIMBAD's plx_value, fetched without plx_err) counts only
    from PARALLAX_WITHOUT_ERROR_MIN_MAS: smaller ones cannot be told from noise (3C 273:
    0.011 mas and NGC 4151: 0.22 +- 0.09 mas, Gaia-noise values of AGN)."""
    data = src.data or {}
    plx = _to_float(data.get("parallax"))
    err = _to_float(data.get("parallax_error"))
    if plx is None:
        plx = _to_float((src.metadata.get("physical") or {}).get("parallax"))
        err = None
    if plx is None or plx <= 0 or plx >= 1000.0:
        return None
    if err is None or not err > 0:
        return plx if plx >= PARALLAX_WITHOUT_ERROR_MIN_MAS else None
    if plx / err < PARALLAX_ADOPTION_MIN_SNR:
        return None
    return plx


def _adoption_chi2(target: Target, src: CatalogSource, target_sigma_arcsec: float, *,
                   moving: bool) -> tuple[float, float]:
    """(chi2, separation) of a row against the target position, for adopting its motion.

    A row with its own motion (``moving``) is propagated to the target's epoch -- for an
    undated target to each of UNDATED_TARGET_EPOCHS (the better one counts), with
    UNDATED_EPOCH_SPREAD_YR of motion along its track -- and its covariance grows by its
    proper-motion error over the interval; other rows are compared where they are. The
    covariance is the association's (:func:`astrometry.detection_covariance`) plus the
    target sigma."""
    base, _ = detection_covariance(src)
    if not moving:
        sep = epoch_separation_arcsec(target, src)[0] if target.epoch is not None else \
            haversine_arcsec(target.ra, target.dec, src.ra, src.dec)
        sigma2 = (base[0] + base[2]) / 2.0 + target_sigma_arcsec**2
        return sep * sep / sigma2, sep
    pm = (float(src.proper_motion_ra_masyr), float(src.proper_motion_dec_masyr))  # type: ignore[arg-type]
    epochs = [float(target.epoch)] if target.epoch is not None else list(UNDATED_TARGET_EPOCHS)
    best = (math.inf, math.inf)
    for epoch in epochs:
        ra, dec = propagate_radec(src.ra, src.dec, pm[0], pm[1], float(src.epoch), epoch)  # type: ignore[arg-type]
        growth = pm_sigma_masyr(src) * abs(epoch - float(src.epoch)) / 1000.0  # type: ignore[arg-type]
        var = growth * growth + target_sigma_arcsec**2
        cov = (base[0] + var, base[1], base[2] + var)
        if target.epoch is None:
            cov = _along_track(cov, pm, math.sqrt(12.0) * UNDATED_EPOCH_SPREAD_YR)
        east, north = tangent_offset_arcsec(target.ra, target.dec, ra, dec)
        det = cov[0] * cov[2] - cov[1] * cov[1]
        chi2 = (cov[2] * east * east - 2.0 * cov[1] * east * north + cov[0] * north * north) / det
        best = min(best, (float(chi2), haversine_arcsec(target.ra, target.dec, ra, dec)))
    return best


def _adopt_proper_motion(
    target: Target,
    successes: list[tuple[str, list[CatalogSource]]],
    radius_arcsec: float,
    *,
    warnings: list[str] | None = None,
    target_sigma_arcsec: float = DEFAULT_TARGET_SIGMA_ARCSEC,
) -> tuple[Target, dict[str, Any]] | None:
    """Adopt the proper motion of the nearest catalog row that has one and IS the target.

    Only rows that coincide with the target qualify: within min(radius,
    PM_ADOPTION_MAX_ARCSEC) of it and consistent with its position within the errors
    (:func:`_adoption_chi2` <= IDENTITY_COINCIDENCE_CHI2 with ``target_sigma_arcsec``) after
    their own propagation -- to the target's epoch, or for undated coordinates to the
    J2000 / J2016 hypothesis that fits better (in-radius rows beyond max_rows included). A
    neighbouring star 1.5" from a galaxy's catalogued position is not the galaxy, whatever
    its motion. Returns the target with that motion (and the row's parallax when
    significant, or that of an agreeing coinciding candidate) and a provenance note.

    Extragalactic targets are stationary: when the nearest coinciding identified row (one
    with an object type or redshift, e.g. SIMBAD or NED) is extragalactic -- M87 is 'AGN' in
    SIMBAD, which lists a Gaia proper motion of its nucleus -- pm = (0, 0) is adopted
    instead ("source": "extragalactic"). Rows that are themselves extragalactic, or
    whose small proper motion comes with an insignificant parallax, are never adopted.

    Crowded fields: when a row (coinciding or not) whose motion DISAGREES with the nearest
    candidate lies within 2 x (its separation) + PM_AMBIGUITY_MARGIN_ARCSEC, the nearest row may
    be a chance alignment (around Sgr A* SIMBAD has ~200 objects within 2"), so nothing
    is adopted and a warning is appended to ``warnings``. Candidates that agree (the same
    star in Gaia, SIMBAD and the Exoplanet Archive) never block adoption.
    """
    limit = min(radius_arcsec, PM_ADOPTION_MAX_ARCSEC)
    candidates: list[tuple[float, str, CatalogSource]] = []
    others: list[tuple[float, str, CatalogSource]] = []
    identity: tuple[float, str, CatalogSource] | None = None
    for name, sources in successes:
        meta = getattr(sources, "meta", {}) or {}
        for src in list(sources) + list(meta.get("excess_sources") or []):
            sep, _ = epoch_separation_arcsec(target, src)
            if sep > limit and target.epoch is not None:
                continue
            with_pm = (src.proper_motion_ra_masyr is not None and src.proper_motion_dec_masyr is not None
                       and src.epoch is not None and not _is_extragalactic_row(src))
            chi2, sep = _adoption_chi2(target, src, target_sigma_arcsec, moving=with_pm)
            if sep > limit:
                continue
            coincides = chi2 <= IDENTITY_COINCIDENCE_CHI2
            physical = src.metadata.get("physical") or {}
            if (coincides and (physical.get("object_type") or physical.get("redshift") is not None)
                    and (identity is None or sep < identity[0])):
                identity = (sep, name, src)
            if not with_pm or _pm_is_noise(src):
                continue
            (candidates if coincides else others).append((sep, name, src))
    if identity is not None and _is_extragalactic_row(identity[2]):
        sep, name, src = identity
        physical = src.metadata.get("physical") or {}
        stationary = replace(target, pm_ra_masyr=0.0, pm_dec_masyr=0.0)
        return stationary, {"source": "extragalactic", "catalog": name, "source_id": src.source_id,
                            "separation_arcsec": sep, "object_type": physical.get("object_type"),
                            "redshift": physical.get("redshift")}
    if not candidates:
        return None
    # Nearest first; coincident rows (SIMBAD lists a star and its planets at one position)
    # prefer the non-planet, then a deterministic catalog/id order.
    candidates.sort(key=lambda c: (round(c[0] / 1e-4), _is_planet(c[2]), c[1], c[2].source_id))
    sep, name, src = candidates[0]
    zone = 2.0 * sep + PM_AMBIGUITY_MARGIN_ARCSEC
    rivals = sorted((c for c in candidates[1:] + others if c[0] <= zone and not _motions_agree(src, c[2])),
                    key=lambda c: c[0])
    if rivals:
        if warnings is not None:
            r_sep, r_name, r_src = rivals[0]
            warnings.append(
                f"Target proper motion not adopted: the nearest candidate {name} {src.source_id} ({sep:.3f} arcsec) "
                f"and {len(rivals)} other row(s) with different motions (e.g. {r_name} {r_src.source_id} at "
                f"{r_sep:.3f} arcsec) lie within {zone:.2f} arcsec, so the match is ambiguous; supply "
                "pm_ra_masyr/pm_dec_masyr to follow the target."
            )
        return None
    parallax: dict[str, Any] | None = None
    for c_sep, c_name, cand in candidates:
        if cand is src or _motions_agree(src, cand):
            plx = _significant_parallax(cand)
            if plx is not None:
                parallax = {"parallax_mas": plx, "source": "adopted", "catalog": c_name,
                            "source_id": cand.source_id, "separation_arcsec": c_sep}
                break
    use_parallax = target.parallax_mas is None and parallax is not None
    adopted = replace(target, pm_ra_masyr=src.proper_motion_ra_masyr, pm_dec_masyr=src.proper_motion_dec_masyr,
                      parallax_mas=parallax["parallax_mas"] if use_parallax else target.parallax_mas)  # type: ignore[index]
    origin: dict[str, Any] = {"source": "adopted", "catalog": name, "source_id": src.source_id, "separation_arcsec": sep}
    if use_parallax:
        origin["parallax"] = parallax
    return adopted, origin

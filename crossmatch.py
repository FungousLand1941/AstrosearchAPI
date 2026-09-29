"""Query building, spatial geometry, proper-motion epoch propagation, and crossmatching engine.

Associations between the target and catalogue rows, and between rows of different
catalogues, are Bayesian (Budavari & Szalay 2008, ApJ 679, 301; NWAY, Salvato et al. 2018,
MNRAS 473, 4937): see :mod:`astrometry`. Every row is first brought to the target's epoch
(its own proper motion, else the target's motion and parallax) with its positional
covariance grown by the proper-motion uncertainty over the epoch difference. A match's
``confidence`` is the posterior probability that the row is the target's counterpart.
:meth:`CrossmatchService.crossmatch_stream` yields results catalogue by catalogue.
"""

from __future__ import annotations

import asyncio
import math
import statistics
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from time import monotonic
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord

from astrometry import (
    COINCIDENT_ARCSEC,
    DEFAULT_TARGET_PM_SIGMA_MASYR,
    LINK_CHI2,
    AssociationConfig,
    AssociationResult,
    Detection,
    associate,
    cone_area_deg2,
    estimate_density_deg2,
    pm_sigma_masyr,
    source_detection,
)
from models import (
    TARGET_PM_METHODS,
    CatalogDefinition,
    CatalogFailure,
    CatalogRegistry,
    CatalogSource,
    CatalogUnavailableError,
    Match,
    QueryPlan,
    QueryTimeoutError,
    Target,
    UnifiedRecord,
    _to_float,
    epoch_separation_arcsec,
    haversine_arcsec,
    is_extragalactic_type,
    source_position_at,
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
# Rows whose parallax could not be removed get the target parallax as extra uncertainty
# when it is at least this large (smaller parallaxes are within catalog errors).
PARALLAX_INFLATION_MIN_MAS = 50.0

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
        return True


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
            if query.profiles and not any(p in catalog.profiles for p in query.profiles):
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
        source_sigma = max(positional_error_arcsec or 0.0, 0.1)
        target_sigma = max(target_uncertainty_arcsec or 0.0, 0.0)
        sigma = math.sqrt(source_sigma**2 + target_sigma**2)
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


def _match_detections(
    matches: list[Match],
    target: Target,
    *,
    target_pm_sigma_masyr: float | None = None,
) -> tuple[list[Detection], list[dict[str, Any]]]:
    """Detections of the matched rows at the common epoch (see :func:`astrometry.source_detection`).

    Extragalactic rows (galaxy/QSO types, redshift >= EXTRAGALACTIC_MIN_REDSHIFT) are not moved.

    Rows compared through the target's own motion without a parallax correction get the
    target's parallax as an extra uncertainty (:func:`parallax_uncertainty_arcsec`).
    """
    pm_sigma = DEFAULT_TARGET_PM_SIGMA_MASYR if target_pm_sigma_masyr is None else float(target_pm_sigma_masyr)
    detections: list[Detection] = []
    infos: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        method = source_position_at(match.source, target.epoch, target.proper_motion, (target.ra, target.dec),
                                    target.parallax_mas)[2]
        det, info = source_detection(match.source, target, target_pm_sigma_masyr=pm_sigma,
                                     extra_sigma_arcsec=parallax_uncertainty_arcsec(target, method),
                                     priority=_row_priority(match.source), label=idx,
                                     extragalactic=_is_extragalactic_row(match.source))
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
        sigma2 = (src.positional_error_arcsec or 0.0) ** 2 + target_sigma_arcsec**2
        if sigma2 > 0 and sep * sep / sigma2 <= LINK_CHI2 and (best is None or sep < best[0]):
            best = (float(sep), src)
    if best is None:
        return 0
    ref = best[1]
    return sum(1 for s in sources if haversine_arcsec(s.ra, s.dec, ref.ra, ref.dec) <= COINCIDENT_ARCSEC)


def catalog_densities(
    successes: list[tuple[str, list[CatalogSource]]],
    target: Target,
    radius_arcsec: float,
    target_sigma_arcsec: float,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    """Field-source density (deg^-2) of every queried catalogue around the target.

    Uses every row fetched from the archive (in-radius, beyond max_rows and epoch pad)
    over the cone actually queried; a cone the archive truncated (rows are returned
    nearest-first) covers only the area inside its farthest returned row. See
    :func:`astrometry.estimate_density_deg2` for the Gamma-Poisson estimate.
    """
    densities: dict[str, float] = {}
    info: dict[str, dict[str, Any]] = {}
    for name, sources in successes:
        meta = getattr(sources, "meta", {}) or {}
        rows = list(sources) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or [])
        radius = float(meta.get("query_radius_arcsec") or radius_arcsec)
        centre = _pair(meta.get("cone_center")) or (target.ra, target.dec)
        truncated = bool(meta.get("archive_truncated"))
        if truncated and rows:
            radius = min(radius, max(haversine_arcsec(centre[0], centre[1], s.ra, s.dec) for s in rows))
        density, details = estimate_density_deg2(
            len(rows), cone_area_deg2(max(radius, 1e-3)), catalog=name,
            n_target_rows=_target_rows(rows, target, target_sigma_arcsec),
        )
        details["radius_arcsec"] = radius
        details["truncated"] = truncated
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
) -> tuple[AssociationResult, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Bayesian association of ``matches`` with the target and with each other.

    Returns (association, per-match propagation info, density provenance). Densities
    default to an estimate from the matches themselves over the cone of
    ``radius_arcsec`` (the largest match separation when not given).
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
                n_target_rows=_target_rows(rows, target, cfg.target_sigma_arcsec))
    else:
        densities = dict(densities)
        for m in matches:  # a catalogue without an estimate (e.g. rows injected by a caller)
            if m.catalog not in densities:
                densities[m.catalog], density_info[m.catalog] = estimate_density_deg2(
                    sum(1 for x in matches if x.catalog == m.catalog),
                    cone_area_deg2(radius_arcsec or max(x.separation_arcsec for x in matches) or 1.0), catalog=m.catalog)
    detections, infos = _match_detections(matches, target, target_pm_sigma_masyr=target_pm_sigma_masyr)
    result = associate(detections, densities, target=(target.ra, target.dec), config=cfg)
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
                                      "sigma_arcsec": info.get("sigma_arcsec"),
                                      "pm_growth_arcsec": info.get("pm_growth_arcsec"),
                                      "covariance_shape": info.get("covariance_shape")},
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


class CrossmatchService:
    """Orchestrates catalog querying, filtering, Bayesian association, and UnifiedRecord assembly.

    ``association_config`` sets the association parameters (target uncertainty, prior
    completeness, ...; see :class:`astrometry.AssociationConfig`). ``max_concurrency``
    bounds the number of targets :meth:`crossmatch_many` runs at once.
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
    ) -> SearchContext:
        """Validate the inputs of a crossmatch and plan its catalogue queries."""
        target = validate_target(ra, dec, epoch=epoch, pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr,
                                 parallax_mas=parallax_mas)
        if query is not None:
            QueryValidator.validate(query, self.registry)
            pm_source = (query.metadata or {}).get("pm_source") or pm_source
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

        sigma = self.association_config.target_sigma_arcsec if target_uncertainty_arcsec is None \
            else _positive(target_uncertainty_arcsec, "target_uncertainty_arcsec")
        pm_sigma = None if target_pm_error_masyr is None else _positive(target_pm_error_masyr, "target_pm_error_masyr",
                                                                         allow_zero=True)

        plans = QueryBuilder(self.registry).build(query) if query else self.planner.plan(search_radius, profile=profile)
        if catalogs:
            enabled = self.registry.enabled_catalogs()
            unknown = [c for c in catalogs if c not in enabled]
            if unknown:
                raise ValueError(f"Unknown catalog(s): {', '.join(unknown)}; known: {', '.join(sorted(enabled))}")
            plans = [p for p in plans if p.catalog in set(catalogs)]
        return SearchContext(target, plans, search_radius, query, profile, pm_source, sigma, pm_sigma)

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
    ) -> UnifiedRecord:
        """Execute full crossmatch pipeline for given coordinates or AdvancedQuery.

        ``parallax_mas`` (the target's parallax; with a query, ``query.target.parallax_mas``)
        removes the annual parallax from single-epoch positions of the target (2MASS, SDSS,
        ...); when not given, a significant parallax is adopted with the proper motion.

        ``epoch`` is the Julian year of (ra, dec); with it, every catalog cone follows
        the target to that catalog's epoch (using ``pm_ra_masyr``/``pm_dec_masyr`` when
        given, else widened by the largest plausible proper motion) and rows are
        compared after propagation. Without it positions are compared as given.
        ``pm_source`` records where a given proper motion came from ("input" by
        default, "resolver" for a name-resolver motion; with a query it is read from
        ``query.metadata["pm_source"]``).

        ``catalogs`` restricts the search to these registry catalogues.
        ``target_uncertainty_arcsec`` is the 1-sigma per-axis uncertainty of the target
        position (default ``association_config.target_sigma_arcsec``, 0.1") and
        ``target_pm_error_masyr`` that of a given proper motion (default 1 mas/yr).

        Every match's ``confidence`` is the posterior probability that the row is the
        target's counterpart (NWAY-style, :mod:`astrometry`); ``crossmatch_groups`` are
        the most probable partition of the matches into physical objects with their
        association probabilities; ``provenance["association"]`` records the densities,
        priors and the target's ``p_any``.

        Catalog statistics: ``row_count``/``status`` count the rows inside the radius
        (the nearest ``max_rows``); with an AdvancedQuery, ``sources`` holds only the rows
        that pass its confidence/type filters and ``returned_count`` is their number.
        """
        ctx = self.prepare(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile, query=query,
                           pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, pm_source=pm_source,
                           parallax_mas=parallax_mas, catalogs=catalogs,
                           target_uncertainty_arcsec=target_uncertainty_arcsec,
                           target_pm_error_masyr=target_pm_error_masyr)
        successes, failures = await self.executor.execute(ctx.plans, ctx.target)
        return self.finalize(ctx, successes, failures)

    def finalize(
        self,
        ctx: SearchContext,
        successes: list[tuple[str, list[CatalogSource]]],
        failures: list[CatalogFailure],
    ) -> UnifiedRecord:
        """Assemble the UnifiedRecord from the catalogue results of ``ctx``'s plans."""
        target, plans, search_radius, query, profile, pm_source = (
            ctx.target, ctx.plans, ctx.search_radius, ctx.query, ctx.profile, ctx.pm_source)

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
            if target.parallax_mas is None:
                found = _adopt_parallax(target, successes, search_radius)
                if found is not None:
                    target, parallax_origin = found
        elif target.epoch is not None:
            adopted = _adopt_proper_motion(target, successes, search_radius, warnings=adoption_warnings)
            if adopted is not None:
                target, pm_origin = adopted
                if pm_origin.get("parallax") is not None:
                    parallax_origin = pm_origin["parallax"]
                if pm_sigma is None:
                    pm_sigma = _adopted_pm_sigma(pm_origin, successes)
        if pm_sigma is None:
            pm_sigma = DEFAULT_TARGET_PM_SIGMA_MASYR

        # Final in-cone / pad split with the final target model over EVERY fetched row
        # (in-radius, beyond max_rows, and pad): rows fetched only because of the epoch
        # pad are never counted as results, and none is lost to a provider-level cut.
        classified: list[tuple[str, QueryResult]] = []
        for name, sources in successes:
            meta = dict(getattr(sources, "meta", {}) or {})
            combined = list(sources) + list(meta.get("excess_sources") or []) + list(meta.get("pad_sources") or [])
            max_rows = int(meta.get("max_rows") or max(len(sources), 1))
            split = classify_sources(
                combined, target, search_radius, catalog_name=name, max_rows=max_rows,
                row_limit=int(meta.get("row_limit") or max_rows),
                archive_truncated=bool(meta.get("archive_truncated", meta.get("truncated", False))),
                cone_center=_pair(meta.get("cone_center")), epoch_span=_pair(meta.get("epoch_span")),
            )
            meta["pad_sources"] = split.pad
            meta["excess_sources"] = split.excess
            meta["truncated"] = split.truncated
            if "base_warnings" in meta:
                meta["warnings"] = list(meta["base_warnings"]) + split.warnings
            classified.append((name, QueryResult(split.inside, meta)))
        successes = classified

        all_sources = [source for _, sources in successes for source in sources]
        all_matches = match_target(target, all_sources, search_radius)

        # Bayesian association over every in-radius row: confidence = posterior that the
        # row is the target's counterpart.
        config = replace(self.association_config, target_sigma_arcsec=ctx.target_sigma_arcsec)
        densities, density_info = catalog_densities(successes, target, search_radius, config.target_sigma_arcsec)
        association, det_infos, extra_density = associate_matches(
            all_matches, target, densities=densities, config=config, radius_arcsec=search_radius,
            target_pm_sigma_masyr=pm_sigma)
        density_info.update(extra_density)
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
                if match.confidence < query.min_confidence or not query.apply_filters(_source_dict(match)):
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
        all_warnings: list[str] = list(adoption_warnings)
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
                "target_pm_error_masyr": pm_sigma if target.proper_motion is not None else None,
                "p_any": _probability(association.p_any),
                "best_match_probability": _probability(target_group.match_probability) if target_group else None,
                "states": association.n_states,
                "exact": association.exact,
                "links": association.n_links,
                "densities": density_info,
                "notes": list(association.notes),
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
    ) -> AsyncIterator[dict[str, Any]]:
        """Crossmatch as an async stream of events, one per catalogue as it completes.

        Same parameters as :meth:`crossmatch`, plus ``name`` (resolved with CDS Sesame --
        ``resolver`` or a :class:`providers.SesameResolver` on the providers' HTTP client --
        which supplies the position, epoch, proper motion, parallax and their errors).
        Events are dicts ``{"event": kind, "data": {...}}``, in this order:

        * ``start``: target, planned catalogues, resolved object (when ``name``);
        * ``catalog`` (one per catalogue, in completion order): ``catalog``, ``status``
          (success / empty / failed), ``count`` (rows inside the radius at the provider's
          first-pass epoch split), ``elapsed_ms``, ``sources`` (serialised rows) and, on
          failure, ``error_type``/``message``;
        * ``group`` (one per crossmatch group of the final record, target group first);
        * ``done``: ``{"record": UnifiedRecord.as_dict()}``.

        Closing the generator (e.g. a disconnected client) cancels the catalogue queries
        still running.
        """
        resolved: dict[str, Any] | None = None
        if name is not None:
            from models import resolved_target
            from providers import SesameResolver

            if resolver is None:
                client = next((getattr(p, "client", None) for p in self.providers.values()
                               if getattr(p, "client", None) is not None), None)
                resolver = SesameResolver(client)
            obj = await resolver.resolve(name)
            resolved = obj.as_dict()
            rt = resolved_target(obj)
            ra, dec = rt.ra, rt.dec
            if rt.epoch is not None and epoch is None:
                epoch = rt.epoch
                if rt.proper_motion is not None and pm_ra_masyr is None and pm_dec_masyr is None:
                    pm_ra_masyr, pm_dec_masyr, pm_source = rt.pm_ra_masyr, rt.pm_dec_masyr, "resolver"
                    if target_pm_error_masyr is None:
                        target_pm_error_masyr = _resolver_pm_error(obj)
                if parallax_mas is None:
                    parallax_mas = rt.parallax_mas
            if target_uncertainty_arcsec is None:
                target_uncertainty_arcsec = _resolver_position_error(obj)
        if ra is None or dec is None:
            raise ValueError("Provide ra and dec, or an object name.")
        ctx = self.prepare(ra, dec, radius_arcsec=radius_arcsec, epoch=epoch, profile=profile, query=query,
                           pm_ra_masyr=pm_ra_masyr, pm_dec_masyr=pm_dec_masyr, pm_source=pm_source,
                           parallax_mas=parallax_mas, catalogs=catalogs,
                           target_uncertainty_arcsec=target_uncertainty_arcsec,
                           target_pm_error_masyr=target_pm_error_masyr)
        yield {"event": "start", "data": {
            "target": ctx.target.as_dict(), "radius_arcsec": ctx.search_radius,
            "catalogs": [p.catalog for p in ctx.plans], "target_sigma_arcsec": ctx.target_sigma_arcsec,
            "resolved_object": resolved,
        }}

        started = monotonic()
        tasks: dict[asyncio.Task[Any], QueryPlan] = {
            asyncio.create_task(self.executor._run_plan(plan, ctx.target), name=f"crossmatch:{plan.catalog}"): plan
            for plan in ctx.plans
        }
        outcomes: dict[str, tuple[str, list[CatalogSource]] | CatalogFailure] = {}
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
                        yield {"event": "catalog", "data": {
                            "catalog": plan.catalog, "wavelength": plan.wavelength, "status": "failed", "count": 0,
                            "elapsed_ms": failure.elapsed_ms, "sources": [], "error_type": failure.error_type,
                            "message": failure.message, "since_start_ms": round((monotonic() - started) * 1000.0, 1),
                        }}
                        continue
                    name_, sources = task.result()
                    outcomes[plan.catalog] = (name_, sources)
                    meta = getattr(sources, "meta", {}) or {}
                    yield {"event": "catalog", "data": {
                        "catalog": plan.catalog, "wavelength": plan.wavelength,
                        "status": "success" if sources else "empty", "count": len(sources),
                        "elapsed_ms": meta.get("elapsed_ms"), "truncated": bool(meta.get("truncated")),
                        "since_start_ms": round((monotonic() - started) * 1000.0, 1),
                        "sources": [_stream_source_dict(s) for s in sources],
                    }}
        finally:
            leftover = [t for t in tasks if not t.done()]
            for task in leftover:
                task.cancel()
            if leftover:
                await asyncio.gather(*leftover, return_exceptions=True)

        successes = [o for p in ctx.plans if isinstance(o := outcomes.get(p.catalog), tuple)]
        failures = [o for p in ctx.plans if isinstance(o := outcomes.get(p.catalog), CatalogFailure)]
        record = self.finalize(ctx, successes, failures)  # type: ignore[arg-type]
        if resolved is not None:
            record.resolved_object = resolved
            record.provenance["resolver"] = resolved.get("resolver")
        for group in record.crossmatch_groups:
            yield {"event": "group", "data": group}
        yield {"event": "done", "data": {"record": record.as_dict(),
                                          "elapsed_ms": round((monotonic() - started) * 1000.0, 1)}}

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
        default ``self.max_concurrency``); results are returned in the input order."""
        limit = self.max_concurrency if max_concurrency is None else int(max_concurrency)
        if limit < 1:
            raise ValueError("max_concurrency must be at least 1")
        semaphore = asyncio.Semaphore(limit)

        async def one(t: dict[str, Any]) -> UnifiedRecord:
            async with semaphore:
                return await self.crossmatch(
                    t["ra"],
                    t["dec"],
                    radius_arcsec=t.get("radius_arcsec", radius_arcsec),
                    epoch=t.get("epoch", epoch),
                    profile=t.get("profile", profile),
                    pm_ra_masyr=t.get("pm_ra_masyr"),
                    pm_dec_masyr=t.get("pm_dec_masyr"),
                    parallax_mas=t.get("parallax_mas"),
                    catalogs=t.get("catalogs"),
                    target_uncertainty_arcsec=t.get("target_uncertainty_arcsec"),
                )

        return list(await asyncio.gather(*(one(t) for t in targets)))


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{label} must be a finite number {'>= 0' if allow_zero else '> 0'}")
    return number


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
    whose own motion agrees with the target's (the same star in Gaia or SIMBAD)."""
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
    """The row's parallax (mas) when positive and at least PARALLAX_ADOPTION_MIN_SNR sigma."""
    plx = _to_float(src.data.get("parallax") if src.data else None)
    if plx is None:
        plx = _to_float((src.metadata.get("physical") or {}).get("parallax"))
    err = _to_float(src.data.get("parallax_error") if src.data else None)
    if plx is None or plx <= 0 or plx >= 1000.0:
        return None
    if err is not None and err > 0 and plx / err < PARALLAX_ADOPTION_MIN_SNR:
        return None
    return plx


def _adopt_proper_motion(
    target: Target,
    successes: list[tuple[str, list[CatalogSource]]],
    radius_arcsec: float,
    *,
    warnings: list[str] | None = None,
) -> tuple[Target, dict[str, Any]] | None:
    """Adopt the proper motion of the nearest catalog row that has one and matches the target.

    Only rows within min(radius, PM_ADOPTION_MAX_ARCSEC) of the target after their own
    propagation qualify (in-radius rows beyond max_rows included). Returns the target
    with that motion (and the row's parallax when significant, or that of an agreeing
    candidate) and a provenance note.

    Extragalactic targets are stationary: when the nearest identified row (one with an
    object type or redshift, e.g. SIMBAD or NED) is extragalactic -- M87 is 'AGN' in
    SIMBAD, which lists a Gaia proper motion of its nucleus -- pm = (0, 0) is adopted
    instead ("source": "extragalactic"). Rows that are themselves extragalactic, or
    whose small proper motion comes with an insignificant parallax, are never adopted.

    Crowded fields: when a candidate whose motion DISAGREES with the nearest one lies
    within 2 x (nearest separation) + PM_AMBIGUITY_MARGIN_ARCSEC, the nearest row may
    be a chance alignment (around Sgr A* SIMBAD has ~200 objects within 2"), so nothing
    is adopted and a warning is appended to ``warnings``. Candidates that agree (the same
    star in Gaia, SIMBAD and the Exoplanet Archive) never block adoption.
    """
    limit = min(radius_arcsec, PM_ADOPTION_MAX_ARCSEC)
    candidates: list[tuple[float, str, CatalogSource]] = []
    identity: tuple[float, str, CatalogSource] | None = None
    for name, sources in successes:
        meta = getattr(sources, "meta", {}) or {}
        for src in list(sources) + list(meta.get("excess_sources") or []):
            sep, _ = epoch_separation_arcsec(target, src)
            if sep > limit:
                continue
            physical = src.metadata.get("physical") or {}
            if (physical.get("object_type") or physical.get("redshift") is not None) and (identity is None or sep < identity[0]):
                identity = (sep, name, src)
            if src.proper_motion_ra_masyr is None or src.proper_motion_dec_masyr is None or src.epoch is None:
                continue
            if _is_extragalactic_row(src) or _pm_is_noise(src):
                continue
            candidates.append((sep, name, src))
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
    rivals = [c for c in candidates[1:] if c[0] <= zone and not _motions_agree(src, c[2])]
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

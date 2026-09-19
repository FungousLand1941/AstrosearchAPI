"""Query building, spatial geometry, proper-motion epoch propagation, and crossmatching engine."""

from __future__ import annotations

import asyncio
import math
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from time import monotonic
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time

from models import (
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
    validate_target,
)
from providers import CatalogProvider

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

        target = validate_target(ra, dec, epoch=target_data.get("epoch", data.get("epoch")))
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

        if query.catalogs:
            active_registry = registry or CatalogRegistry()
            for name in query.catalogs:
                if name not in active_registry.enabled_catalogs():
                    raise ValueError(f"Unknown catalog: {name}")
        return True


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


def _source_coordinate(source: CatalogSource, epoch: float | None = None) -> SkyCoord:
    """Propagate catalog source coordinate to target epoch using proper motion."""
    coordinate = SkyCoord(ra=source.ra * u.deg, dec=source.dec * u.deg, frame="icrs")
    if (
        epoch is not None
        and source.epoch is not None
        and source.proper_motion_ra_masyr is not None
        and source.proper_motion_dec_masyr is not None
    ):
        try:
            moving = SkyCoord(
                ra=source.ra * u.deg,
                dec=source.dec * u.deg,
                pm_ra_cosdec=source.proper_motion_ra_masyr * u.mas / u.yr,
                pm_dec=source.proper_motion_dec_masyr * u.mas / u.yr,
                obstime=Time(source.epoch, format="jyear"),
                frame="icrs",
            )
            coordinate = moving.apply_space_motion(new_obstime=Time(epoch, format="jyear"))
        except (TypeError, ValueError, u.UnitConversionError):
            pass
    return coordinate


def angular_separation_arcsec(target: Target, source: CatalogSource | Target) -> float:
    """Compute spherical angular separation in arcseconds."""
    target_coord = SkyCoord(ra=target.ra * u.deg, dec=target.dec * u.deg, frame=target.frame)
    source_coord = (
        _source_coordinate(source, target.epoch)
        if isinstance(source, CatalogSource)
        else SkyCoord(ra=source.ra * u.deg, dec=source.dec * u.deg, frame="icrs")
    )
    return float(target_coord.separation(source_coord).to(u.arcsec).value)


def match_score(
    separation_arcsec: float,
    *,
    positional_error_arcsec: float | None = None,
    target_uncertainty_arcsec: float | None = None,
) -> float:
    """Calculate match confidence using Gaussian positional uncertainty quadrature."""
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


def match_target(target: Target, sources: list[CatalogSource], radius_arcsec: float) -> list[Match]:
    """Filter sources by radius and rank by separation."""
    matches = []
    for source in sources:
        sep = angular_separation_arcsec(target, source)
        if sep <= radius_arcsec:
            matches.append(
                Match(
                    source.catalog,
                    source,
                    sep,
                    match_score(sep, positional_error_arcsec=source.positional_error_arcsec),
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
        "proper_motion_ra_masyr": source.proper_motion_ra_masyr,
        "proper_motion_dec_masyr": source.proper_motion_dec_masyr,
        "physical": source.metadata.get("physical", {}),
        "links": {name: url for name, url in source.metadata.get("links", {}).items() if url},
    }


def _group_matches(matches: list[Match], target: Target, radius_arcsec: float) -> list[dict[str, Any]]:
    """Cluster multi-catalog detections into coherent physical objects using Disjoint-Set Union."""
    if not matches:
        return []
    parent = list(range(len(matches)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(len(matches)):
        for j in range(i + 1, len(matches)):
            coord_i = _source_coordinate(matches[i].source, target.epoch)
            coord_j = _source_coordinate(matches[j].source, target.epoch)
            sep = float(coord_i.separation(coord_j).to(u.arcsec).value)
            allowed = max(
                radius_arcsec,
                matches[i].source.positional_error_arcsec or 0.0,
                matches[j].source.positional_error_arcsec or 0.0,
            )
            if sep <= allowed:
                union(i, j)

    grouped: dict[int, list[Match]] = {}
    for idx, match in enumerate(matches):
        grouped.setdefault(find(idx), []).append(match)

    result = []
    for num, group in enumerate(grouped.values(), start=1):
        result.append({
            "group_id": f"object-{num}",
            "catalogs": sorted({m.catalog for m in group}),
            "wavelengths": sorted({str(m.source.metadata.get("wavelength", "unknown")) for m in group}),
            "members": [_source_dict(m) for m in group],
        })
    return result


# ---------------------------------------------------------------------------
# Query Planner & Concurrent Executor
# ---------------------------------------------------------------------------


class QueryPlanner:
    """Creates default catalog query plans based on profile selections."""

    def __init__(self, registry: CatalogRegistry) -> None:
        self.registry = registry

    def plan(self, radius_arcsec: float, profile: str | None = None) -> list[QueryPlan]:
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
    """Executes catalog queries concurrently with timeouts and error isolation."""

    def __init__(self, providers: dict[str, CatalogProvider], *, timeout: float = 30.0) -> None:
        self.providers = providers
        self.timeout = timeout

    async def execute(
        self, plans: list[QueryPlan], target: Target
    ) -> tuple[list[tuple[str, list[CatalogSource]]], list[CatalogFailure]]:
        async def run(plan: QueryPlan) -> tuple[str, list[CatalogSource]]:
            provider = self.providers.get(plan.provider)
            if provider is None:
                raise CatalogUnavailableError(f"No provider configured for {plan.provider}")
            catalog = CatalogDefinition(
                name=plan.catalog,
                provider=plan.provider,
                wavelength=plan.wavelength,
                endpoint=plan.endpoint,
                table=plan.parameters.get("table"),
                catalog=plan.parameters.get("catalog"),
                parameters=dict(plan.parameters),
            )
            try:
                sources = await asyncio.wait_for(provider.query(catalog, target, plan.radius_arcsec), timeout=self.timeout)
                return plan.catalog, sources
            except TimeoutError as exc:
                raise QueryTimeoutError(f"Catalog {plan.catalog} timed out after {self.timeout:g}s") from exc

        gathered = await asyncio.gather(*(run(p) for p in plans), return_exceptions=True)
        successes: list[tuple[str, list[CatalogSource]]] = []
        failures: list[CatalogFailure] = []

        for plan, item in zip(plans, gathered):
            if isinstance(item, BaseException):
                failures.append(CatalogFailure(plan.catalog, error_type=item.__class__.__name__, message=str(item)))
            else:
                successes.append(item)
        return successes, failures


# ---------------------------------------------------------------------------
# Crossmatch Service
# ---------------------------------------------------------------------------


class CrossmatchService:
    """Orchestrates catalog querying, filtering, grouping, and UnifiedRecord assembly."""

    def __init__(
        self,
        registry: CatalogRegistry,
        providers: dict[str, CatalogProvider],
        *,
        radius_arcsec: float = 3.0,
        timeout: float = 30.0,
    ) -> None:
        self.registry = registry
        self.providers = providers
        self.radius_arcsec = radius_arcsec
        self.planner = QueryPlanner(registry)
        self.executor = QueryExecutor(providers, timeout=timeout)

    async def crossmatch(
        self,
        ra: float | str,
        dec: float | str,
        *,
        radius_arcsec: float | None = None,
        epoch: float | None = None,
        profile: str | None = None,
        query: AdvancedQuery | None = None,
    ) -> UnifiedRecord:
        """Execute full crossmatch pipeline for given coordinates or AdvancedQuery."""
        target = validate_target(ra, dec, epoch=epoch)
        if query is not None:
            QueryValidator.validate(query, self.registry)
            target = validate_target(
                query.target.ra,
                query.target.dec,
                epoch=query.target.epoch if query.proper_motion else None,
            )

        try:
            search_radius = (
                query.radius_arcsec if query else self.radius_arcsec if radius_arcsec is None else float(radius_arcsec)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("radius_arcsec must be a finite number greater than zero.") from exc
        if not math.isfinite(search_radius) or search_radius <= 0:
            raise ValueError("radius_arcsec must be a finite number greater than zero.")

        plans = QueryBuilder(self.registry).build(query) if query else self.planner.plan(search_radius, profile=profile)
        successes, failures = await self.executor.execute(plans, target)

        all_sources = [source for _, sources in successes for source in sources]
        matches = match_target(target, all_sources, search_radius)

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
        catalog_results = {
            name: {
                "sources": [s for s in sources if (s.catalog, s.source_id) in allowed],
                "status": "success",
            }
            for name, sources in successes
        } if query else {name: {"sources": sources, "status": "success"} for name, sources in successes}

        counterparts: dict[str, list[dict[str, Any]]] = {}
        for match in matches:
            wave = str(match.source.metadata.get("wavelength", "unknown"))
            counterparts.setdefault(wave, []).append(_source_dict(match))

        failures_list = [f.as_dict() for f in failures]
        groups = _group_matches(matches, target, search_radius)

        provenance = {
            "query_radius_arcsec": search_radius,
            "effective_radius_arcsec": effective_radius,
            "target_epoch": target.epoch,
            "profile": profile,
            "advanced_query": query.to_dict() if query else None,
            "catalogs_planned": [p.catalog for p in plans],
            "matches": [
                {
                    "catalog": m.catalog,
                    "source_id": m.source.source_id,
                    "separation_arcsec": m.separation_arcsec,
                    "confidence": m.confidence,
                }
                for m in matches
            ],
        }

        return UnifiedRecord(
            target={"ra": target.ra, "dec": target.dec, "frame": target.frame},
            catalogs_queried=len(catalog_results) + len(failures_list),
            catalog_results=catalog_results,
            counterparts=counterparts,
            failures=failures_list,
            provenance=provenance,
            crossmatch_groups=groups,
        )

    async def crossmatch_many(
        self,
        targets: list[dict[str, Any]],
        *,
        radius_arcsec: float | None = None,
        epoch: float | None = None,
        profile: str | None = None,
    ) -> list[UnifiedRecord]:
        """Execute crossmatch pipeline sequentially or concurrently for multiple targets."""
        return [
            await self.crossmatch(
                t["ra"],
                t["dec"],
                radius_arcsec=t.get("radius_arcsec", radius_arcsec),
                epoch=t.get("epoch", epoch),
                profile=t.get("profile", profile),
            )
            for t in targets
        ]

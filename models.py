"""AstroSearch data models, coordinate astrometry, parsers, and embedded catalog registry."""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii as astropy_ascii
from astropy.io.votable import parse as parse_votable

# ---------------------------------------------------------------------------
# Error Hierarchy
# ---------------------------------------------------------------------------


class AstroSearchError(Exception):
    """Base exception for all AstroSearch errors."""


class InvalidCoordinateError(AstroSearchError):
    """Raised when RA, DEC, or epoch coordinates fail validation."""


class CatalogUnavailableError(AstroSearchError):
    """Raised when an astronomy catalog or provider service is unavailable."""


class QueryTimeoutError(AstroSearchError):
    """Raised when a catalog query exceeds its allotted timeout."""


class CatalogQueryError(AstroSearchError):
    """Raised when an upstream catalog HTTP request returns an error."""


class ResponseParseError(AstroSearchError):
    """Raised when a provider response cannot be parsed."""


class ObjectResolutionError(AstroSearchError):
    """Raised when an astronomical object name cannot be resolved to coordinates."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Target:
    """Canonical representation of a search target position."""

    ra: float
    dec: float
    frame: str = "icrs"
    epoch: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ResolvedObject:
    """Canonical identity and position returned by an object-name resolver."""

    query: str
    canonical_name: str | None
    ra_deg: float
    dec_deg: float
    aliases: list[str]
    object_type: str | None
    redshift: float | None
    pm_ra_masyr: float | None
    pm_dec_masyr: float | None
    epoch: float | None
    resolver: str
    resolver_metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogDefinition:
    """Specification of an astronomical catalog and its access parameters."""

    name: str
    provider: str
    wavelength: str
    enabled: bool = True
    endpoint: str | None = None
    table: str | None = None
    catalog: str | None = None
    description: str | None = None
    query_method: str | None = None
    units: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    profiles: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogSource:
    """A detected astronomical source returned by a catalog."""

    catalog: str
    source_id: str
    ra: float
    dec: float
    positional_error_arcsec: float | None
    data: dict[str, Any]
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    epoch: float | None = None
    proper_motion_ra_masyr: float | None = None
    proper_motion_dec_masyr: float | None = None
    position_uncertainty_arcsec: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QueryPlan:
    """Planned query for a single catalog."""

    catalog: str
    provider: str
    endpoint: str | None
    parameters: dict[str, Any]
    radius_arcsec: float
    wavelength: str = "unknown"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Match:
    """Crossmatch association between a target and an archive detection."""

    catalog: str
    source: CatalogSource
    separation_arcsec: float
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "catalog": self.catalog,
            "source": self.source.as_dict(),
            "separation_arcsec": self.separation_arcsec,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class CatalogFailure:
    """Error record for a catalog that failed during execution."""

    catalog: str
    status: str = "failed"
    error_type: str | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UnifiedRecord:
    """Provenance-rich unified result aggregating all catalog crossmatches."""

    target: dict[str, Any]
    catalogs_queried: int
    catalog_results: dict[str, Any]
    counterparts: dict[str, list[dict[str, Any]]]
    failures: list[dict[str, Any]]
    provenance: dict[str, Any]
    crossmatch_groups: list[dict[str, Any]] = field(default_factory=list)
    resolved_object: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Runtime Settings
# ---------------------------------------------------------------------------


class Settings:
    """Runtime configuration read from environment variables or overrides."""

    def __init__(self, **overrides: Any) -> None:
        def val(name: str, default: Any) -> Any:
            return overrides.get(name, os.getenv(name, default))

        self.app_name = str(val("APP_NAME", "astro-crossmatch"))
        self.debug = str(val("DEBUG", "false")).lower() == "true"
        self.default_radius_arcsec = float(val("DEFAULT_RADIUS_ARCSEC", 3.0))
        self.request_timeout_seconds = float(val("REQUEST_TIMEOUT_SECONDS", 30.0))
        self.max_response_bytes = int(val("MAX_RESPONSE_BYTES", 10_000_000))
        self.resolver_endpoint = str(val("SESAME_ENDPOINT", "https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNV"))
        self.catalog_registry_path = val("CATALOG_REGISTRY_PATH", None)
        self.log_level = str(val("LOG_LEVEL", "INFO"))

        if not math.isfinite(self.default_radius_arcsec) or self.default_radius_arcsec <= 0:
            raise ValueError("DEFAULT_RADIUS_ARCSEC must be finite and greater than zero.")
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("REQUEST_TIMEOUT_SECONDS must be finite and greater than zero.")
        if self.max_response_bytes <= 0:
            raise ValueError("MAX_RESPONSE_BYTES must be greater than zero.")


# ---------------------------------------------------------------------------
# Coordinate Validation
# ---------------------------------------------------------------------------


def _as_float(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidCoordinateError(f"{name} must be numeric.") from exc
    if not math.isfinite(numeric):
        raise InvalidCoordinateError(f"{name} must be finite.")
    return numeric


def validate_target(ra: float | str, dec: float | str, *, frame: str = "icrs", epoch: float | None = None) -> Target:
    """Validate coordinates and normalize right ascension to [0, 360)."""
    ra_val = _as_float(ra, "ra") % 360.0
    dec_val = _as_float(dec, "dec")
    if not -90.0 <= dec_val <= 90.0:
        raise InvalidCoordinateError("DEC must be within [-90, 90] degrees.")
    if epoch is not None:
        try:
            epoch = float(epoch)
        except (TypeError, ValueError) as exc:
            raise InvalidCoordinateError("epoch must be numeric.") from exc
        if not math.isfinite(epoch) or epoch < 1800 or epoch > 2200:
            raise InvalidCoordinateError("epoch must be a finite Julian year between 1800 and 2200.")
    try:
        coord = SkyCoord(ra=ra_val * u.deg, dec=dec_val * u.deg, frame=frame)
    except Exception as exc:
        raise InvalidCoordinateError(f"Unsupported coordinate frame: {frame}") from exc
    if not math.isfinite(coord.ra.deg) or not math.isfinite(coord.dec.deg):
        raise InvalidCoordinateError("Coordinate values are not finite.")
    return Target(ra=ra_val, dec=dec_val, frame=frame, epoch=epoch)


# ---------------------------------------------------------------------------
# Normalization & Parsers
# ---------------------------------------------------------------------------


def normalize_field_names(record: Mapping[str, Any]) -> dict[str, Any]:
    """Map heterogeneous astronomical catalog field names to canonical keys."""
    aliases = {
        "ra": "ra", "ra_icrs": "ra", "raj2000": "ra", "ramean": "ra",
        "dec": "dec", "dec_icrs": "dec", "dej2000": "dec", "decmean": "dec",
        "source_id": "source_id", "objid": "source_id", "designation": "source_id",
        "id": "source_id", "sourceid": "source_id", "main_id": "source_id",
        "objname": "source_id", "prefname": "source_id", "pl_name": "source_id", "oid": "source_id",
        "pmra": "pmra", "pm_ra": "pmra", "pmra_cosdec": "pmra",
        "pmdec": "pmdec", "pm_dec": "pmdec",
        "ref_epoch": "epoch", "epoch": "epoch", "obsepoch": "epoch",
        "poserr": "position_uncertainty_arcsec", "pos_error": "position_uncertainty_arcsec",
        "ra_error": "position_uncertainty_arcsec", "dec_error": "position_uncertainty_arcsec",
        "raerror": "position_uncertainty_arcsec", "decerror": "position_uncertainty_arcsec",
        "uncmaja": "position_uncertainty_arcsec", "err_pos": "position_uncertainty_arcsec",
        "parallax": "parallax", "plx_value": "parallax",
        "z": "redshift", "redshift": "redshift", "z_value": "redshift",
        "otype": "object_type", "objtype": "object_type", "prefphytype": "object_type",
        "morphology": "morphology", "sp_type": "spectral_type", "spectral_type": "spectral_type",
        "obsdate": "observation_date", "obs_date": "observation_date",
        "observation_date": "observation_date", "date_obs": "observation_date",
        "quality": "quality_flags", "quality_flag": "quality_flags", "flags": "quality_flags",
    }
    normalized: dict[str, Any] = {}
    for key, val in record.items():
        clean_key = str(key).strip().lower()
        normalized[aliases.get(clean_key, str(key).strip())] = val
    return normalized


def normalize_source_record(raw_record: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize provider record values without hallucinating missing fields."""
    values = normalize_field_names(raw_record)
    if "ra" in values:
        try:
            values["ra"] = float(values["ra"])
        except (TypeError, ValueError):
            pass
    if "dec" in values:
        try:
            values["dec"] = float(values["dec"])
        except (TypeError, ValueError):
            pass
    if "source_id" in values:
        values["source_id"] = str(values["source_id"])
    for key in ("pmra", "pmdec", "epoch", "position_uncertainty_arcsec", "parallax", "redshift"):
        if key in values and values[key] not in (None, ""):
            try:
                values[key] = float(values[key])
            except (TypeError, ValueError):
                values.pop(key, None)
    return values


def parse_json_records(payload: str | bytes | dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    """Parse JSON records returned by public astronomy APIs."""
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    if isinstance(data, dict):
        rows = data.get("data", data.get("results"))
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return [data]
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    return []


def parse_csv_records(payload: str | bytes) -> list[dict[str, Any]]:
    """Parse CSV text rows, skipping comment lines."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    csv_text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    return [dict(row) for row in csv.DictReader(io.StringIO(csv_text))]


def parse_ipac_records(payload: str | bytes) -> list[dict[str, Any]]:
    """Parse an IPAC ASCII table (such as IRSA Gator responses) via Astropy."""
    text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    table = astropy_ascii.read(text, format="ipac")
    return [{name: row[name] for name in table.colnames} for row in table]


def parse_votable_records(payload: bytes | io.BytesIO | str) -> list[dict[str, Any]]:
    """Parse an IVOA VOTable XML stream via Astropy."""
    raw = io.BytesIO(payload.encode("utf-8") if isinstance(payload, str) else payload)
    table = parse_votable(raw)
    first_table = table.get_first_table()
    return [{key: val for key, val in zip(first_table.columns.names, row)} for row in first_table.array]


def build_provenance(
    catalog: str,
    *,
    provider: str,
    source_id: str,
    endpoint: str | None,
    query_parameters: Mapping[str, object],
    search_radius_arcsec: float,
) -> dict[str, object]:
    """Assemble complete provenance metadata for an archive observation."""
    return {
        "catalog": catalog,
        "provider": provider,
        "source_id": source_id,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "endpoint": endpoint,
        "search_radius_arcsec": search_radius_arcsec,
        "query_parameters": dict(query_parameters),
    }


# ---------------------------------------------------------------------------
# Embedded 19-Catalog Registry
# ---------------------------------------------------------------------------

DEFAULT_CATALOGS: dict[str, dict[str, Any]] = {
    "gaia_dr3": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "optical",
        "endpoint": "https://gea.esac.esa.int/tap-server/tap/sync",
        "table": "gaiadr3.gaia_source",
        "description": "Gaia DR3 TAP positional and astrometric search",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {"columns": ["source_id", "ra", "dec", "pmra", "pmdec", "ref_epoch", "parallax"]},
        "profiles": ["full", "optical", "stellar"],
    },
    "simbad": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "multi",
        "endpoint": "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
        "table": "basic",
        "description": "SIMBAD astronomical object identities, types, and measurements",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": ["main_id", "ra", "dec", "otype", "sp_type", "plx_value", "pmra", "pmdec"],
            "id_field": "main_id",
            "positional_error_field": "err_pos",
        },
        "profiles": ["full", "optical", "stellar", "identity"],
    },
    "ned": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "extragalactic",
        "endpoint": "https://ned.ipac.caltech.edu/tap/sync",
        "table": "objdir",
        "description": "NASA/IPAC Extragalactic Database object directory",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": ["prefname", "ra", "dec", "uncmaja", "z", "zflag", "prefphytype"],
            "id_field": "prefname",
            "positional_error_field": "uncmaja",
        },
        "profiles": ["full", "optical", "extragalactic", "identity"],
    },
    "exoplanet_archive": {
        "enabled": True,
        "provider": "tap",
        "wavelength": "exoplanet",
        "endpoint": "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
        "table": "ps",
        "description": "NASA Exoplanet Archive planetary systems",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": ["pl_name", "hostname", "ra", "dec", "discoverymethod", "pl_orbper", "pl_rade"],
            "id_field": "pl_name",
        },
        "profiles": ["full", "exoplanet", "stellar"],
    },
    "vizier_2mass_reference": {
        "enabled": False,
        "provider": "tap",
        "wavelength": "infrared",
        "endpoint": "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
        "table": "II/246/out",
        "description": "VizieR TAP 2MASS reference catalog",
        "query_method": "ADQL",
        "units": "degrees",
        "parameters": {
            "columns": ["2MASS", "RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag"],
            "id_field": "2MASS",
            "ra_field": "RAJ2000",
            "dec_field": "DEJ2000",
        },
        "profiles": ["full", "infrared", "stellar"],
    },
    "twomass_psc": {
        "enabled": True,
        "provider": "irsa_gator",
        "wavelength": "infrared",
        "catalog": "fp_psc",
        "endpoint": "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query",
        "description": "2MASS Point Source Catalog",
        "query_method": "cone-search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "infrared", "stellar"],
    },
    "allwise": {
        "enabled": True,
        "provider": "irsa_gator",
        "wavelength": "infrared",
        "catalog": "allwise_p3as_psd",
        "endpoint": "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query",
        "description": "AllWISE mid-infrared source catalog",
        "query_method": "cone-search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "infrared", "stellar"],
    },
    "panstarrs_dr2": {
        "enabled": True,
        "provider": "mast",
        "wavelength": "optical",
        "endpoint": "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/mean",
        "description": "Pan-STARRS DR2 mean object catalog",
        "query_method": "positional API",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "optical", "extragalactic"],
    },
    "sdss": {
        "enabled": True,
        "provider": "sdss",
        "wavelength": "optical",
        "endpoint": "https://skyserver.sdss.org/dr18/SkyServerWS/ConeSearch/ConeSearchService",
        "description": "Sloan Digital Sky Survey cone search",
        "query_method": "cone-search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "optical", "extragalactic", "spectroscopy"],
    },
    "first": {
        "enabled": True,
        "provider": "heasarc_xamin",
        "wavelength": "radio",
        "table": "first",
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query",
        "description": "FIRST radio catalogue (1.4 GHz)",
        "query_method": "Xamin positional search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "radio", "extragalactic"],
    },
    "nvss": {
        "enabled": True,
        "provider": "heasarc_xamin",
        "wavelength": "radio",
        "table": "nvss",
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query",
        "description": "NVSS radio catalogue (1.4 GHz all-sky)",
        "query_method": "Xamin positional search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "radio", "extragalactic"],
    },
    "vlass": {
        "enabled": True,
        "provider": "heasarc_xamin",
        "wavelength": "radio",
        "table": "vlass",
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query",
        "description": "VLASS radio catalogue (3 GHz)",
        "query_method": "Xamin positional search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "radio", "extragalactic"],
    },
    "lotss": {
        "enabled": True,
        "provider": "heasarc_xamin",
        "wavelength": "radio",
        "table": "lotss",
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query",
        "description": "LoTSS radio catalogue (120-168 MHz)",
        "query_method": "Xamin positional search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "radio", "extragalactic"],
    },
    "rosat": {
        "enabled": True,
        "provider": "heasarc_xamin",
        "wavelength": "xray",
        "table": "rosmaster",
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query",
        "description": "ROSAT all-sky survey master catalog",
        "query_method": "Xamin positional search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "xray", "high-energy"],
    },
    "chandra": {
        "enabled": True,
        "provider": "heasarc_xamin",
        "wavelength": "xray",
        "table": "chandra",
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query",
        "description": "Chandra source catalogue",
        "query_method": "Xamin positional search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "xray", "high-energy"],
    },
    "xmm": {
        "enabled": True,
        "provider": "heasarc_xamin",
        "wavelength": "xray",
        "table": "xmm",
        "endpoint": "https://heasarc.gsfc.nasa.gov/xamin/query",
        "description": "XMM-Newton source catalogue",
        "query_method": "Xamin positional search",
        "units": "arcseconds",
        "parameters": {},
        "profiles": ["full", "xray", "high-energy"],
    },
}


class CatalogRegistry:
    """Catalog registry managing enabled astronomy catalogs with optional YAML override."""

    def __init__(self, registry_path: str | os.PathLike[str] | None = None) -> None:
        self.registry_path = Path(registry_path) if registry_path else None
        self._catalogs: dict[str, CatalogDefinition] = {}
        self.reload()

    def reload(self) -> None:
        """Load catalog definitions from YAML file if existing, or default embedded registry."""
        catalogs_data: dict[str, Any] = {}
        if self.registry_path and self.registry_path.exists():
            try:
                content = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
                catalogs_data = content.get("catalogs", {})
            except Exception:
                catalogs_data = DEFAULT_CATALOGS
        else:
            catalogs_data = DEFAULT_CATALOGS

        self._catalogs = {}
        for name, entry in catalogs_data.items():
            if not isinstance(entry, dict):
                continue
            self._catalogs[name] = CatalogDefinition(
                name=name,
                provider=str(entry.get("provider", "unknown")),
                wavelength=str(entry.get("wavelength", "unknown")),
                enabled=bool(entry.get("enabled", True)),
                endpoint=entry.get("endpoint"),
                table=entry.get("table"),
                catalog=entry.get("catalog"),
                description=entry.get("description"),
                query_method=entry.get("query_method"),
                units=entry.get("units"),
                parameters=dict(entry.get("parameters", {})),
                profiles=tuple(str(item) for item in entry.get("profiles", ())),
            )

    @property
    def catalogs(self) -> dict[str, CatalogDefinition]:
        return dict(self._catalogs)

    def enabled_catalogs(self) -> dict[str, CatalogDefinition]:
        return {name: cat for name, cat in self._catalogs.items() if cat.enabled}

    def get(self, name: str) -> CatalogDefinition:
        if name not in self._catalogs:
            raise KeyError(f"Catalog '{name}' not found in registry.")
        return self._catalogs[name]

    def by_profile(self, profile: str) -> dict[str, CatalogDefinition]:
        return {
            name: cat for name, cat in self.enabled_catalogs().items()
            if not cat.profiles or profile in cat.profiles
        }

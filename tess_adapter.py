"""TESS light-curve FITS adapter for the canonical signal contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from representations import RepresentationError

ALLOWED_FLUX = {"SAP_FLUX", "PDCSAP_FLUX"}


def _json_numbers(values: np.ndarray) -> list[float | None]:
    """Preserve missing FITS samples as JSON null rather than non-standard NaN."""
    return [float(value) if np.isfinite(value) else None for value in values]


def read_tess_light_curve(path: Path, *, flux_kind: str = "PDCSAP_FLUX", source_url: str | None = None) -> dict[str, Any]:
    """Read a SPOC light-curve product without erasing flags or provenance."""
    flux_kind = flux_kind.upper()
    if flux_kind not in ALLOWED_FLUX:
        raise RepresentationError(f"flux_kind must be one of {sorted(ALLOWED_FLUX)}")
    with fits.open(path, memmap=False) as hdus:
        if len(hdus) < 2 or hdus[1].data is None:
            raise RepresentationError("TESS product has no light-curve table")
        primary, table_header, data = hdus[0].header, hdus[1].header, hdus[1].data
        names = {name.upper() for name in data.names}
        error_name = flux_kind + "_ERR"
        required = {"TIME", flux_kind, error_name, "QUALITY"}
        if not required <= names:
            raise RepresentationError(f"TESS product lacks required columns: {sorted(required - names)}")
        tic_id = primary.get("TICID") or table_header.get("TICID") or primary.get("OBJECT") or path.stem
        sector = primary.get("SECTOR") or table_header.get("SECTOR")
        ra = primary.get("RA_OBJ", table_header.get("RA_OBJ"))
        dec = primary.get("DEC_OBJ", table_header.get("DEC_OBJ"))
        if ra is None or dec is None:
            raise RepresentationError("TESS product lacks RA_OBJ/DEC_OBJ")
        times = np.asarray(data["TIME"], dtype=float)
        values = np.asarray(data[flux_kind], dtype=float)
        errors = np.asarray(data[error_name], dtype=float)
        quality = np.asarray(data["QUALITY"], dtype=np.int64)
        observation_id = f"tess-sector-{sector or 'unknown'}-tic-{tic_id}-{flux_kind.lower()}"
        return {
            "observation_id": observation_id,
            "object_id": f"TIC {tic_id}",
            "title": f"TIC {tic_id}, TESS sector {sector or 'unknown'} ({flux_kind})",
            "modality": "light_curve",
            "ra_deg": float(ra),
            "dec_deg": float(dec),
            "axis": _json_numbers(times),
            "values": _json_numbers(values),
            "uncertainties": _json_numbers(errors),
            "quality": quality.tolist(),
            "instrument": "TESS",
            "band": "TESS",
            "observed_at": primary.get("DATE-OBS") or table_header.get("DATE-OBS"),
            "source_url": source_url or "",
            "provenance": {
                "archive": "MAST",
                "local_product": path.name,
                "mission": primary.get("TELESCOP", "TESS"),
                "pipeline": primary.get("ORIGIN", "SPOC"),
                "pipeline_version": primary.get("PROCVER"),
                "sector": sector,
                "camera": primary.get("CAMERA"),
                "ccd": primary.get("CCD"),
                "axis_unit": table_header.get("TUNIT1", "BTJD day"),
                "value_unit": table_header.get("TUNIT" + str(list(data.names).index(flux_kind) + 1), "electron / s"),
                "value_kind": flux_kind,
                "quality_policy": "QUALITY == 0",
            },
        }

"""Count sources per catalog within a radius for canary targets (live network).

Usage: python scripts/catalog_census.py [radius_arcsec] [--json out.json]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from crossmatch import CrossmatchService, angular_separation_arcsec
from models import CatalogRegistry, Target
from providers import provider_map

TARGETS = {
    "3C 273": (187.2779154, 2.0523883),
    "M87": (187.7059308, 12.3911233),
}


async def census(radius: float) -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        registry = CatalogRegistry()
        service = CrossmatchService(registry, provider_map(client, timeout=120.0), radius_arcsec=radius, timeout=150.0)
        for label, (ra, dec) in TARGETS.items():
            record = await service.crossmatch(ra, dec, radius_arcsec=radius)
            target = Target(ra, dec)
            col: dict[str, str] = {}
            for name, result in record.catalog_results.items():
                sources = result.get("sources", [])
                n = sum(1 for s in sources if angular_separation_arcsec(target, s) <= radius)
                status = result.get("status", "success")
                col[name] = f"{n} ({status})" if status != "success" else str(n)
            for failure in record.failures:
                col[failure["catalog"]] = f"FAILED: {failure.get('error_type')}"
            table[label] = col
    return table


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    radius = float(args[0]) if args else 10.0
    table = asyncio.run(census(radius))
    names = sorted({n for col in table.values() for n in col})
    print(f"{'catalog':<24}" + "".join(f"{label:>28}" for label in table))
    for name in names:
        print(f"{name:<24}" + "".join(f"{table[label].get(name, '-'):>28}" for label in table))
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        Path(out).write_text(json.dumps(table, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

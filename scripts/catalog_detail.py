"""Print the nearest source per catalog with converted errors/epochs (live network)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from crossmatch import CrossmatchService
from models import CatalogRegistry
from providers import provider_map


async def run(ra: float, dec: float, radius: float, only: list[str] | None) -> None:
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
        registry = CatalogRegistry()
        service = CrossmatchService(registry, provider_map(client, timeout=120.0), radius_arcsec=radius, timeout=150.0)
        plans = service.planner.plan(radius)
        if only:
            plans = [p for p in plans if p.catalog in only]
        successes, failures = await service.executor.execute(plans, service_target(ra, dec))
        for name, sources in sorted(successes):
            meta = getattr(sources, "meta", {})
            if not sources:
                print(f"{name:<18} EMPTY  {meta.get('elapsed_ms')} ms")
                continue
            s = sources[0]
            print(f"{name:<18} n={len(sources):<3} id={s.source_id!s:<32} sep={s.metadata['query_separation_arcsec']:.4f}\" "
                  f"err={s.positional_error_arcsec if s.positional_error_arcsec is None else round(s.positional_error_arcsec, 4)} "
                  f"epoch={s.epoch if s.epoch is None else round(s.epoch, 3)} phys={s.metadata['physical']} "
                  f"{meta.get('elapsed_ms')} ms fb={bool(meta.get('fallback'))}")
        for f in failures:
            print(f"{f.catalog:<18} FAILED {f.error_type}: {f.message[:300]}")


def service_target(ra: float, dec: float):
    from models import validate_target

    return validate_target(ra, dec)


if __name__ == "__main__":
    ra, dec, radius = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    asyncio.run(run(ra, dec, radius, sys.argv[4].split(",") if len(sys.argv) > 4 else None))

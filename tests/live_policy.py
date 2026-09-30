"""The skip policy of every live test module: skip only on network errors and HTTP 5xx.

An archive or the name resolver being unreachable (a transport error, a timeout, HTTP 5xx or 429)
is not a failure of this code, so the test is skipped with the reason. Anything else -- a parse
error, an HTTP 4xx, a crash reported as HTTP 500, a record with a non-network catalog failure --
fails the test: a regression must never pass as a skip.

* :func:`skip_on_network_failures` -- a record's (or a batch/dataset's) catalog failures;
* :func:`api_ok` -- an API answer: 502/503/504 skip only when the detail names a network error;
* :func:`skip_if_resolver_degraded` -- CDS Sesame answering a name SIMBAD knows with 'Nothing found'
  or with a VizieR-local fallback (seen live) is the resolver's outage, shared by every harness.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable, Mapping
from typing import Any

import pytest

# Error types the providers raise for an unreachable archive (network error, timeout, HTTP 5xx/429).
NETWORK_ERRORS = frozenset({"CatalogUnavailableError", "QueryTimeoutError", "RateLimitedError", "TimeoutError",
                            "ConnectError", "ResolverUnavailableError", "ReadTimeout", "ConnectTimeout", "ReadError",
                            "RemoteProtocolError", "PoolTimeout", "WriteTimeout"})
# Words of an error text that describe an unreachable service (never a parse or query error).
_NETWORK_TEXT = re.compile(
    r"CatalogUnavailableError|QueryTimeoutError|RateLimitedError|TimeoutError|ResolverUnavailableError|ConnectError|"
    r"ConnectTimeout|ReadTimeout|ReadError|RemoteProtocolError|PoolTimeout|WriteTimeout|timed out|time budget|"
    r"HTTP 5\d\d|HTTP 429|\b5\d\d\b [A-Z][a-z]+ (?:Error|Unavailable|Gateway)|Name resolver unavailable|"
    r"unreachable|network error|circuit is open|Connection (?:reset|refused|aborted)")


def network_text(text: Any) -> bool:
    """True when an error text describes an unreachable service."""
    return bool(_NETWORK_TEXT.search(str(text or "")))


def network_failure(failure: Mapping[str, Any] | str) -> bool:
    """True for a catalog failure (dict with ``error_type``/``message``, or its text) caused by the network."""
    if isinstance(failure, Mapping):
        if failure.get("error_type") in NETWORK_ERRORS:
            return True
        return failure.get("error_type") is None and network_text(failure.get("message") or failure.get("error"))
    return network_text(failure)


def _describe(failure: Mapping[str, Any] | str) -> str:
    if isinstance(failure, Mapping):
        return (f"{failure.get('catalog') or failure.get('survey') or '?'}: {failure.get('error_type')}: "
                f"{str(failure.get('message') or failure.get('error'))[:160]}")
    return str(failure)[:200]


def skip_on_network_failures(record: Any, needed: Iterable[str] | None = None) -> None:
    """Skip when a (needed) catalog failed for a network reason; fail on any other failure.

    ``record``: a UnifiedRecord, its dict, or a list of failures (dicts or texts). ``needed``: the
    catalogs the test depends on (default: every failure counts)."""
    if isinstance(record, list):
        failures = record
    else:
        failures = (record.get("failures") if isinstance(record, Mapping) else record.failures) or []
    wanted = set(needed) if needed is not None else None
    relevant = [f for f in failures
                if wanted is None or not isinstance(f, Mapping) or f.get("catalog") in wanted]
    down = [f for f in relevant if network_failure(f)]
    if down:
        pytest.skip("archive unreachable: " + "; ".join(_describe(f) for f in down))
    assert not relevant, relevant


def api_ok(response: Any, expected: int = 200) -> Any:
    """The JSON body of a successful API answer. A 502/503/504 whose detail names a network error or an
    unreachable service skips; any other status (a 500 crash, a 502 for a parse error) fails."""
    if response.status_code in {502, 503, 504} and network_text(response.text):
        pytest.skip(f"upstream unavailable ({response.status_code}): {response.text[:300]}")
    assert response.status_code == expected, response.text[:2000]
    return response.json() if response.headers.get("content-type", "").startswith("application/json") else None


def skip_on_network_error_event(data: Mapping[str, Any]) -> None:
    """An SSE ``error`` event: skip when it reports a network error, fail otherwise."""
    if data.get("error_type") in NETWORK_ERRORS or network_text(data.get("detail") or data.get("message")):
        pytest.skip(f"stream ended with an upstream error: {data}")
    pytest.fail(f"stream ended with an error: {data}")


async def _resolver_state(name: str) -> str | None:
    import httpx

    from models import ObjectResolutionError, ResolverUnavailableError
    from providers import SesameResolver

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        try:
            answer = await SesameResolver(client).resolve(name)
        except ResolverUnavailableError as exc:
            return f"Sesame unreachable: {exc}"
        except ObjectResolutionError as exc:
            if "No coordinates found" not in str(exc):
                return f"Sesame answered unusably: {exc}"
            # Does SIMBAD itself know the name? Then Sesame's 'Nothing found' is its outage.
            query = ("SELECT TOP 1 oidref FROM ident WHERE id = '" + " ".join(name.split()).replace("'", "''") + "'")
            try:
                response = await client.get("https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
                                            params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json",
                                                    "QUERY": query})
                known = response.status_code == 200 and bool(response.json().get("data"))
            except (httpx.HTTPError, ValueError) as tap_exc:
                return f"Sesame found nothing and SIMBAD TAP is unreachable ({tap_exc})"
            if known:
                return f"Sesame answered 'Nothing found' for {name!r}, which SIMBAD knows (a Sesame outage)"
            return None  # a genuinely unknown name: the test itself is wrong, let it fail
        kind = SesameResolver.answer_kind(answer.resolver_metadata)
        if kind not in ("simbad", "ned"):
            return (f"Sesame answered {name!r} from {answer.resolver_metadata.get('resolver_name')} (SIMBAD retry: "
                    f"{answer.resolver_metadata.get('simbad_retry')}), not SIMBAD: an undated position without motion")
    return None


def resolver_degraded(name: str) -> str | None:
    """Why CDS Sesame cannot resolve ``name`` properly right now, or None when it answers from SIMBAD/NED."""
    return asyncio.run(_resolver_state(name))


def skip_if_resolver_degraded(name: str) -> None:
    """Skip when Sesame is degraded for ``name`` (see :func:`resolver_degraded`)."""
    reason = resolver_degraded(name)
    if reason:
        pytest.skip(reason)


async def skip_if_resolver_degraded_async(name: str) -> None:
    """:func:`skip_if_resolver_degraded` for async tests."""
    reason = await _resolver_state(name)
    if reason:
        pytest.skip(reason)

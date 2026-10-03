"""Offline checks of the live-test skip policy (tests/live_policy.py): only an unreachable upstream
(network error, timeout, HTTP 5xx/429) may turn a live test into a skip. A parse error, a crash
reported as HTTP 500 or any other catalog failure must fail it, so a regression never passes as a skip.
"""

from __future__ import annotations

import httpx
import pytest
from live_policy import (
    api_ok,
    network_failure,
    skip_on_network_error_event,
    skip_on_network_failures,
    skip_on_network_messages,
)

import api
from models import CatalogQueryError, CatalogUnavailableError, QueryTimeoutError, ResponseParseError

Skipped = pytest.skip.Exception
Failed = pytest.fail.Exception

# The failure the review reproduced: an archive answering HTTP 200 with an HTML page.
PARSE_FAILURE = {"catalog": "gaia_dr3", "error_type": "CatalogQueryError",
                 "message": "TAP gaia_dr3 returned an HTML page instead of data"}
# The failure the integrator saw live and recorded per catalog.
LOOP_FAILURE = {"catalog": "simbad", "error_type": "RuntimeError", "message": "Event loop is closed"}
NETWORK_FAILURE = {"catalog": "twomass_psc", "error_type": "CatalogUnavailableError",
                   "message": "IRSA returned HTTP 503 Service Unavailable"}
TIMEOUT_FAILURE = {"catalog": "allwise", "error_type": "QueryTimeoutError", "message": "timed out after 90 s"}


def _response(status: int, detail: str) -> httpx.Response:
    return httpx.Response(status, json={"detail": detail})


@pytest.mark.parametrize("failure", [PARSE_FAILURE, LOOP_FAILURE])
def test_a_non_network_catalog_failure_fails_the_live_test(failure: dict) -> None:
    with pytest.raises(AssertionError, match=failure["error_type"]):
        skip_on_network_failures({"failures": [failure]})


def test_an_outage_elsewhere_does_not_hide_a_parse_error() -> None:
    with pytest.raises(AssertionError, match="CatalogQueryError"):
        skip_on_network_failures({"failures": [NETWORK_FAILURE, PARSE_FAILURE]})


@pytest.mark.parametrize("failure", [NETWORK_FAILURE, TIMEOUT_FAILURE])
def test_a_network_failure_skips(failure: dict) -> None:
    with pytest.raises(Skipped, match="archive unreachable"):
        skip_on_network_failures({"failures": [failure]})


def test_needed_catalogs_limit_which_failures_count() -> None:
    skip_on_network_failures({"failures": [PARSE_FAILURE]}, needed=["simbad"])  # not needed: passes
    with pytest.raises(AssertionError):
        skip_on_network_failures({"failures": [PARSE_FAILURE]}, needed=["gaia_dr3"])
    skip_on_network_failures({"failures": []})  # no failures: passes


def test_api_answers_of_a_crash_or_a_parse_error_fail_and_an_outage_skips() -> None:
    # The API's own mapping (api.search_error) of each exception, fed to the live helper.
    crash = api.search_error(TypeError("unexpected keyword argument"))
    assert crash[0] == 500
    with pytest.raises(AssertionError):
        api_ok(_response(crash[0], crash[1]))
    for exc in (CatalogQueryError("TAP gaia_dr3 returned an HTML page instead of data"),
                ResponseParseError("VOTable has no RESOURCE")):
        status, detail, _ = api.search_error(exc)
        assert status == 502
        with pytest.raises(AssertionError):
            api_ok(_response(status, detail))
    for exc in (CatalogUnavailableError("gaia_dr3: HTTP 503"), QueryTimeoutError("gaia_dr3 timed out"),
                httpx.ConnectError("connection refused")):
        status, detail, _ = api.search_error(exc)
        with pytest.raises(Skipped, match="upstream unavailable"):
            api_ok(_response(status, detail))
    with pytest.raises(Skipped):
        api_ok(_response(503, "Name resolver unavailable: Sesame request failed: ReadTimeout"))
    assert api_ok(httpx.Response(200, json={"ok": True})) == {"ok": True}


def test_sse_error_events() -> None:
    with pytest.raises(Failed):
        skip_on_network_error_event({"error_type": "TypeError", "detail": "Internal error during search"})
    with pytest.raises(Skipped):
        skip_on_network_error_event({"error_type": "QueryTimeoutError", "detail": "timed out"})


def test_text_failures_skip_only_when_every_one_is_a_network_error() -> None:
    skip_on_network_messages("batch", [])
    with pytest.raises(Skipped):
        skip_on_network_messages("batch", ["CatalogUnavailableError: HTTP 502 Bad Gateway", "ReadTimeout: timed out"])
    with pytest.raises(Failed, match="not a network error"):
        skip_on_network_messages("batch", ["ReadTimeout: timed out", "ResponseParseError: no TABLEDATA"])
    with pytest.raises(Failed):
        skip_on_network_messages("mirror tiles", ["1 tile(s) failed"])
    assert network_failure("HTTP 429 Too Many Requests") and not network_failure("HTTP 400 Bad Request")

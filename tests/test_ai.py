"""Offline tests for ai.py: a scripted fake Anthropic client plus replayed real Sesame/SIMBAD traffic.

The fake client records every Messages API request, so these tests check the request
shape (strict tools, adaptive thinking, fallbacks, prompt caching), the validation
retry loop, hallucinated-catalog rejection and citation mapping without network.
Sesame / SIMBAD TAP answers are real recordings (tests/fixtures/ai, see
record_ai_fixtures.py) replayed strictly: a changed request fails the test.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.util
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import httpx2
import pytest
import respx
from astropy.cosmology import Planck18
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fixture_io import load_exchanges, replay_side_effect, request_signature, target_for
from helpers import make_service, offline_client

import ai
from crossmatch import AdvancedQuery, QueryValidator
from models import (
    CatalogRegistry,
    InvalidCoordinateError,
    ResolvedObject,
    ResponseParseError,
    UnifiedRecord,
    haversine_arcsec,
)
from providers import SesameResolver

M87_SESAME = (187.70593077, 12.39112325)
C3C273_SESAME = (187.27791594, 2.05238823)
BOGUS_NAME = "Qzxv Nonexistent Object 999"
UNSUPPORTED_SCHEMA_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength",
                           "maxLength", "pattern", "minItems", "maxItems", "default"}

_spec = importlib.util.spec_from_file_location(
    "record_ai_fixtures", Path(__file__).parent / "fixtures" / "ai" / "record_ai_fixtures.py"
)
recorder = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(recorder)


# ---------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------


def tool_use(name: str, payload: dict[str, Any], block_id: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id or f"toolu_{name}_{id(payload)}", name=name, input=payload)


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def reply(*blocks: SimpleNamespace, stop_reason: str = "tool_use", model: str = "claude-opus-5",
          stop_details: Any = None) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, model=model, stop_details=stop_details)


class FakeMessages:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(copy.deepcopy(kwargs))
        if not self.script:
            raise AssertionError("fake Anthropic client called more times than scripted")
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step(kwargs) if callable(step) else step


class FakeAnthropic:
    def __init__(self, *script: Any) -> None:
        self.messages = FakeMessages(list(script))
        self.beta = SimpleNamespace(messages=self.messages)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


def submission(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "scope": "cone",
        "target": {"name": "M87", "coordinates": None},
        "radius_arcsec": 900.0,
        "catalogs": ["first", "nvss", "vlass", "lotss", "simbad", "ned"],
        "profiles": [],
        "object_types": [],
        "spectral_types": [],
        "search_mode": "cone",
        "min_radius_arcsec": 0.0,
        "min_distance_pc": None,
        "max_distance_pc": None,
        "start_year": None,
        "end_year": None,
        "max_results": None,
        "min_confidence": 0.5,
        "adql": None,
        "plan": ["Resolve M87 with Sesame.", "Query radio catalogs and SIMBAD/NED in a 15 arcmin cone.",
                 "Keep crossmatch groups that contain a SIMBAD/NED quasar and a radio source."],
        "explanation": "Radio emission is taken as a detection in FIRST, NVSS, VLASS or LoTSS; the quasar type is "
                       "checked on the SIMBAD/NED members of each crossmatch group.",
    }
    base.update(overrides)
    return base


SETTINGS = ai.AISettings()  # defaults, independent of the environment


def tool_results(call: dict[str, Any]) -> list[dict[str, Any]]:
    last = call["messages"][-1]
    assert last["role"] == "user"
    return [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_result"]


@pytest.fixture
def sesame_replay():
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/sesame")))
        yield router


async def compile_with(fake: FakeAnthropic, text: str = "quasars near M87 with radio emission", **kwargs: Any):
    async with offline_client() as http_client:
        return await ai.compile_query(
            text, anthropic_client=fake, registry=kwargs.pop("registry", CatalogRegistry()), http_client=http_client,
            settings=kwargs.pop("settings", SETTINGS), **kwargs,
        )


# ---------------------------------------------------------------------------
# Tool schema & request shape
# ---------------------------------------------------------------------------


def _walk_keys(schema: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key != "properties":
                keys.add(key)
            keys |= _walk_keys(value)
    elif isinstance(schema, list):
        for item in schema:
            keys |= _walk_keys(item)
    return keys


def test_query_tools_are_strict_and_bound_to_registry(registry: CatalogRegistry) -> None:
    tools = {t["name"]: t for t in ai.build_query_tools(registry)}
    assert set(tools) == {"resolve_object", "submit_query"}
    for tool in tools.values():
        assert tool["strict"] is True
        for node in ai._iter_schema_objects(tool["input_schema"]):
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
        assert not (_walk_keys(tool["input_schema"]) & UNSUPPORTED_SCHEMA_KEYS)
    props = tools["submit_query"]["input_schema"]["properties"]
    assert props["catalogs"]["items"]["enum"] == sorted(registry.enabled_catalogs())
    assert "lotss_dr2" not in props["catalogs"]["items"]["enum"]  # disabled in the registry
    assert props["profiles"]["items"]["enum"] == sorted({p for c in registry.enabled_catalogs().values() for p in c.profiles})
    adql_catalogs = props["adql"]["anyOf"][1]["properties"]["catalog"]["enum"]
    assert "gaia_dr3" in adql_catalogs and "simbad" in adql_catalogs
    assert "panstarrs_dr2" not in adql_catalogs and "sdss" not in adql_catalogs  # MAST / SkyServer are not TAP
    assert props["search_mode"]["enum"] == ["cone", "shell", "cylinder"]


async def test_request_shape_caching_thinking_and_fallbacks(sesame_replay) -> None:
    fake = FakeAnthropic(reply(tool_use("submit_query", submission())))
    compiled = await compile_with(fake)
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert call["tool_choice"] == {"type": "auto"}
    assert call["max_tokens"] == 16000
    assert "temperature" not in call
    system = call["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    for name in CatalogRegistry().enabled_catalogs():
        assert f'"name":"{name}"' in system[0]["text"]
    assert "Never supply coordinates for a named object" in system[0]["text"]
    assert "quasars near M87 with radio emission" in call["messages"][0]["content"]
    assert compiled.attempts == 1 and compiled.model == "claude-opus-5"


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTROSEARCH_AI_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("ASTROSEARCH_AI_EFFORT", "low")
    monkeypatch.setenv("ASTROSEARCH_AI_FALLBACKS", "off")
    settings = ai.AISettings.from_env()
    assert settings.model == "claude-haiku-4-5" and settings.effort == "low"
    assert not settings.uses_fallbacks and not settings.supports_adaptive_thinking
    monkeypatch.setenv("ASTROSEARCH_AI_EFFORT", "extreme")
    with pytest.raises(ai.AIConfigurationError):
        ai.AISettings.from_env()
    monkeypatch.setenv("ASTROSEARCH_AI_EFFORT", "high")
    monkeypatch.setenv("ASTROSEARCH_AI_MAX_TOKENS", "16k")
    with pytest.raises(ai.AIConfigurationError, match="must be numbers"):
        ai.AISettings.from_env()
    monkeypatch.delenv("ASTROSEARCH_AI_MAX_TOKENS")
    monkeypatch.setenv("ASTROSEARCH_AI_MODEL", "claude-opus-4-6")
    monkeypatch.setenv("ASTROSEARCH_AI_EFFORT", "xhigh")  # xhigh arrived with Opus 4.7
    with pytest.raises(ai.AIConfigurationError, match="does not accept effort"):
        ai.AISettings.from_env()
    monkeypatch.delenv("ASTROSEARCH_AI_MODEL")
    monkeypatch.delenv("ASTROSEARCH_AI_EFFORT")
    monkeypatch.delenv("ASTROSEARCH_AI_FALLBACKS")
    default = ai.AISettings.from_env()
    assert default.model == "claude-opus-5" and default.uses_fallbacks and default.supports_adaptive_thinking


@pytest.mark.parametrize(
    ("model", "adaptive", "fallbacks"),
    [
        ("claude-opus-5", True, True),
        ("claude-fable-5-1", True, True),
        ("claude-opus-5-5", True, False),  # fallback targets not published for Opus 5.5
        ("claude-sonnet-5", True, False),
        ("claude-opus-4-8", True, False),
        ("claude-opus-4-6", True, False),
        ("claude-sonnet-4-6", True, False),
        ("claude-opus-4-5", False, False),  # budget_tokens generation: no adaptive thinking
        ("claude-opus-4-5-20251101", False, False),
        ("claude-opus-4-1", False, False),
        ("claude-sonnet-4-5", False, False),  # effort errors on Sonnet 4.5
        ("claude-haiku-4-5", False, False),
        ("claude-3-7-sonnet-latest", False, False),
    ],
)
def test_adaptive_thinking_and_fallbacks_are_allow_listed(model: str, adaptive: bool, fallbacks: bool) -> None:
    settings = ai.AISettings(model=model)
    assert settings.supports_adaptive_thinking is adaptive
    assert settings.uses_fallbacks is fallbacks


@pytest.mark.parametrize("model", ["claude-haiku-4-5", "claude-sonnet-4-5", "claude-opus-4-5", "claude-opus-4-1"])
async def test_older_models_get_no_thinking_effort_or_fallbacks(model: str, sesame_replay) -> None:
    fake = FakeAnthropic(reply(tool_use("submit_query", submission())))
    await compile_with(fake, settings=ai.AISettings(model=model, fallbacks=True))
    call = fake.calls[0]
    assert "thinking" not in call and "betas" not in call and "fallbacks" not in call and "output_config" not in call


# ---------------------------------------------------------------------------
# Compilation: names, validation loop, hallucinations
# ---------------------------------------------------------------------------


async def test_named_target_is_resolved_by_sesame_not_by_the_model(sesame_replay) -> None:
    fake = FakeAnthropic(
        reply(tool_use("resolve_object", {"name": "M87"}, "toolu_1")),
        reply(tool_use("submit_query", submission(), "toolu_2")),
    )
    compiled = await compile_with(fake)
    # The resolve_object answer (real Sesame data) went back to Claude.
    (result,) = tool_results(fake.calls[1])
    assert result["tool_use_id"] == "toolu_1" and "is_error" not in result
    resolved = json.loads(result["content"])
    assert resolved["canonical_name"] == "M 87" and resolved["object_type"] == "AGN"
    q = compiled.advanced_query
    assert q is not None and compiled.scope == "cone"
    assert math.isclose(q["target"]["ra"], M87_SESAME[0], abs_tol=1e-9)
    assert math.isclose(q["target"]["dec"], M87_SESAME[1], abs_tol=1e-9)
    assert q["target"]["epoch"] == 2000.0  # SIMBAD positions are J2000
    assert (q["target"]["pm_ra_masyr"], q["target"]["pm_dec_masyr"]) == (0.0, 0.0)  # extragalactic: stationary
    assert q["use_resolved_name"] is True and q["resolved_name"] == "M 87"
    assert q["catalogs"] == ["first", "nvss", "vlass", "lotss", "simbad", "ned"]
    assert q["metadata"]["target_source"] == "sesame"
    rebuilt = AdvancedQuery.from_dict(q)
    assert QueryValidator.validate(rebuilt, CatalogRegistry())
    assert compiled.resolved_objects[0]["canonical_name"] == "M 87"
    assert len(compiled.plan) == 3 and compiled.adql is None


async def test_hallucinated_catalog_is_rejected_and_fed_back(sesame_replay) -> None:
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(catalogs=["first", "hubble_ultra_deep_field"]), "toolu_bad")),
        reply(tool_use("submit_query", submission(catalogs=["first"]), "toolu_good")),
    )
    compiled = await compile_with(fake)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] is True and result["tool_use_id"] == "toolu_bad"
    assert "Unknown catalog: hubble_ultra_deep_field" in result["content"]
    assert "call submit_query again" in result["content"]
    assert compiled.attempts == 2
    assert any("hubble_ultra_deep_field" in e for e in compiled.validation_history[0])
    assert compiled.advanced_query["catalogs"] == ["first"]


async def test_retry_budget_is_enforced(sesame_replay) -> None:
    bad = submission(catalogs=["made_up_survey"])
    fake = FakeAnthropic(*(reply(tool_use("submit_query", bad)) for _ in range(3)))
    with pytest.raises(ai.AIQueryCompilationError) as info:
        await compile_with(fake)
    assert len(fake.calls) == 3  # first attempt + 2 corrections
    assert any("made_up_survey" in e for e in info.value.errors)
    assert len(info.value.history) == 3

    once = FakeAnthropic(reply(tool_use("submit_query", bad)))
    with pytest.raises(ai.AIQueryCompilationError):
        await compile_with(once, max_retries=0)
    assert len(once.calls) == 1


async def test_query_validator_errors_are_fed_back(sesame_replay) -> None:
    fake = FakeAnthropic(
        # cylinder without distance bound (on parallax catalogs, so only QueryValidator objects)
        reply(tool_use("submit_query", submission(search_mode="cylinder", catalogs=["gaia_dr3", "simbad"]))),
        reply(tool_use("submit_query", submission(radius_arcsec=7200.0))),  # beyond the 30' cap
        # empty selection: a named catalog outside the profile is an input error, not a silent drop
        reply(tool_use("submit_query", submission(catalogs=["first"], profiles=["xray"]))),
    )
    with pytest.raises(ai.AIQueryCompilationError) as info:
        await compile_with(fake)
    history = info.value.history
    assert any("cylinder searches require a distance bound" in e for e in history[0])
    assert any("radius_arcsec must be in (0, 1800]" in e for e in history[1])
    assert any("AdvancedQuery validation failed" in e and "first are not in profile 'xray'" in e
               and "intersected with the profile" in e for e in history[2])


def coords(text: str, frame: str = "icrs", equinox: str | None = None) -> dict[str, Any]:
    return {"name": None, "coordinates": {"text": text, "frame": frame, "equinox": equinox}}


async def test_unresolvable_name_is_fed_back_then_typed_coordinates_accepted(sesame_replay) -> None:
    text = f"radio sources within 60 arcsec of {BOGUS_NAME} at 12h30m49.42s +12d23m28.0s"
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(target={"name": BOGUS_NAME, "coordinates": None}, radius_arcsec=60.0))),
        reply(tool_use("submit_query", submission(target=coords("12h30m49.42s +12d23m28.0s"), radius_arcsec=60.0))),
    )
    compiled = await compile_with(fake, text)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] is True and "could not be resolved by CDS Sesame" in result["content"]
    assert "only if the user typed coordinates" in result["content"]
    q = compiled.advanced_query
    assert q["metadata"]["target_source"] == "user_coordinates"
    assert q["metadata"]["user_coordinates"] == {"text": "12h30m49.42s +12d23m28.0s", "frame": "icrs", "equinox": None}
    # Parsed by astropy from the typed text (RA hours x 15), not taken from the model.
    assert haversine_arcsec(q["target"]["ra"], q["target"]["dec"], *M87_SESAME) < 0.2
    assert q["target"]["epoch"] is None


async def test_model_invented_coordinates_are_rejected(sesame_replay) -> None:
    """Coordinates the user never typed (e.g. M31's position for an M87 request) are refused."""
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(target=coords("10.6847 41.269")))),
        reply(tool_use("submit_query", submission())),
    )
    compiled = await compile_with(fake, "quasars near M87 with radio emission")
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] is True and "does not appear in the request" in result["content"]
    assert compiled.advanced_query["metadata"]["target_source"] == "sesame"


async def test_model_cannot_skip_the_hours_to_degrees_conversion(sesame_replay) -> None:
    """The degrees come from parsing the typed text, whatever the model believes they are."""
    text = "radio sources within 60 arcsec of 12h30m49.42s +12d23m28.0s"
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(target=coords("12h30m49.42s +12d23m28.0s"),
                                                                   radius_arcsec=60.0))))
    compiled = await compile_with(fake, text)
    target = compiled.advanced_query["target"]
    assert target["ra"] == pytest.approx(187.705917, abs=1e-5) and target["dec"] == pytest.approx(12.391111, abs=1e-5)


async def test_galactic_coordinates_are_transformed_to_icrs(sesame_replay) -> None:
    text = "X-ray sources within 120 arcsec of Galactic l=0.0, b=0.0"
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(target=coords("l=0.0, b=0.0", "galactic"),
                                                                   radius_arcsec=120.0, catalogs=["chandra"]))))
    compiled = await compile_with(fake, text)
    target = compiled.advanced_query["target"]
    # IAU Galactic centre (l, b) = (0, 0) is ICRS RA 266.405, Dec -28.936 (astropy).
    assert target["ra"] == pytest.approx(266.40499, abs=1e-4) and target["dec"] == pytest.approx(-28.93617, abs=1e-4)


@pytest.mark.parametrize(
    ("text", "frame", "equinox", "expected"),
    [
        ("12h30m49.42s +12d23m28.0s", "icrs", None, (187.705917, 12.391111)),
        ("12:30:49.42 +12:23:28.0", "fk5", "J2000", (187.705917, 12.391111)),
        ("12 30 49.42 +12 23 28.0", "icrs", None, (187.705917, 12.391111)),
        ("RA=187.7059, Dec=+12.3911", "icrs", None, (187.7059, 12.3911)),
        ("187d42m21.3s +12d23m28s", "icrs", None, (187.705917, 12.391111)),
        ("RA 12h30m49.4s, Dec +12°23′28″", "icrs", None, (187.705833, 12.391111)),
        # M87's B1950 position (FK4) precesses to its J2000 position.
        ("12h28m17.6s +12d40m02s", "fk4", "B1950", (187.70603, 12.39116)),
        ("l=0.0, b=0.0", "galactic", None, (266.40499, -28.93617)),
    ],
)
def test_parse_user_coordinates(text: str, frame: str, equinox: str | None, expected: tuple[float, float]) -> None:
    target = ai.parse_user_coordinates(text, frame, equinox)
    assert haversine_arcsec(target.ra, target.dec, *expected) < 0.1


@pytest.mark.parametrize("text", ["12h", "hello world", "12h30m +95d00m"])
def test_parse_user_coordinates_rejects_garbage(text: str) -> None:
    with pytest.raises((ValueError, InvalidCoordinateError)):
        ai.parse_user_coordinates(text, "icrs")


async def test_name_and_conflicting_coordinates_rejected(sesame_replay) -> None:
    text = "radio sources near M87 at 00h42m44.3s +41d16m09s"
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(target={"name": "M87", "coordinates": {
            "text": "00h42m44.3s +41d16m09s", "frame": "icrs", "equinox": None}}))),
        reply(tool_use("submit_query", submission())),
    )
    await compile_with(fake, text)
    (result,) = tool_results(fake.calls[1])
    assert "from Sesame's position of 'M87'" in result["content"]


async def test_type_filter_on_untyped_catalogs_is_fed_back(sesame_replay) -> None:
    """object_types=['quasar'] with radio catalogs would delete every radio row: rejected."""
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(object_types=["quasar"]))),
        reply(tool_use("submit_query", submission())),
    )
    compiled = await compile_with(fake)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] is True
    assert "would remove every row of first, nvss, vlass, lotss" in result["content"]
    assert "ned, sdss, simbad" in result["content"]
    assert not compiled.advanced_query["object_types"]
    # Allowed when every selected catalog reports a type.
    ok = FakeAnthropic(reply(tool_use("submit_query", submission(object_types=["quasar"], catalogs=["simbad", "ned"]))))
    compiled_ok = await compile_with(ok)
    # 'quasar' is expanded into SIMBAD's QSO subtree (blazars, BL Lacs, candidates); NED uses QSO.
    assert compiled_ok.advanced_query["object_types"] == ["BL?", "BLL", "Bla", "Bz?", "Q?", "QSO"]
    assert compiled_ok.advanced_query["metadata"]["object_types_requested"] == ["quasar"]


def test_type_capable_catalogs_come_from_the_registry(registry: CatalogRegistry) -> None:
    assert ai.typed_catalogs(registry, "object_type") == ["ned", "sdss", "simbad"]
    assert ai.typed_catalogs(registry, "spectral_type") == ["simbad"]
    tools = {t["name"]: t for t in ai.build_query_tools(registry)}
    props = tools["submit_query"]["input_schema"]["properties"]
    assert "radio: first, lotss, nvss, vlass" in props["catalogs"]["description"]
    assert "xray: chandra, rosat, xmm" in props["catalogs"]["description"]


async def test_sesame_outage_is_an_upstream_error_not_feedback() -> None:
    """A Sesame failure must not be fed back to Claude (which could then invent coordinates)."""
    fake = FakeAnthropic(reply(tool_use("submit_query", submission())), reply(tool_use("submit_query", submission())))
    with respx.mock(assert_all_mocked=True) as router:
        route = router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").mock(
            return_value=httpx.Response(503, text="Service Unavailable"))
        with pytest.raises(ai.UpstreamServiceError, match="Sesame"):
            await compile_with(fake)
    assert len(fake.calls) == 1 and route.call_count == 1


async def test_sesame_outage_is_not_cached() -> None:
    names = ai._NameCache(SesameResolver(httpx.AsyncClient()))
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").mock(
            side_effect=[httpx.Response(503), httpx.Response(503)])
        for _ in range(2):
            with pytest.raises(ai.UpstreamServiceError):
                await names.resolve("M87")
        assert router.calls.call_count == 2
    await names.resolver.client.aclose()


async def test_all_sky_scope_requires_adql_and_returns_no_advanced_query() -> None:
    adql = ("SELECT TOP 500 b.main_id AS main_id, b.sp_type AS sp_type, b.plx_value AS plx_value FROM basic AS b "
            "JOIN alltypes AS a ON a.oidref = b.oid WHERE b.sp_type LIKE 'M%' AND b.plx_value >= 50 "
            "AND a.otypes LIKE '%X%'")
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(scope="all_sky", target=None))),
        reply(tool_use("submit_query", submission(scope="all_sky", target=None, adql={"catalog": "simbad", "query": adql + ";"}))),
    )
    compiled = await compile_with(fake, "X-ray bright M dwarfs within 20 pc")
    (result,) = tool_results(fake.calls[1])
    assert "scope all_sky requires an adql query" in result["content"]
    assert compiled.scope == "all_sky" and compiled.advanced_query is None
    assert compiled.adql == adql and compiled.adql_catalog == "simbad"
    assert compiled.adql_endpoint == "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"


async def test_remote_adql_verification_feeds_back_service_error() -> None:
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(scope="all_sky", target=None, adql={"catalog": "simbad", "query": recorder.BAD_ADQL}))),
        reply(tool_use("submit_query", submission(scope="all_sky", target=None, adql={"catalog": "simbad", "query": recorder.GOOD_ADQL}))),
    )
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/adql")))
        compiled = await compile_with(fake, "the stars with the largest parallaxes", verify_adql=True)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] is True
    assert 'Unknown column "b.no_such_column"' in result["content"]
    assert compiled.adql == recorder.GOOD_ADQL


async def test_end_turn_without_tool_call_is_nudged(sesame_replay) -> None:
    fake = FakeAnthropic(
        reply(text_block("Here is my plan..."), stop_reason="end_turn"),
        reply(tool_use("submit_query", submission())),
    )
    compiled = await compile_with(fake)
    assert fake.calls[1]["messages"][-1] == {"role": "user", "content": "Call the submit_query tool with the compiled search now."}
    assert compiled.validation_history == [["No submit_query call was made."]]


async def test_refusal_is_reported() -> None:
    fake = FakeAnthropic(reply(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber", explanation=None)))
    with pytest.raises(ai.AIRefusalError) as info:
        await compile_with(fake)
    assert info.value.category == "cyber"


async def test_anthropic_api_errors_become_upstream_errors() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    for exc in (
        anthropic.APIConnectionError(request=request),
        anthropic.RateLimitError("rate limited", response=httpx2.Response(429, request=request), body=None),
        anthropic.InternalServerError("overloaded", response=httpx2.Response(500, request=request), body=None),
    ):
        with pytest.raises(ai.AIUpstreamError):
            await compile_with(FakeAnthropic(exc))


CREDENTIAL_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID",
                  "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_SERVICE_ACCOUNT_ID", "ANTHROPIC_IDENTITY_TOKEN_FILE",
                  "ANTHROPIC_IDENTITY_TOKEN")


def _no_credentials(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    for key in CREDENTIAL_ENV:
        monkeypatch.delenv(key, raising=False)
    for key in ("HOME", "USERPROFILE", "XDG_CONFIG_HOME"):  # no `ant auth login` profile on disk either
        monkeypatch.setenv(key, str(home))


def test_build_client_requires_credentials(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_credentials(monkeypatch, tmp_path)
    assert not ai.anthropic_configured()
    with pytest.raises(ai.AINotConfiguredError, match="ANTHROPIC_API_KEY"):
        ai.build_anthropic_client()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    client = ai.build_anthropic_client(ai.AISettings())
    assert isinstance(client, anthropic.AsyncAnthropic)


def test_sdk_resolved_credentials_count_as_configured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An `ant auth login` profile or WIF (resolved by the SDK, no env key) is accepted."""
    _no_credentials(monkeypatch, tmp_path)

    class ProfileClient:
        api_key = None
        auth_token = None
        credentials = object()  # what the SDK sets for a profile / federation credential

        def __init__(self, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(ai.anthropic, "Anthropic", ProfileClient)
    assert ai.anthropic_configured()


async def test_unresolvable_sdk_credentials_become_503_error() -> None:
    error = TypeError('"Could not resolve authentication method. Expected one of api_key, auth_token, or credentials"')
    with pytest.raises(ai.AINotConfiguredError):
        await compile_with(FakeAnthropic(error))


@pytest.mark.parametrize(
    ("catalog", "query", "fragment"),
    [
        ("gaia_dr3", "SELECT source_id FROM gaiadr3.gaia_source WHERE parallax > 50", "SELECT TOP n"),
        ("gaia_dr3", "SELECT TOP 10 source_id FROM gaiadr3.gaia_source LIMIT 10", "no LIMIT"),
        ("gaia_dr3", "SELECT TOP 5000 source_id FROM gaiadr3.gaia_source", "TOP must be between"),
        ("gaia_dr3", "DELETE FROM gaiadr3.gaia_source", "read-only"),
        ("gaia_dr3", "SELECT TOP 10 ra FROM gaiadr2.gaia_source", "must read its table gaiadr3.gaia_source"),
        ("gaia_dr3", "SELECT TOP 10 ra FROM gaiadr3.gaia_source; SELECT TOP 1 ra FROM gaiadr3.gaia_source", "single statement"),
        ("gaia_dr3", "SELECT TOP 10 ra FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < (12", "unbalanced parentheses"),
        ("simbad", "SELECT TOP 10 main_id FROM basic WHERE main_id = 'M 87", "unbalanced single quote"),
        ("panstarrs_dr2", "SELECT TOP 10 objID FROM mean", "not a TAP catalog"),
        ("hubble_legacy_archive", "SELECT TOP 10 * FROM hla", "not a TAP catalog"),
    ],
)
def test_validate_adql_rejects(catalog: str, query: str, fragment: str, registry: CatalogRegistry) -> None:
    errors = ai.validate_adql(catalog, query, registry)
    assert any(fragment in e for e in errors), errors


@pytest.mark.parametrize(
    ("catalog", "query"),
    [
        ("gaia_dr3", "SELECT TOP 100 source_id, parallax FROM gaiadr3.gaia_source WHERE parallax > 50 ORDER BY parallax DESC"),
        ("vlass", 'SELECT TOP 10 Name, Flux FROM "J/ApJ/914/42/table5" WHERE Flux > 100'),
        ("simbad", "select distinct top 20 b.main_id from basic as b where b.otype = 'QSO'"),
        ("xmm", "SELECT TOP 50 name, ep_flux FROM xmmssc WHERE ep_flux > 1e-12"),
    ],
)
def test_validate_adql_accepts(catalog: str, query: str, registry: CatalogRegistry) -> None:
    assert ai.validate_adql(catalog, query, registry) == []


async def test_execute_compiled_adql_replay(registry: CatalogRegistry) -> None:
    compiled = ai.CompiledQuery(
        text="nearest stars", scope="all_sky", advanced_query=None, plan=["run"], explanation="x",
        adql=recorder.GOOD_ADQL, adql_catalog="simbad", adql_endpoint=None, resolved_objects=[], attempts=1,
        validation_history=[], model="test",
    )
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/adql")))
        async with offline_client() as client:
            result = await ai.execute_compiled_query(compiled, http_client=client, registry=registry, row_limit=5)
    assert result["kind"] == "adql" and result["catalog"] == "simbad"
    names = [r["main_id"] for r in result["rows"]]
    assert "NAME Proxima Centauri" in names and "* alf Cen" in names
    assert all(r["plx_value"] > 500 for r in result["rows"])  # Proxima: ~768 mas


async def test_execute_compiled_cone_query_uses_service() -> None:
    calls: list[tuple[Any, ...]] = []

    class StubService:
        async def crossmatch(self, ra, dec, **kwargs):
            calls.append((ra, dec, kwargs))
            return UnifiedRecord(target={"ra": ra, "dec": dec}, catalogs_queried=0, catalog_results={}, counterparts={},
                                 failures=[], provenance={})

    payload = AdvancedQuery.from_dict({"ra": 187.7, "dec": 12.39, "radius_arcsec": 60.0, "catalogs": ["first"]}).to_dict()
    compiled = ai.CompiledQuery(text="t", scope="cone", advanced_query=payload, plan=["p"], explanation="e", adql=None,
                                adql_catalog=None, adql_endpoint=None, resolved_objects=[], attempts=1,
                                validation_history=[], model="m")
    result = await ai.execute_compiled_query(compiled, service=StubService())
    assert result["kind"] == "crossmatch" and result["record"]["target"] == {"ra": 187.7, "dec": 12.39}
    assert isinstance(calls[0][2]["query"], AdvancedQuery) and calls[0][2]["query"].catalogs == ["first"]


# ---------------------------------------------------------------------------
# Citations & number checks
# ---------------------------------------------------------------------------


def _sources(n: int) -> list[ai.Source]:
    book = ai.SourceBook()
    for i in range(n):
        book.add(kind="reference", bibcode=f"20{10 + i}ApJ...{900 + i}..{10 + i}X", title=f"Paper {i}")
    return book.sources


def test_map_citations_renumbers_and_drops_unknown() -> None:
    sources = _sources(5)
    text, cited, warnings = ai.map_citations(
        "It is a quasar [3]. It is bright [1, 3] and variable [9]. Redshift is known [5; 12][2].", sources
    )
    assert text == "It is a quasar [1]. It is bright [2, 1] and variable. Redshift is known [3][4]."
    assert [c.bibcode for c in cited] == [sources[2].bibcode, sources[0].bibcode, sources[4].bibcode, sources[1].bibcode]
    assert [c.n for c in cited] == [1, 2, 3, 4]
    assert cited[0].url == f"https://ui.adsabs.harvard.edu/abs/{sources[2].bibcode}/abstract"
    assert warnings == ["Removed citation [9]: it matches no provided source.",
                        "Removed citation [12]: it matches no provided source."]
    assert sources[2].n == 3  # originals untouched


def test_map_citations_expands_ranges_before_renumbering() -> None:
    sources = _sources(8)
    text, cited, warnings = ai.map_citations("A fact [5]. Another [2-4]. Third [7].", sources)
    assert text == "A fact [1]. Another [2, 3, 4]. Third [5]."
    assert [c.bibcode for c in cited] == [sources[i].bibcode for i in (4, 1, 2, 3, 6)]
    assert warnings == []


@pytest.mark.parametrize("marker", ["[1-2]", "[1–2]", "[1—2]", "[1 - 2]", "[refs 1-2]"])
def test_map_citations_range_forms(marker: str) -> None:
    sources = _sources(5)
    text, cited, warnings = ai.map_citations(f"A [3]. B {marker}. C [4].", sources)
    assert text == "A [1]. B [2, 3]. C [4]."
    assert [c.bibcode for c in cited] == [sources[2].bibcode, sources[0].bibcode, sources[1].bibcode, sources[3].bibcode]
    assert warnings == []


def test_map_citations_ref_prefix_and_leftovers_are_reported() -> None:
    sources = _sources(5)
    text, cited, warnings = ai.map_citations("A [ref 2]. B [4-2]. Colour [3.6]-[4.5]. C [1, 9].", sources)
    assert text.startswith("A [1].") and "[4-2]" not in text and "C [2]." in text
    assert [c.bibcode for c in cited] == [sources[1].bibcode, sources[0].bibcode]
    assert "Removed citation range [4-2]: not a valid ascending range." in warnings
    assert "Removed citation [9]: it matches no provided source." in warnings
    assert any("[3.6]" in w and "not a mapped citation" in w for w in warnings)
    # Every surviving marker maps to a returned citation.
    numbers = {int(n) for group in re.findall(r"\[([\d, ]+)\]", text) for n in group.split(",")}
    assert numbers == {c.n for c in cited}


def test_map_citations_flags_uncited_text() -> None:
    _text, cited, warnings = ai.map_citations("No markers here.", _sources(2))
    assert cited == [] and warnings == ["The explanation cites no sources."]


def test_ads_url_escapes_ampersand() -> None:
    assert ai.ads_url("2020A&A...641A...6P") == "https://ui.adsabs.harvard.edu/abs/2020A%26A...641A...6P/abstract"


def test_unverified_numbers_heuristic() -> None:
    facts = {"z": 0.15756751, "dl": 777.7, "name": "3C 273", "t": 2.037, "nbref": 6060, "year": 1963}
    text = ("3C 273 [1] has redshift 0.158 [2], about 780 Mpc [3], 2.04 Gyr [3], 6,060 papers since 1963 [4]; "
            "it is 2.4 billion light years away and 12.9 mag.")
    assert ai.unverified_numbers(text, facts) == ["2.4", "12.9"]


def test_unverified_numbers_ignores_source_indices_ids_and_titles() -> None:
    """Small invented numbers must not pass just because a source number or title contains them."""
    facts = {
        "object": {"simbad_oid": 1940765, "main_id": "3C 273"},
        "identity": {"main_id": "3C 273", "identifiers": ["3C 273", "2MASS J12290665+0203085", "PKS 1226+023"], "ref": 1},
        "measurements": [{"quantity": "redshift", "value": 0.15756751, "ref": 3}],
        "sources": [{"n": i, "bibcode": f"20{10 + i}ApJ...{900 + i}..{10 + i}X",
                     "title": "The 12 kpc jet and the 27 knots of a 7-component quasar", "year": 2000 + i}
                    for i in range(1, 30)],
    }
    for claim, bad in [("Its jet extends about 12 kpc [2].", "12"),
                       ("The black hole mass is 7 billion solar masses.", "7"),
                       ("The host galaxy is 27 kpc across.", "27"),
                       ("The jet speed is 0.99c.", "0.99"),
                       ("It was observed at 5GHz.", "5")]:
        assert ai.unverified_numbers(claim, facts) == [bad], claim
    ok = ("3C 273 (PKS 1226+023, 2MASS J12290665+0203085) has redshift 0.158 [3]; a paper titled "
          "The 12 kpc jet and the 27 knots of a 7-component quasar studies it [2].")
    assert ai.unverified_numbers(ok, facts) == []


# ---------------------------------------------------------------------------
# Explanation: facts from replayed SIMBAD + real crossmatch record
# ---------------------------------------------------------------------------


@pytest.fixture
async def crossmatch_record_3c273() -> UnifiedRecord:
    """A real crossmatch record for 3C 273, built from the recorded catalog fixtures."""
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("3c273")))
        async with offline_client() as client:
            target = target_for("3c273")
            return await make_service(client).crossmatch(target.ra, target.dec, radius_arcsec=10.0)


class RecordingService:
    def __init__(self, record: UnifiedRecord) -> None:
        self.record = record
        self.calls: list[tuple[Any, ...]] = []

    async def crossmatch(self, ra, dec, **kwargs):
        self.calls.append((ra, dec, kwargs))
        return self.record


def _explainer(bogus_ref: int = 99) -> Callable[[dict[str, Any]], SimpleNamespace]:
    """Fake Claude that writes a summary citing refs found in the facts it was sent."""

    def respond(kwargs: dict[str, Any]) -> SimpleNamespace:
        payload = json.loads(kwargs["messages"][0]["content"].split("Facts (JSON):\n", 1)[1].rsplit("\n\nWrite", 1)[0])
        by_quantity = {m["quantity"]: m for m in payload["measurements"] + payload["derived"]}
        schmidt = next(e for e in payload["bibliography"]["foundational"] if e["bibcode"] == "1963Natur.197.1040S")
        chandra = next(d for d in payload["crossmatch"]["detections"] if d["catalog"] == "chandra")
        summary = (
            f"3C 273 is classified in SIMBAD as a {payload['identity']['otype_description']} [{payload['identity']['ref']}]. "
            f"Its redshift is 0.1576 [{by_quantity['redshift']['ref']}], corresponding to a luminosity distance of "
            f"{by_quantity['luminosity_distance']['value']} Mpc [{by_quantity['luminosity_distance']['ref']}]. "
            f"It is detected by Chandra [{chandra['ref']}]. "
            f"An early study reported it as a star-like object with large red-shift [{schmidt['ref']}]. "
            f"It is 2.4 billion light-years away [{bogus_ref}]."
        )
        return reply(SimpleNamespace(type="thinking", thinking=""), text_block(json.dumps({"summary": summary})),
                     stop_reason="end_turn")

    return respond


async def test_explain_object_offline_maps_citations_to_bibcodes(crossmatch_record_3c273: UnifiedRecord) -> None:
    fake = FakeAnthropic(_explainer())
    service = RecordingService(crossmatch_record_3c273)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_3c273")))
        async with offline_client() as client:
            result = await ai.explain_object(
                http_client=client, name="3C 273", anthropic_client=fake, service=service, settings=SETTINGS,
                radius_arcsec=5.0, max_references=8, ads_token="",
            )
    # Crossmatch ran at Sesame's position with the resolver epoch and zero (extragalactic) motion.
    ra, dec, kwargs = service.calls[0]
    assert (ra, dec) == pytest.approx(C3C273_SESAME, abs=1e-8)
    assert kwargs["epoch"] == 2000.0 and (kwargs["pm_ra_masyr"], kwargs["pm_dec_masyr"]) == (0.0, 0.0)
    assert kwargs["radius_arcsec"] == ai.DEFAULT_CROSSMATCH_RADIUS_ARCSEC

    call = fake.calls[0]
    assert "Use ONLY the provided facts" in call["system"]
    assert "never attach a unit" in call["system"] and "NOT non-detections" in call["system"]
    assert call["output_config"]["format"]["schema"]["required"] == ["summary"]
    assert call["output_config"]["effort"] == "high" and call["fallbacks"] == "default"

    assert result.summary.startswith("3C 273 is classified in SIMBAD as a BL Lac [1].")
    markers = [int(m) for m in re.findall(r"\[(\d+)\]", result.summary)]
    assert markers == [1, 2, 3, 4, 5]  # renumbered by first appearance; the bogus [99] removed
    assert "[99]" not in result.summary and any("[99]" in w for w in result.warnings)
    by_n = {c.n: c for c in result.citations}
    assert by_n[1].bibcode == "2000A&AS..143....9W"  # SIMBAD database paper
    assert by_n[2].bibcode == "2022ApJS..261....2K"  # SIMBAD's redshift reference (BASS DR2)
    assert by_n[3].bibcode == "2020A&A...641A...6P" and by_n[3].title == "Planck 2018 results. VI. Cosmological parameters."
    # The registry's Chandra citation carries no bibcode, only a DOI.
    assert by_n[4].kind == "catalog" and by_n[4].label == "chandra catalog" and by_n[4].bibcode is None
    assert by_n[4].title.startswith("Evans et al. 2024, ApJS 274, 22") and by_n[4].url == "https://doi.org/10.25574/csc2.1"
    assert by_n[5].bibcode == "1963Natur.197.1040S" and by_n[5].title == "3C 273: a star-like object with large red-shift."
    assert by_n[5].url == "https://ui.adsabs.harvard.edu/abs/1963Natur.197.1040S/abstract"
    assert result.unverified_numbers == ["2.4"]

    facts = result.facts
    assert facts.object["main_id"] == "3C 273" and facts.object["simbad_oid"] == 1940765
    assert "QSO" in facts.identity["all_otypes"]
    redshift = next(m for m in facts.measurements if m["quantity"] == "redshift")
    assert redshift["value"] == pytest.approx(0.15756751)
    assert redshift["error"] == 0.0005 and redshift["quality"] == "C" and redshift["nature"] == "spectroscopic"
    # SIMBAD's relativistic rvz_radvel (43555 km/s) is never reported as a velocity; at z > 0.1 no cz either.
    assert not [m for m in facts.measurements if m["quantity"] in {"radial_velocity", "cz"}]
    z_cmb = next(m for m in facts.derived if m["quantity"] == "redshift_cmb_frame")
    # 3C 273 lies 21 deg from the CMB dipole apex, so its CMB-frame redshift is larger.
    assert 0.1585 < z_cmb["value"] < 0.1592 and z_cmb["error"] == 0.0005
    dl = next(m for m in facts.derived if m["quantity"] == "luminosity_distance")
    # D_L = (1 + z_hel) D_M(z_cmb) (Davis et al. 2011), not astropy's (1 + z_cmb) D_M(z_cmb).
    expected = (1 + 0.15756751) * Planck18.comoving_transverse_distance(z_cmb["value"]).value
    assert dl["value"] == pytest.approx(expected, abs=0.1) and 780 < dl["value"] < 790
    # dz = hypot(0.0005, 250 km/s / c) = 0.00097: the peculiar-velocity scatter is in the error.
    assert 4.0 < dl["plus_error"] < 5.5 and 4.0 < dl["minus_error"] < 5.5
    assert "peculiar-velocity scatter of 250 km/s" in dl["note"]
    pec = next(s for s in facts.sources if s.n == dl["error_refs"][0])
    assert pec.bibcode == "2022ApJ...938..110B"  # Brout et al. 2022 (Pantheon+)
    assert not [d for d in facts.derived if d["quantity"] == "parallax_distance"]
    assert facts.crossmatch["detected_in"] >= 10
    assert {"radio", "xray", "infrared", "optical"} <= set(facts.crossmatch["wavelengths_detected"])
    assert {p["band"] for p in facts.photometry} >= {"V", "J", "H", "K"}
    assert result.as_dict()["citations"][0]["url"] == "https://ui.adsabs.harvard.edu/abs/2000A%26AS..143....9W/abstract"


async def test_gather_facts_by_coordinates_uses_simbad_cone() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_m87_coords")))
        async with offline_client() as client:
            facts = await ai.gather_facts(http_client=client, ra=recorder.M87[0], dec=recorder.M87[1],
                                          include_crossmatch=False, max_references=3)
    assert facts.object["simbad_oid"] == 1937204 and facts.identity["main_id"] == "M 87"
    assert facts.query == {"ra_deg": recorder.M87[0], "dec_deg": recorder.M87[1], "radius_arcsec": 5.0,
                           "simbad_match_separation_arcsec": 0.0}
    redshift = next(m for m in facts.measurements if m["quantity"] == "redshift")
    assert redshift["value"] < ai.HUBBLE_FLOW_MIN_Z and redshift["quality"] == "E"
    # SIMBAD's M 87 redshift has quality E: no Hubble-flow distance (and z < 0.01 anyway).
    assert not [d for d in facts.derived if d["quantity"] == "luminosity_distance"]
    assert any("quality is E" in w for w in facts.warnings)
    cz = next(m for m in facts.measurements if m["quantity"] == "cz")
    assert cz["value"] == pytest.approx(ai.SPEED_OF_LIGHT_KMS * redshift["value"], abs=0.1)
    assert all(len(v) <= 3 for k, v in facts.bibliography.items() if isinstance(v, list))


async def test_simbad_bibliography_from_recording_has_real_bibcodes() -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_3c273")))
        async with offline_client() as client:
            biblio = await ai.fetch_simbad_bibliography(client, 1940765, limit=8)
    for entries in biblio.values():
        assert len(entries) == 8
        for entry in entries:
            assert ai.BIBCODE_RE.fullmatch(entry["bibcode"]), entry["bibcode"]
            assert int(entry["bibcode"][:4]) == entry["year"]
    foundational = [e["year"] for e in biblio["foundational"]]
    assert foundational == sorted(foundational) and foundational[0] == 1963
    assert "1963Natur.197.1040S" in {e["bibcode"] for e in biblio["foundational"]}  # Schmidt 1963
    assert "1963Natur.197.1037H" in {e["bibcode"] for e in biblio["foundational"]}  # Hazard et al. 1963
    recent = [e["year"] for e in biblio["recent"]]
    assert recent == sorted(recent, reverse=True)
    # Only papers ABOUT the object: SIMBAD ref_flag says it is in the title or abstract.
    for key in ("focused", "recent"):
        assert all(e["object_in_title"] or e["object_in_abstract"] for e in biblio[key]), key
    freqs = [e["simbad_obj_freq"] for e in biblio["focused"]]
    assert freqs == sorted(freqs, reverse=True) and freqs[-1] >= 20  # ranked by occurrences in the paper
    # The three lists are disjoint: 'focused' never just repeats the newest papers.
    lists = [{e["bibcode"] for e in biblio[k]} for k in ("focused", "recent", "foundational")]
    assert not (lists[0] & lists[1]) and not (lists[0] & lists[2])


def test_summarize_crossmatch_real_record(crossmatch_record_3c273: UnifiedRecord) -> None:
    book = ai.SourceBook()
    summary = ai.summarize_crossmatch(crossmatch_record_3c273, book)
    by_catalog = {d["catalog"]: d for d in summary["detections"]}
    # The exoplanet archive lists known hosts only: absence there is not a non-detection.
    assert "exoplanet_archive" in [d["catalog"] for d in summary["coverage_unknown"]]
    # VLASS: Dec > -40 but "very bright sources (e.g. 3C 273) may be absent" -> inconclusive, not undetected.
    vlass = next(d for d in summary["coverage_unknown"] if d["catalog"] == "vlass")
    assert "3C 273" in vlass["caveat"]
    assert "non_detections" not in summary
    # NVSS (45" beam) blends 3C 273's core and jet: its centroid is 5.6" away, beyond the 5" association radius.
    # It is the only NVSS row, so the local density is unmeasured: never called "unlikely to be a chance
    # coincidence", never a detection; reported only as the nearest row outside the association radius.
    assert "nvss" not in {d["catalog"] for d in summary["detections"] + summary["ambiguous"]}
    nvss = next(d for d in summary["no_source_within_radius"] if d["catalog"] == "nvss")
    assert 5.0 < nvss["nearest_row_arcsec"] < 6.0 and "not associated" in nvss["nearest_row_note"]
    assert "unlikely to be a chance coincidence" not in json.dumps(summary)
    first = by_catalog["first"]
    assert first["wavelength"] == "radio" and first["fields"]["flux_20_cm"] == pytest.approx(35558.44)
    assert first["fields"]["flux_20_cm_error"] == pytest.approx(6.196)  # errors are kept with their values
    assert "ra" not in first["fields"] and "match_dist" not in first["fields"]
    assert summary["field_units"].startswith("not provided")
    ps1 = by_catalog["panstarrs_dr2"]["fields"]
    assert "distance" not in ps1  # the match distance (deg) is not a measurement
    assert "gMeanPSFMag" in ps1 and "gMeanPSFMagErr" in ps1
    assert by_catalog["ned"]["fields"]["prefphytype"] == "QSO"
    for d in summary["detections"]:
        p_chance = d["chance_coincidence_probability"]
        if p_chance is None:  # a lone row: density unmeasured, stated in the note
            assert d["rows_in_search_cone"] == 1 and "not measured" in d["note"]
        else:
            assert p_chance <= ai.MAX_CHANCE_PROBABILITY
    # Literature compilations are not observing bands.
    assert set(summary["listed_in"]) == {"ned", "simbad"}
    assert not {"multi", "extragalactic", "exoplanet"} & set(summary["wavelengths_detected"])
    gaia_source = book.get(by_catalog["gaia_dr3"]["ref"])
    assert gaia_source is not None and gaia_source.bibcode == "2023A&A...674A...1G"


def _recorded_crossmatch(key: str) -> dict[str, Any]:
    path = Path(__file__).parent / "fixtures" / "ai" / "crossmatch_records" / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_positions_outside_survey_footprints_are_not_non_detections() -> None:
    """Centaurus A (Dec -43): NVSS/VLASS (Dec > -40) and Pan-STARRS (Dec > -30) never observed it."""
    summary = ai.summarize_crossmatch(_recorded_crossmatch("centaurus_a"), ai.SourceBook())
    outside = {d["catalog"] for d in summary["outside_coverage"]}
    assert outside == {"nvss", "vlass", "panstarrs_dr2"}
    unknown = {d["catalog"] for d in summary["coverage_unknown"]}
    assert {"first", "lotss", "sdss"} <= unknown  # footprints that cannot be evaluated here
    no_source = {d["catalog"] for d in summary["no_source_within_radius"]}
    assert not (no_source & (outside | unknown))
    detected = {d["catalog"] for d in summary["detections"]}
    assert {"chandra", "xmm", "simbad", "ned"} <= detected


def test_counterpart_beyond_the_old_5_arcsec_radius_is_found() -> None:
    """NGC 4151: the ROSAT 2RXS source ~6 arcsec from the nucleus is within 3 sigma of its position error."""
    summary = ai.summarize_crossmatch(_recorded_crossmatch("ngc4151"), ai.SourceBook())
    rosat = next(d for d in summary["detections"] if d["catalog"] == "rosat")
    assert 5.0 < rosat["separation_arcsec"] < 6.5
    assert rosat["association_radius_arcsec"] >= rosat["separation_arcsec"]
    assert rosat["association_radius_arcsec"] == pytest.approx(3 * rosat["positional_error_arcsec"], abs=0.01)
    assert summary["search_radius_arcsec"] == ai.DEFAULT_CROSSMATCH_RADIUS_ARCSEC
    assert "rosat" not in {d["catalog"] for d in summary["no_source_within_radius"]}


def test_galactic_centre_field_stars_are_ambiguous_not_detections() -> None:
    """Sgr A*: the 2MASS (K ~ 7) and Pan-STARRS rows ~1 arcsec away are chance coincidences in a dense field."""
    summary = ai.summarize_crossmatch(_recorded_crossmatch("sgr_a_star"), ai.SourceBook())
    ambiguous = {d["catalog"]: d for d in summary["ambiguous"]}
    assert {"twomass_psc", "panstarrs_dr2"} <= set(ambiguous)
    assert ambiguous["twomass_psc"]["chance_coincidence_probability"] > 0.05
    assert ambiguous["twomass_psc"]["rows_in_search_cone"] > 20
    detected = {d["catalog"] for d in summary["detections"]}
    assert not ({"twomass_psc", "panstarrs_dr2"} & detected)
    assert {"chandra", "simbad"} <= detected  # the X-ray / radio source itself


def test_lone_distant_rows_are_not_associated_gn_z11() -> None:
    """GN-z11 (z ~ 10.6): single Gaia / 2MASS / Chandra / XMM rows 25-27 arcsec away are unrelated field sources.

    With one row in the cone the local density is unmeasured; P must not be reported as 0 and the rows must
    not be called "unlikely to be a chance coincidence" (the true Gaia P at 26 arcsec is ~0.5).
    """
    summary = ai.summarize_crossmatch(_recorded_crossmatch("gn_z11"), ai.SourceBook())
    associated = {d["catalog"] for d in summary["detections"] + summary["ambiguous"]}
    assert not associated & {"gaia_dr3", "twomass_psc", "chandra", "xmm"}
    nearest = {d["catalog"]: d for d in summary["no_source_within_radius"] + summary["coverage_unknown"]}
    for catalog in ("gaia_dr3", "twomass_psc", "chandra", "xmm"):
        assert 24.0 < nearest[catalog]["nearest_row_arcsec"] < 28.0, catalog
    assert "unlikely to be a chance coincidence" not in json.dumps(summary)
    assert all(d["chance_coincidence_probability"] != 0.0 or d["separation_arcsec"] == 0.0
               for d in summary["detections"] + summary["ambiguous"])
    assert summary["wavelengths_detected"] == [] and summary["listed_in"] == ["ned", "simbad"]


def test_vega_lone_rows_have_unknown_not_zero_chance_probability() -> None:
    summary = ai.summarize_crossmatch(_recorded_crossmatch("vega"), ai.SourceBook())
    detections = {d["catalog"]: d for d in summary["detections"]}
    assert "gaia_dr3" not in detections  # Vega (G ~ 0) is too bright for Gaia DR3
    lone = [d for d in detections.values() if d["rows_in_search_cone"] == 1]
    assert lone and all(d["chance_coincidence_probability"] is None for d in lone)
    assert all("not measured" in d["note"] for d in lone)


@pytest.mark.parametrize("separation", [7.0, 20.0, 29.0])
def test_single_row_beyond_association_radius_is_never_associated(separation: float) -> None:
    """NGC 4151's lone FIRST row moved outside the association radius: reported only as the nearest row."""
    record = _recorded_crossmatch("ngc4151")
    for sources in record["counterparts"].values():
        for src in sources:
            if src["catalog"] == "first":
                src["separation_arcsec"] = separation
    summary = ai.summarize_crossmatch(record, ai.SourceBook())
    assert "first" not in {d["catalog"] for d in summary["detections"] + summary["ambiguous"]}
    first = next(d for d in summary["coverage_unknown"] + summary["no_source_within_radius"] if d["catalog"] == "first")
    assert first["nearest_row_arcsec"] == separation
    assert "unlikely to be a chance coincidence" not in json.dumps(summary)


def test_coverage_status_parsing() -> None:
    assert ai.coverage_status("Dec > -40", -43.0) == ("outside", None)
    assert ai.coverage_status("Dec > -40", 2.0) == ("covered", None)
    assert ai.coverage_status("Dec > -40; very bright sources may be absent", 2.0) == (
        "unknown", "very bright sources may be absent")
    assert ai.coverage_status("all-sky", -89.0) == ("covered", None)
    assert ai.coverage_status("all-sky, G < ~21", 0.0) == ("covered", None)
    assert ai.coverage_status("pointed observations only (~1% of sky)", 0.0)[0] == "unknown"
    assert ai.coverage_status(None, 0.0)[0] == "unknown"


async def _facts_from(set_name: str, **kwargs: Any) -> ai.ObjectFacts:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges(f"ai/{set_name}")))
        async with offline_client() as client:
            return await ai.gather_facts(http_client=client, include_crossmatch=False, max_references=2, **kwargs)


def _quantities(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {e["quantity"]: e for e in entries}


async def test_no_parallax_distance_for_a_seyfert_galaxy() -> None:
    """NGC 4639 (Sy1, Cepheid distance ~20 Mpc) has a 5.6-sigma Gaia parallax of 2.65 mas: not 377 pc."""
    facts = await _facts_from("explain_ngc4639", name="NGC 4639")
    assert facts.identity["otype"] == "Sy1"
    measured = _quantities(facts.measurements)
    assert measured["parallax"]["value"] == pytest.approx(2.6549) and "not a distance" in measured["parallax"]["note"]
    assert "parallax_distance" not in _quantities(facts.derived)
    assert any("Parallax distance withheld" in w for w in facts.warnings)
    # A measured velocity (rvz_type v) stays a velocity, with its km/s error.
    assert measured["radial_velocity"]["value"] == 981 and measured["radial_velocity"]["error"] == 3
    mpc = [d["value"] for d in facts.distances if d["unit"] == "Mpc"]
    assert mpc and all(10 < v < 40 for v in mpc)


async def test_seyfert_with_redshift_gets_hubble_flow_but_no_parallax_distance() -> None:
    facts = await _facts_from("explain_eso140_43", name="ESO 140-43")
    derived = _quantities(facts.derived)
    assert "parallax_distance" not in derived
    assert 55 < derived["luminosity_distance"]["value"] < 70  # z_cmb ~ 0.0138
    assert derived["luminosity_distance"]["plus_error"] > 1.5  # dz = 0.0005 at z ~ 0.014 is ~3.6 %
    measured = _quantities(facts.measurements)
    assert "radial_velocity" not in measured
    assert measured["cz"]["value"] == pytest.approx(ai.SPEED_OF_LIGHT_KMS * 0.0138641, abs=0.1)
    assert "not a measured Doppler velocity" in measured["cz"]["note"]


async def test_galactic_binary_with_photometric_redshift_gets_no_cosmology() -> None:
    """HM Cnc (RX J0806.3+1527, a double white dwarf) has a photometric quality-E 'z = 2.19' in SIMBAD."""
    facts = await _facts_from("explain_hmcnc", name="HM Cnc")
    assert facts.identity["otype"] == "XB*"
    measured = _quantities(facts.measurements)
    z = measured["photometric_redshift"]
    assert z["value"] == 2.19 and z["nature"] == "photometric" and z["quality"] == "E"
    assert "redshift" not in measured and "cz" not in measured and "radial_velocity" not in measured
    assert facts.derived == []
    assert any("not one of the galaxy/AGN classes" in w for w in facts.warnings)


async def test_explain_by_coordinates_applies_proper_motion() -> None:
    """Barnard's star at its Gaia DR3 J2016.0 position is ~166 arcsec from its SIMBAD J2000 position."""
    ra, dec = recorder.BARNARD_GAIA2016
    facts = await _facts_from("explain_barnard_2016", ra=ra, dec=dec, epoch=2016.0)
    assert facts.identity["main_id"] == "NAME Barnard's star"  # the star, not its planets at the same position
    assert facts.query["epoch"] == 2016.0 and facts.query["simbad_match_separation_arcsec"] < 0.5
    assert haversine_arcsec(ra, dec, facts.object["ra_deg"], facts.object["dec_deg"]) > 150
    parallax_distance = _quantities(facts.derived)["parallax_distance"]
    assert parallax_distance["value"] == pytest.approx(1.828, abs=0.002)  # 1000 / 546.98 mas


async def test_crossmatch_is_cancelled_when_simbad_fails() -> None:
    state = {"started": 0, "cancelled": 0, "finished": 0}

    class SlowService:
        async def crossmatch(self, *args: Any, **kwargs: Any) -> UnifiedRecord:
            state["started"] += 1
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                state["cancelled"] += 1
                raise
            state["finished"] += 1
            raise RuntimeError("late crossmatch failure")

    class StubResolver:
        async def resolve(self, name: str) -> ResolvedObject:
            return ResolvedObject(query=name, canonical_name="3C 273", ra_deg=C3C273_SESAME[0],
                                  dec_deg=C3C273_SESAME[1], aliases=[], object_type="QSO", redshift=0.158,
                                  pm_ra_masyr=None, pm_dec_masyr=None, epoch=2000.0, resolver="Sesame",
                                  resolver_metadata={"resolver_name": "Simbad", "raw_fields": {"oid": ["1940765"]}})

    with respx.mock(assert_all_mocked=True) as router:
        router.post(ai.SIMBAD_TAP).mock(return_value=httpx.Response(503, text="Service Unavailable"))
        async with offline_client() as client:
            with pytest.raises(ai.UpstreamServiceError):
                await ai.gather_facts(http_client=client, name="3C 273", service=SlowService(),
                                      resolver=StubResolver())  # type: ignore[arg-type]
    assert state == {"started": 1, "cancelled": 1, "finished": 0}


async def test_ads_most_cited_request_shape() -> None:
    # Response body follows the documented ADS search API shape (no token available to record one).
    body = {"responseHeader": {"status": 0}, "response": {"numFound": 2, "docs": [
        {"bibcode": "1963Natur.197.1040S", "title": ["3C 273: a star-like object with large red-shift"], "year": "1963",
         "citation_count": 1},
        {"bibcode": "1963Natur.197.1037H", "title": ["Investigation of the radio source 3C 273"], "year": "1963",
         "citation_count": 1},
    ]}}
    with respx.mock(assert_all_mocked=True) as router:
        route = router.get("https://api.adsabs.harvard.edu/v1/search/query").mock(return_value=httpx.Response(200, json=body))
        async with offline_client() as client:
            docs = await ai.fetch_ads_most_cited(client, "token-123", "3C 273", rows=2)
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer token-123"
    assert dict(request.url.params) == {"q": 'object:"3C 273"', "fl": "bibcode,title,year,citation_count", "rows": "2",
                                        "sort": "citation_count desc"}
    assert docs[0] == {"bibcode": "1963Natur.197.1040S", "title": "3C 273: a star-like object with large red-shift",
                       "year": 1963, "citation_count": 1}


async def test_simbad_failure_is_upstream_error() -> None:
    with respx.mock(assert_all_mocked=True) as router:
        router.post(ai.SIMBAD_TAP).mock(return_value=httpx.Response(503, text="Service Unavailable"))
        async with offline_client() as client:
            with pytest.raises(ai.UpstreamServiceError, match="HTTP 503"):
                await ai.gather_facts(http_client=client, ra=10.0, dec=10.0, include_crossmatch=False)


async def test_explain_rejects_bad_json_from_model() -> None:
    fake = FakeAnthropic(reply(text_block("not json"), stop_reason="end_turn"))
    facts = ai.ObjectFacts(query={}, object={"simbad_oid": 1})
    with pytest.raises(ai.AIUpstreamError, match="not the expected JSON"):
        await ai.write_explanation(facts, anthropic_client=fake, settings=SETTINGS)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@pytest.fixture
def app_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _no_credentials(monkeypatch, tmp_path)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    for key in ("ASTROSEARCH_AI_MODEL", "ASTROSEARCH_AI_EFFORT", "ASTROSEARCH_AI_FALLBACKS"):
        monkeypatch.delenv(key, raising=False)

    def build(fake: FakeAnthropic | None = None) -> FastAPI:
        app = FastAPI()
        app.include_router(ai.router)
        app.state.registry = CatalogRegistry()
        if fake is not None:
            app.state.anthropic = fake
        return app

    return build


def test_router_query_success(app_factory, sesame_replay) -> None:
    fake = FakeAnthropic(reply(tool_use("submit_query", submission())))
    with TestClient(app_factory(fake)) as client:
        response = client.post("/api/v1/ai/query", json={"text": "quasars near M87 with radio emission"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert set(data) >= {"advanced_query", "plan", "explanation", "adql"}
    assert data["advanced_query"]["target"]["ra"] == pytest.approx(M87_SESAME[0])
    assert data["adql"] is None and data["scope"] == "cone"


def test_router_query_validation_and_errors(app_factory, sesame_replay) -> None:
    with TestClient(app_factory(FakeAnthropic())) as client:
        assert client.post("/api/v1/ai/query", json={"text": ""}).status_code == 422
        assert client.post("/api/v1/ai/query", json={"text": "ok query", "max_retries": 5}).status_code == 422
        assert client.post("/api/v1/ai/query", json={"text": "   "}).status_code == 422
    with TestClient(app_factory()) as client:  # no key configured
        response = client.post("/api/v1/ai/query", json={"text": "quasars near M87"})
    assert response.status_code == 503 and "ANTHROPIC_API_KEY" in response.json()["detail"]
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    with TestClient(app_factory(FakeAnthropic(anthropic.APIConnectionError(request=request)))) as client:
        response = client.post("/api/v1/ai/query", json={"text": "quasars near M87"})
    assert response.status_code == 502 and "Anthropic" in response.json()["detail"]
    bad = reply(tool_use("submit_query", submission(catalogs=["imaginary"])))
    with TestClient(app_factory(FakeAnthropic(bad))) as client:
        response = client.post("/api/v1/ai/query", json={"text": "quasars near M87", "max_retries": 0})
    assert response.status_code == 422
    assert any("Unknown catalog: imaginary" in e for e in response.json()["detail"]["errors"])


def test_router_explain_facts_only_and_errors(app_factory) -> None:
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_3c273")))
        with TestClient(app_factory()) as client:
            response = client.post("/api/v1/ai/explain", json={"name": "3C 273", "facts_only": True, "include_crossmatch": False})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["summary"] is None and data["citations"] == []
    assert data["facts"]["identity"]["main_id"] == "3C 273"
    assert data["object"]["simbad_oid"] == 1940765
    with TestClient(app_factory()) as client:
        assert client.post("/api/v1/ai/explain", json={"name": "3C 273", "ra": 1.0, "dec": 2.0}).status_code == 422
        assert client.post("/api/v1/ai/explain", json={"ra": 1.0}).status_code == 422
        assert client.post("/api/v1/ai/explain", json={}).status_code == 422
        assert client.post("/api/v1/ai/explain", json={"name": "3C 273", "radius_arcsec": 600}).status_code == 422
        no_key = client.post("/api/v1/ai/explain", json={"name": "3C 273"})
    assert no_key.status_code == 503


def test_router_explain_with_claude(app_factory, crossmatch_record_3c273: UnifiedRecord) -> None:
    app = app_factory(FakeAnthropic(_explainer()))
    app.state.service = RecordingService(crossmatch_record_3c273)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_3c273")))
        with TestClient(app) as client:
            response = client.post("/api/v1/ai/explain", json={"name": "3C 273"})
    assert response.status_code == 200, response.text
    data = response.json()
    assert [c["n"] for c in data["citations"]] == [1, 2, 3, 4, 5]
    assert data["citations"][4]["bibcode"] == "1963Natur.197.1040S" and data["citations"][4]["year"] == 1963
    assert data["unverified_numbers"] == ["2.4"]


def test_router_unresolvable_name_is_404(app_factory, sesame_replay) -> None:
    """models.resolution_failure_status: a well-formed Sesame answer that knows no such object is a 404."""
    with TestClient(app_factory()) as client:
        response = client.post("/api/v1/ai/explain", json={"name": BOGUS_NAME, "facts_only": True})
    assert response.status_code == 404 and "No coordinates found" in response.json()["detail"]
    assert "retry-after" not in response.headers


def test_router_sesame_outage_is_503_with_retry_after(app_factory) -> None:
    """models.resolution_failure_status: Sesame answering HTTP 5xx is an outage -- 503 + Retry-After."""
    with respx.mock(assert_all_mocked=True) as router:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").mock(return_value=httpx.Response(503))
        with TestClient(app_factory()) as client:
            explain = client.post("/api/v1/ai/explain", json={"name": "3C 273", "facts_only": True,
                                                              "include_crossmatch": False})
        with TestClient(app_factory(FakeAnthropic(reply(tool_use("submit_query", submission()))))) as client:
            query = client.post("/api/v1/ai/query", json={"text": "quasars near M87 with radio emission"})
    assert explain.status_code == 503 and "Sesame request failed" in explain.json()["detail"]
    assert explain.headers["retry-after"] == "30"
    assert query.status_code == 503 and "Sesame" in query.json()["detail"]
    assert query.headers["retry-after"] == "30"


def test_router_blank_sky_is_404_without_calling_claude(app_factory) -> None:
    fake = FakeAnthropic()  # any Claude call would fail the test ("called more times than scripted")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_blank_sky")))
        with TestClient(app_factory(fake)) as client:
            response = client.post("/api/v1/ai/explain", json={"ra": recorder.BLANK_SKY[0], "dec": recorder.BLANK_SKY[1],
                                                               "include_crossmatch": False})
    assert response.status_code == 404 and "Nothing is catalogued" in response.json()["detail"]
    assert fake.calls == []


def test_router_misconfiguration_is_500_not_422(app_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASTROSEARCH_AI_EFFORT", "ultra")
    with TestClient(app_factory(FakeAnthropic())) as client:
        response = client.post("/api/v1/ai/query", json={"text": "quasars near M87"})
    assert response.status_code == 500 and "ASTROSEARCH_AI_EFFORT" in response.json()["detail"]
    monkeypatch.setenv("ASTROSEARCH_AI_EFFORT", "high")
    monkeypatch.setenv("ASTROSEARCH_AI_MAX_TOKENS", "16k")
    with TestClient(app_factory(FakeAnthropic())) as client:
        response = client.post("/api/v1/ai/explain", json={"name": "3C 273"})
    assert response.status_code == 500


def test_router_explain_epoch_validation(app_factory) -> None:
    with TestClient(app_factory()) as client:
        assert client.post("/api/v1/ai/explain", json={"name": "3C 273", "epoch": 2016.0}).status_code == 422
        assert client.post("/api/v1/ai/explain", json={"ra": 1.0, "dec": 2.0, "epoch": 3016.0}).status_code == 422


async def test_state_clients_are_closed() -> None:
    closed: list[bool] = []

    class Closable:
        async def close(self) -> None:
            closed.append(True)

    state = SimpleNamespace(anthropic=Closable())
    await ai.aclose_state_clients(state)
    assert closed == [True] and state.anthropic is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="astrosearch")
    ai.register_cli(parser.add_subparsers(dest="command"))
    return parser


def test_cli_ask_compiles_against_the_services_registry(monkeypatch: pytest.MonkeyPatch, sesame_replay,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    """The tool enums come from the registry that will execute the query (CATALOG_REGISTRY_PATH)."""
    registry = CatalogRegistry()
    registry.get("vlass").enabled = False
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(catalogs=["first", "nvss"]))))
    monkeypatch.setattr(ai, "anthropic_factory", lambda: fake)
    monkeypatch.setattr(ai, "_http_client_factory", offline_client)
    monkeypatch.setattr(ai, "_service_factory", lambda client: SimpleNamespace(registry=registry))
    args = _parser().parse_args(["ask", "radio sources near M87"])
    assert args.handler(args) == 0
    submit = next(t for t in fake.calls[0]["tools"] if t["name"] == "submit_query")
    assert "vlass" not in submit["input_schema"]["properties"]["catalogs"]["items"]["enum"]
    assert "first" in submit["input_schema"]["properties"]["catalogs"]["items"]["enum"]
    capsys.readouterr()


def test_cli_explain_rejects_bad_coordinates_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    for argv, message in [
        (["explain", "--ra", "10", "--dec", "95", "--facts-only", "--no-crossmatch"], "--dec must be within"),
        (["explain", "--ra", "400", "--dec", "5", "--facts-only", "--no-crossmatch"], "--ra must be within"),
        (["explain", "--name", "M87", "--ra", "10", "--dec", "5", "--facts-only"], "Give either --name"),
        (["explain", "--name", "M87", "--epoch", "2016", "--facts-only"], "--epoch applies"),
    ]:
        args = _parser().parse_args(argv)
        assert args.handler(args) == 2
        err = capsys.readouterr().err
        assert message in err and "Traceback" not in err


def test_cli_catches_every_astrosearch_error(capsys: pytest.CaptureFixture[str]) -> None:
    async def boom(args: argparse.Namespace) -> int:
        raise ResponseParseError("upstream returned HTML")

    assert ai._cli(boom)(argparse.Namespace()) == 1
    assert "upstream returned HTML" in capsys.readouterr().err


def test_cli_ask(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], sesame_replay) -> None:
    fake = FakeAnthropic(reply(tool_use("submit_query", submission())), reply(tool_use("submit_query", submission())))
    monkeypatch.setattr(ai, "anthropic_factory", lambda: fake)
    monkeypatch.setattr(ai, "_http_client_factory", offline_client)
    args = _parser().parse_args(["ask", "quasars near M87 with radio emission"])
    assert args.handler(args) == 0
    out = capsys.readouterr().out
    assert "Plan:" in out and "Resolved: M87 -> M 87" in out and "catalogs=['first', 'nvss'" in out
    args = _parser().parse_args(["ask", "quasars near M87 with radio emission", "--json"])
    assert args.handler(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["compiled"]["advanced_query"]["target"]["dec"] == pytest.approx(M87_SESAME[1])


def test_cli_explain_facts_only_and_missing_key(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                                tmp_path: Path) -> None:
    monkeypatch.setattr(ai, "_http_client_factory", offline_client)
    _no_credentials(monkeypatch, tmp_path)
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_3c273")))
        args = _parser().parse_args(["explain", "--name", "3C 273", "--facts-only", "--no-crossmatch", "--json"])
        assert args.handler(args) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["facts"]["identity"]["main_id"] == "3C 273" and data["summary"] is None
    args = _parser().parse_args(["explain", "--name", "3C 273", "--no-crossmatch"])
    assert args.handler(args) == 3
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_cli_explain_with_claude(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                 crossmatch_record_3c273: UnifiedRecord) -> None:
    monkeypatch.setattr(ai, "anthropic_factory", lambda: FakeAnthropic(_explainer()))
    monkeypatch.setattr(ai, "_http_client_factory", offline_client)
    monkeypatch.setattr(ai, "_service_factory", lambda client: RecordingService(crossmatch_record_3c273))
    monkeypatch.delenv("ADS_API_TOKEN", raising=False)
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_3c273")))
        args = _parser().parse_args(["explain", "--name", "3C 273"])
        assert args.handler(args) == 0
    captured = capsys.readouterr()
    assert "3C 273 is classified in SIMBAD as a BL Lac [1]." in captured.out
    assert "[5] 1963Natur.197.1040S" in captured.out
    assert "numbers not traced to facts: 2.4" in captured.err


# ---------------------------------------------------------------------------
# Real SDK wiring (anthropic.AsyncAnthropic over a mock HTTP transport)
# ---------------------------------------------------------------------------


def _api_message(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    """Messages API response body (documented shape) for the mock transport."""
    return {"id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5", "content": content,
            "stop_reason": stop_reason, "stop_sequence": None, "usage": {"input_tokens": 100, "output_tokens": 20}}


async def test_real_sdk_request_and_tool_loop_serialization(sesame_replay) -> None:
    sent: list[httpx2.Request] = []
    answers = [
        _api_message([{"type": "thinking", "thinking": "", "signature": "sig-1"},
                      {"type": "tool_use", "id": "toolu_01", "name": "resolve_object", "input": {"name": "M87"}}], "tool_use"),
        _api_message([{"type": "tool_use", "id": "toolu_02", "name": "submit_query", "input": submission()}], "tool_use"),
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=answers[len(sent) - 1])

    sdk = anthropic.AsyncAnthropic(
        api_key="sk-ant-test", base_url="https://api.anthropic.test", max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    compiled = await compile_with(sdk)
    assert compiled.advanced_query["target"]["ra"] == pytest.approx(M87_SESAME[0])
    assert len(sent) == 2
    first = json.loads(sent[0].content)
    assert sent[0].url.path == "/v1/messages"
    assert "server-side-fallback-2026-07-01" in sent[0].headers["anthropic-beta"]
    assert first["fallbacks"] == "default" and first["thinking"] == {"type": "adaptive"}
    assert first["output_config"] == {"effort": "high"} and first["model"] == "claude-opus-5"
    assert [t["strict"] for t in first["tools"]] == [True, True]
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    second = json.loads(sent[1].content)
    assistant, results = second["messages"][1], second["messages"][2]
    # The SDK's response blocks (thinking with signature + tool_use) are sent back unchanged.
    assert assistant["role"] == "assistant"
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-1"}
    assert assistant["content"][1]["type"] == "tool_use" and assistant["content"][1]["id"] == "toolu_01"
    assert results["content"][0]["tool_use_id"] == "toolu_01"
    assert json.loads(results["content"][0]["content"])["canonical_name"] == "M 87"


async def test_real_sdk_explanation_request() -> None:
    sent: list[httpx2.Request] = []
    summary = json.dumps({"summary": "M 87 is listed in SIMBAD [1]."})

    def handler(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=_api_message([{"type": "text", "text": summary}], "end_turn"))

    sdk = anthropic.AsyncAnthropic(
        api_key="sk-ant-test", base_url="https://api.anthropic.test", max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    book = ai.SourceBook()
    ref = book.add(kind="database", bibcode=ai.SIMBAD_BIBCODE, label="SIMBAD database")
    facts = ai.ObjectFacts(query={"name": "M87"}, object={"simbad_oid": 1937204, "main_id": "M 87"},
                           identity={"main_id": "M 87", "ref": ref}, sources=list(book.sources))
    text, cited, warnings, unverified, model = await ai.write_explanation(facts, anthropic_client=sdk, settings=SETTINGS)
    body = json.loads(sent[0].content)
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["effort"] == "high"
    assert text == "M 87 is listed in SIMBAD [1]." and cited[0].bibcode == ai.SIMBAD_BIBCODE
    assert warnings == [] and unverified == [] and model == "claude-opus-5"


# ---------------------------------------------------------------------------
# Regression tests (review round 2)
# ---------------------------------------------------------------------------


async def test_ned_only_sesame_answer_is_identified_by_simbad_identifier() -> None:
    """Sesame answering 'M 1' from NED only (the Crab Pulsar's position) must still give SIMBAD's M 1 (SNR)."""
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_m1_ned")))
        async with offline_client() as client:
            facts = await ai.gather_facts(http_client=client, name="M 1", include_crossmatch=False, max_references=2,
                                          resolver=recorder.ned_only_resolver(client))
    assert facts.object["simbad_oid"] == 795871 and facts.identity["main_id"] == "M 1"
    assert facts.identity["otype"] == "SNR" and facts.identity["otype_path"] == "ISM > SNR"
    assert facts.query["identification"]["method"] == "SIMBAD identifier"
    # The crossmatch/position is SIMBAD's nebula centre, not NED's pulsar position 10.7 arcsec away.
    assert haversine_arcsec(facts.object["ra_deg"], facts.object["dec_deg"], 83.6324, 22.0174) < 1.0
    assert haversine_arcsec(facts.object["ra_deg"], facts.object["dec_deg"], *recorder.NED_M1) > 5.0
    assert facts.warnings == []


async def test_unknown_name_positional_fallback_is_flagged() -> None:
    """A name SIMBAD does not list: the nearest SIMBAD object is used only with an explicit warning."""
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=replay_side_effect(load_exchanges("ai/explain_ned_unknown_name")))
        async with offline_client() as client:
            facts = await ai.gather_facts(http_client=client, name=recorder.UNKNOWN_NED_NAME, include_crossmatch=False,
                                          max_references=2, resolver=recorder.UnknownNameNedResolver())
    assert facts.identity["main_id"] == "V* CM Tau" and facts.identity["otype"] == "Psr"
    note = facts.query["identification"]["note"]
    assert "may be a DIFFERENT object" in note and recorder.UNKNOWN_NED_NAME in note
    assert note in facts.warnings
    # The note reaches Claude (the facts payload keeps "query"), and the prompt says to state it.
    assert facts.prompt_payload()["query"]["identification"]["note"] == note
    assert '"identification" note' in ai._EXPLAIN_SYSTEM_PROMPT


async def test_hubble_flow_error_includes_peculiar_velocity_ngc4993() -> None:
    """NGC 4993 (z_cmb 0.0108): a 250 km/s peculiar-velocity scatter is ~8 % of D_L, not the 0.5 % of dz alone."""
    facts = await _facts_from("explain_ngc4993", name="NGC 4993")
    derived = _quantities(facts.derived)
    dl, z_cmb = derived["luminosity_distance"], derived["redshift_cmb_frame"]
    assert 46 < dl["value"] < 51
    assert dl["plus_error"] >= 3.5 and dl["minus_error"] >= 3.5
    z_hel = _quantities(facts.measurements)["redshift"]["value"]
    assert dl["value"] == pytest.approx((1 + z_hel) * Planck18.comoving_transverse_distance(z_cmb["value"]).value,
                                        abs=0.1)
    # The SBF distance (Cantiello et al. 2018, 40.7 +- 1.5 Mpc) lies within ~2 sigma of the Hubble-flow value.
    sbf = [d for d in facts.distances if d["unit"] == "Mpc" and d["plus_error"]]
    assert sbf and all(d["minus_error"] > 0 for d in sbf)  # SIMBAD's negative minus_err stored as a magnitude
    assert any(abs(d["value"] - dl["value"]) < 2.5 * dl["minus_error"] for d in sbf)


@pytest.mark.parametrize(("set_name", "name", "otype"), [("explain_ngc6621", "NGC 6621", "AG?"),
                                                           ("explain_ngc4807", "NGC 4807", "GiP")])
async def test_simbad_candidate_and_pair_galaxy_types_are_extragalactic(set_name: str, name: str, otype: str) -> None:
    facts = await _facts_from(set_name, name=name)
    assert facts.identity["otype"] == otype and facts.identity["otype_path"].startswith("G")
    assert not any("not one of the galaxy/AGN classes" in w for w in facts.warnings)
    derived = _quantities(facts.derived)
    assert 80 < derived["luminosity_distance"]["value"] < 120
    assert "cz" in _quantities(facts.measurements)


@pytest.mark.parametrize(
    ("otype", "path", "expected"),
    [("AG?", "G > AGN", True), ("Q?", "G > AGN > QSO", True), ("Bz?", "G > AGN > QSO > Bla", True),
     ("BL?", "G > AGN > QSO > Bla > BLL", True), ("GiP", "G > GiP", True), ("C?G", "ClG", True), ("Gr?", "GrG", True),
     ("SC?", "SCG", True), ("PCG?", "PCG", True), ("IG", "IG", True), ("PaG", "PaG", True), ("LeQ", None, True),
     ("Lev", "grv > Lev", False), ("Psr", "* > Psr", False), ("SNR", "ISM > SNR", False), ("X", "X", False),
     ("BH", "grv > BH", False)],
)
def test_simbad_type_hierarchy_decides_extragalactic(otype: str, path: str | None, expected: bool) -> None:
    assert ai.simbad_type_is_extragalactic(otype, path) is expected


async def test_proxima_bibliography_and_parallax_distance_errors() -> None:
    facts = await _facts_from("explain_proxima", name="Proxima Centauri")
    for key in ("focused", "recent"):
        entries = facts.bibliography[key]
        assert entries and all(e["object_in_title"] or e["object_in_abstract"] for e in entries), key
    assert "title or abstract" in facts.bibliography["selection"]
    plx = _quantities(facts.measurements)["parallax"]
    dist = _quantities(facts.derived)["parallax_distance"]
    assert dist["value"] == pytest.approx(1000 / plx["value"], abs=0.001)  # 1.302 pc
    assert 0 < dist["minus_error"] <= dist["plus_error"] < 0.01
    ref = next(s for s in facts.sources if s.n == dist["ref"])
    if ref.bibcode in ai.GAIA_EDR3_DR3_BIBCODES:
        assert "zero-point" in dist["note"]
        assert next(s for s in facts.sources if s.n == dist["zero_point_ref"]).bibcode == "2021A&A...649A...4L"


async def test_barnard_parallax_distance_has_asymmetric_errors() -> None:
    ra, dec = recorder.BARNARD_GAIA2016
    facts = await _facts_from("explain_barnard_2016", ra=ra, dec=dec, epoch=2016.0)
    plx = _quantities(facts.measurements)["parallax"]
    dist = _quantities(facts.derived)["parallax_distance"]
    assert dist["plus_error"] == pytest.approx(1000 / (plx["value"] - plx["error"]) - 1000 / plx["value"], rel=0.05)
    assert dist["minus_error"] == pytest.approx(1000 / plx["value"] - 1000 / (plx["value"] + plx["error"]), rel=0.05)


def test_wavelengths_detected_excludes_compilations(crossmatch_record_3c273: UnifiedRecord) -> None:
    summary = ai.summarize_crossmatch(crossmatch_record_3c273, ai.SourceBook())
    assert set(summary["wavelengths_detected"]) <= {"radio", "infrared", "optical", "xray"}
    assert set(summary["listed_in"]) <= {"simbad", "ned", "exoplanet_archive"}


@pytest.mark.parametrize(
    ("text", "frame", "equinox", "expected"),
    [
        ("J123049.42+122328.0", "icrs", None, M87_SESAME),
        ("J122906.70+020308.6", "icrs", None, C3C273_SESAME),
        ("123049.42+122328.0", "icrs", None, M87_SESAME),
        ("J123049.42+122328.0", "fk5", "J2000", M87_SESAME),
        ("B1950 12h28m17.6s +12d40m02s", "fk4", None, M87_SESAME),
    ],
)
def test_parse_compact_iau_positions_as_sexagesimal(text: str, frame: str, equinox: str | None,
                                                    expected: tuple[float, float]) -> None:
    target = ai.parse_user_coordinates(text, frame, equinox)
    assert haversine_arcsec(target.ra, target.dec, *expected) < 1.0


@pytest.mark.parametrize(
    ("text", "frame", "equinox", "fragment"),
    [
        ("l=0.0, b=0.0", "icrs", None, "use frame galactic"),
        ("glon=121.17 glat=-21.57", "icrs", None, "use frame galactic"),
        ("l=121.17, b=-21.57", "fk5", None, "use frame galactic"),
        ("RA=187.7 Dec=12.4", "galactic", None, "frame is galactic"),
        ("12h28m17.6s +12d40m02s", "icrs", "B1950", "takes no equinox"),
        ("l=0.0 b=0.0", "galactic", "J2000", "takes no equinox"),
        ("12h30m49s +12d23m28s", "fk5", "B1950", "B1950 coordinates are FK4"),
        ("12h28m17.6s +12d40m02s", "fk4", "J2000", "not FK4"),
        ("B1950 12h28m17.6s +12d40m02s", "icrs", None, "use frame fk4"),
        ("J123049+122328", "fk4", None, "not fk4"),
        ("370.0 10.0", "icrs", None, "outside [0, 360)"),
        ("25h00m00s +10d00m00s", "icrs", None, "outside [0, 24)"),
    ],
)
def test_parse_user_coordinates_rejects_inconsistent_frames(text: str, frame: str, equinox: str | None,
                                                           fragment: str) -> None:
    with pytest.raises(ValueError, match=re.escape(fragment)):
        ai.parse_user_coordinates(text, frame, equinox)


async def test_compact_designation_is_parsed_end_to_end(sesame_replay) -> None:
    text = "radio sources within 60 arcsec of J123049.42+122328.0"
    fake = FakeAnthropic(reply(tool_use("submit_query", submission(target=coords("J123049.42+122328.0"),
                                                                   radius_arcsec=60.0))))
    compiled = await compile_with(fake, text)
    target = compiled.advanced_query["target"]
    assert haversine_arcsec(target["ra"], target["dec"], *M87_SESAME) < 0.2  # not RA 12.51 deg


async def test_mislabelled_frames_are_fed_back(sesame_replay) -> None:
    text = "X-ray sources within 120 arcsec of Galactic l=0.0, b=0.0"
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(target=coords("l=0.0, b=0.0", "icrs"), radius_arcsec=120.0,
                                                  catalogs=["chandra"]))),
        reply(tool_use("submit_query", submission(target=coords("l=0.0, b=0.0", "galactic"), radius_arcsec=120.0,
                                                  catalogs=["chandra"]))),
    )
    compiled = await compile_with(fake, text)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] is True and "use frame galactic" in result["content"]
    assert compiled.advanced_query["target"]["ra"] == pytest.approx(266.40499, abs=1e-4)
    # An equinox stated elsewhere in the request must be honoured too.
    text = "radio sources within 60 arcsec of the B1950 position 12h28m17.6s +12d40m02s"
    fake = FakeAnthropic(
        reply(tool_use("submit_query", submission(target=coords("12h28m17.6s +12d40m02s"), radius_arcsec=60.0))),
        reply(tool_use("submit_query", submission(target=coords("12h28m17.6s +12d40m02s", "fk4", "B1950"),
                                                  radius_arcsec=60.0))),
    )
    compiled = await compile_with(fake, text)
    (result,) = tool_results(fake.calls[1])
    assert result["is_error"] is True and "use frame fk4" in result["content"]
    target = compiled.advanced_query["target"]
    assert haversine_arcsec(target["ra"], target["dec"], *M87_SESAME) < 1.0


@pytest.mark.parametrize(
    ("summary", "expected_text", "expected_cited"),
    [
        ("First fact [6]. Second [1, 2, and 3].", "First fact [1]. Second [2, 3, 4].", [6, 1, 2, 3]),
        ("A [4]. B [1, 2 & 3].", "A [1]. B [2, 3, 4].", [4, 1, 2, 3]),
        ("A [4]. B [2,].", "A [1]. B [2].", [4, 2]),
        ("A [4]. B [#2].", "A [1]. B [2].", [4, 2]),
        ("A [4]. B [source 2].", "A [1]. B [2].", [4, 2]),
        ("A [4]. B [see 2].", "A [1]. B [2].", [4, 2]),
    ],
)
def test_map_citations_list_forms(summary: str, expected_text: str, expected_cited: list[int]) -> None:
    sources = _sources(8)
    text, cited, warnings = ai.map_citations(summary, sources)
    assert text == expected_text
    assert [c.bibcode for c in cited] == [sources[i - 1].bibcode for i in expected_cited]
    assert warnings == []


def test_map_citations_never_leaves_stale_numbers() -> None:
    """Citation-like text the grammar cannot map is removed, never left beside renumbered markers."""
    sources = _sources(8)
    text, cited, warnings = ai.map_citations("First [6]. Then [nos. 2 and 9 or 3]. Band [3.6].", sources)
    assert text == "First [1]. Then. Band [3.6]."
    assert [c.bibcode for c in cited] == [sources[5].bibcode]
    assert any("Removed unparseable citation marker [nos. 2 and 9 or 3]" in w for w in warnings)
    numbers = {int(n) for group in re.findall(r"\[([\d, ]+)\]", text) for n in group.split(",")}
    assert numbers == {c.n for c in cited}


def _wif_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _no_credentials(monkeypatch, tmp_path)
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_test")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_test")
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_test")
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN_FILE", str(tmp_path / "missing" / "token.jwt"))


def _offline_sdk() -> anthropic.AsyncAnthropic:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError(f"no request may reach the network: {request.url}")

    return anthropic.AsyncAnthropic(max_retries=0, http_client=anthropic.DefaultAsyncHttpxClient(
        transport=httpx2.MockTransport(handler)))


async def test_request_time_credential_failure_is_not_configured(monkeypatch: pytest.MonkeyPatch,
                                                                 tmp_path: Path) -> None:
    """WIF configured but the identity-token file is missing: a 503-class error, not a crash."""
    _wif_env(monkeypatch, tmp_path)
    assert ai.anthropic_configured()
    with pytest.raises(ai.AINotConfiguredError, match="Identity token file"):
        await ai._create_message(_offline_sdk(), SETTINGS, messages=[{"role": "user", "content": "hi"}])


async def test_other_sdk_errors_are_upstream_errors() -> None:
    fake = FakeAnthropic(anthropic.WorkloadIdentityError("token exchange rejected"))
    with pytest.raises(ai.AIUpstreamError, match="WorkloadIdentityError"):
        await compile_with(fake)


def test_router_and_cli_map_credential_failure_to_503_and_exit_3(app_factory, monkeypatch: pytest.MonkeyPatch,
                                                                 tmp_path: Path,
                                                                 capsys: pytest.CaptureFixture[str]) -> None:
    app = app_factory()
    _wif_env(monkeypatch, tmp_path)
    app.state.anthropic = _offline_sdk()
    with TestClient(app) as client:
        response = client.post("/api/v1/ai/query", json={"text": "quasars near M87"})
    assert response.status_code == 503 and "Identity token file" in response.json()["detail"]
    monkeypatch.setattr(ai, "anthropic_factory", _offline_sdk)
    monkeypatch.setattr(ai, "_http_client_factory", offline_client)
    monkeypatch.setattr(ai, "_service_factory", lambda client: SimpleNamespace(registry=CatalogRegistry()))
    args = _parser().parse_args(["ask", "quasars near M87"])
    assert args.handler(args) == 3
    err = capsys.readouterr().err
    assert "Identity token file" in err and "Traceback" not in err


async def test_simbad_fan_out_is_cancelled_on_first_failure() -> None:
    state = {"finished": 0, "cancelled": 0}

    async def respond(request: httpx.Request) -> httpx.Response:
        query = request_signature(b"", request.content)["QUERY"][0]
        if "FROM basic AS b" in query and "WHERE b.oid" in query:
            return httpx.Response(503, text="Service Unavailable")
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            state["cancelled"] += 1
            raise
        state["finished"] += 1
        return httpx.Response(200, json={"metadata": [], "data": []})

    facts = ai.ObjectFacts(query={}, object={})
    with respx.mock(assert_all_mocked=True) as router:
        router.post(ai.SIMBAD_TAP).mock(side_effect=respond)
        async with offline_client() as client:
            started = asyncio.get_running_loop().time()
            with pytest.raises(ai.UpstreamServiceError, match="HTTP 503"):
                await ai._add_simbad_facts(facts, client, 1940765, ai.SourceBook(), 1, [], max_references=2,
                                           ads_token=None, endpoint=ai.SIMBAD_TAP)
            assert asyncio.get_running_loop().time() - started < 1.5
    # 8 TAP requests (5 fact queries + 3 bibliography lists): the slow ones are cancelled (or never sent);
    # none keeps running to completion after the failure.
    await asyncio.sleep(2.3)
    assert state["finished"] == 0 and 1 <= state["cancelled"] <= 7


@pytest.mark.parametrize("body", [b"<html><body>Maintenance</body></html>", b"", b'{"status": "ok"}'])
async def test_non_votable_200_does_not_verify_adql(body: bytes) -> None:
    simbad = CatalogRegistry().get("simbad")
    with respx.mock(assert_all_mocked=True) as router:
        router.post(simbad.endpoint).mock(return_value=httpx.Response(200, content=body))
        async with offline_client() as client:
            with pytest.raises(ai.UpstreamServiceError, match="without a VOTable result"):
                await ai.verify_adql_remote(simbad, recorder.GOOD_ADQL, client)
        fake = FakeAnthropic(reply(tool_use("submit_query", submission(
            scope="all_sky", target=None, adql={"catalog": "simbad", "query": recorder.GOOD_ADQL}))))
        compiled = await compile_with(fake, "the stars with the largest parallaxes", verify_adql=True)
    assert any("could not be verified" in w for w in compiled.warnings)


async def test_max_references_is_validated() -> None:
    for bad in (0, -3, 26, 2.5):
        with pytest.raises(ValueError, match="max_references"):
            await ai.gather_facts(http_client=None, ra=10.0, dec=10.0, max_references=bad)  # type: ignore[arg-type]


def test_cli_bad_input_exit_codes(sesame_replay, monkeypatch: pytest.MonkeyPatch,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(ai, "_http_client_factory", offline_client)
    for argv, fragment in [
        (["explain", "--name", "3C273", "--max-references", "-3", "--facts-only", "--no-crossmatch"], "max_references"),
        (["explain", "--name", BOGUS_NAME, "--facts-only", "--no-crossmatch"], "No coordinates found"),
    ]:
        args = _parser().parse_args(argv)
        assert args.handler(args) == 2, argv
        err = capsys.readouterr().err
        assert fragment in err and "Traceback" not in err


async def test_state_anthropic_is_built_once_under_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[object] = []

    def slow_build() -> object:
        time.sleep(0.2)
        client = object()
        built.append(client)
        return client

    monkeypatch.setattr(ai, "build_anthropic_client", slow_build)
    state = SimpleNamespace()
    clients = await asyncio.gather(*(ai._state_anthropic(state) for _ in range(5)))
    assert len(built) == 1 and all(c is built[0] for c in clients) and state.anthropic is built[0]

"""The ``astrosearch`` CLI with every feature module registered (main.py).

Every subcommand (nested ones included) prints its help and exits 0; every module's commands
dispatch through ``args.handler`` with its documented exit codes; and real offline runs
(recorded archive answers, temporary stores) produce the documented output.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import respx
from fixture_io import FIXTURES, TARGETS, load_exchanges, replay_side_effect

import main

ROOT = Path(__file__).resolve().parents[1]
RA, DEC = TARGETS["3c273"]
SESAME_3C273 = (FIXTURES / "sesame" / "3c273.xml").read_text(encoding="utf-8")


def command_paths(parser: argparse.ArgumentParser) -> list[list[str]]:
    """Every subcommand path of the parser: ['alerts'], ['alerts', 'poll'], ..."""
    paths: list[list[str]] = []

    def walk(current: argparse.ArgumentParser, prefix: list[str]) -> None:
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    paths.append([*prefix, name])
                    walk(sub, [*prefix, name])

    walk(parser, [])
    return paths


COMMANDS = command_paths(main.build_parser())
TOP_LEVEL = {"serve", "search", "dataset", "catalogs", "benchmark", "verify", "stream", "xmatch-calibrate", "batch",
             "mirror", "skycache", "vizier", "sed", "lightcurve", "solar-system", "cutout", "ask", "explain", "replay",
             "cite", "manifest", "vo", "alerts"}


@pytest.fixture
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for name, value in {
        "DATASET_STORAGE_PATH": tmp_path / "datasets", "SKYCACHE_PATH": tmp_path / "skycache",
        "CATALOG_REGISTRY_PATH": tmp_path / "catalogs.yaml", "CUTOUT_CACHE_DIR": tmp_path / "cutouts",
        "ASTROSEARCH_CACHE_DIR": tmp_path / "cache", "ALERTS_DATABASE_URL": f"sqlite:///{(tmp_path / 'a.db').as_posix()}",
    }.items():
        monkeypatch.setenv(name, str(value))
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "REDIS_URL", "SKYCACHE_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def run(argv: list[str]) -> int:
    """main.main(argv): its exit status (0 when it returns normally)."""
    try:
        main.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def upstream(*exchanges: Any, sesame: bool = True, strict: bool = False, ads: bool = False) -> respx.MockRouter:
    """A respx router replaying ``exchanges`` (Sesame answers 3C 273 unless ``sesame=False``).
    ``ads``: the ADS link gateway answers 404 (no DOI link for a bibcode), as a registered
    table's paper lookup (vizier.citation_references) then finds -- it is cited unverified."""
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    if sesame:
        router.get(url__startswith="https://cds.unistra.fr/cgi-bin/nph-sesame").respond(
            200, text=SESAME_3C273, headers={"content-type": "text/xml"})
    if ads:
        router.get(url__startswith="https://ui.adsabs.harvard.edu/link_gateway/").respond(404, text="Not Found")
    handler = replay_side_effect(list(exchanges), strict=strict) if exchanges else None
    router.route().mock(side_effect=handler or AssertionError("no upstream request expected"))
    return router


# ---------------------------------------------------------------------------
# Registration and help
# ---------------------------------------------------------------------------


def test_every_module_registers_its_commands() -> None:
    top = {path[0] for path in COMMANDS}
    assert top == TOP_LEVEL
    nested = {tuple(path) for path in COMMANDS if len(path) == 2}
    assert {("alerts", "poll"), ("alerts", "watch"), ("alerts", "list"), ("alerts", "show"), ("skycache", "status"),
            ("skycache", "cone"), ("skycache", "delete"), ("vizier", "search"), ("vizier", "describe"),
            ("vizier", "add"), ("vizier", "list"), ("vo", "cone"), ("vo", "adql"), ("vo", "tables")} <= nested


@pytest.mark.parametrize("path", COMMANDS, ids=[" ".join(p) for p in COMMANDS])
def test_every_subcommand_prints_help(path: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert run([*path, "--help"]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"usage: astrosearch {' '.join(path)}"), out[:200]


def test_the_entry_point_runs_as_a_process() -> None:
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    done = subprocess.run([sys.executable, "main.py", "--help"], cwd=ROOT, capture_output=True, text=True, env=env,
                          timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    for command in TOP_LEVEL:
        assert command in done.stdout
    done = subprocess.run([sys.executable, "main.py", "catalogs", "--name", "gaia_dr3"], cwd=ROOT,
                          capture_output=True, text=True, env=env, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["name"] == "gaia_dr3"
    missing = subprocess.run([sys.executable, "main.py", "vizier"], cwd=ROOT, capture_output=True, text=True, env=env,
                             timeout=300, check=False)
    assert missing.returncode == 2 and "usage: astrosearch vizier" in missing.stdout


def test_verify_passes(isolated: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.chdir(isolated)
    assert run(["verify"]) == 0
    out = capsys.readouterr().out
    assert "Verification Results: 14/14 tests passed" in out
    assert "No AstroSearch module shadowed" in out


def test_verify_names_a_shadowed_module(isolated: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    """A module of the same name elsewhere (Hugging Face `datasets`, a user's batch.py in the
    working directory) replaces an AstroSearch module: `verify` reports it by name."""
    import types

    import cli

    elsewhere = isolated / "site"
    elsewhere.mkdir()
    (elsewhere / "sed.py").write_text("VALUE = 1\n", encoding="utf-8")
    imposter = types.ModuleType("ai")
    imposter.__file__ = str(elsewhere / "ai" / "__init__.py")
    monkeypatch.setitem(sys.modules, "ai", imposter)  # imported from another distribution
    monkeypatch.delitem(sys.modules, "sed", raising=False)  # not imported yet: found first on sys.path
    monkeypatch.syspath_prepend(str(elsewhere))
    shadowed = cli.shadowed_modules()
    assert set(shadowed) == {"ai", "sed"}, shadowed
    assert Path(shadowed["sed"]) == elsewhere / "sed.py"
    assert cli.shadowed_modules(["models", "cli", "main"]) == {}

    monkeypatch.chdir(isolated)
    assert run(["verify"]) == 1
    out = capsys.readouterr().out
    assert "Verification Results: 14/14" not in out  # (the imposters break the CLI registration check too)
    failed = next(line for line in out.splitlines() if "No AstroSearch module shadowed" in line)
    assert "FAILED" in failed and "'ai' resolves to" in failed and "'sed' resolves to" in failed

    # When the shadowing breaks main.py's own imports, the CLI names the module instead of a traceback.
    def broken(name: str) -> Any:
        raise ImportError("cannot import name 'MetadataStore' from 'datasets'")

    monkeypatch.setattr(cli, "_module", broken)
    assert run(["verify"]) == 1
    err = capsys.readouterr().err
    assert "AstroSearch module 'ai' resolves to" in err and "AstroSearch module 'sed' resolves to" in err


# ---------------------------------------------------------------------------
# Core commands
# ---------------------------------------------------------------------------


def test_catalogs_and_benchmark(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["catalogs"]) == 0
    assert "gaia_dr3" in capsys.readouterr().out
    assert run(["catalogs", "--name", "nope"]) == 1
    assert run(["benchmark", "--rows", "200", "--format", "csv"]) == 0
    assert "Wrote 200 rows" in capsys.readouterr().out


def test_search_json_output_is_pure_json(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with upstream(*load_exchanges("3c273")):
        assert run(["search", "--name", "3C 273", "--radius", "10", "--catalogs", "simbad,gaia_dr3",
                    "--format", "json"]) == 0
    record = json.loads(capsys.readouterr().out)  # no 'Resolving ...' line mixed in
    assert record["resolved_object"]["canonical_name"] == "3C 273"
    assert record["provenance"]["association"]["target_sigma_source"] == "resolver"
    with upstream(*load_exchanges("3c273")):
        assert run(["search", "--ra", str(RA), "--dec", str(DEC), "--radius", "10", "--catalogs", "simbad"]) == 0
    out = capsys.readouterr().out
    assert "--- Crossmatch Summary ---" in out and "simbad:3C 273" in out
    # Invalid input exits 2 (as stream, dataset, batch); 1 is kept for upstream failures.
    assert run(["search", "--ra", "10"]) == 2
    assert run(["search", "--ra", "10", "--dec", "10", "--catalogs", ","]) == 2


def _no_upstream() -> respx.MockRouter:
    router = respx.mock(assert_all_called=False, assert_all_mocked=True)
    router.route().mock(side_effect=AssertionError("no upstream request expected"))
    return router


@pytest.mark.parametrize("argv", [
    ["--ra", "400", "--dec", "2"],  # was searched at RA 40 without a warning
    ["--ra", "-5", "--dec", "2"],  # was searched at RA 355
    ["--ra", "10", "--dec", "95"],
    ["--ra", "1", "--dec", "2", "--radius", "1801"],  # API_MAX_RADIUS_ARCSEC: exited 1
    ["--ra", "150.1", "--dec", "2.2", "--profile", "radio", "--catalogs", "gaia_dr3"],  # exited 0, 0 catalogs
    ["--name", "3C 273", "--ra", "1", "--dec", "2"],  # the coordinates were dropped silently
])
def test_search_invalid_input_exits_2_before_any_request(isolated: Path, capsys: pytest.CaptureFixture[str],
                                                         argv: list[str]) -> None:
    with _no_upstream():
        assert run(["search", *argv]) == 2
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and "Traceback" not in err


def test_stream_rejects_catalogs_outside_the_profile(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Finding: `stream --profile radio --catalogs gaia_dr3` streamed an empty, successful result."""
    with _no_upstream():
        assert run(["stream", "--ra", "150.1", "--dec", "2.2", "--profile", "radio", "--catalogs", "gaia_dr3"]) == 2
        assert run(["stream", "--ra", "1", "--dec", "2", "--radius", "1801"]) == 2
    assert "not in profile 'radio'" in capsys.readouterr().out


def test_lightcurve_refuses_a_name_with_coordinates(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with _no_upstream():
        assert run(["lightcurve", "--name", "RR Lyr", "--ra", "1", "--dec", "2"]) == 2
    assert "not both" in capsys.readouterr().err


def test_verify_report_is_not_interleaved_with_request_logs(capsys: pytest.CaptureFixture[str]) -> None:
    """Finding: request logs (structlog JSON, httpx 'HTTP Request') split check [11] from its PASSED."""
    assert main.run_verification()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert all(line.startswith(("=", "[", "AstroSearch", "Verification Results")) for line in lines), lines
    check = next(line for line in lines if "FastAPI REST endpoints" in line)
    assert check.endswith("PASSED"), check


def test_dataset_command(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    targets = isolated / "targets.json"
    targets.write_text(json.dumps([{"ra": RA, "dec": DEC}]), encoding="utf-8")
    out_path = isolated / "datasets" / "cli.csv"
    with upstream(*load_exchanges("3c273")):
        assert run(["dataset", "--name", "cli", "--profile", "full", "--radius", "10", "--targets", str(targets),
                    "--catalogs", "gaia_dr3,simbad", "--format", "csv", "--output", str(out_path)]) == 0
    assert "method: per-target" in capsys.readouterr().out
    with open(out_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {"match_probability", "target_probability", "contains_target"} <= set(rows[0])
    assert run(["dataset", "--name", "x", "--profile", "full", "--targets", str(isolated / "missing.json")]) == 1
    bad = isolated / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert run(["dataset", "--name", "x", "--profile", "full", "--targets", str(bad)]) == 2


# ---------------------------------------------------------------------------
# Feature commands (offline)
# ---------------------------------------------------------------------------


def test_stream_command(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with upstream(*load_exchanges("3c273")):
        assert run(["stream", "--ra", str(RA), "--dec", str(DEC), "--radius", "10", "--catalogs", "simbad"]) == 0
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert events[0]["event"] == "start" and events[-1]["event"] == "done"
    assert run(["stream", "--ra", "10"]) == 2


def test_xmatch_calibrate_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["xmatch-calibrate", "--fields", "20", "--seed", "3"]) == 0
    assert capsys.readouterr().out.strip()


def test_batch_command(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from test_batch_fixtures import batch_replay_side_effect, load_batch_exchanges

    targets = isolated / "targets.csv"
    targets.write_text("id,ra,dec\n" + "".join(f"{k},{ra},{dec}\n" for k, (ra, dec) in TARGETS.items()),
                       encoding="utf-8")
    out = isolated / "matches.csv"
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route().mock(side_effect=batch_replay_side_effect(load_batch_exchanges("canary", ["simbad"])))
        assert run(["batch", "--targets", str(targets), "--catalogs", "simbad", "--radius", "10",
                    "--out", str(out)]) == 0
    assert "simbad" in capsys.readouterr().out
    with open(out, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert any(r["target_id"] == "3c273" and r["source_id"] == "3C 273" for r in rows)
    bad = isolated / "bad.csv"
    bad.write_text("id,ra,dec\nx,10,95\n", encoding="utf-8")
    assert run(["batch", "--targets", str(bad), "--catalogs", "simbad"]) == 2


def test_skycache_commands(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store = str(isolated / "store")
    assert run(["mirror", "--catalog", "gaia_dr3", "--store", store]) == 2  # no region given
    assert run(["skycache", "status", "--store", store, "--json"]) == 0
    capsys.readouterr()
    assert run(["skycache", "cone", "--catalog", "gaia_dr3", "--ra", "10", "--dec", "10", "--radius-arcsec", "5",
                "--store", store]) == 1
    assert run(["skycache", "cone", "--catalog", "nope", "--ra", "10", "--dec", "10", "--radius-arcsec", "5",
                "--store", store]) == 2
    assert run(["skycache", "delete", "--catalog", "gaia_dr3", "--store", store]) == 1
    capsys.readouterr()
    with upstream(*load_exchanges("skycache/3c273_mirror", ["gaia_dr3"]), sesame=False, strict=True):
        assert run(["mirror", "--catalog", "gaia_dr3", "--ra", str(RA), "--dec", str(DEC), "--radius-deg", "0.2",
                    "--store", store, "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["tiles_failed"] == 0 and report["rows_stored"] > 400
    assert run(["skycache", "cone", "--catalog", "gaia_dr3", "--ra", str(RA), "--dec", str(DEC),
                "--radius-arcsec", "10", "--store", store, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)
    assert run(["skycache", "delete", "--catalog", "gaia_dr3", "--store", store]) == 0


def test_vizier_commands(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry = str(isolated / "user.yaml")
    assert run(["vizier"]) == 2
    capsys.readouterr()
    assert run(["vizier", "list", "--registry-path", registry, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["catalogs"] == {}
    with upstream(*load_exchanges("vizier/describe_2sxps"), sesame=False, strict=True, ads=True):
        assert run(["vizier", "describe", "IX/58/2sxps", "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["table_id"] == "IX/58/2sxps"
        assert run(["vizier", "add", "IX/58/2sxps", "--name", "swift_2sxps", "--registry-path", registry]) == 0
    added = capsys.readouterr().out
    assert "Registered 'swift_2sxps'" in added
    # ADS has no DOI link for the table's paper: registered anyway, the paper cited unverified.
    assert "ADS gives no DOI for" in added and "unverified" in added
    assert run(["vizier", "list", "--registry-path", registry, "--json"]) == 0
    assert list(json.loads(capsys.readouterr().out)["catalogs"]) == ["swift_2sxps"]


def test_science_commands_validate_their_input(isolated: Path) -> None:
    with upstream() as router:  # no archive is contacted for invalid input
        assert run(["sed", "--ra", "10", "--dec", "95"]) == 1  # sed: 1 for any error, 2 for missing arguments
        assert run(["sed", "--ra", "10"]) == 2
        assert run(["lightcurve"]) == 2
        assert run(["solar-system", "--ra", "10", "--dec", "10", "--epoch-mjd", "1e7"]) == 2
        assert run(["cutout", "--ra", "10", "--dec", "10"]) == 2  # no --out
        assert run(["ask", "hi"]) == 2
        assert run(["ask", "quasars near M87"]) == 3  # no Anthropic credentials
        assert run(["explain"]) == 2
        assert run(["replay", str(isolated / "missing.json")]) == 2
        assert run(["cite", "--catalogs", "nope"]) == 2
        assert run(["alerts", "show", "ZTF00nothere"]) == 1
        assert run(["alerts", "list", "--db", "mysql://nope"]) == 2
    assert not router.calls, [str(call.request.url) for call in router.calls]


def test_cutout_surveys_citations_vo_and_alert_listing(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with upstream():
        assert run(["cutout", "--list-surveys"]) == 0
        assert "dss2" in capsys.readouterr().out
        assert run(["cite", "--catalogs", "gaia_dr3,simbad", "--json"]) == 0
        bundle = json.loads(capsys.readouterr().out)
        assert bundle["unknown"] == [] and len(bundle["keys"]) >= 2
        assert run(["vo", "tables"]) == 0
        assert "astrosearch.matches" in capsys.readouterr().out
        assert run(["vo", "adql", "SELECT name FROM astrosearch.catalogs", "--format", "csv"]) == 0
        assert "gaia_dr3" in capsys.readouterr().out
        assert run(["alerts", "list", "--format", "json"]) == 0


def test_manifest_from_a_search_record_and_its_replay(isolated: Path, capsys: pytest.CaptureFixture[str]) -> None:
    record_path, manifest_path = isolated / "record.json", isolated / "manifest.json"
    with upstream(*load_exchanges("3c273")):
        assert run(["search", "--ra", str(RA), "--dec", str(DEC), "--radius", "10", "--catalogs", "simbad,gaia_dr3",
                    "--format", "json"]) == 0
        record_path.write_text(capsys.readouterr().out, encoding="utf-8")
        assert run(["manifest", "--record", str(record_path), "--no-live-release", "-o", str(manifest_path)]) == 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert set(manifest["catalogs"]) == {"simbad", "gaia_dr3"}
        assert run(["replay", str(manifest_path)]) == 0
    assert run(["cite", "--from", str(manifest_path), "--json"]) == 0
    assert "unknown" in capsys.readouterr().out


def test_run_handler_awaits_coroutines() -> None:
    async def handler(args: argparse.Namespace) -> int:
        return 7

    assert main.run_handler(handler, argparse.Namespace()) == 7
    assert main.run_handler(lambda args: None, argparse.Namespace()) == 0


def command_ids() -> Iterator[str]:  # pragma: no cover - documentation helper
    yield from (" ".join(p) for p in COMMANDS)

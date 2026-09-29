"""The ``astrosearch`` command line: a light entry point that imports only what a command needs.

Importing every feature module costs about ten seconds (the science stack, the Anthropic SDK,
astropy's cosmology ...), so the parser is built lazily: the core commands are defined here,
and a feature command's module is imported only when that command runs. Every other feature
command appears in ``--help`` from :data:`FEATURE_COMMANDS` (module and help text), which
``tests/test_integration_cli.py`` checks against what the modules register. ``astrosearch
--help`` therefore starts in a fraction of a second, and ``astrosearch skycache cone ...``
imports the core plus :mod:`skycache` only.

:func:`build_parser` with ``full=True`` (``main.build_parser``) registers every module, as the
``verify`` suite and the tests use it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import sys
from collections.abc import Callable, Sequence
from typing import Any

DESCRIPTION = "AstroSearch: astronomical catalog cross-matching, time-domain, imaging and dataset engine."

# The core commands, defined here (handlers in main.py), in help order.
CORE_COMMANDS = ("serve", "search", "dataset", "catalogs", "benchmark", "verify")

# Feature modules in help order, each with the subcommands its register_cli() adds and their help.
FEATURE_COMMANDS: dict[str, dict[str, str]] = {
    "streaming": {"stream": "Stream a crossmatch catalogue by catalogue (JSON lines or SSE)"},
    "astrometry": {"xmatch-calibrate": "Monte-Carlo completeness/purity/calibration of the Bayesian crossmatch"},
    "batch": {"batch": "Batch crossmatch a target list (TAP upload / CDS XMatch / cones)"},
    "skycache": {"mirror": "Mirror a sky region of a catalog into the local HATS sky cache",
                 "skycache": "Inspect or query the local sky cache"},
    "vizier": {"vizier": "Discover, describe and register any VizieR table"},
    "sed": {"sed": "Multi-wavelength SED, classification and redshift for one object"},
    "timedomain": {"lightcurve": "Multi-survey light curves, variability metrics and period search",
                   "solar-system": "Known asteroids/comets near a position at an epoch (IMCCE SkyBoT)"},
    "imaging": {"cutout": "Download a sky cutout (PNG/JPEG/FITS) from a HiPS survey via CDS hips2fits"},
    "ai": {"ask": "Compile a natural-language request into a validated query (Claude)",
           "explain": "Explain an object from SIMBAD/crossmatch facts with citations (Claude)"},
    "provenance": {"replay": "Re-run a provenance manifest and diff the science",
                   "cite": "Acknowledgements and BibTeX for catalogs/services used",
                   "manifest": "Build a provenance manifest from a record or a new search"},
    "vo_server": {"vo": "Run Virtual Observatory (Cone Search / ADQL) queries locally"},
    "alerts": {"alerts": "Ingest live transient alerts (ALeRCE, Fink) and crossmatch them"},
}

COMMAND_MODULES: dict[str, str] = {command: module for module, commands in FEATURE_COMMANDS.items()
                                   for command in commands}


def _module(name: str) -> Any:
    """A feature (or core) module of this package, imported on demand."""
    package = __package__ or ""
    return importlib.import_module(f"{package}.{name}" if package else name)


def add_core_commands(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    """The core subcommands (serve, search, dataset, catalogs, benchmark, verify)."""
    serve_parser = subparsers.add_parser("serve", help="Launch the FastAPI REST server and web UI")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable automatic code reloading")

    search_parser = subparsers.add_parser("search", help="Execute single astronomical search")
    search_parser.add_argument("--ra", type=float, help="Right ascension in degrees [0, 360)")
    search_parser.add_argument("--dec", type=float, help="Declination in degrees [-90, 90]")
    search_parser.add_argument("--name", type=str, help="Astronomical object name to resolve (e.g. M87, Vega)")
    search_parser.add_argument("--radius", type=float, default=3.0, help="Search radius in arcseconds (default: 3.0)")
    search_parser.add_argument("--profile", type=str, help="Catalog profile: optical, infrared, radio, xray, etc.")
    search_parser.add_argument("--catalogs", type=str, help="Comma-separated registry catalog names (default: all)")
    search_parser.add_argument("--epoch", type=float,
                               help="Julian year of --ra/--dec (with --name: move the resolved position to it)")
    search_parser.add_argument("--pm-ra", type=float, help="Target proper motion in RA*cos(Dec), mas/yr")
    search_parser.add_argument("--pm-dec", type=float, help="Target proper motion in Dec, mas/yr")
    search_parser.add_argument("--parallax", type=float, help="Target parallax (mas)")
    search_parser.add_argument("--format", choices=["json", "summary"], default="summary", help="Output format")

    dataset_parser = subparsers.add_parser("dataset", help="Generate a filtered crossmatched dataset")
    dataset_parser.add_argument("--name", required=True, help="Dataset identification name")
    dataset_parser.add_argument("--profile", required=True, help="Catalog profile (e.g. stellar, optical)")
    dataset_parser.add_argument("--radius", type=float, default=5.0, help="Search radius in arcseconds")
    dataset_parser.add_argument("--targets", required=True, help="Path to JSON file containing array of targets")
    dataset_parser.add_argument("--catalogs", type=str,
                                help="Comma-separated registry catalog names (each must belong to --profile)")
    dataset_parser.add_argument("--min-confidence", type=float, default=0.0,
                                help="Minimum posterior probability of a row (default 0)")
    dataset_parser.add_argument("--count-threshold", type=int, default=1,
                                help="Minimum detections per crossmatched object (default 1)")
    dataset_parser.add_argument("--format", choices=["parquet", "csv", "json", "fits"], default="parquet")
    dataset_parser.add_argument("--output", help="Path of the export file to write (any unused path whose extension "
                                                 "matches --format; default: DATASET_STORAGE_PATH/<id>.<format>)")

    catalogs_parser = subparsers.add_parser("catalogs", help="List and inspect configured catalog definitions")
    catalogs_parser.add_argument("--name", help="Specific catalog to inspect")

    bench_parser = subparsers.add_parser("benchmark", help="Benchmark streaming dataset export performance")
    bench_parser.add_argument("--rows", type=int, default=50000, help="Number of synthetic records to stream")
    bench_parser.add_argument("--format", choices=["parquet", "csv", "json", "fits"], default="parquet")

    subparsers.add_parser("verify", help="Run the offline self-test and verification suite")


def build_parser(command: str | None = None, *, full: bool = False) -> argparse.ArgumentParser:
    """The ``astrosearch`` parser: the core commands, the full parser of ``command``'s feature
    module (every module's with ``full``), and a help-only entry for every other feature command."""
    parser = argparse.ArgumentParser(prog="astrosearch", description=DESCRIPTION)
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")
    add_core_commands(subparsers)
    selected = COMMAND_MODULES.get(command or "")
    for module_name, commands in FEATURE_COMMANDS.items():
        if full or module_name == selected:
            _module(module_name).register_cli(subparsers)
        else:
            for name, help_text in commands.items():
                subparsers.add_parser(name, help=help_text, description=f"{help_text} (see: astrosearch {name} -h)")
    return parser


def run_handler(handler: Callable[[argparse.Namespace], Any], args: argparse.Namespace) -> int:
    """Run a subcommand handler; a coroutine is awaited. Returns the exit status."""
    result = handler(args)
    if inspect.isawaitable(result):
        result = asyncio.run(result)  # type: ignore[arg-type]
    return int(result or 0)


def log_to_stderr() -> None:
    """Send structlog output (the feature modules' diagnostics) to stderr: stdout carries the
    command's result (JSON, CSV, a summary) and must stay parseable. The API configures its
    own (JSON) logging when it is imported."""
    try:
        import structlog
    except ImportError:  # pragma: no cover - a dependency
        return
    if not structlog.is_configured():
        structlog.configure(logger_factory=structlog.PrintLoggerFactory(file=sys.stderr))


def _command_of(argv: Sequence[str]) -> str | None:
    """The subcommand named on the command line (the top-level parser has no options that take a value)."""
    return next((arg for arg in argv if not arg.startswith("-")), None)


def main(argv: list[str] | None = None) -> None:
    """Entry point of the ``astrosearch`` console script."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    log_to_stderr()
    parser = build_parser(_command_of(arguments))
    args = parser.parse_args(arguments)

    if args.command == "serve":
        import uvicorn

        print(f"Starting AstroSearch REST API and web UI on http://{args.host}:{args.port}")
        app_path = f"{__package__}.api:app" if __package__ else "api:app"
        uvicorn.run(app_path, host=args.host, port=args.port, reload=args.reload)
        return
    if args.command is None:
        parser.print_help()
        return
    if args.command in CORE_COMMANDS:
        core = _module("main")
        if args.command == "verify":
            sys.exit(0 if core.run_verification() else 1)
        handler = {"search": core._cmd_search, "dataset": core._cmd_dataset, "catalogs": core._cmd_catalogs,
                   "benchmark": core._cmd_benchmark}[args.command]
    else:
        handler = getattr(args, "handler", None)
        if handler is None:
            parser.print_help()
            return
    try:
        status = run_handler(handler, args)
    except KeyboardInterrupt:
        status = 130
    if status:
        sys.exit(status)


if __name__ == "__main__":
    main()

"""Packaging: what `pip install .[dev]` declares and installs.

* every module the application imports from this repository is shipped (py-modules);
* the dev extra declares every third-party package the offline suite imports (lxml was missing);
* the dependency floors are installable on Python 3.12 and keep the invariants the code relies on;
* a wheel's web UI is found when its data files land outside sys.prefix (--user, --prefix, distro schemes).
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
MODULES = {p.stem for p in ROOT.glob("*.py")}


def _requirements(names: list[str]) -> dict[str, Requirement]:
    out = {}
    for text in names:
        req = Requirement(text)
        out[re.sub(r"[-_.]+", "-", req.name).lower()] = req
    return out


def _floor(req: Requirement) -> Version:
    floors = [Version(s.version) for s in req.specifier if s.operator == ">="]
    assert floors, f"{req} has no lower bound"
    return max(floors)


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def test_every_application_module_is_shipped() -> None:
    """A module that main/api import but the wheel leaves out breaks the installed console script
    (cli.py, the console script's entry point, was missing)."""
    shipped = set(PYPROJECT["tool"]["setuptools"]["py-modules"])
    import cli

    # The console script's module, plus what cli.py imports on demand (importlib, not an import statement).
    lazy = {"main", *cli.FEATURE_COMMANDS}
    needed = {"cli"} | lazy
    pending = ["cli", "api", *sorted(lazy)]
    while pending:
        module = pending.pop()
        for name in _imports(ROOT / f"{module}.py") & MODULES:
            if name not in needed:
                needed.add(name)
                pending.append(name)
    assert not needed - shipped, f"imported by the application but not in py-modules: {sorted(needed - shipped)}"
    assert PYPROJECT["project"]["scripts"]["astrosearch"] == "cli:main"


def test_verify_checks_every_shipped_module_for_shadowing() -> None:
    import cli

    assert list(cli.DISTRIBUTION_MODULES) == PYPROJECT["tool"]["setuptools"]["py-modules"]


def test_dev_extra_declares_what_the_offline_suite_imports() -> None:
    dev = _requirements(PYPROJECT["project"]["optional-dependencies"]["dev"])
    runtime = _requirements(PYPROJECT["project"]["dependencies"])
    # Imported unguarded by the offline tests (tests/test_vo_server.py validates against the IVOA schemas).
    for name in ("lxml", "pytest", "pytest-asyncio", "respx", "httpx-sse", "pyvo", "cdshealpix", "matplotlib"):
        assert name in dev or name in runtime, name
    import_name = {"pytest-asyncio": "pytest_asyncio", "httpx-sse": "httpx_sse"}
    used: set[str] = set()
    for path in (ROOT / "tests").glob("test_*.py"):
        used |= _imports(path)
    for name in dev:
        if name in {"pytest-cov", "ruff", "mypy", "pytest-asyncio"}:
            continue
        assert import_name.get(name, name) in used, f"dev extra {name} is not used by the tests"


def test_dependency_floors_install_on_python_312() -> None:
    """Floors below these have no Python 3.12 wheel and fail to build (scipy 1.11.1, PyYAML 6.0.0,
    cdshealpix 0.6.0), or break an invariant the code relies on (astropy < 8: the chunked fast
    Lomb-Scargle of timedomain.ls_power differs from the single-grid one by up to 7e-3). astropy 8
    needs numpy 2, so the compiled packages start at their first numpy-2 builds."""
    runtime = _requirements(PYPROJECT["project"]["dependencies"])
    dev = _requirements(PYPROJECT["project"]["optional-dependencies"]["dev"])
    minimum = {"scipy": "1.13.0", "pyyaml": "6.0.1", "astropy": "8.0", "numpy": "2.0", "pandas": "2.2.2",
               "pyarrow": "16.0.0", "pillow": "10.1", "pyerfa": "2.0.1.3", "astropy-healpix": "1.0.3"}
    for name, version in minimum.items():
        assert _floor(runtime[name]) >= Version(version), (name, str(runtime[name]))
    assert _floor(dev["cdshealpix"]) >= Version("0.7")
    assert _floor(dev["matplotlib"]) >= Version("3.9")  # the first numpy-2 build
    assert _floor(dev["lxml"]) >= Version("5.0")
    assert _floor(dev["pyvo"]) >= Version("1.5.3")  # 1.5.0-1.5.2 fail on the VOSI capabilities
    assert PYPROJECT["project"]["requires-python"] == ">=3.12"


def test_installed_astropy_meets_the_declared_floor() -> None:
    import astropy
    import numpy

    runtime = _requirements(PYPROJECT["project"]["dependencies"])
    assert Version(astropy.__version__) >= _floor(runtime["astropy"])
    assert Version(numpy.__version__) >= _floor(runtime["numpy"])


def test_web_ui_of_a_user_or_prefix_scheme_install_is_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A wheel installed with --user (or into a distro prefix scheme) puts share/astrosearch/web under
    that scheme's data path, not sys.prefix: the UI must still be served."""
    import sysconfig

    import api

    user_base = tmp_path / "userbase"
    web = user_base / "share" / "astrosearch" / "web"
    web.mkdir(parents=True)
    (web / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.delenv("ASTROSEARCH_WEB_DIR", raising=False)
    monkeypatch.setattr(api.imaging, "WEB_DIR", tmp_path / "not-a-checkout" / "web")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    real_get_path = sysconfig.get_path

    def get_path(name: str, scheme: str | None = None, *args, **kwargs):
        if name == "data" and scheme and "user" in scheme:
            return str(user_base)
        if name == "data":
            return str(tmp_path / "venv")
        return real_get_path(name, scheme, *args, **kwargs)

    monkeypatch.setattr(sysconfig, "get_path", get_path)
    import site

    monkeypatch.setattr(site, "USER_BASE", str(tmp_path / "other-user-base"), raising=False)
    assert api.web_ui_dir() == web
    # Without it, nothing is found (the UI is then reported missing, not served from elsewhere).
    (web / "index.html").unlink()
    found = api.web_ui_dir()
    assert found is None or found != web


def test_the_installed_console_script_runs_outside_the_checkout(tmp_path: Path) -> None:
    """`astrosearch` as installed (the console-script entry point cli:main), run from a directory
    without the sources, not `python main.py` from the checkout."""
    import importlib.metadata
    import json
    import os
    import subprocess

    try:
        entry = next(e for e in importlib.metadata.distribution("astrosearch").entry_points
                     if e.group == "console_scripts" and e.name == "astrosearch")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("astrosearch is not installed in this environment (pip install -e '.[dev]')")
    assert entry.value == "cli:main"
    script = Path(sys.executable).parent / ("astrosearch.exe" if os.name == "nt" else "astrosearch")
    assert script.is_file(), f"the console script is missing next to {sys.executable}"
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1"}
    done = subprocess.run([str(script), "--help"], cwd=tmp_path, capture_output=True, text=True, env=env,
                          timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    assert "verify" in done.stdout and "search" in done.stdout
    done = subprocess.run([str(script), "catalogs", "--name", "gaia_dr3"], cwd=tmp_path, capture_output=True,
                          text=True, env=env, timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["name"] == "gaia_dr3"

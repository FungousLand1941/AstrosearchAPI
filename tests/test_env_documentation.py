"""Every environment variable the code reads is documented (DOCUMENTATION.md section 9), and the
example configuration does not silently change behaviour.

The variables are collected from the source with the AST, not from a hand-kept list: direct
``os.getenv``/``os.environ`` reads, module-level name constants passed to them, and helper
functions that forward one of their parameters to ``os.getenv`` (``_env_int(name, default)``,
``Settings.__init__.val(name, default)``), to any depth.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODULES = sorted(p for p in ROOT.glob("*.py"))
ENV_NAME = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")


def _is_env_read(call: ast.Call) -> bool:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "getenv"
    if isinstance(func, ast.Attribute):
        if func.attr == "getenv":
            return True
        if func.attr in {"get", "pop", "setdefault"}:
            owner = func.value
            return (isinstance(owner, ast.Attribute) and owner.attr == "environ") or (
                isinstance(owner, ast.Name) and owner.id == "environ")
    return False


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _constants(tree: ast.Module) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value.value
    return out


def _literal(node: ast.AST | None, constants: dict[str, str]) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    return None


def _functions(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


# Reads whose variable name is computed at run time: {(module, enclosing function): the names it can read}.
# The collector fails on any computed read that is not listed here, so a new one cannot go unnoticed.
DYNAMIC_READS = {
    ("batch.py", "_chunk_size"): {"BATCH_CHUNK_SIMBAD", "BATCH_CHUNK_VIZIER", "BATCH_CHUNK_IRSA",
                                 "BATCH_CHUNK_HEASARC", "BATCH_CHUNK_XMATCH"},
    ("vo_server.py", "_pool"): {"VO_CPU_THREADS", "VO_CROSSMATCH_THREADS"},
}


def _enclosing(tree: ast.Module) -> dict[int, str]:
    """{id(call node): name of the innermost enclosing function ('<module>' at top level)}."""
    owner: dict[int, str] = {}

    def visit(node: ast.AST, current: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else current
            owner[id(child)] = name
            visit(child, name)

    visit(tree, "<module>")
    return owner


@functools.lru_cache(maxsize=1)
def _collect() -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    parsed = {path.name: ast.parse(path.read_text(encoding="utf-8")) for path in MODULES}
    constants = {name: _constants(tree) for name, tree in parsed.items()}
    # Helper functions that pass one of their parameters on to os.getenv: {function name: parameter index}.
    wrappers: dict[str, int] = {}
    changed = True
    while changed:
        changed = False
        for tree in parsed.values():
            for func in _functions(tree):
                params = [a.arg for a in func.args.posonlyargs + func.args.args]
                offset = 1 if params and params[0] in {"self", "cls"} else 0
                for call in ast.walk(func):
                    if not isinstance(call, ast.Call) or func.name in wrappers:
                        continue
                    index = 0 if _is_env_read(call) else wrappers.get(_called_name(call) or "")
                    if index is None or len(call.args) <= index:
                        continue
                    arg = call.args[index]
                    if isinstance(arg, ast.Name) and arg.id in params:
                        wrappers[func.name] = params.index(arg.id) - offset
                        changed = True
    found: dict[str, set[str]] = {}
    dynamic: set[tuple[str, str]] = set()
    for module, tree in parsed.items():
        owner = _enclosing(tree)
        for call in ast.walk(tree):
            if not isinstance(call, ast.Call):
                continue
            index = 0 if _is_env_read(call) else wrappers.get(_called_name(call) or "")
            if index is None or len(call.args) <= index:
                continue
            arg = call.args[index]
            name = _literal(arg, constants[module])
            if name is not None:
                if ENV_NAME.match(name):
                    found.setdefault(name, set()).add(module)
                continue
            function = owner.get(id(call), "<module>")
            if function in wrappers:
                continue  # the wrapper's own forwarding read, resolved at its call sites
            dynamic.add((module, function))
        # os.environ["NAME"]
        for node in ast.walk(tree):
            if (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "environ"):
                name = _literal(node.slice, constants[module])
                if name and ENV_NAME.match(name):
                    found.setdefault(name, set()).add(module)
    return found, dynamic


def environment_variables() -> dict[str, set[str]]:
    """{variable: {module, ...}} for every environment variable the application modules read."""
    found, dynamic = _collect()
    found = {name: set(modules) for name, modules in found.items()}
    unlisted = dynamic - set(DYNAMIC_READS)
    assert not unlisted, f"environment reads with a computed name; list their variables in DYNAMIC_READS: {unlisted}"
    for (module, _function), names in DYNAMIC_READS.items():
        for name in names:
            found.setdefault(name, set()).add(module)
    return found


def test_the_collector_finds_direct_wrapped_and_constant_reads() -> None:
    names = environment_variables()
    # Direct reads, a Settings.val(...) wrapper, a batch/vo_server/skycache helper and a module constant.
    for expected in ("LOG_LEVEL", "API_MAX_RADIUS_ARCSEC", "REQUEST_TIMEOUT_SECONDS", "BATCH_CHUNK_SIMBAD",
                     "VO_CPU_THREADS", "SKYCACHE_PATH", "HIPS2FITS_URLS", "TIMEDOMAIN_REDIS_TIMEOUT_S"):
        assert expected in names, (expected, sorted(names))
    assert len(names) > 100


def test_every_environment_variable_is_documented() -> None:
    documentation = (ROOT / "DOCUMENTATION.md").read_text(encoding="utf-8")
    section = documentation[documentation.index("## 9"):]
    section = section[:section.index("\n## ", 1)]
    missing = {name: sorted(modules) for name, modules in environment_variables().items()
               if not re.search(rf"`{re.escape(name)}`", section)}
    assert not missing, f"environment variables read by the code but not documented in DOCUMENTATION.md section 9: {missing}"


def _example_settings() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def test_env_example_names_only_variables_the_code_reads() -> None:
    read = environment_variables()
    active = set(_example_settings())
    commented = set(re.findall(r"^#\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(encoding="utf-8"), re.MULTILINE))
    unknown = sorted((active | commented) - set(read))
    assert not unknown, f".env.example sets variables no module reads: {unknown}"


def test_env_example_defaults_do_not_change_the_catalog_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """.env.example says its active values are the defaults; loading it (docker --env-file) must not
    cap the catalogs' own 60-90 s timeouts (an explicit REQUEST_TIMEOUT_SECONDS would)."""
    import models

    example = _example_settings()
    assert "REQUEST_TIMEOUT_SECONDS" not in example and "CATALOG_TIMEOUT_CAP_SECONDS" not in example
    for key in ("REQUEST_TIMEOUT_SECONDS", "CATALOG_TIMEOUT_CAP_SECONDS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in example.items():
        monkeypatch.setenv(key, value)
    default = models.Settings()
    assert default.catalog_timeout_cap_seconds is None
    assert default.max_radius_arcsec == 1800.0

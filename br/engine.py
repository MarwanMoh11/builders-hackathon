"""Run sqlglot tests against a patched sqlglot, in memory, in forked children.

The parent imports sqlglot and the test modules once. For each commit a child is forked; it
applies the patch either by swapping the code of the changed functions (cheap) or by
re-importing every sqlglot and test module from patched sources (import-time changes), runs the
candidate tests with unittest, and reports which failed.
"""

from __future__ import annotations

import builtins
import gc
import importlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pickle
import signal
import sys
import time
import types
import unittest
from concurrent.futures import Future

ROOT = os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ------------------------------------------------------------------ process pools, in-process


class InlineExecutor:
    """ProcessPoolExecutor stand-in: runs work in-process, with the pickle round trip a real
    pool does (pickling runs sqlglot's serializer). Tests then see the patched in-memory code."""

    def __init__(self, *args, initializer=None, initargs=(), **kwargs):
        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:  # noqa: BLE001
            future.set_exception(error)
        return future

    def map(self, fn, *iterables, **kwargs):
        return [self.submit(fn, *args).result() for args in zip(*iterables)]

    def shutdown(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def module_name(relpath: str) -> str:
    name = relpath[:-3].replace("/", ".")
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    return name


def is_ours(name: str) -> bool:
    return name in ("sqlglot", "tests") or name.startswith("sqlglot.") or name.startswith("tests.")


# ------------------------------------------------------------------ parent setup

FUNCS_BY_FILE: dict[str, list[types.FunctionType]] = {}
TESTS: list[dict] = []


def setup_parent(tests: list[dict], import_tests: bool = True) -> None:
    import concurrent.futures

    os.environ.setdefault("SKIP_INTEGRATION", "0")
    concurrent.futures.ProcessPoolExecutor = InlineExecutor
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    TESTS[:] = tests

    import pkgutil

    # Import order matters: some dialects mutate shared class state at import time. pytest
    # imports the test files first, in collection order (the order of tests.json), so do that.
    if import_tests:
        seen = []
        for t in tests:
            if t["file"] not in seen:
                seen.append(t["file"])
        for relpath in seen:
            try:
                importlib.import_module(module_name(relpath))
            except Exception as error:  # noqa: BLE001
                log(f"import {relpath} failed: {error!r}")
    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except Exception as error:  # noqa: BLE001
            log(f"import {info.name} failed: {error!r}")
    build_dependents()
    for obj in gc.get_objects():
        if type(obj) is types.FunctionType:
            filename = obj.__code__.co_filename
            if filename.startswith(ROOT):
                FUNCS_BY_FILE.setdefault(filename, []).append(obj)
    gc.collect()
    gc.freeze()


# ------------------------------------------------------------------ applying patches in a child


def _code_key(code: types.CodeType):
    return (code.co_qualname, code.co_firstlineno)


def hot_patch(filename: str, analysis) -> bool:
    """Swap the code of changed functions in every live function object. False if some changed
    function cannot be found alive (the caller then re-imports instead)."""
    from pyanalyze import is_function

    targets: dict = {}
    required: set = set()
    for change in analysis.changes:
        if change.kind != "function":
            continue
        chain_old = change.path + [change.old]
        chain_new = change.new_path + [change.new]
        for depth, (old, new) in enumerate(zip(chain_old, chain_new)):
            if is_function(old):
                targets[_code_key(old)] = new
                # a function right under a module or class body must exist as a live object
                if depth > 0 and not is_function(chain_old[depth - 1]):
                    required.add(_code_key(old))
    found: set = set()
    for func in FUNCS_BY_FILE.get(filename, ()):
        key = _code_key(func.__code__)
        new = targets.get(key)
        if new is not None:
            if len(new.co_freevars) != len(func.__code__.co_freevars):
                return False
            func.__code__ = new
            found.add(key)
    return required <= found


class PatchedLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, source):
        super().__init__(fullname, path)
        self._source = source

    def get_data(self, path):
        if path == self.path:
            return self._source.encode("utf-8")
        return super().get_data(path)

    def get_code(self, fullname):
        return compile(self._source, self.path, "exec", dont_inherit=True)


class PatchedFinder:
    def __init__(self, sources: dict[str, str]):
        # module name -> (absolute filename, source)
        self.by_module = {module_name(rel): (os.path.join(ROOT, rel), src) for rel, src in sources.items()}

    def find_spec(self, fullname, path=None, target=None):
        found = self.by_module.get(fullname)
        if found is None:
            return None
        filename, source = found
        loader = PatchedLoader(fullname, filename, source)
        is_package = filename.endswith("__init__.py")
        return importlib.util.spec_from_file_location(
            fullname,
            filename,
            loader=loader,
            submodule_search_locations=[os.path.dirname(filename)] if is_package else None,
        )


DEPENDENTS: dict[str, set[str]] = {}  # module -> modules whose import-time state refers to it


def _owner(value) -> str | None:
    if isinstance(value, types.ModuleType):
        return value.__name__
    owner = getattr(value, "__module__", None) if isinstance(value, (type, types.FunctionType)) else None
    if owner is None:
        owner = getattr(type(value), "__module__", None)
    return owner if isinstance(owner, str) else None


def build_dependents() -> None:
    """Who refers to whom, read off live module (and class) namespaces after import.

    A module that binds nothing from another cannot hold state built from it, so re-importing a
    changed module only needs its dependents re-imported. A package's submodule attributes do not
    count: re-importing a submodule rebinds that attribute by itself."""
    ours = {n: m for n, m in sys.modules.items() if is_ours(n) and m is not None}
    for name, module in ours.items():
        refs: set[str] = set()
        namespaces = [vars(module)]
        for value in list(vars(module).values()):
            if isinstance(value, type) and value.__module__ == name:
                namespaces.append(vars(value))
        for namespace in namespaces:
            for key, value in list(namespace.items()):
                owner = _owner(value)
                if isinstance(value, types.ModuleType) and owner == f"{name}.{key}":
                    continue  # package -> its own submodule attribute
                if owner is not None and owner != name and owner in ours:
                    refs.add(owner)
                # a container built elsewhere (e.g. ENV) carries its maker's objects
                if isinstance(value, (dict, list, tuple, set, frozenset)):
                    items = value.values() if isinstance(value, dict) else value
                    for count, item in enumerate(items):
                        if count >= 64:
                            break
                        item_owner = _owner(item)
                        if item_owner is not None and item_owner != name and item_owner in ours:
                            refs.add(item_owner)
        for ref in refs:
            DEPENDENTS.setdefault(ref, set()).add(name)


def dependent_closure(names) -> set[str]:
    out = set(names)
    stack = list(names)
    while stack:
        for dep in DEPENDENTS.get(stack.pop(), ()):
            if dep not in out:
                out.add(dep)
                stack.append(dep)
    return out


def reimport(sources: dict[str, str], everything: bool = False):
    """Drop the changed modules and every module depending on them, and import them again,
    patched sources first in line. Returns (old modules, new modules, names that failed)."""
    changed = {module_name(rel) for rel in sources}
    if everything or not DEPENDENTS:
        selected = None
    else:
        selected = dependent_closure(changed)
    old = {
        n: m
        for n, m in sys.modules.items()
        if is_ours(n) and m is not None and (selected is None or n in selected)
    }
    for n in old:
        del sys.modules[n]
    importlib.invalidate_caches()
    sys.meta_path.insert(0, PatchedFinder(sources))
    failed = []
    for name in old:  # first-import order, as sys.modules remembers it
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001
            failed.append(name)
            log(f"re-import {name} failed: {error!r}")
    # A re-imported package only gets attributes for submodules imported after it: re-attach the
    # submodules that were kept.
    for name, module in list(sys.modules.items()):
        if module is None or not is_ours(name) or "." not in name:
            continue
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        # only when missing: a package may rebind a name its submodule also has (sqlglot.diff)
        if parent is not None and parent_name in old and not hasattr(parent, child):
            try:
                setattr(parent, child, module)
            except (AttributeError, TypeError):
                pass
    _repoint_registries(set(old))
    new = {n: sys.modules[n] for n in old if n in sys.modules}
    return old, new, failed


_DROP = object()


def _resolve_new(obj, purged: set[str]):
    """The re-imported counterpart of a class or function from a purged module (None if obj is
    not from one, _DROP if its module no longer imports or no longer defines it)."""
    if not isinstance(obj, type):
        return None
    module_name_ = getattr(obj, "__module__", None)
    if module_name_ not in purged:
        return None
    current = sys.modules.get(module_name_)
    if current is None:
        return _DROP  # its module no longer imports
    for part in getattr(obj, "__qualname__", "").split("."):
        current = getattr(current, part, None)
        if current is None:
            return None  # not reachable by name: leave it alone
    return current if isinstance(current, type) else None


def _repoint_registries(purged: set[str]) -> None:
    """Kept modules may hold registries (e.g. Dialect._classes) filled with classes from modules
    we re-imported: point them at the new classes, and drop entries whose module now fails."""
    for name, module in list(sys.modules.items()):
        if module is None or not is_ours(name) or name in purged:
            continue
        namespaces = [vars(module)]
        for value in list(vars(module).values()):
            if isinstance(value, type) and value.__module__ == name:
                namespaces.append(vars(value))
        for namespace in namespaces:
            for value in list(namespace.values()):
                if not isinstance(value, dict) or not value:
                    continue
                for key, item in list(value.items()):
                    replacement = _resolve_new(item, purged)
                    if replacement is None or replacement is item:
                        continue
                    if replacement is _DROP:
                        del value[key]
                    else:
                        value[key] = replacement


def install_overlays(overlays: dict[str, str]) -> None:
    if not overlays:
        return
    os.makedirs("/tmp/br-overlay", exist_ok=True)
    redirect = {}
    for i, (rel, text) in enumerate(overlays.items()):
        target = f"/tmp/br-overlay/{os.getpid()}-{i}"
        with open(target, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        redirect[os.path.realpath(os.path.join(ROOT, rel))] = target
    real_open = io.open

    def opener(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)):
            try:
                key = os.path.realpath(os.fspath(file))
            except (TypeError, ValueError):
                key = None
            if key in redirect:
                file = redirect[key]
        return real_open(file, *args, **kwargs)

    builtins.open = opener
    io.open = opener


# ------------------------------------------------------------------ running tests


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


CLASS_SETUP_CPU: dict[str, float] = {}


def run_tests(numbers, budget_seconds: float | None = None) -> tuple[set[int], dict[int, float]]:
    """Run tests by number; return (failing numbers, CPU seconds per test). Class setup CPU is
    recorded in CLASS_SETUP_CPU."""
    signal.signal(signal.SIGALRM, _alarm)
    groups: dict[tuple[str, str], list[dict]] = {}
    for n in sorted(numbers):
        t = TESTS[n]
        parts = t["id"].split("::")
        groups.setdefault((module_name(t["file"]), parts[1] if len(parts) > 2 else ""), []).append(t)
    failing: set[int] = set()
    cpu: dict[int, float] = {}
    for (modname, clsname), items in groups.items():
        try:
            module = importlib.import_module(modname)
            cls = getattr(module, clsname)
        except BaseException:  # noqa: BLE001 - import errors fail every test of the file
            failing.update(t["number"] for t in items)
            continue
        if getattr(cls, "__unittest_skip__", False):
            continue
        setup_started = time.process_time()
        try:
            cls.setUpClass()
        except BaseException:  # noqa: BLE001
            failing.update(t["number"] for t in items)
            try:
                cls.doClassCleanups()
            except BaseException:  # noqa: BLE001
                pass
            continue
        CLASS_SETUP_CPU[f"{modname}::{clsname}"] = time.process_time() - setup_started
        for t in items:
            method = t["id"].split("::")[-1]
            result = unittest.TestResult()
            started = time.process_time()
            limit = max(5.0, 15.0 * t["seconds"])
            signal.setitimer(signal.ITIMER_REAL, limit)
            try:
                cls(method).run(result)
            except _Timeout:
                result.errors.append((None, "timeout"))
            except BaseException as error:  # noqa: BLE001
                result.errors.append((None, repr(error)))
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            cpu[t["number"]] = time.process_time() - started
            if result.failures or result.errors or result.unexpectedSuccesses:
                failing.add(t["number"])
        try:
            cls.tearDownClass()
        except BaseException:  # noqa: BLE001
            pass
        try:
            cls.doClassCleanups()
        except BaseException:  # noqa: BLE001
            pass
    return failing, cpu

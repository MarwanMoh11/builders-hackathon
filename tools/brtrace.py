"""pytest plugin that records, per test, which sqlglot lines ran and which data files were opened.

Load with `-p brtrace` (tools/ on PYTHONPATH). Writes one JSON file per process to $BR_OUT.
Uses sys.monitoring (Python 3.12+): each line location fires once per test, then is disabled
until the next test restarts events, so the overhead stays low.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

ROOT = os.path.realpath(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")) + os.sep
OUT = os.environ.get("BR_OUT", "/tmp/brtrace")
WORKER = os.environ.get("PYTEST_XDIST_WORKER", "main")

mon = sys.monitoring
TOOL = mon.COVERAGE_ID

_files: dict[str, int] = {}
_current_lines: set = set()
_current_opens: set = set()
_results: dict = {}
_collect_lines: set = set()


def _fid(filename: str) -> int:
    fid = _files.get(filename)
    if fid is None:
        fid = _files[filename] = len(_files)
    return fid


def _on_line(code, line):
    filename = code.co_filename
    if filename.startswith(ROOT):
        _current_lines.add((_fid(filename), line))
    return mon.DISABLE


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = args[0]
        if isinstance(path, bytes):
            path = path.decode(errors="replace")
        if path.startswith(ROOT):
            rel = path[len(ROOT):]
            # .py files under sqlglot/ and tests/ are opened by imports; anything else (data,
            # sqlglotc/setup.py run by a test, pyproject.toml ...) is a file a test depends on
            if rel.endswith(".pyc") or (rel.endswith(".py") and rel.startswith(("sqlglot/", "tests/"))):
                return
            _current_opens.add(rel)


class InlineExecutor:
    """Stands in for ProcessPoolExecutor so pool work runs, and is traced, inside the test."""

    # Everything crosses a pickle round trip, as it would between processes: pickling runs
    # sqlglot's own serializer (sqlglot/serde.py), so tests depend on it.

    def __init__(self, *args, initializer=None, initargs=(), **kwargs):
        import pickle

        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, *args, **kwargs):
        import pickle
        from concurrent.futures import Future

        future = Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:  # noqa: BLE001 - mirrored into the future like a pool
            future.set_exception(error)
        return future

    def shutdown(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
        return False


def _preimport_sqlglot():
    # Import every sqlglot module up front so import-time lines land in "collect" instead of
    # being credited to whichever test happened to import a module lazily first.
    import importlib
    import pkgutil

    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except Exception:
            pass


_attr_keys: dict[str, int] = {}
_current_attrs: set = set()
# Read straight from a class __dict__ by sqlglot's metaclasses, so they must stay unwrapped.
_ATTR_SKIP = {"INVERSE_TIME_MAPPING", "tokenizer_class", "jsonpath_tokenizer_class",
              "parser_class", "generator_class"}


class _Rec:
    """Non-data descriptor standing in for a class constant; records which test reads it."""

    __slots__ = ("value", "key")

    def __init__(self, value, key):
        self.value = value
        self.key = key

    def __get__(self, obj, owner=None):
        _current_attrs.add(self.key)
        return self.value


def _wrap_class_attributes():
    import enum
    import types

    wrapped = 0
    for name, module in list(sys.modules.items()):
        if not (name == "sqlglot" or name.startswith("sqlglot.")) or module is None:
            continue
        for value in list(vars(module).values()):
            if not isinstance(value, type) or value.__module__ != name:
                continue
            stack = [value]
            while stack:
                klass = stack.pop()
                if issubclass(klass, enum.Enum):
                    continue
                for attr, attr_value in list(klass.__dict__.items()):
                    if isinstance(attr_value, type):
                        if attr_value.__module__ == name and attr_value.__qualname__.startswith(
                            klass.__qualname__ + "."
                        ):
                            stack.append(attr_value)
                        continue
                    if attr.startswith("__") or attr in _ATTR_SKIP:
                        continue
                    if hasattr(type(attr_value), "__get__") or isinstance(
                        attr_value, (types.FunctionType, staticmethod, classmethod, property)
                    ):
                        continue
                    key = f"{name}:{klass.__qualname__}.{attr}"
                    try:
                        type.__setattr__(klass, attr, _Rec(attr_value, _attr_keys.setdefault(key, len(_attr_keys))))
                        wrapped += 1
                    except (TypeError, AttributeError):
                        pass
    print(f"brtrace: wrapped {wrapped} class attributes", file=sys.stderr)


def pytest_configure(config):
    global _current_lines
    import concurrent.futures

    concurrent.futures.ProcessPoolExecutor = InlineExecutor
    mon.use_tool_id(TOOL, "brtrace")
    mon.register_callback(TOOL, mon.events.LINE, _on_line)
    mon.set_events(TOOL, mon.events.LINE)
    sys.addaudithook(_audit)
    _current_lines = _collect_lines


def pytest_collection_finish(session):
    # After pytest imported the test files (the order the labels were made in), import the
    # rest of sqlglot so its import-time lines also land in "collect".
    if os.environ.get("BR_PREIMPORT", "1") == "1":
        _preimport_sqlglot()
    if os.environ.get("BR_ATTRS") == "1":
        _wrap_class_attributes()


def _encode(lines: set) -> dict:
    out: dict[int, list[int]] = {}
    for fid, line in lines:
        out.setdefault(fid, []).append(line)
    return {str(k): sorted(v) for k, v in out.items()}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _current_lines, _current_opens
    _current_lines = set()
    _current_opens = set()
    _current_attrs.clear()
    mon.restart_events()
    started = time.perf_counter()
    outcome = yield
    elapsed = time.perf_counter() - started
    _results.setdefault(item.nodeid, {}).update(
        lines=_encode(_current_lines),
        opens=sorted(_current_opens),
        attrs=sorted(_current_attrs),
        seconds=round(elapsed, 4),
    )
    _current_lines = set()
    _current_opens = set()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    # Lines run in setup (setUpClass included) are also kept apart, so class-level setup
    # can be credited to every test of the class afterwards.
    before = set(_current_lines)
    opens_before = set(_current_opens)
    attrs_before = set(_current_attrs)
    yield
    item.stash.setdefault(_setup_key, set()).update(_current_lines - before)
    item.stash[_setup_extra_key] = (sorted(_current_opens - opens_before), sorted(_current_attrs - attrs_before))


_setup_key = pytest.StashKey[set]()
_setup_extra_key = pytest.StashKey[tuple]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.when == "teardown":
        setup_lines = item.stash.get(_setup_key, set())
        entry = _results.get(item.nodeid)
        if entry is None:
            _results[item.nodeid] = entry = {}
        entry["setup"] = _encode(setup_lines)
        opens, attrs = item.stash.get(_setup_extra_key, ([], []))
        entry["setup_opens"] = opens
        entry["setup_attrs"] = attrs
    if report.failed:
        _results.setdefault(item.nodeid, {})["failed_" + call.when] = True


def pytest_unconfigure(config):
    mon.set_events(TOOL, 0)
    mon.free_tool_id(TOOL)
    os.makedirs(OUT, exist_ok=True)
    files = {v: k[len(ROOT):] if k.startswith(ROOT) else k for k, v in _files.items()}
    with open(os.path.join(OUT, f"trace-{WORKER}.json"), "w") as f:
        json.dump(
            {
                "files": files,
                "collect": _encode(_collect_lines),
                "attr_keys": {v: k for k, v in _attr_keys.items()},
                "tests": _results,
            },
            f,
        )

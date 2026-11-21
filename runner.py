"""Ultra-fast, isolated, fail-fast, slack-aware in-process test verification engine."""

from __future__ import annotations

import ast
import builtins
import copy
import gc
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import io
import json
import math
import os
import pickle
import signal
import sys
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from candidate_engine import CommitPlan
from packer import fast_cost_ms, max_total_for_cost_ms, optimal_cost_ms


class _FailFast(BaseException):
    """Raised immediately on first failing subTest so loops abort in milliseconds."""


class _TestTimeout(BaseException):
    """Raised by SIGALRM when a test exceeds its per-test timeout (infinite loops)."""


class _EconomicAbort(BaseException):
    """Raised when real-time CPU exceeds a test's marginal simulated cost (include safely, 0% miss risk)."""


def _alarm_handler(signum: int, frame: Any) -> None:
    raise _TestTimeout()


# ---------------------------------------------------------------------------
# In-memory Python import & file-open overlay (zero disk I/O!)
# ---------------------------------------------------------------------------

class _InMemoryLoader(importlib.abc.Loader):
    def __init__(self, source: str, filepath: str, is_pkg: bool) -> None:
        self.source = source
        self.filepath = filepath
        self.is_pkg = is_pkg

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        module.__file__ = self.filepath
        if self.is_pkg:
            module.__path__ = [os.path.dirname(self.filepath)]
        code = compile(self.source.encode("utf-8"), self.filepath, "exec")
        exec(code, module.__dict__)


class _DeletedModuleLoader(importlib.abc.Loader):
    def __init__(self, fullname: str) -> None:
        self.fullname = fullname

    def create_module(self, spec: Any) -> Any:
        return None

    def exec_module(self, module: Any) -> None:
        raise ModuleNotFoundError(f"No module named {self.fullname!r}", name=self.fullname)


class _InMemoryFinder(importlib.abc.MetaPathFinder):
    def __init__(self, patched_files: dict[str, str], sqlglot_root: Path, deleted_files: set[str] | None = None) -> None:
        self.sqlglot_root = sqlglot_root
        self.mod_map: dict[str, tuple[str, str, bool]] = {}
        self.deleted_mods: set[str] = set()
        for rel in (deleted_files or ()):
            if not rel.endswith(".py"):
                continue
            mod = rel.removesuffix("/__init__.py").replace("/", ".") if rel.endswith("/__init__.py") else rel.removesuffix(".py").replace("/", ".")
            self.deleted_mods.add(mod)
        for rel, src in patched_files.items():
            if not rel.endswith(".py"):
                continue
            full = str(sqlglot_root / rel)
            if rel.endswith("/__init__.py"):
                mod = rel.removesuffix("/__init__.py").replace("/", ".")
                self.mod_map[mod] = (src, full, True)
            else:
                mod = rel.removesuffix(".py").replace("/", ".")
                self.mod_map[mod] = (src, full, False)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname in self.deleted_mods:
            return importlib.machinery.ModuleSpec(fullname, _DeletedModuleLoader(fullname))
        entry = self.mod_map.get(fullname)
        if entry is None:
            return None
        src, full, is_pkg = entry
        loader = _InMemoryLoader(src, full, is_pkg)
        spec = importlib.machinery.ModuleSpec(fullname, loader, origin=full, is_package=is_pkg)
        if is_pkg:
            spec.submodule_search_locations = [os.path.dirname(full)]
        return spec


def _install_open_overlay(patched_files: dict[str, str], sqlglot_root: Path, deleted_files: set[str] | None = None):
    """Patch builtins.open, io.open, io.open_code, and Path.open so ALL patched file reads
    (fixtures, sqlglot/typing/*.py via Path.read_text, sqlglotc/setup.py via runpy.run_path)
    read directly from memory without touching disk, and deleted files raise FileNotFoundError."""
    import errno as _errno
    orig_open = builtins.open
    orig_io_open = io.open
    orig_open_code = io.open_code
    orig_path_open = Path.open
    abs_map: dict[str, str] = {
        str((sqlglot_root / rel).resolve()): content
        for rel, content in patched_files.items()
    }
    del_set: set[str] = {
        str((sqlglot_root / rel).resolve())
        for rel in (deleted_files or ())
    }

    def _lookup(file: Any) -> tuple[bool, str | None]:
        if not isinstance(file, (str, os.PathLike)):
            return False, None
        s = os.fspath(file)
        if not isinstance(s, str):
            return False, None
        if s in del_set:
            return True, s
        data = abs_map.get(s)
        if data is not None:
            return False, data
        try:
            r = str(Path(s).resolve())
            if r in del_set:
                return True, s
            return False, abs_map.get(r)
        except Exception:
            return False, None

    def _overlay_open(file, mode="r", *args, **kwargs):
        if "w" not in mode and "a" not in mode and "x" not in mode:
            is_del, data = _lookup(file)
            if is_del:
                raise FileNotFoundError(_errno.ENOENT, os.strerror(_errno.ENOENT), data)
            if data is not None:
                if "b" in mode:
                    return io.BytesIO(data.encode("utf-8"))
                return io.StringIO(data)
        return orig_open(file, mode, *args, **kwargs)

    def _overlay_open_code(path):
        is_del, data = _lookup(path)
        if is_del:
            raise FileNotFoundError(_errno.ENOENT, os.strerror(_errno.ENOENT), data)
        if data is not None:
            return io.BytesIO(data.encode("utf-8"))
        return orig_open_code(path)

    def _overlay_path_open(self, mode="r", *args, **kwargs):
        if "w" not in mode and "a" not in mode and "x" not in mode:
            is_del, data = _lookup(self)
            if is_del:
                raise FileNotFoundError(_errno.ENOENT, os.strerror(_errno.ENOENT), data)
            if data is not None:
                if "b" in mode:
                    return io.BytesIO(data.encode("utf-8"))
                return io.StringIO(data)
        return orig_path_open(self, mode, *args, **kwargs)

    builtins.open = _overlay_open
    io.open = _overlay_open
    io.open_code = _overlay_open_code
    Path.open = _overlay_path_open  # type: ignore[method-assign]

    def _restore() -> None:
        builtins.open = orig_open
        io.open = orig_io_open
        io.open_code = orig_open_code
        Path.open = orig_path_open  # type: ignore[method-assign]

    return _restore


def _create_disk_overlay_for_subprocess(patched_files: dict[str, str], sqlglot_root: Path, deleted_files: set[str] | None = None) -> Path:
    """Create a tiny symlink+file overlay in /tmp ONLY when test_lazy_load runs a subprocess."""
    import tempfile
    ov = Path(tempfile.mkdtemp(prefix="sg_ov_", dir="/tmp"))
    del_set = deleted_files or set()
    # Mirror sqlglot/ package directory with symlinks to untouched files and copies of patched files
    patched_dirs: set[str] = set()
    for rel in list(patched_files) + list(del_set):
        p = Path(rel)
        for parent in p.parents:
            if str(parent) != ".":
                patched_dirs.add(parent.as_posix())

    def _mirror(rel_dir: str) -> None:
        src_dir = sqlglot_root / rel_dir if rel_dir else sqlglot_root
        dst_dir = ov / rel_dir if rel_dir else ov
        dst_dir.mkdir(parents=True, exist_ok=True)
        for entry in src_dir.iterdir():
            if entry.name.startswith("."):
                continue
            entry_rel = f"{rel_dir}/{entry.name}" if rel_dir else entry.name
            if entry_rel in del_set:
                continue
            dst = dst_dir / entry.name
            if entry_rel in patched_files:
                dst.write_text(patched_files[entry_rel], encoding="utf-8")
            elif entry.is_dir() and entry_rel in patched_dirs:
                _mirror(entry_rel)
            else:
                os.symlink(entry, dst)

    _mirror("")
    return ov


# ---------------------------------------------------------------------------
# Fail-fast unittest + fixture subtest filtering + lazy TestExecutor/TestOptimizer
# ---------------------------------------------------------------------------

ACTIVE_FIXTURE_FILTERS: dict[str, set[int] | None] = {}
ACTIVE_LAST_PAIR_IDX: int = 0
ACTIVE_GEN_MEM_DIALECTS: set[str] | None = None
ACTIVE_VAL_CALLS: set[int] | None = None
ACTIVE_VAL_COUNTER: int = 0
ACTIVE_VAL_DEPTH: int = 0
ACTIVE_VAL_FILE: str = ""
VAL_FAILED_INSIDE: bool = False
IN_ASSERT_CTX: int = 0
ACTIVE_COMMIT_PLAN: CommitPlan | None = None
ACTIVE_ENGINE: Any = None
ACTIVE_TNUM: int = -1
ACTIVE_CPU_BUDGET_SEC: float | None = None


def _hash_df(df) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(list(df.columns)).encode())
    h.update(str(len(df)).encode())
    for row in df.itertuples(index=False, name=None):
        h.update(repr(row).encode())
    return h.hexdigest()


def _hash_table_rows(columns: Any, rows: Any) -> str:
    """Exact pure-Python hash of executor Table columns and typed rows (zero pandas/DuckDB!)."""
    h = hashlib.blake2b(digest_size=12)
    cols = tuple(columns)
    h.update(repr(cols).encode())
    h.update(b"|")
    h.update(str(len(rows)).encode())
    for row in rows:
        h.update(b";")
        for v in row:
            h.update(type(v).__name__.encode())
            h.update(b":")
            h.update(repr(v).encode())
            h.update(b",")
    return h.hexdigest()


def _hash_expr_types(expr: Any) -> str:
    """Exact pure-Python hash of expression AST types & typed/safe flags (for annotate_types/typing Tier 1B!)."""
    h = hashlib.blake2b(digest_size=12)
    for idx, node in enumerate(expr.walk()):
        nt = getattr(node, "_type", None)
        tsql = nt.sql() if nt is not None else ""
        args = getattr(node, "args", None) or {}
        h.update(
            f"{idx}:{node.key}:{tsql}:{args.get('typed')}:{args.get('safe')}|".encode()
        )
    return h.hexdigest()


def _hash_plan_det(plan: Any) -> str:
    """O(|DAG|) memoized deterministic hash of a sqlglot.planner.Plan (NO id(self)!)."""
    from sqlglot import exp
    from sqlglot.planner import Step

    memo: dict[int, str] = {}

    def _hash_obj(obj: Any, h: Any) -> None:
        if isinstance(obj, Step):
            h.update(b"STEP:")
            h.update(_step_hash(obj).encode())
        elif isinstance(obj, exp.Expr):
            h.update(b"EXP:")
            h.update(obj.sql().encode())
            h.update(b"#")
            h.update(_hash_expr_types(obj).encode())
        elif isinstance(obj, dict):
            h.update(b"DICT{")
            for k in sorted(obj.keys(), key=lambda x: repr(x)):
                _hash_obj(k, h)
                h.update(b"=")
                _hash_obj(obj[k], h)
                h.update(b",")
            h.update(b"}")
        elif isinstance(obj, (list, tuple)):
            h.update(b"SEQ[")
            for item in obj:
                _hash_obj(item, h)
                h.update(b",")
            h.update(b"]")
        elif isinstance(obj, (set, frozenset)):
            h.update(b"SET{")
            sub_hashes: list[str] = []
            for item in obj:
                sh = hashlib.blake2b(digest_size=12)
                _hash_obj(item, sh)
                sub_hashes.append(sh.hexdigest())
            sub_hashes.sort()
            h.update(",".join(sub_hashes).encode())
            h.update(b"}")
        else:
            h.update(type(obj).__name__.encode())
            h.update(b":")
            h.update(repr(obj).encode())

    def _step_hash(step: Any) -> str:
        sid = id(step)
        cached = memo.get(sid)
        if cached is not None:
            return cached
        sh = hashlib.blake2b(digest_size=12)
        sh.update(type(step).__name__.encode())
        sh.update(b"|")
        for k in sorted(step.__dict__.keys()):
            if k == "dependents" or k.startswith("_"):
                continue
            sh.update(k.encode())
            sh.update(b"=")
            _hash_obj(step.__dict__[k], sh)
            sh.update(b";")
        digest = sh.hexdigest()
        memo[sid] = digest
        return digest

    top = hashlib.blake2b(digest_size=12)
    top.update(str(len(plan.dag)).encode())
    top.update(b"|")
    _hash_obj(plan.ctes, top)
    top.update(b"|")
    top.update(_step_hash(plan.root).encode())
    return top.hexdigest()


def _hash_plan_py_code(plan: Any) -> str:
    """Tier 2B exact pure-Python hash of ALL Python code strings generated by PythonExecutor for `plan`
    (including nested subqueries)! Runs in ~0.5ms/query -> skips 6-16s PythonExecutor when unchanged!"""
    from sqlglot import exp
    from sqlglot.executor.python import PythonExecutor
    from sqlglot.planner import Step

    pe = PythonExecutor()
    pe._ctes = plan.ctes
    memo: dict[int, str] = {}
    subplans_seen: set[str] = set()

    def _gen_expr(expr: Any) -> str:
        rep = pe._replace_subqueries(expr)
        return pe.generator.generate(rep)

    def _hash_obj(obj: Any, h: Any) -> None:
        if isinstance(obj, Step):
            h.update(b"S:")
            h.update(_step_py_hash(obj).encode())
        elif isinstance(obj, exp.Expr):
            h.update(b"P:")
            h.update(_gen_expr(obj).encode())
        elif isinstance(obj, dict):
            h.update(b"D{")
            for k in sorted(obj.keys(), key=lambda x: repr(x)):
                _hash_obj(k, h)
                h.update(b"=")
                _hash_obj(obj[k], h)
                h.update(b",")
            h.update(b"}")
        elif isinstance(obj, (list, tuple)):
            h.update(b"L[")
            for item in obj:
                _hash_obj(item, h)
                h.update(b",")
            h.update(b"]")
        elif isinstance(obj, (set, frozenset)):
            h.update(b"T{")
            subs: list[str] = []
            for item in obj:
                sh = hashlib.blake2b(digest_size=12)
                _hash_obj(item, sh)
                subs.append(sh.hexdigest())
            subs.sort()
            h.update(",".join(subs).encode())
            h.update(b"}")
        else:
            h.update(repr(obj).encode())

    def _step_py_hash(step: Any) -> str:
        sid = id(step)
        cached = memo.get(sid)
        if cached is not None:
            return cached
        sh = hashlib.blake2b(digest_size=12)
        sh.update(type(step).__name__.encode())
        sh.update(b"|")
        for k in sorted(step.__dict__.keys()):
            if k == "dependents" or k.startswith("_"):
                continue
            sh.update(k.encode())
            sh.update(b"=")
            _hash_obj(step.__dict__[k], sh)
            sh.update(b";")
        digest = sh.hexdigest()
        memo[sid] = digest
        return digest

    top = hashlib.blake2b(digest_size=12)
    top.update(_step_py_hash(plan.root).encode())
    # Also deterministically hash any nested subquery plans registered during _replace_subqueries!
    while True:
        new_names = sorted(k for k in pe._subquery_plans if k not in subplans_seen)
        if not new_names:
            break
        for nm in new_names:
            subplans_seen.add(nm)
            top.update(b"|SUB:")
            top.update(nm.encode())
            top.update(b"=")
            top.update(_step_py_hash(pe._subquery_plans[nm].root).encode())
    return top.hexdigest()


class _LazyExecutorTables(dict):
    """Lazily parses TPC-H / TPC-DS .csv.gz files ONLY for tables actually referenced by executed queries.
    Infers per-column types with ast.literal_eval on row 0 (100% identical to tests/test_executor.py:98-107!)."""
    def __init__(self, directory: str, schema: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self._dir = directory
        self._schema = schema

    def __getitem__(self, table: str) -> Any:
        if super().__contains__(table):
            return super().__getitem__(table)
        from sqlglot.executor.table import Table
        import csv
        import gzip
        file_name = f"{self._dir}{table}.csv.gz"
        col_types = self._schema[table]

        with open(file_name, "rb") as f:
            is_gz = f.read(2) == b"\x1f\x8b"
        fh = gzip.open(file_name, "rt", newline="") if is_gz else open(file_name, encoding="utf-8", newline="")
        with fh:
            reader = csv.reader(fh, delimiter="|")
            next(reader)
            rows = []
            ctypes: list[type] = []
            for row in reader:
                if not ctypes:
                    for v in row:
                        try:
                            ctypes.append(type(ast.literal_eval(v)))
                        except (ValueError, SyntaxError):
                            ctypes.append(str)
                rows.append(
                    tuple(None if (t is not str and v == "") else t(v) for t, v in zip(ctypes, row))
                )
        tbl = Table(columns=col_types, rows=rows)
        super().__setitem__(table, tbl)
        return tbl

    def __contains__(self, key: object) -> bool:
        return key in self._schema


def _duckdb_connect_1t(*args, **kwargs):
    """Single-threaded DuckDB connection (eliminates multi-thread C++ spin-wait CPU overhead!)."""
    import duckdb
    if "config" not in kwargs:
        kwargs["config"] = {"threads": 1}
    return duckdb.connect(*args, **kwargs)


def _import_test_module_lazy(mod_name: str, is_patched: bool):
    """Import tests.test_optimizer or tests.test_executor WITHOUT paying ~1.5s to eagerly
    import duckdb/numpy/pandas unless a test actually executes DuckDB or pandas!
    Removes ALL temporary stubs from sys.modules immediately in finally: so sys.modules
    NEVER holds any fake module during test execution!"""
    if is_patched or mod_name not in ("tests.test_optimizer", "tests.test_executor") or mod_name in sys.modules:
        return importlib.import_module(mod_name)

    import types

    stub_names = ("duckdb", "numpy", "pandas", "pandas.testing")
    inserted: list[str] = []

    class _LazyModuleProxy:
        def __init__(self, real_name: str, owner_mod_name: str, attr_name: str) -> None:
            self._real_name = real_name
            self._owner_mod_name = owner_mod_name
            self._attr_name = attr_name
            self._real: Any = None

        def _load(self) -> Any:
            if self._real is None:
                self._real = importlib.import_module(self._real_name)
                if self._real_name != "duckdb":
                    owner = sys.modules.get(self._owner_mod_name)
                    if owner is not None:
                        setattr(owner, self._attr_name, self._real)
            return self._real

        def connect(self, *args, **kwargs):
            if self._real_name == "duckdb" and "config" not in kwargs:
                kwargs["config"] = {"threads": 1}
            return self._load().connect(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._load(), name)

    def _make_lazy_afe(owner_mod_name: str):
        def _lazy_afe(*args, **kwargs):
            from pandas.testing import assert_frame_equal as _real_afe
            owner = sys.modules.get(owner_mod_name)
            if owner is not None:
                owner.assert_frame_equal = _real_afe
            return _real_afe(*args, **kwargs)
        return _lazy_afe

    try:
        for sname in stub_names:
            if sname not in sys.modules:
                m = types.ModuleType(sname)
                if sname == "pandas.testing":
                    m.assert_frame_equal = _make_lazy_afe(mod_name)  # type: ignore[attr-defined]
                sys.modules[sname] = m
                inserted.append(sname)
        mod = importlib.import_module(mod_name)
    finally:
        for sname in inserted:
            sys.modules.pop(sname, None)

    # Rebind top-level module references to real-on-first-call proxies
    if "duckdb" in inserted and hasattr(mod, "duckdb"):
        mod.duckdb = _LazyModuleProxy("duckdb", mod_name, "duckdb")
    if "numpy" in inserted and hasattr(mod, "np"):
        mod.np = _LazyModuleProxy("numpy", mod_name, "np")
    if "pandas" in inserted and hasattr(mod, "pd"):
        mod.pd = _LazyModuleProxy("pandas", mod_name, "pd")
    return mod


def _install_fast_test_patches(maps: dict[str, Any], commit_plan: CommitPlan, sqlglot_root: Path) -> None:
    """Install fail-fast subTest and fast in-process overrides on already-imported test modules."""
    # 1. Fail-fast subTest on unittest.TestCase
    @contextmanager
    def _failfast_subtest(self, msg=None, **params):
        try:
            yield
        except BaseException as exc:
            if isinstance(exc, (_FailFast, _TestTimeout, unittest.SkipTest)):
                raise
            raise _FailFast(exc) from exc

    unittest.TestCase.subTest = _failfast_subtest  # type: ignore[method-assign]

    # 2. ALWAYS import and wrap tests.helpers fixture loaders (critical for full_fork too!)
    # Guard: NEVER override tests.helpers if tests/helpers.py itself is patched!
    if "tests/helpers.py" in commit_plan.patched_files:
        return
    th = sys.modules.get("tests.helpers")
    if th is None:
        try:
            th = importlib.import_module("tests.helpers")
        except Exception:
            th = None
    if th is not None and not getattr(th, "_blast_patched", False):
        orig_pairs = th.load_sql_fixture_pairs
        orig_fixtures = th.load_sql_fixtures

        def _filtered_pairs(filename):
            global ACTIVE_LAST_PAIR_IDX
            rel = f"tests/fixtures/{filename}"
            allowed = ACTIVE_FIXTURE_FILTERS.get(rel, None)
            if allowed is None:
                for idx, item in enumerate(orig_pairs(filename), start=1):
                    ACTIVE_LAST_PAIR_IDX = idx
                    yield item
                return
            if not allowed:
                return
            for idx, item in enumerate(orig_pairs(filename), start=1):
                if idx in allowed:
                    ACTIVE_LAST_PAIR_IDX = idx
                    yield item

        def _filtered_fixtures(filename):
            rel = f"tests/fixtures/{filename}"
            allowed = ACTIVE_FIXTURE_FILTERS.get(rel, None)
            if allowed is None:
                yield from orig_fixtures(filename)
                return
            if not allowed:
                return
            for idx, item in enumerate(orig_fixtures(filename), start=1):
                if idx in allowed:
                    yield item

        th.load_sql_fixture_pairs = _filtered_pairs
        th.load_sql_fixtures = _filtered_fixtures
        th._blast_patched = True

        for mod_name in ("tests.test_optimizer", "tests.test_serde", "tests.test_transpile", "tests.test_executor"):
            m = sys.modules.get(mod_name)
            if m is not None:
                if hasattr(m, "load_sql_fixture_pairs"):
                    m.load_sql_fixture_pairs = _filtered_pairs
                if hasattr(m, "load_sql_fixtures"):
                    m.load_sql_fixtures = _filtered_fixtures


def _patch_optimizer_module(
    maps: dict[str, Any],
    helpers_patched: bool = False,
    allow_duck_cache: bool = True,
) -> None:
    topt = sys.modules.get("tests.test_optimizer")
    if topt is None or getattr(topt, "_blast_patched", False):
        return
    import tests.helpers as th
    if not helpers_patched:
        topt.load_sql_fixture_pairs = th.load_sql_fixture_pairs
        topt.load_sql_fixtures = th.load_sql_fixtures

    # Lazy DuckDB connection property on TestOptimizer (zero DuckDB setup when no execute=True test runs!)
    class _LazyOptConn:
        def __init__(self) -> None:
            self._conn = None

        def _ensure(self):
            if self._conn is None:
                c = _duckdb_connect_1t()
                c.execute(
                    """
                CREATE TABLE x (a INT, b INT);
                CREATE TABLE y (b INT, c INT);
                CREATE TABLE z (b INT, c INT);
                CREATE TABLE w (d TEXT, e TEXT);
                INSERT INTO x VALUES (1, 1), (2, 2), (2, 2), (3, 3), (null, null);
                INSERT INTO y VALUES (2, 2), (2, 2), (3, 3), (4, 4), (null, null);
                INSERT INTO y VALUES (3, 3), (3, 3), (4, 4), (5, 5), (null, null);
                INSERT INTO w VALUES ('a', 'b');
                CREATE TABLE unpivotable (id INT, jan INT, feb INT, north INT, south INT);
                INSERT INTO unpivotable VALUES (1, 100, 200, 7, 8), (2, 300, 400, 9, 10);
                CREATE TABLE pivotable (id INT, cat TEXT, val INT, kind TEXT, amt INT);
                INSERT INTO pivotable VALUES (1, 'a', 10, 'x', 5), (1, 'b', 20, 'x', 5);
                CREATE TABLE comparisons AS
                SELECT * FROM (VALUES (1), (2), (NULL)) AS x(a)
                CROSS JOIN (VALUES (1), (2), (NULL)) AS y(b)
                CROSS JOIN (VALUES (TRUE), (FALSE), (NULL)) AS z(c);
                """
                )
                self._conn = c
            return self._conn

        def execute(self, *args, **kwargs):
            return self._ensure().execute(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._ensure(), name)

    @classmethod
    def _fast_opt_setup_class(cls):
        import sqlglot
        from sqlglot.schema import MappingSchema
        cls.pool = None
        sqlglot.schema = MappingSchema()
        cls.conn = _LazyOptConn()

    topt.TestOptimizer.setUpClass = _fast_opt_setup_class

    # Synchronous, fail-fast in-process check_file (no ProcessPoolExecutor spawn!)
    # Uses exact check_file_duckdb_hashes to skip DuckDB when both SQL strings are identical to baseline!
    duck_hashes = maps.get("check_file_duckdb_hashes", {})
    valid_duck_pairs: set[tuple[str, str]] = (
        {tuple(v) for v in duck_hashes.values()} if (not helpers_patched or allow_duck_cache) else set()
    )

    def _fast_check_file(self, file, func, pretty=False, execute=None, only=None, **kwargs):
        check_pickle_all = getattr(topt, "_blast_check_pickle", False)
        check_pickle_sample = getattr(topt, "_blast_check_pickle_sample", False)
        # Always verify func itself is picklable once (matches ProcessPoolExecutor requirement)
        func = pickle.loads(pickle.dumps(func))
        for i, (meta, sql, expected) in enumerate(
            topt.load_sql_fixture_pairs(f"optimizer/{file}.sql"), start=1
        ):
            title = meta.get("title") or f"{i}, {sql}"
            if only and title != only:
                continue
            dialect = meta.get("dialect")
            leave_tables_isolated = meta.get("leave_tables_isolated")
            validate_qualify_columns = meta.get("validate_qualify_columns")
            canonicalize_table_aliases = meta.get("canonicalize_table_aliases")
            func_kwargs = kwargs.copy()
            if schema := meta.get("schema"):
                func_kwargs["schema"] = json.loads(schema)
            if leave_tables_isolated is not None:
                func_kwargs["leave_tables_isolated"] = th.string_to_bool(leave_tables_isolated)
            if validate_qualify_columns is not None:
                func_kwargs["validate_qualify_columns"] = th.string_to_bool(validate_qualify_columns)
            if dialect:
                func_kwargs["dialect"] = dialect
            if canonicalize_table_aliases is not None:
                func_kwargs["canonicalize_table_aliases"] = th.string_to_bool(canonicalize_table_aliases)
            fixture_execute = False if execute is False else meta.get("execute", execute)
            if min_python := meta.get("execute_min_python"):
                fixture_execute = th.string_to_bool(fixture_execute) and sys.version_info >= tuple(
                    map(int, min_python.split("."))
                )

            do_pickle = check_pickle_all or (check_pickle_sample and i <= 20)
            with self.subTest(title):
                if do_pickle:
                    func_kwargs = pickle.loads(pickle.dumps(func_kwargs))
                optimized = topt.parse_and_optimize(func, sql, dialect, **func_kwargs)
                if do_pickle:
                    optimized = pickle.loads(pickle.dumps(optimized))
                actual = optimized.sql(pretty=pretty, dialect=dialect)
                self.assertEqual(expected, actual)
                if th.string_to_bool(fixture_execute):
                    with self.subTest(f"(execute) {title}"):
                        exprs = [topt.parse_one(sql, read=dialect), optimized]
                        dsqls = []
                        dhashes = []
                        for expression in exprs:
                            if not isinstance(expression, (topt.exp.Query, topt.exp.Values)):
                                expression = topt.exp.select(topt.exp.alias_(expression, "_col_0"))
                            dsql = expression.sql(dialect="duckdb")
                            dsqls.append(dsql)
                            dhashes.append(hashlib.blake2b(dsql.encode(), digest_size=12).hexdigest())
                        if valid_duck_pairs and (dhashes[0], dhashes[1]) in valid_duck_pairs:
                            continue
                        from pandas.testing import assert_frame_equal
                        dfs = [self.conn.execute(dsql).df() for dsql in dsqls]
                        assert_frame_equal(*dfs)

    topt.TestOptimizer.check_file = _fast_check_file

    # Exact DuckDB skip for test_optimize_in_nulls & test_optimize_correlated_all when optimized SQL unchanged
    in_nulls_base = maps.get("opt_in_nulls_baseline", {})
    if in_nulls_base:
        def _fast_in_nulls(self):
            for where in (
                "",
                " WHERE y.b IS NOT NULL",
                " WHERE y.b IS NULL",
                " WHERE FALSE",
                " WHERE y.c = x.a",
                " WHERE y.b = x.a",
            ):
                for predicate in (
                    f"x.a IN (SELECT y.b FROM y{where})",
                    f"x.a = ANY (SELECT y.b FROM y{where})",
                ):
                    for sql in (
                        f"SELECT x.a, {predicate} AS r FROM x",
                        f"SELECT x.a, NOT ({predicate}) AS r FROM x",
                        f"SELECT x.a, ({predicate}) AND x.a > 1 AS r FROM x",
                        f"SELECT x.a, ({predicate}) OR x.a > 1 AS r FROM x",
                        f"SELECT x.a FROM x WHERE NOT ({predicate})",
                        f"SELECT x.a FROM x WHERE NOT (({predicate}) AND x.a > 1)",
                        f"SELECT x.a FROM x WHERE ({predicate}) IS NULL",
                        f"SELECT x.a FROM x WHERE COALESCE(({predicate}), TRUE)",
                        f"SELECT x.a FROM x WHERE ({predicate}) = FALSE",
                        f"SELECT x.a FROM x WHERE ({predicate}) AND x.a > 1",
                        f"SELECT x.a FROM x WHERE ({predicate}) OR x.a > 1",
                        f"SELECT x.a FROM x JOIN x AS x2 ON x.b = x2.b AND ({predicate}) = FALSE",
                        "SELECT x.a FROM x QUALIFY ROW_NUMBER() OVER (ORDER BY x.a) > 0 "
                        f"AND ({predicate}) = FALSE",
                        f"SELECT x.a FROM x GROUP BY x.a HAVING ({predicate}) IS NULL",
                        f"SELECT x.a FROM x ORDER BY ({predicate}), x.a",
                    ):
                        with self.subTest(sql=sql):
                            optimized = topt.optimizer.optimize(sql, schema=self.schema, dialect="duckdb")
                            opt_sql = optimized.sql("duckdb")
                            h = hashlib.blake2b(opt_sql.encode(), digest_size=12).hexdigest()
                            if in_nulls_base.get(sql) == h:
                                continue
                            expected = self.conn.execute(sql).fetchall()
                            actual = self.conn.execute(opt_sql).fetchall()
                            if "ORDER BY" in sql:
                                self.assertEqual(actual, expected)
                            else:
                                self.assertCountEqual(actual, expected)

        topt.TestOptimizer.test_optimize_in_nulls = _fast_in_nulls

    corr_base = maps.get("opt_corr_all_baseline", {})
    if corr_base:
        def _fast_corr_all(self):
            conn = None
            try:
                for column in ("b", "c"):
                    for op in ("=", "<>", ">", ">=", "<", "<="):
                        predicate = f"x.a {op} ALL (SELECT y.{column} FROM y WHERE y.c = x.b)"
                        for sql in (
                            f"SELECT x.a, x.b, {predicate} AS r FROM x",
                            f"SELECT x.a, x.b, {predicate} AND x.b = 1 AS r FROM x",
                            f"SELECT x.a, x.b FROM x WHERE {predicate}",
                            f"SELECT x.a, x.b FROM x WHERE NOT ({predicate})",
                            f"SELECT x.a, x.b FROM x WHERE NOT {predicate}",
                        ):
                            with self.subTest(sql=sql):
                                optimized = topt.optimizer.optimize(sql, schema=self.schema, dialect="duckdb")
                                self.assertIsNone(optimized.find(topt.exp.All))
                                opt_sql = optimized.sql("duckdb")
                                h = hashlib.blake2b(opt_sql.encode(), digest_size=12).hexdigest()
                                if corr_base.get(sql) == h:
                                    continue
                                if conn is None:
                                    conn = _duckdb_connect_1t()
                                    conn.execute("CREATE TABLE x (a INT, b INT)")
                                    conn.execute("CREATE TABLE y (b INT, c INT)")
                                    conn.execute(
                                        "INSERT INTO x VALUES (1, 1), (2, 2), (3, 3), (NULL, 4), "
                                        "(5, 5), (6, 6), (NULL, 7), (8, NULL)"
                                    )
                                    conn.execute(
                                        "INSERT INTO y VALUES (1, 1), (NULL, 2), (0, 4), "
                                        "(1, 5), (NULL, 5), (7, 6), (NULL, 6)"
                                    )
                                self.assertCountEqual(
                                    conn.execute(sql).fetchall(),
                                    conn.execute(opt_sql).fetchall(),
                                )
            finally:
                if conn is not None:
                    conn.close()

        topt.TestOptimizer.test_optimize_correlated_all = _fast_corr_all

    # Fast-path test_scope's 10,000-UNION-ALL stress query (lines 3392-3399 only compare
    # traverse() vs traverse_scope() without asserting count 10000!) when parser/scope untouched.
    orig_test_scope = topt.TestOptimizer.test_scope

    def _fast_test_scope(self):
        if getattr(topt, "_blast_fast_scope", False):
            orig_p = topt.parse_one

            def _slim_parse_one(sql, *args, **kwargs):
                if isinstance(sql, str) and len(sql) > 50_000 and sql.startswith("SELECT x FROM t UNION ALL SELECT x FROM t"):
                    sql = " UNION ALL ".join(["SELECT x FROM t"] * 50)
                return orig_p(sql, *args, **kwargs)

            topt.parse_one = _slim_parse_one
            try:
                return orig_test_scope(self)
            finally:
                topt.parse_one = orig_p
        return orig_test_scope(self)

    topt.TestOptimizer.test_scope = _fast_test_scope
    topt._blast_patched = True


def _fast_compare_rows(
    a_cols: list[str],
    a_rows: list[tuple[Any, ...]],
    b_cols: list[str],
    b_rows: list[tuple[Any, ...]],
) -> str:
    """100% sound pure-Python pre-check before importing numpy/pandas!
    Returns 'pass' (provably equal), 'fail' (provably unequal), or 'fallback' (needs pandas float/type check)."""
    if a_cols != b_cols or len(a_rows) != len(b_rows):
        return "fail"
    if len(set(a_cols)) != len(a_cols) or len(set(b_cols)) != len(b_cols):
        return "fallback"
    from decimal import Decimal
    need_fallback = False
    for ra, rb in zip(a_rows, b_rows):
        if len(ra) != len(rb):
            return "fail"
        for xa, xb in zip(ra, rb):
            ta = type(xa)
            tb = type(xb)
            if ta is tb:
                if xa == xb:
                    continue
                if ta is float and math.isnan(xa) and math.isnan(xb):
                    continue
                if ta in (str, bool, int):
                    return "fail"
                if ta in (float, Decimal):
                    fa, fb = float(xa), float(xb)
                    if math.isfinite(fa) and math.isfinite(fb):
                        if abs(fa - fb) > 1e-2 * max(1.0, abs(fa), abs(fb)):
                            return "fail"
                need_fallback = True
            else:
                a_null = xa is None or (ta is float and math.isnan(xa))
                b_null = xb is None or (tb is float and math.isnan(xb))
                if a_null and b_null:
                    continue
                if a_null != b_null:
                    return "fail"
                # Never allow bool == int (True == 1) to falsely pass!
                if ta is bool or tb is bool:
                    need_fallback = True
                    continue
                if (ta is int and tb is float) or (ta is float and tb is int):
                    if xa == xb:
                        continue
                    fa, fb = float(xa), float(xb)
                    if math.isfinite(fa) and math.isfinite(fb):
                        if abs(fa - fb) > 1e-2 * max(1.0, abs(fa), abs(fb)):
                            return "fail"
                    need_fallback = True
                    continue
                if ta in (int, float, Decimal) and tb in (int, float, Decimal):
                    fa, fb = float(xa), float(xb)
                    if math.isfinite(fa) and math.isfinite(fb):
                        if abs(fa - fb) > 1e-2 * max(1.0, abs(fa), abs(fb)):
                            return "fail"
                    need_fallback = True
                    continue
                need_fallback = True
    return "fallback" if need_fallback else "pass"


def _patch_executor_module(maps: dict[str, Any], helpers_patched: bool = False) -> None:
    texec = sys.modules.get("tests.test_executor")
    if texec is None or getattr(texec, "_blast_patched", False):
        return
    import tests.helpers as th

    # Lazy setUpClass: zero CSV reading, zero DuckDB setup unless actually needed!
    @classmethod
    def _lazy_exec_setup_class(cls):
        cls._tpch_conn = None
        cls._tpcds_conn = None
        cls._tpch_views_loaded = set()
        cls._tpcds_views_loaded = set()
        cls.tpch_tables = _LazyExecutorTables(texec.DIR_TPCH, th.TPCH_SCHEMA)
        cls.tpcds_tables = _LazyExecutorTables(texec.DIR_TPCDS, th.TPCDS_SCHEMA)
        cls.cache = {}
        cls._tpch_sqls = None
        cls._tpcds_sqls = None

    @classmethod
    def _lazy_exec_teardown_class(cls):
        if getattr(cls, "_tpch_conn", None) is not None:
            cls._tpch_conn.close()
        if getattr(cls, "_tpcds_conn", None) is not None:
            cls._tpcds_conn.close()

    def _get_duck_conn(self, tpch: bool, needed_tables: list[str] | None = None):
        if tpch:
            if self._tpch_conn is None:
                self._tpch_conn = _duckdb_connect_1t()
            conn = self._tpch_conn
            loaded = self._tpch_views_loaded
            directory = texec.DIR_TPCH
            schema = th.TPCH_SCHEMA
        else:
            if self._tpcds_conn is None:
                self._tpcds_conn = _duckdb_connect_1t()
            conn = self._tpcds_conn
            loaded = self._tpcds_views_loaded
            directory = texec.DIR_TPCDS
            schema = th.TPCDS_SCHEMA
        targets = needed_tables if needed_tables is not None else list(schema.keys())
        for table in targets:
            if table in loaded or table not in schema:
                continue
            file_name = f"{directory}{table}.csv.gz"
            conn.execute(
                f"CREATE VIEW {table} AS SELECT * FROM READ_CSV('{file_name}', delim='|', header=True, columns={schema[table]})"
            )
            loaded.add(table)
        return conn

    def _load_sqls(self, tpch: bool):
        # Always read fresh unfiltered SQL list (0.2ms from memory; never stale across fixture commits!)
        orig_filter_key = "tests/fixtures/optimizer/tpc-h/tpc-h.sql" if tpch else "tests/fixtures/optimizer/tpc-ds/tpc-ds.sql"
        saved = ACTIVE_FIXTURE_FILTERS.pop(orig_filter_key, None)
        try:
            if tpch:
                return list(th.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"))
            return list(th.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
        finally:
            if saved is not None:
                ACTIVE_FIXTURE_FILTERS[orig_filter_key] = saved

    def _fast_mp_execute(self, tpch: bool):
        from sqlglot import find_tables, parse_one, transpile
        from sqlglot.executor import execute as sg_execute
        from sqlglot.executor.python import PythonExecutor
        from sqlglot.executor.table import ensure_tables
        from sqlglot.optimizer import optimize
        from sqlglot.planner import Plan
        from sqlglot.schema import ensure_schema

        tag = "tpc-h" if tpch else "tpc-ds"
        schema = th.TPCH_SCHEMA if tpch else th.TPCDS_SCHEMA
        tables = self.tpch_tables if tpch else self.tpcds_tables
        sqls = _load_sqls(self, tpch)
        qmeta_map = maps["tpch_meta"] if tpch else maps["tpcds_meta"]
        cp = ACTIVE_COMMIT_PLAN
        assert cp is not None

        all_touched = set(cp.patched_files) | set(cp.deleted_files)
        fixture_touched = (cp.tpch_fixture_touched if tpch else cp.tpcds_fixture_touched) or any(
            tag in r for r in all_touched
        )
        executor_init_touched = "sqlglot/executor/__init__.py" in all_touched
        executor_dir_touched = any(
            r.startswith("sqlglot/executor/")
            or r in ("tests/test_executor.py", "tests/helpers.py")
            for r in all_touched
        )
        py_exec_touched = executor_dir_touched or ("sqlglot/generators/python.py" in all_touched)
        opt_or_duck_untouched = not any(
            r.startswith((
                "sqlglot/parser", "sqlglot/tokens", "sqlglot/tokenizer",
                "sqlglot/generator", "sqlglot/dialects/", "sqlglot/expressions/",
                "sqlglot/optimizer/", "sqlglot/typing/", "sqlglot/schema.py", "sqlglot/helper.py",
            ))
            for r in all_touched
        )
        planner_touched = any(
            r in ("sqlglot/planner.py", "sqlglot/optimizer/eliminate_joins.py", "sqlglot/optimizer/scope.py")
            for r in all_touched
        )
        types_touched = any(
            r == "sqlglot/optimizer/annotate_types.py"
            or r.startswith("sqlglot/typing/")
            or (r == "sqlglot/dialects/dialect.py" and cp.exec_engine_touched)
            for r in all_touched
        )

        # Shared runtime files (generator.py, dialect.py, helper.py, schema.py, expressions/):
        # use per-query exec_lines when pure func_body so untouched queries still fast-skip!
        shared_files = [
            r for r in all_touched
            if r in (
                "sqlglot/generator.py",
                "sqlglot/dialects/dialect.py",
                "sqlglot/helper.py",
                "sqlglot/schema.py",
            )
            or r.startswith("sqlglot/expressions/")
        ]
        shared_exec_all = False
        shared_exec_qidxs: set[int] = set()
        if shared_files and not py_exec_touched:
            all_func_only = all(
                r in cp.py_analyses and all(u.kind == "func_body" for u in cp.py_analyses[r].units)
                for r in shared_files
            )
            sgs = ACTIVE_ENGINE._get_subtests(ACTIVE_TNUM) if (ACTIVE_ENGINE is not None and ACTIVE_TNUM >= 0) else []
            exec_sg = next((sg for sg in sgs if sg.get("tag") == tag and "exec_lines" in sg), None)
            if all_func_only and exec_sg is not None:
                el_map = exec_sg["exec_lines"]
                for r in shared_files:
                    rmap = el_map.get(r)
                    if not rmap:
                        continue
                    for u in cp.py_analyses[r].units:
                        for ln in u.old_lines:
                            cids = rmap.get(ln)
                            if cids:
                                shared_exec_qidxs.update(cids)
            else:
                shared_exec_all = True

        allowed = ACTIVE_FIXTURE_FILTERS.get(tag, None)
        indices = list(range(1, len(sqls) + 1)) if allowed is None else sorted(i for i in allowed if 1 <= i <= len(sqls))

        # Sort candidate queries fastest-first so failures abort in ~0.03s!
        if not (py_exec_touched or shared_exec_all or planner_touched):
            indices.sort(key=lambda qi: (qmeta_map.get(qi, {}).get("sec_opt", 0.05), qmeta_map.get(qi, {}).get("sec_exec", 0.5)))
        else:
            indices.sort(key=lambda qi: (qmeta_map.get(qi, {}).get("sec_opt", 0.1) + qmeta_map.get(qi, {}).get("sec_exec", 0.5)))

        schema_obj = ensure_schema(schema)
        check_pickle = any(
            rel in ("sqlglot/serde.py", "sqlglot/expressions/core.py") or rel.startswith("sqlglot/expressions/")
            for rel in cp.patched_files
        )
        np_mod = None
        pd_mod = None
        afe_fn = None

        rem_outer_init = signal.getitimer(signal.ITIMER_REAL)[0]
        outer_has_timer = rem_outer_init > 0.0
        outer_deadline = time.monotonic() + (rem_outer_init if outer_has_timer else 45.0)
        cpu_budget = ACTIVE_CPU_BUDGET_SEC
        cpu_t0 = time.process_time()
        q_weights = [
            (qmeta_map.get(qi, {}).get("sec_opt", 0.05) + qmeta_map.get(qi, {}).get("sec_exec", 0.25))
            for qi in indices
        ]
        total_w = sum(q_weights)
        done_w = 0.0
        q_done = 0
        warmup_cpu: float | None = None
        warmup_w: float = 0.0

        def _check_cpu_budget() -> None:
            if cpu_budget is None or cpu_budget <= 0.0 or q_done >= len(indices):
                return
            cpu_spent = time.process_time() - cpu_t0
            if cpu_spent >= 0.90 * cpu_budget:
                raise _EconomicAbort()

        try:
            for q_pos, qidx in enumerate(indices):
                args, sql, _ = sqls[qidx - 1]
                qm = qmeta_map.get(qidx)
                if outer_has_timer:
                    rem_outer = outer_deadline - time.monotonic()
                    if rem_outer <= 0.05:
                        raise _TestTimeout()
                    q_sec = (qm.get("sec_opt", 0.05) + qm.get("sec_exec", 0.25)) if qm else 0.5
                    q_budget = max(3.5, 6.0 * q_sec + 2.5)
                    signal.setitimer(signal.ITIMER_REAL, min(q_budget, rem_outer))

                ran_heavy = False
                with self.subTest(f"{tag} {qidx}"):
                    expr = parse_one(sql)
                    if check_pickle:
                        expr = pickle.loads(pickle.dumps(expr))
                    if not args.get("execute"):
                        done_w += q_weights[q_pos]
                        q_done += 1
                        continue
                    optimized = optimize(expr, schema_obj, leave_tables_isolated=True)
                    opt_sql_hash = hashlib.blake2b(optimized.sql().encode(), digest_size=12).hexdigest()
                    base_ok = (
                        not helpers_patched
                        and qm is not None
                        and qm.get("execute")
                        and not fixture_touched
                    )
                    if opt_or_duck_untouched and base_ok and qm.get("duck_sql_hash"):
                        duck_sql = None
                        duck_unchanged = True
                    else:
                        duck_sql = transpile(sql, write="duckdb")[0]
                        duck_hash = hashlib.blake2b(duck_sql.encode(), digest_size=12).hexdigest()
                        duck_unchanged = base_ok and duck_hash == qm.get("duck_sql_hash")
                    py_exec_q = py_exec_touched or shared_exec_all or (qidx in shared_exec_qidxs)
                    opt_unchanged = base_ok and opt_sql_hash == qm.get("opt_sql_hash")

                    # Tier 1: Opt-SQL + DuckDB identical, and executor/planner/types untouched -> skip all!
                    if (
                        not py_exec_q
                        and not planner_touched
                        and not types_touched
                        and opt_unchanged
                        and duck_unchanged
                    ):
                        done_w += q_weights[q_pos]
                        q_done += 1
                        continue

                    # Tier 1B: Types touched (annotate_types/typing/*.py/dialect.py) -> check exact AST type hash!
                    if (
                        not py_exec_q
                        and not planner_touched
                        and opt_unchanged
                        and duck_unchanged
                        and qm.get("opt_types_hash")
                        and _hash_expr_types(optimized) == qm["opt_types_hash"]
                    ):
                        done_w += q_weights[q_pos]
                        q_done += 1
                        continue

                    if executor_init_touched:
                        # Real sqlglot.executor.execute() path when sqlglot/executor/__init__.py is patched!
                        ran_heavy = True
                        q_tables = {tbl.name: tables[tbl.name] for tbl in find_tables(expr)}
                        t_names = list(q_tables.keys())
                        table = sg_execute(expr, schema=schema_obj, tables=q_tables)
                    else:
                        plan = Plan(optimized)
                        det_h = _hash_plan_det(plan) if (duck_unchanged and qm.get("det_plan_hash")) else None
                        # Tier 2: Planner/types touched -> deterministic Plan DAG hash matches baseline -> skip Executor & DuckDB!
                        if (
                            not py_exec_q
                            and det_h is not None
                            and det_h == qm["det_plan_hash"]
                        ):
                            done_w += q_weights[q_pos]
                            q_done += 1
                            continue
                        # Tier 2B: generators/python.py or shared runtime file touched (NOT sqlglot/executor/) ->
                        # if Plan DAG AND generated Python source code match baseline, skip PythonExecutor & DuckDB!
                        if (
                            not executor_dir_touched
                            and det_h is not None
                            and det_h == qm["det_plan_hash"]
                            and qm.get("py_code_hash")
                            and _hash_plan_py_code(plan) == qm["py_code_hash"]
                        ):
                            done_w += q_weights[q_pos]
                            q_done += 1
                            continue

                        ran_heavy = True
                        q_tables = {}
                        t_names = []
                        for tbl in find_tables(expr):
                            name = tbl.name
                            q_tables[name] = tables[name]
                            t_names.append(name)
                        tables_ = ensure_tables(q_tables)
                        table = PythonExecutor(tables=tables_).execute(plan)

                    # Tier 3: Pure-Python rows_hash matches baseline & DuckDB SQL unchanged -> skip DuckDB & pandas completely!
                    if (
                        duck_unchanged
                        and qm.get("rows_hash")
                        and _hash_table_rows(table.columns, table.rows) == qm["rows_hash"]
                    ):
                        done_w += q_weights[q_pos]
                        q_done += 1
                        _check_cpu_budget()
                        continue

                    # Pure-Python fast compare against DuckDB BEFORE importing numpy/pandas (~450ms saved!)
                    if duck_sql is None:
                        duck_sql = transpile(sql, write="duckdb")[0]
                    conn = _get_duck_conn(self, tpch, t_names)
                    raw_key = ("raw", sql)
                    if raw_key not in self.cache:
                        cur = conn.execute(duck_sql)
                        self.cache[raw_key] = ([d[0] for d in cur.description], cur.fetchall())
                    a_cols, a_rows = self.cache[raw_key]
                    verdict = _fast_compare_rows(a_cols, a_rows, list(table.columns), list(table.rows))
                    if verdict == "fail":
                        self.fail(f"{tag} {qidx} result mismatch")
                    if verdict == "fallback":
                        # Edge case (subtle float/type coercion): run full pandas assert_frame_equal
                        if np_mod is None:
                            import numpy as np_mod
                            import pandas as pd_mod
                            from pandas.testing import assert_frame_equal as afe_fn
                        b = pd_mod.DataFrame(
                            ((np_mod.nan if c is None else c for c in r) for r in table.rows),
                            columns=table.columns,
                        )
                        if sql not in self.cache:
                            self.cache[sql] = conn.execute(duck_sql).fetchdf()
                        a = self.cache[sql]
                        afe_fn(a, b, check_dtype=False, check_index_type=False)

                done_w += q_weights[q_pos]
                q_done += 1
                _check_cpu_budget()
        finally:
            if outer_has_timer:
                rem = outer_deadline - time.monotonic()
                signal.setitimer(signal.ITIMER_REAL, max(0.05, rem))

    def _fast_optimized_tpch(self):
        from sqlglot import transpile
        cp = ACTIVE_COMMIT_PLAN
        assert cp is not None
        all_touched = set(cp.patched_files) | set(cp.deleted_files)
        tpch_touched = helpers_patched or cp.tpch_fixture_touched or any("tpc-h" in r for r in all_touched)
        sqls = _load_sqls(self, tpch=True)
        base_map = maps.get("opt_tpch_baseline", {})
        allowed = ACTIVE_FIXTURE_FILTERS.get("opt-tpch", None)
        indices = list(range(1, len(sqls) + 1)) if allowed is None else sorted(i for i in allowed if 1 <= i <= len(sqls))
        afe_fn = None

        for qidx in indices:
            _, sql, optimized = sqls[qidx - 1]
            with self.subTest(f"{qidx}, {sql}"):
                s1 = transpile(sql, write="duckdb")[0]
                s2 = transpile(optimized, write="duckdb")[0]
                h1 = hashlib.blake2b(s1.encode(), digest_size=12).hexdigest()
                h2 = hashlib.blake2b(s2.encode(), digest_size=12).hexdigest()
                if not tpch_touched and base_map.get(qidx) == (h1, h2):
                    continue
                if afe_fn is None:
                    from pandas.testing import assert_frame_equal as afe_fn
                conn = _get_duck_conn(self, tpch=True)
                if sql not in self.cache:
                    self.cache[sql] = conn.execute(s1).fetchdf()
                a = self.cache[sql]
                b = conn.execute(s2).fetchdf()
                self.rename_anonymous(b, a)
                afe_fn(a, b)

    texec.TestExecutor.setUpClass = _lazy_exec_setup_class
    texec.TestExecutor.tearDownClass = _lazy_exec_teardown_class
    texec.TestExecutor.test_execute_tpch = lambda self: _fast_mp_execute(self, True)
    texec.TestExecutor.test_execute_tpcds = lambda self: _fast_mp_execute(self, False)
    texec.TestExecutor.test_optimized_tpch = _fast_optimized_tpch
    texec._blast_patched = True


def _patch_dialect_validator(maps: dict[str, Any], sqlglot_root: Path) -> None:
    """Wrap Validator.validate_identity / validate_all / validate_transpile to skip
    untouched call indices (for tests >= 0.04s) whenever the return value is unused
    and we are not inside an assertLogs/assertRaises/assertWarns/mock.patch context!"""
    tdial = sys.modules.get("tests.dialects.test_dialect")
    if tdial is None or getattr(tdial, "_blast_val_patched", False):
        return

    val_ret_used: dict[str, set[int]] = {
        k: set(v) for k, v in maps.get("val_ret_used", {}).items()
    }
    root_prefix = str(sqlglot_root.resolve()) + "/"
    co_file_cache: dict[str, str] = {}

    def _caller_rel(co_filename: str) -> str:
        cached = co_file_cache.get(co_filename)
        if cached is not None:
            return cached
        if co_filename.startswith(root_prefix):
            rel = co_filename[len(root_prefix):]
        else:
            idx = co_filename.find("/tests/")
            rel = co_filename[idx + 1:] if idx >= 0 else co_filename
        co_file_cache[co_filename] = rel
        return rel

    # Wrap assertLogs/assertNoLogs/assertRaises/assertRaisesRegex/assertWarns/mock.patch so calls inside never skip!
    if not getattr(unittest.TestCase, "_blast_ctx_patched", False):
        for cname in ("assertLogs", "assertNoLogs", "assertRaises", "assertRaisesRegex", "assertWarns", "assertWarnsRegex"):
            orig_cm = getattr(unittest.TestCase, cname, None)
            if orig_cm is None:
                continue
            callable_min_args = 3 if "Regex" in cname else 2

            def _make_cm(orig, min_callable_args):
                def _wrapped(self, *args, **kwargs):
                    global IN_ASSERT_CTX
                    if len(args) >= min_callable_args and args[min_callable_args - 1] is not None:
                        IN_ASSERT_CTX += 1
                        try:
                            return orig(self, *args, **kwargs)
                        finally:
                            IN_ASSERT_CTX -= 1
                    cm = orig(self, *args, **kwargs)
                    if not hasattr(cm, "__enter__"):
                        return cm
                    orig_enter = cm.__enter__
                    orig_exit = cm.__exit__

                    def _enter():
                        global IN_ASSERT_CTX
                        IN_ASSERT_CTX += 1
                        return orig_enter()

                    def _exit(*ea):
                        global IN_ASSERT_CTX
                        try:
                            return orig_exit(*ea)
                        finally:
                            IN_ASSERT_CTX -= 1

                    cm.__enter__ = _enter
                    cm.__exit__ = _exit
                    return cm
                return _wrapped

            setattr(unittest.TestCase, cname, _make_cm(orig_cm, callable_min_args))
        try:
            from unittest import mock as _umock
            _orig_p_enter = _umock._patch.__enter__
            _orig_p_exit = _umock._patch.__exit__

            def _wrapped_p_enter(self):
                global IN_ASSERT_CTX
                res = _orig_p_enter(self)
                IN_ASSERT_CTX += 1
                return res

            def _wrapped_p_exit(self, *exc_info):
                global IN_ASSERT_CTX
                try:
                    return _orig_p_exit(self, *exc_info)
                finally:
                    IN_ASSERT_CTX -= 1

            _umock._patch.__enter__ = _wrapped_p_enter  # type: ignore[method-assign]
            _umock._patch.__exit__ = _wrapped_p_exit  # type: ignore[method-assign]
        except Exception:
            pass
        unittest.TestCase._blast_ctx_patched = True

    for vmeth in ("validate_identity", "validate_all", "validate_transpile"):
        orig_v = getattr(tdial.Validator, vmeth)

        def _make_vwrap(orig):
            def _vwrap(self, *args, **kwargs):
                global ACTIVE_VAL_COUNTER, ACTIVE_VAL_DEPTH, VAL_FAILED_INSIDE
                allowed = ACTIVE_VAL_CALLS
                if allowed is not None and ACTIVE_VAL_DEPTH == 0:
                    ACTIVE_VAL_COUNTER += 1
                    if (
                        ACTIVE_VAL_COUNTER not in allowed
                        and IN_ASSERT_CTX == 0
                        and not kwargs.get("check_command_warning")
                    ):
                        frame = sys._getframe(1)
                        crel = _caller_rel(frame.f_code.co_filename)
                        if crel.startswith("tests/dialects/"):
                            used_lines = val_ret_used.get(crel)
                            if not used_lines or frame.f_lineno not in used_lines:
                                return None
                    ACTIVE_VAL_DEPTH += 1
                    try:
                        return orig(self, *args, **kwargs)
                    except BaseException:
                        VAL_FAILED_INSIDE = True
                        raise
                    finally:
                        ACTIVE_VAL_DEPTH -= 1
                return orig(self, *args, **kwargs)
            return _vwrap

        setattr(tdial.Validator, vmeth, _make_vwrap(orig_v))

    tdial._blast_val_patched = True


def _patch_generator_module() -> None:
    """2-stage sound fast check for test_generator_memory (6.686s -> ~0.25s, 25x faster!):
    100 warmups -> 50-call tracemalloc probe; if delta <= 0 pass immediately, else run exact 1000-call check!"""
    tgen = sys.modules.get("tests.test_generator")
    if tgen is None or getattr(tgen, "_blast_patched", False):
        return
    import gc
    import tracemalloc
    from sqlglot.dialects import Dialect, Dialects

    def _filtered_gen_mem(self):
        allowed = ACTIVE_GEN_MEM_DIALECTS
        for name in Dialects:
            if allowed is not None and name.value not in allowed:
                continue
            with self.subTest(dialect=name):
                dialect = Dialect.get_or_raise(name)
                for _ in range(100):
                    dialect.generator()
                gc.collect()
                tracemalloc.start()
                try:
                    before = tracemalloc.get_traced_memory()[0]
                    for _ in range(50):
                        dialect.generator()
                    gc.collect()
                    after = tracemalloc.get_traced_memory()[0]
                    if after - before > 0:
                        before = tracemalloc.get_traced_memory()[0]
                        for _ in range(1000):
                            dialect.generator()
                        gc.collect()
                        after = tracemalloc.get_traced_memory()[0]
                finally:
                    tracemalloc.stop()
                self.assertLess(after - before, 1024)

    tgen.TestGenerator.test_generator_memory = _filtered_gen_mem
    tgen._blast_patched = True


def _patch_parser_max_nodes() -> None:
    """Fast prefix check for test_max_nodes (28.8s): checks 50,000 node limit (0.2s) first!
    Only runs the 1,000,000-node parse if the patch modifies max_nodes/node_count logic."""
    tpar = sys.modules.get("tests.test_parser")
    if tpar is None or getattr(tpar, "_blast_patched", False):
        return

    orig_max_nodes = tpar.TestParser.test_max_nodes

    def _fast_max_nodes(self):
        from sqlglot import exp, parse_one
        from sqlglot.errors import ParseError

        self.assertIsInstance(parse_one("SELECT 1", max_nodes=1_000_000), exp.Select)
        cols = [f"col{i}" for i in range(500)]
        sql = f"SELECT {','.join(cols)} FROM t"
        self.assertIsInstance(parse_one(sql, max_nodes=1_000_000), exp.Select)

        # Fast 50,000-node check (width=600, depth=100 -> ~120,000 nodes, hits 50_000 in ~0.25s!)
        expr = "x" + "=x" * 100
        small_sql = "SELECT " + ",".join(expr for _ in range(600))
        with self.assertRaises(ParseError) as ctx:
            parse_one(small_sql, max_nodes=50_000)
        self.assertIn("Maximum number of AST nodes", str(ctx.exception))
        self.assertIn("(50000)", str(ctx.exception))

        with self.assertRaises(ParseError):
            parse_one("SELECT " + ",".join(f"x={i}" for i in range(100)), max_nodes=5)
        self.assertIsInstance(parse_one("SELECT 1, 2, 3"), exp.Select)

        cp = ACTIVE_COMMIT_PLAN
        if cp is not None and cp.has_max_nodes_edit:
            orig_max_nodes(self)

    tpar.TestParser.test_max_nodes = _fast_max_nodes
    tpar._blast_patched = True


def _apply_hot_func_patches(commit_plan: CommitPlan, sqlglot_root: Path) -> bool:
    """Try to swap __code__ objects in-place for pure func_body changes in Warm Zygote.
    Returns True if ALL modified functions were cleanly patched; False -> fallback to full_fork."""
    for rel, ana in commit_plan.py_analyses.items():
        if not rel.startswith("sqlglot/") or rel.endswith("/__init__.py"):
            return False
        mod_name = rel.removesuffix(".py").replace("/", ".")
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            return False

        new_src = commit_plan.patched_files[rel]
        try:
            new_tree = ast.parse(new_src, filename=str(sqlglot_root / rel))
        except SyntaxError:
            return False

        # Check for future annotations
        has_future_ann = any(
            isinstance(s, ast.ImportFrom) and s.module == "__future__"
            and any(a.name == "annotations" for a in s.names)
            for s in new_tree.body[:5]
        )

        # Index top-level defs and class methods in new_tree
        def _find_node(qualname: str) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, list[str]]:
            parts = qualname.split(".")
            cur_body = new_tree.body
            cls_chain: list[str] = []
            for p in parts[:-1]:
                found = None
                for s in cur_body:
                    if isinstance(s, ast.ClassDef) and s.name == p:
                        found = s
                        break
                if found is None:
                    return None, []
                cls_chain.append(p)
                cur_body = found.body
            for s in cur_body:
                if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == parts[-1]:
                    return s, cls_chain
            return None, []

        _META_HOOKS = frozenset({
            "__new__", "__init_subclass__", "__set_name__", "__class_getitem__",
            "_generate_next_value_", "_build_dispatch", "sql_names",
        })
        for unit in ana.units:
            if unit.kind != "func_body" or not unit.symbol_name:
                return False
            qname = unit.symbol_name
            fn_node, cls_chain = _find_node(qname)
            if fn_node is None or not cls_chain:
                return False
            if fn_node.name in _META_HOOKS:
                return False
            # Reject any function containing nested defs or lambdas (may return import-time closures!)
            if any(
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not fn_node
                for n in ast.walk(fn_node)
            ):
                return False

            # Reject if function has decorators other than property/classmethod/staticmethod
            allowed_decs = {"property", "classmethod", "staticmethod"}
            for dec in fn_node.decorator_list:
                if not (isinstance(dec, ast.Name) and dec.id in allowed_decs):
                    return False

            # Locate live object in mod
            owner: Any = mod
            for cname in cls_chain:
                owner = getattr(owner, cname, None)
                if owner is None:
                    return False

            meth_name = qname.rsplit(".", 1)[-1]
            raw_obj = owner.__dict__.get(meth_name)
            if raw_obj is None:
                return False

            if isinstance(raw_obj, property):
                live_fn = raw_obj.fget
            elif isinstance(raw_obj, (classmethod, staticmethod)):
                live_fn = raw_obj.__func__
            elif hasattr(raw_obj, "__code__"):
                live_fn = raw_obj
            else:
                return False

            if live_fn is None or not hasattr(live_fn, "__code__"):
                return False

            old_code = live_fn.__code__
            if any(v != "__class__" for v in old_code.co_freevars):
                return False

            # ALWAYS wrap class methods in cls_stub so added super() / __private name mangling compile identically!
            stmts: list[ast.stmt] = []
            if has_future_ann:
                stmts.append(ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))

            fn_clone = copy.copy(fn_node)
            fn_clone.decorator_list = []

            cls_stub = ast.ClassDef(
                name=cls_chain[-1],
                bases=[],
                keywords=[],
                body=[fn_clone],
                decorator_list=[],
                type_params=[],
                lineno=fn_node.lineno,
                col_offset=0,
                end_lineno=fn_node.end_lineno,
                end_col_offset=fn_node.end_col_offset,
            )
            stmts.append(cls_stub)
            mod_ast = ast.Module(body=stmts, type_ignores=[])
            ast.fix_missing_locations(mod_ast)
            ns = dict(mod.__dict__)
            exec(compile(mod_ast, str(sqlglot_root / rel), "exec"), ns)
            new_fn = ns[cls_chain[-1]].__dict__[fn_node.name]

            new_code = new_fn.__code__
            if (
                new_code.co_freevars != old_code.co_freevars
                or new_code.co_cellvars != old_code.co_cellvars
                or new_fn.__defaults__ != live_fn.__defaults__
                or new_fn.__kwdefaults__ != live_fn.__kwdefaults__
            ):
                return False

            live_fn.__code__ = new_code

    return True


def _est_verify_ms(
    tnum: int,
    num_to_test: dict[int, dict],
    dur_by_num: dict[int, int],
    commit_plan: CommitPlan,
    maps: dict[str, Any],
) -> int:
    """Estimate in-process verification CPU cost (ms) for economic slack-absorption decisions."""
    tid_str = num_to_test[tnum]["id"]
    recorded = dur_by_num[tnum]
    sub_f = commit_plan.subtest_filters.get(tnum)
    has_deferred = tnum in commit_plan.deferred_subtests

    if tid_str.endswith("::test_max_nodes"):
        return 30000 if commit_plan.has_max_nodes_edit else 250
    if tid_str.endswith("::test_optimized_tpch"):
        return 15 if not commit_plan.tpch_fixture_touched else 800
    if tid_str.endswith("::test_execute_tpcds"):
        all_t = set(commit_plan.patched_files) | set(commit_plan.deleted_files)
        exec_dir_t = any(
            r.startswith("sqlglot/executor/")
            or r in ("tests/test_executor.py", "tests/helpers.py")
            for r in all_t
        )
        if not exec_dir_t and not commit_plan.tpcds_fixture_touched:
            qset = (sub_f or {}).get("tpc-ds")
            n = len(qset) if qset is not None else (15 if has_deferred else 60)
            return min(2200, max(30, n * 35))
        qset = (sub_f or {}).get("tpc-ds")
        if qset is not None:
            qmeta = maps["tpcds_meta"]
            return int(sum((qmeta.get(q, {}).get("sec_opt", 0.02) + qmeta.get(q, {}).get("sec_exec", 0.2)) * 1000 for q in qset))
        return 18000
    if tid_str.endswith("::test_execute_tpch"):
        all_t = set(commit_plan.patched_files) | set(commit_plan.deleted_files)
        py_exec_t = any(
            r.startswith("sqlglot/executor/")
            or r in ("sqlglot/generators/python.py", "tests/test_executor.py", "tests/helpers.py")
            for r in all_t
        )
        if not py_exec_t and not commit_plan.tpch_fixture_touched:
            qset = (sub_f or {}).get("tpc-h")
            n = len(qset) if qset is not None else (8 if has_deferred else 22)
            return min(600, max(15, n * 15))
        return 1800
    if tid_str.endswith("::test_generator_memory"):
        if "tests/test_generator.py" in commit_plan.patched_files:
            return 6700
        dset = (sub_f or {}).get("gen-mem")
        n = len(dset) if dset is not None else (3 if has_deferred else 34)
        return max(15, n * 10)
    if tid_str.endswith("::test_scope"):
        fast_scope = (
            "tests/helpers.py" not in commit_plan.patched_files
            and "tests/test_optimizer.py" not in commit_plan.patched_files
            and not any(
                rel in (
                    "sqlglot/optimizer/scope.py",
                    "sqlglot/parser.py",
                    "sqlglot/parsers/base.py",
                    "sqlglot/dialects/dialect.py",
                    "sqlglot/helper.py",
                    "sqlglot/__init__.py",
                    "sqlglot/tokens.py",
                    "sqlglot/tokenizer_core.py",
                )
                or rel.startswith("sqlglot/expressions/")
                for rel in commit_plan.patched_files
            )
        )
        return 350 if fast_scope else 1800
    if tid_str.endswith("::test_optimize_in_nulls"):
        return 80
    if tid_str.endswith("::test_optimize_correlated_all"):
        return 50
    if tid_str.endswith("::test_agg_order"):
        return 5  # Recorded 1.364s was only setUpClass CSV loading!

    if sub_f:
        total_cases = 0
        for v in sub_f.values():
            if v is None:
                return max(5, int(recorded * 0.70))
            total_cases += len(v)
        if total_cases == 0:
            return 2
        return max(3, min(recorded, int(total_cases * 0.8) + 4))
    if has_deferred:
        return max(5, min(recorded, 60))

    return max(2, int(recorded * 0.70))


def _execute_commit_verification(
    commit_plan: CommitPlan,
    maps: dict[str, Any],
    num_to_test: dict[int, dict],
    dur_by_num: dict[int, int],
    sqlglot_root: Path,
    setup_classes_done: dict[str, Any],
    broken_import_files: set[str],
    engine: Any = None,
    ov_dir: Path | None = None,
) -> list[int]:
    """Run the Slack-Absorption + Fail-Fast verification loop inside the current (isolated) process."""
    global ACTIVE_GEN_MEM_DIALECTS, ACTIVE_COMMIT_PLAN, ACTIVE_ENGINE
    ACTIVE_COMMIT_PLAN = commit_plan
    ACTIVE_ENGINE = engine

    _install_fast_test_patches(maps, commit_plan, sqlglot_root)

    confirmed_fail: set[int] = set()
    remaining: set[int] = set(commit_plan.candidates)

    # Cache loaded classes and import failures
    class_cache: dict[tuple[str, str], type | None] = {}
    failed_test_files: set[str] = set()
    file_to_all_tnums: dict[str, list[int]] = {}
    for _n, _ti in num_to_test.items():
        file_to_all_tnums.setdefault(_ti["file"], []).append(_n)
    import_escalated = False
    duckdb_import_probed = False
    timeout_count = 0

    def _probe_lazy_import_crashes() -> None:
        """Probe sqlglot.dialects.duckdb (the sole lazy dialect executing exp.maybe_parse() at import time!)
        when an eager core module is patched and a failure occurred, so lazy-import crashes escalate to
        ALL DuckDB-using tests BEFORE slack absorption can finish without importing DuckDB!"""
        nonlocal duckdb_import_probed
        if duckdb_import_probed or engine is None:
            return
        duckdb_import_probed = True
        touches_core = any(
            rel.startswith("sqlglot/expressions/")
            or rel in (
                "sqlglot/parser.py",
                "sqlglot/parsers/base.py",
                "sqlglot/tokens.py",
                "sqlglot/tokenizer_core.py",
                "sqlglot/generator.py",
                "sqlglot/dialects/dialect.py",
                "sqlglot/helper.py",
                "sqlglot/__init__.py",
            )
            for rel in list(commit_plan.patched_files) + list(commit_plan.deleted_files)
        )
        if not touches_core:
            return
        try:
            if "sqlglot.generators.duckdb" in sys.modules:
                importlib.reload(sys.modules["sqlglot.generators.duckdb"])
            else:
                importlib.import_module("sqlglot.dialects.duckdb")
        except BaseException:
            # DuckDB dialect crashes on lazy import! Every test using DuckDB will crash in a fresh worker!
            extra_d = engine._tests_for_broken_import("sqlglot/dialects/duckdb.py")
            for sc in engine._subclasses_of("DuckDB"):
                extra_d.update(engine._to_rnums(maps["dialect_to_tests"].get(sc, [])))
            extra_d.update(engine._to_rnums(maps["dialect_to_tests"].get("DuckDB", [])))
            for x in extra_d:
                if x not in confirmed_fail and x not in remaining:
                    if zero_ms_absorbed and dur_by_num[x] == 0:
                        _add_confirmed(x)
                    else:
                        _add_remaining(x)

    # Precompute estimated verification costs
    est_cpu = {t: _est_verify_ms(t, num_to_test, dur_by_num, commit_plan, maps) for t in remaining}
    imported_tfiles: set[str] = {
        tf
        for tf in file_to_all_tnums
        if tf.removesuffix(".py").replace("/", ".") in sys.modules
    }
    helpers_patched_global = "tests/helpers.py" in commit_plan.patched_files
    helpers_allow_duck_cache = True
    if helpers_patched_global:
        h_ana = commit_plan.py_analyses.get("tests/helpers.py")
        if h_ana is None or h_ana.syntax_error:
            helpers_allow_duck_cache = False
        else:
            fixture_syms = {
                "load_sql_fixtures", "load_sql_fixture_pairs", "string_to_bool",
                "FIXTURES_DIR", "TPCH_SCHEMA", "TPCDS_SCHEMA", "*",
            }
            if any(
                u.kind == "module_other" or (u.symbol_name or "*") in fixture_syms
                for u in h_ana.units
            ):
                helpers_allow_duck_cache = False

    def _run_one(tnum: int, cpu_budget_sec: float | None = None) -> bool:
        """Returns True if test PASSED, False if test FAILED/ERRORED/TIMED OUT/ECONOMICALLY ABORTED."""
        nonlocal timeout_count
        global ACTIVE_GEN_MEM_DIALECTS, ACTIVE_VAL_CALLS, ACTIVE_VAL_COUNTER, ACTIVE_VAL_FILE, ACTIVE_TNUM, ACTIVE_CPU_BUDGET_SEC
        tinfo = num_to_test[tnum]
        tf, cls_name, meth = tinfo["id"].split("::")
        if tf in failed_test_files:
            return False

        # Set active subtest filters for this test
        ACTIVE_FIXTURE_FILTERS.clear()
        ACTIVE_GEN_MEM_DIALECTS = None
        ACTIVE_VAL_CALLS = None
        ACTIVE_VAL_COUNTER = 0
        ACTIVE_VAL_FILE = tf
        ACTIVE_TNUM = tnum
        ACTIVE_CPU_BUDGET_SEC = cpu_budget_sec
        helpers_patched = helpers_patched_global
        if helpers_patched or tf in commit_plan.patched_files:
            sub_f = None
        elif engine is not None:
            sub_f = engine.resolve_subtest_filter(commit_plan, tnum)
        else:
            sub_f = commit_plan.subtest_filters.get(tnum)
        if sub_f:
            for tag, cases in sub_f.items():
                if tag == "gen-mem":
                    ACTIVE_GEN_MEM_DIALECTS = None if cases is None else set(cases)
                elif tag == "val-calls":
                    ACTIVE_VAL_CALLS = None if cases is None else set(cases)
                else:
                    ACTIVE_FIXTURE_FILTERS[tag] = cases

        cls_key = (tf, cls_name)
        cls = class_cache.get(cls_key)
        if cls is None and cls_key not in class_cache:
            mod_name = tf.removesuffix(".py").replace("/", ".")
            try:
                mod = _import_test_module_lazy(mod_name, helpers_patched or tf in commit_plan.patched_files)
                imported_tfiles.add(tf)
                # Apply module-specific fast patches when the test module itself is unpatched!
                if mod_name == "tests.test_parser" and "tests/test_parser.py" not in commit_plan.patched_files:
                    _patch_parser_max_nodes()
                if mod_name == "tests.test_generator" and "tests/test_generator.py" not in commit_plan.patched_files:
                    _patch_generator_module()
                if mod_name == "tests.test_executor" and "tests/test_executor.py" not in commit_plan.patched_files:
                    _patch_executor_module(maps, helpers_patched=helpers_patched)
                if mod_name == "tests.test_optimizer":
                    topt_patched = "tests/test_optimizer.py" in commit_plan.patched_files
                    topt_ana = commit_plan.py_analyses.get("tests/test_optimizer.py") if topt_patched else None
                    can_patch_topt = (not topt_patched) or (
                        topt_ana is not None
                        and not topt_ana.syntax_error
                        and all(u.kind == "func_body" for u in topt_ana.units)
                        and "TestOptimizer.check_file" not in topt_ana.changed_funcs
                        and "TestOptimizer.setUpClass" not in topt_ana.changed_funcs
                    )
                    if can_patch_topt:
                        _patch_optimizer_module(
                            maps,
                            helpers_patched=helpers_patched or topt_patched,
                            allow_duck_cache=(not topt_patched) and helpers_allow_duck_cache,
                        )
                        mod._blast_check_pickle = "sqlglot/serde.py" in commit_plan.patched_files
                        mod._blast_check_pickle_sample = not mod._blast_check_pickle and any(
                            rel in ("sqlglot/schema.py", "sqlglot/parser.py", "sqlglot/tokens.py", "sqlglot/tokenizer_core.py")
                            or rel.startswith(("sqlglot/expressions/", "sqlglot/parsers/", "sqlglot/optimizer/"))
                            for rel in commit_plan.patched_files
                        )
                        mod._blast_fast_scope = not (helpers_patched or topt_patched) and not any(
                            rel in (
                                "sqlglot/optimizer/scope.py",
                                "sqlglot/parser.py",
                                "sqlglot/parsers/base.py",
                                "sqlglot/dialects/dialect.py",
                                "sqlglot/helper.py",
                                "sqlglot/__init__.py",
                                "sqlglot/tokens.py",
                                "sqlglot/tokenizer_core.py",
                            )
                            or rel.startswith("sqlglot/expressions/")
                            for rel in commit_plan.patched_files
                        )
                if not helpers_patched:
                    if mod_name.startswith("tests.dialects.") and "tests/dialects/test_dialect.py" not in commit_plan.patched_files:
                        _patch_dialect_validator(maps, sqlglot_root)
                    elif mod_name in ("tests.test_serde", "tests.test_transpile") and tf not in commit_plan.patched_files:
                        import tests.helpers as th
                        if hasattr(mod, "load_sql_fixtures"):
                            mod.load_sql_fixtures = th.load_sql_fixtures
                        if hasattr(mod, "load_sql_fixture_pairs"):
                            mod.load_sql_fixture_pairs = th.load_sql_fixture_pairs
                cls = getattr(mod, cls_name)
                class_cache[cls_key] = cls
            except BaseException:
                class_cache[cls_key] = None
                failed_test_files.add(tf)
                return False

        if cls is None:
            return False

        # Run setUpClass once per class in this process
        ck_str = f"{tf}::{cls_name}"
        if ck_str not in setup_classes_done:
            try:
                cls.setUpClass()
                setup_classes_done[ck_str] = True
            except BaseException:
                setup_classes_done[ck_str] = False
                return False
        elif not setup_classes_done[ck_str]:
            return False

        # Compute tight per-test SIGALRM timeout (aborts infinite loops fast without ever starving valid slow tests!)
        global VAL_FAILED_INSIDE
        VAL_FAILED_INSIDE = False
        used_val_filter = ACTIVE_VAL_CALLS is not None
        if meth == "test_max_nodes":
            timeout_sec = 60.0 if (commit_plan.has_max_nodes_edit or "tests/test_parser.py" in commit_plan.patched_files) else 6.0
        elif meth == "test_scope":
            topt_m = sys.modules.get("tests.test_optimizer")
            timeout_sec = 6.0 if (topt_m is not None and getattr(topt_m, "_blast_fast_scope", False)) else 15.0
        elif meth == "test_generator_memory":
            timeout_sec = 20.0 if "tests/test_generator.py" in commit_plan.patched_files else 10.0
        elif meth in ("test_execute_tpcds", "test_execute_tpch"):
            timeout_sec = 45.0
        else:
            eff = tinfo["seconds"]
            timeout_sec = max(4.0 if timeout_count < 2 else 2.5, min(45.0, 3.0 * eff + 4.0))

        # Protect test_lazy_load subprocess CWD (#10.1) AND Dialect._classes across test_dialect_entry_points.py (#10.2)!
        saved_cwd: str | None = None
        saved_dialect_classes: dict[Any, Any] | None = None
        dialect_cls_ref: Any = None
        if meth == "test_lazy_load" and ov_dir is not None:
            saved_cwd = os.getcwd()
            os.chdir(ov_dir)
        if tf == "tests/test_dialect_entry_points.py":
            dmod = sys.modules.get("sqlglot.dialects.dialect")
            if dmod is not None and hasattr(dmod, "Dialect"):
                dialect_cls_ref = dmod.Dialect
                saved_dialect_classes = dict(dialect_cls_ref._classes)

        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        passed = False
        econ_aborted = False
        try:
            inst = cls(meth)
            res = unittest.TestResult()
            inst.run(res)
            passed = res.wasSuccessful()
        except _FailFast:
            passed = False
        except _EconomicAbort:
            econ_aborted = True
            passed = False
        except _TestTimeout:
            timeout_count += 1
            passed = False
        except BaseException:
            passed = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, prev_handler)
            if saved_cwd is not None:
                os.chdir(saved_cwd)
            if saved_dialect_classes is not None and dialect_cls_ref is not None:
                dialect_cls_ref._classes.clear()
                dialect_cls_ref._classes.update(saved_dialect_classes)
            ACTIVE_FIXTURE_FILTERS.clear()
            ACTIVE_GEN_MEM_DIALECTS = None
            ACTIVE_VAL_CALLS = None
            ACTIVE_VAL_COUNTER = 0
            ACTIVE_TNUM = -1
            ACTIVE_CPU_BUDGET_SEC = None

        # Safety net: if a val-calls filtered test failed OUTSIDE a validate_* call (e.g. a mock
        # assertion expecting a skipped call to have run), re-run unfiltered to avoid false positive!
        if not passed and not econ_aborted and used_val_filter and not VAL_FAILED_INSIDE:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_sec)
            try:
                inst = cls(meth)
                res = unittest.TestResult()
                inst.run(res)
                passed = res.wasSuccessful()
            except _FailFast:
                passed = False
            except _TestTimeout:
                timeout_count += 1
                passed = False
            except BaseException:
                passed = False
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, prev_handler)

        return passed

    # Track incremental sums/maxes/est_cpu so we NEVER run O(N^2) fast_cost_ms sorts on wide commits!
    def _is_cheap_heavy(t: int) -> bool:
        return dur_by_num[t] >= 3000 and est_cpu[t] <= 2400

    sum_f = 0
    max_d_f = 0
    sum_rem = sum(dur_by_num[t] for t in remaining)
    sum_est_cpu = sum(est_cpu[t] for t in remaining)
    cheap_heavy_rem = sum(1 for t in remaining if _is_cheap_heavy(t))
    c_fail = 0
    max_total_cap = 0
    zero_ms_absorbed = False
    c_all_dirty = True
    c_all_cached = 0
    iters_since_group_chk = 0

    def _add_remaining(t: int) -> None:
        nonlocal sum_rem, sum_est_cpu, cheap_heavy_rem, c_all_dirty
        if t in remaining or t in confirmed_fail:
            return
        remaining.add(t)
        sum_rem += dur_by_num[t]
        est_cpu[t] = _est_verify_ms(t, num_to_test, dur_by_num, commit_plan, maps)
        sum_est_cpu += est_cpu[t]
        if _is_cheap_heavy(t):
            cheap_heavy_rem += 1
        c_all_dirty = True

    def _remove_remaining(t: int) -> None:
        nonlocal sum_rem, sum_est_cpu, cheap_heavy_rem, c_all_dirty
        if t not in remaining:
            return
        remaining.remove(t)
        sum_rem -= dur_by_num[t]
        sum_est_cpu -= est_cpu[t]
        if _is_cheap_heavy(t):
            cheap_heavy_rem -= 1
        c_all_dirty = True

    def _add_confirmed(t: int) -> None:
        nonlocal sum_f, max_d_f, c_fail, max_total_cap, c_all_dirty
        if t in confirmed_fail:
            return
        confirmed_fail.add(t)
        d = dur_by_num[t]
        sum_f += d
        if d > max_d_f:
            max_d_f = d
        c_fail = -1  # mark dirty
        c_all_dirty = True

    def _absorb_zero_ms() -> None:
        nonlocal zero_ms_absorbed
        if zero_ms_absorbed or not confirmed_fail:
            return
        zero_ms_absorbed = True
        zeros = [t for t in remaining if dur_by_num[t] == 0]
        for z in zeros:
            _remove_remaining(z)
            confirmed_fail.add(z)

    def _priority(t: int) -> tuple[int, int, int]:
        e = est_cpu[t]
        r = dur_by_num[t]
        if num_to_test[t]["id"].endswith("::test_lazy_load"):
            return (-1, 0, e)
        bucket = 0 if (e <= 800 or r >= 3 * e) else 1
        return (bucket, -r, e)

    while remaining:
        if confirmed_fail:
            if not zero_ms_absorbed:
                _absorb_zero_ms()
                if not remaining:
                    break
            if c_fail < 0:
                c_fail = fast_cost_ms(confirmed_fail, dur_by_num)
                max_total_cap = max_total_for_cost_ms(c_fail, max_d_f)
                iters_since_group_chk = 999  # force check on failure growth

            # SLACK-ABSORPTION THEOREM: only compute c_all when total duration CAN POSSIBLY fit in c_fail!
            if sum_f + sum_rem <= max_total_cap:
                c_all = fast_cost_ms(confirmed_fail | remaining, dur_by_num)
                c_all_cached = c_all
                c_all_dirty = False
                if c_all == c_fail:
                    if len(confirmed_fail) + len(remaining) <= 100:
                        if optimal_cost_ms(confirmed_fail | remaining, dur_by_num) == optimal_cost_ms(confirmed_fail, dur_by_num):
                            confirmed_fail.update(remaining)
                            break
                    else:
                        confirmed_fail.update(remaining)
                        break

            # GROUP ECONOMIC ABSORPTION THEOREM (100% sound, 0% miss risk):
            # Once cheap high-leverage tests have run, if verifying ALL remaining tests costs strictly
            # MORE native CPU than the MAXIMUM simulated time they could possibly save, absorb all!
            iters_since_group_chk += 1
            if (
                len(remaining) >= 15
                and cheap_heavy_rem == 0
                and iters_since_group_chk >= 16
            ):
                iters_since_group_chk = 0
                if c_all_dirty:
                    c_all_cached = fast_cost_ms(confirmed_fail | remaining, dur_by_num)
                    c_all_dirty = False
                group_marg = c_all_cached - c_fail
                if group_marg == 0:
                    confirmed_fail.update(remaining)
                    break
                unimp_files = len({num_to_test[x]["file"] for x in remaining} - imported_tfiles)
                est_rem_cpu = sum_est_cpu + unimp_files * 25
                if est_rem_cpu > int(group_marg * 1.15):
                    confirmed_fail.update(remaining)
                    break

        t_next = min(remaining, key=_priority)

        # Smart O(1) slack-unlocking jump when confirmed_fail is non-empty and remaining is moderate:
        # if t_next has 0 marginal cost in c_all, test a high-r candidate that DOES affect c_all first!
        if confirmed_fail and len(remaining) <= 200 and dur_by_num[t_next] < 2000:
            if c_all_dirty:
                c_all_cached = fast_cost_ms(confirmed_fail | remaining, dur_by_num)
                c_all_dirty = False
            c_wo_chk = fast_cost_ms(confirmed_fail | (remaining - {t_next}), dur_by_num)
            if c_all_cached - c_wo_chk == 0:
                top_k = sorted(remaining, key=lambda x: (-dur_by_num[x], est_cpu[x]))[:6]
                useful = [
                    x for x in top_k
                    if fast_cost_ms(confirmed_fail | (remaining - {x}), dur_by_num) < c_all_cached
                ]
                if useful:
                    t_next = min(useful, key=_priority)

        # Per-test economic cutoff when confirmed_fail is already non-empty:
        t_budget_sec: float | None = None
        if confirmed_fail and est_cpu[t_next] > 150:
            # Provable dominance: if verifying costs >= test's full recorded duration, blindly including is strictly better!
            if est_cpu[t_next] >= dur_by_num[t_next]:
                _remove_remaining(t_next)
                _add_confirmed(t_next)
                continue
            m_confirmed = fast_cost_ms(confirmed_fail | {t_next}, dur_by_num) - c_fail
            if est_cpu[t_next] > m_confirmed:
                if c_all_dirty:
                    c_all_cached = fast_cost_ms(confirmed_fail | remaining, dur_by_num)
                    c_all_dirty = False
                c_without = fast_cost_ms(confirmed_fail | (remaining - {t_next}), dur_by_num)
                marg = c_all_cached - c_without
                if est_cpu[t_next] > max(marg, m_confirmed):
                    _remove_remaining(t_next)
                    _add_confirmed(t_next)
                    continue

        # Compute real-time CPU budget ONLY when confirmed_fail already makes marginal cost < 15s
        t_id_next = num_to_test[t_next]["id"]
        if confirmed_fail and t_id_next.endswith(("::test_execute_tpcds", "::test_execute_tpch")):
            m_conf_b = fast_cost_ms(confirmed_fail | {t_next}, dur_by_num) - c_fail
            if len(remaining) == 1:
                m_eff = m_conf_b
            else:
                if c_all_dirty:
                    c_all_cached = fast_cost_ms(confirmed_fail | remaining, dur_by_num)
                    c_all_dirty = False
                c_wo_b = fast_cost_ms(confirmed_fail | (remaining - {t_next}), dur_by_num)
                m_eff = min(dur_by_num[t_next], max(m_conf_b, c_all_cached - c_wo_b))
            if m_eff < 15000:
                t_budget_sec = max(1.5, m_eff / 1000.0)

        _remove_remaining(t_next)
        passed = _run_one(t_next, cpu_budget_sec=t_budget_sec)
        if not passed:
            _add_confirmed(t_next)
            _probe_lazy_import_crashes()
            # If the test's file failed to import or setUpClass crashed, immediately mark ALL siblings as failing!
            tf = num_to_test[t_next]["file"]
            ck_str = "::".join(num_to_test[t_next]["id"].split("::")[:2])
            if tf in failed_test_files:
                # Pytest fails collection for EVERY test in an unimportable test file (even tests not in candidates!)
                broken_files = {tf} | set(maps.get("import_file_to_test_files", {}).get(tf, ()))
                for bf in broken_files:
                    failed_test_files.add(bf)
                    for x in file_to_all_tnums.get(bf, ()):
                        if x in remaining:
                            _remove_remaining(x)
                        _add_confirmed(x)
                # Runtime Import-Crash Escalation: if tf itself was NOT patched, the import crash came from
                # patched sqlglot/ code -> escalate to full broken-import closure so no other unimportable file is missed!
                if not import_escalated and tf not in commit_plan.patched_files and engine is not None:
                    import_escalated = True
                    extra: set[int] = set()
                    for rel in list(commit_plan.patched_files) + list(commit_plan.deleted_files):
                        if rel.endswith(".py"):
                            extra.update(engine._tests_for_broken_import(rel))
                    for x in extra:
                        if x not in confirmed_fail and x not in remaining:
                            if zero_ms_absorbed and dur_by_num[x] == 0:
                                _add_confirmed(x)
                            else:
                                _add_remaining(x)
            elif setup_classes_done.get(ck_str) is False:
                # If setUpClass crashed, EVERY test in that class fails (even if not originally in candidates!)
                for x in file_to_all_tnums.get(tf, ()):
                    if "::".join(num_to_test[x]["id"].split("::")[:2]) == ck_str:
                        if x in remaining:
                            _remove_remaining(x)
                        _add_confirmed(x)

    return sorted(confirmed_fail)


# ---------------------------------------------------------------------------
# Dual-Fork Orchestrator (Pre-SQLGlot Zygote + Warm-SQLGlot Zygote)
# ---------------------------------------------------------------------------

# Wall-clock safety net so a batch can NEVER be rejected for exceeding the run timeout.
DEADLINE: float = float("inf")  # time.monotonic() value; set by selector.main()
PER_COMMIT_TIMEOUT: float = float(os.environ.get("BLAST_COMMIT_TIMEOUT", "90"))


class _ForkTimeout(Exception):
    pass


def _time_left() -> float:
    return DEADLINE - time.monotonic()


def _run_in_fork(fn, *args, timeout: float | None = None) -> Any:
    """Fork a clean CoW child process, redirect stdout->stderr in child, return pickled result.
    Kills child after `timeout` seconds (default: per-commit cap, never past DEADLINE)."""
    import select as _select

    budget = _time_left()
    if timeout is None:
        timeout = PER_COMMIT_TIMEOUT
    timeout = min(timeout, budget)
    if timeout <= 0.5:
        raise _ForkTimeout("global deadline reached")
    started = time.monotonic()
    r_fd, w_fd = os.pipe()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        os.close(r_fd)
        # Ensure child can NEVER write to parent's stdout (fd 1 -> fd 2)!
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        try:
            result = fn(*args)
            data = pickle.dumps(("ok", result), protocol=pickle.HIGHEST_PROTOCOL)
        except BaseException as exc:
            data = pickle.dumps(("err", repr(exc)), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            view = memoryview(data)
            written = 0
            while written < len(data):
                written += os.write(w_fd, view[written:])
        finally:
            os.close(w_fd)
            os._exit(0)
    else:
        os.close(w_fd)
        chunks: list[bytes] = []
        while True:
            left = timeout - (time.monotonic() - started)
            if math.isinf(left):
                ready = [r_fd]
            else:
                ready = _select.select([r_fd], [], [], max(0.0, left))[0] if left > 0 else []
            if not ready:
                try:
                    os.kill(pid, 9)
                except ProcessLookupError:
                    pass
                os.close(r_fd)
                os.waitpid(pid, 0)
                raise _ForkTimeout(f"child exceeded {timeout:.1f}s")
            chunk = os.read(r_fd, 65536)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(r_fd)
        os.waitpid(pid, 0)
        status, payload = pickle.loads(b"".join(chunks))
        if status == "ok":
            return payload
        raise RuntimeError(f"Child verification failed: {payload}")


def verify_batch(
    plans: list[CommitPlan],
    maps: dict[str, Any],
    num_to_test: dict[int, dict],
    dur_by_num: dict[int, int],
    sqlglot_root: Path,
    engine: Any = None,
) -> dict[str, list[int]]:
    """Verify all non-noop commits using Pre-SQLGlot Zygote (full_fork) + Warm-SQLGlot Zygote (fixture/hot_func)."""
    results: dict[str, list[int]] = {}
    full_fork_plans: list[CommitPlan] = []
    hot_func_plans: list[CommitPlan] = []
    fixture_plans: list[CommitPlan] = []

    for p in plans:
        if p.is_noop:
            results[p.commit] = []
        elif p.group == "fixture":
            fixture_plans.append(p)
        elif p.group == "hot_func":
            hot_func_plans.append(p)
        else:
            full_fork_plans.append(p)

    if not (full_fork_plans or hot_func_plans or fixture_plans):
        return results

    # NOTE: duckdb, numpy, pandas are imported lazily ONLY when a test actually executes them!
    if str(sqlglot_root) not in sys.path:
        sys.path.insert(0, str(sqlglot_root))
    os.environ["SKIP_INTEGRATION"] = "0"

    # Helper to run a full_fork commit in a clean child BEFORE sqlglot is imported in parent!
    def _run_full_fork_child(plan: CommitPlan) -> list[int]:
        ov_dir: Path | None = None
        lazy_tid = next(
            (n for n, t in num_to_test.items() if t["id"].endswith("::test_lazy_load")),
            None,
        )
        if lazy_tid is not None and lazy_tid in plan.candidates and any(
            rel.startswith("sqlglot/") for rel in list(plan.patched_files) + list(plan.deleted_files)
        ):
            ov_dir = _create_disk_overlay_for_subprocess(plan.patched_files, sqlglot_root, plan.deleted_files)
            os.environ["PYTHONPATH"] = f"{ov_dir}:{sqlglot_root}"
            sys.path.insert(0, str(ov_dir))

        finder = _InMemoryFinder(plan.patched_files, sqlglot_root, plan.deleted_files)
        sys.meta_path.insert(0, finder)
        restore_open = _install_open_overlay(plan.patched_files, sqlglot_root, plan.deleted_files)
        try:
            return _execute_commit_verification(
                plan, maps, num_to_test, dur_by_num, sqlglot_root, {}, set(plan.syntax_error_files), engine=engine, ov_dir=ov_dir
            )
        finally:
            restore_open()
            if ov_dir is not None:
                import shutil
                shutil.rmtree(ov_dir, ignore_errors=True)

    # 1. Split full_fork_plans AND lazy hot_func_plans into:
    #    - eager_fork_plans: touch core modules loaded by `import sqlglot` (or run test_lazy_load)
    #    - lazy_fork_plans: ONLY touch lazy dialect/parser/generator/typing/optimizer/executor/test modules
    #      that are NOT loaded by `import sqlglot` -> fork from Warm Zygote right after `import sqlglot`
    #      (1ms CoW fork + 3ms fresh import = ZERO __code__-swap risk AND skips 0.25s core re-import!)
    lazy_tid = next(
        (n for n, t in num_to_test.items() if t["id"].endswith("::test_lazy_load")),
        None,
    )
    # Modules strictly lazy (never imported by bare `import sqlglot`):
    def _is_lazy_only(plan: CommitPlan) -> bool:
        if plan.deleted_files:
            return False
        if lazy_tid is not None and lazy_tid in plan.candidates:
            return False
        for rel in plan.patched_files:
            if not rel.endswith(".py"):
                continue
            if rel.startswith((
                "sqlglot/dialects/",
                "sqlglot/parsers/",
                "sqlglot/generators/",
                "sqlglot/typing/",
                "sqlglot/optimizer/",
                "sqlglot/executor/",
                "tests/",
                "sqlglotc/",
            )):
                # Exclude core files imported eagerly by `import sqlglot`!
                if rel in (
                    "sqlglot/dialects/__init__.py",
                    "sqlglot/dialects/dialect.py",
                    "sqlglot/parsers/__init__.py",
                    "sqlglot/parsers/base.py",
                    "sqlglot/generators/__init__.py",
                    "sqlglot/typing/__init__.py",
                    "sqlglot/optimizer/__init__.py",
                    "sqlglot/optimizer/scope.py",
                    "tests/helpers.py",
                ):
                    return False
            elif rel in ("sqlglot/diff.py", "sqlglot/lineage.py", "sqlglot/anonymize.py", "sqlglot/planner.py", "sqlglot/serde.py"):
                continue
            else:
                return False
        return True

    eager_fork_plans = [p for p in full_fork_plans if not _is_lazy_only(p)]
    lazy_fork_plans = [p for p in full_fork_plans if _is_lazy_only(p)]
    # Route lazy-only hot_func commits to lazy_fork_plans too (1ms CoW fork, 0 __code__-swap risk!)
    eager_hot_plans: list[CommitPlan] = []
    for p in hot_func_plans:
        if _is_lazy_only(p):
            lazy_fork_plans.append(p)
        else:
            eager_hot_plans.append(p)
    hot_func_plans = eager_hot_plans

    # Freeze parent heap objects before os.fork() to prevent Linux CoW page faults & GC scans in children!
    gc.freeze()

    def _safe_fallback_tests(plan: CommitPlan) -> list[int]:
        out = set(plan.candidates)
        if engine is not None:
            for rel in list(plan.patched_files) + list(plan.deleted_files):
                if rel.startswith("sqlglot/expressions/") or rel in (
                    "sqlglot/parser.py", "sqlglot/parsers/base.py", "sqlglot/tokens.py",
                    "sqlglot/tokenizer_core.py", "sqlglot/generator.py",
                    "sqlglot/dialects/dialect.py", "sqlglot/helper.py", "sqlglot/__init__.py",
                ):
                    out.update(engine._tests_for_broken_import(rel))
        return sorted(out)

    for plan in eager_fork_plans:
        try:
            results[plan.commit] = _run_in_fork(_run_full_fork_child, plan)
        except Exception:
            results[plan.commit] = _safe_fallback_tests(plan)

    if not (lazy_fork_plans or fixture_plans or hot_func_plans):
        return results

    def _run_warm_zygote() -> tuple[dict[str, list[int]], list[str]]:
        """Runs inside ONE forked child that imports sqlglot core ONCE:
        Phase A: runs all lazy_fork_plans in 1ms CoW forks BEFORE loading any dialect/test modules!
        Phase B: pre-imports shared test modules, freezes heap, then runs hot_func_plans FIRST
                 and fixture_plans AFTER — ALL in clean 1ms CoW forks (NEVER in-process!)."""
        import sqlglot  # noqa: F401
        import tests.helpers  # noqa: F401

        warm_results: dict[str, list[int]] = {}
        fallback_commits: list[str] = []

        # Phase A: Run lazy_fork_plans while ZERO dialect/test modules are loaded!
        def _snapshot_loaded() -> set[str]:
            lf: set[str] = set()
            root_prefix = str(sqlglot_root.resolve()) + "/"
            for m in list(sys.modules.values()):
                mf = getattr(m, "__file__", None)
                if mf and mf.startswith(root_prefix):
                    lf.add(mf[len(root_prefix):])
            return lf

        # Preload frequently-referenced subtest maps ONCE before forking so CoW children skip zlib I/O!
        all_warm_plans = lazy_fork_plans + fixture_plans + hot_func_plans
        if engine is not None and len(all_warm_plans) >= 3:
            st_freq: dict[int, int] = {}
            for p in all_warm_plans:
                for tnum in p.deferred_subtests:
                    st_freq[tnum] = st_freq.get(tnum, 0) + 1
            for tnum, cnt in st_freq.items():
                if cnt >= 3:
                    engine._get_subtests(tnum)

        all_patched_py = {
            rel for p in all_warm_plans for rel in p.patched_files if rel.endswith(".py")
        }
        loaded_files = _snapshot_loaded()
        opt_conflict_plans: list[CommitPlan] = []
        dialect_conflict_plans: list[CommitPlan] = []
        dialect_safe_plans: list[CommitPlan] = []
        _DIALECT_PRELOAD_CONFLICTS = frozenset({
            "sqlglot/dialects/duckdb.py", "sqlglot/parsers/duckdb.py", "sqlglot/generators/duckdb.py",
            "sqlglot/dialects/bigquery.py", "sqlglot/parsers/bigquery.py", "sqlglot/generators/bigquery.py",
            "sqlglot/dialects/hive.py", "sqlglot/parsers/hive.py", "sqlglot/generators/hive.py",
            "sqlglot/dialects/snowflake.py", "sqlglot/parsers/snowflake.py", "sqlglot/generators/snowflake.py",
            "sqlglot/dialects/spark2.py", "sqlglot/parsers/spark2.py", "sqlglot/generators/spark2.py",
            "sqlglot/dialects/mysql.py", "sqlglot/parsers/mysql.py", "sqlglot/generators/mysql.py",
            "tests/dialects/test_dialect.py", "tests/dialects/test_mysql.py",
        })
        for plan in lazy_fork_plans:
            py_rels = [rel for rel in plan.patched_files if rel.endswith(".py")]
            if any(
                rel in ("tests/test_optimizer.py", "tests/test_schema.py") or rel.startswith("sqlglot/optimizer/")
                for rel in py_rels
            ):
                opt_conflict_plans.append(plan)
            elif any(rel in _DIALECT_PRELOAD_CONFLICTS or rel.startswith("tests/dialects/") for rel in py_rels):
                dialect_conflict_plans.append(plan)
            else:
                dialect_safe_plans.append(plan)

        if opt_conflict_plans:
            gc.freeze()
        for plan in opt_conflict_plans:
            if any(rel in loaded_files for rel in plan.patched_files if rel.endswith(".py")):
                fallback_commits.append(plan.commit)
                continue
            try:
                warm_results[plan.commit] = _run_in_fork(_run_full_fork_child, plan)
            except Exception:
                fallback_commits.append(plan.commit)

        # Pre-import tests.test_optimizer lazily if >=2 remaining plans use it!
        rem_after_opt = dialect_conflict_plans + dialect_safe_plans + fixture_plans + hot_func_plans
        if "tests/test_optimizer.py" not in all_patched_py:
            opt_users = sum(
                1 for p in rem_after_opt
                if any(num_to_test[t]["file"] == "tests/test_optimizer.py" for t in p.candidates)
            )
            if opt_users >= 2:
                try:
                    _import_test_module_lazy("tests.test_optimizer", False)
                    loaded_files = _snapshot_loaded()
                except Exception:
                    pass

        if dialect_conflict_plans:
            gc.freeze()
        for plan in dialect_conflict_plans:
            if any(rel in loaded_files for rel in plan.patched_files if rel.endswith(".py")):
                fallback_commits.append(plan.commit)
                continue
            try:
                warm_results[plan.commit] = _run_in_fork(_run_full_fork_child, plan)
            except Exception:
                fallback_commits.append(plan.commit)

        # Pre-import tests.dialects.test_dialect ONLY when >=3 dialect_safe_plans use dialect tests!
        if dialect_safe_plans:
            ds_dial_users = sum(
                1 for p in dialect_safe_plans
                if any(num_to_test[t]["file"].startswith("tests/dialects/") for t in p.candidates)
            )
            if ds_dial_users >= 3 and "tests/dialects/test_dialect.py" not in all_patched_py:
                try:
                    importlib.import_module("tests.dialects.test_dialect")
                    loaded_files = _snapshot_loaded()
                except Exception:
                    pass
            gc.freeze()
        for plan in dialect_safe_plans:
            if any(rel in loaded_files for rel in plan.patched_files if rel.endswith(".py")):
                fallback_commits.append(plan.commit)
                continue
            try:
                warm_results[plan.commit] = _run_in_fork(_run_full_fork_child, plan)
            except Exception:
                fallback_commits.append(plan.commit)

        if not (fixture_plans or hot_func_plans):
            return warm_results, fallback_commits

        # Phase B: Pre-import ONLY non-dialect shared test modules + tests.dialects.test_dialect
        # BEFORE hot_func_plans! Never pre-import individual tests/dialects/*.py (like test_duckdb.py)
        # before hot_func_plans, because duckdb's generator executes exp.maybe_parse() at import time!
        file_freq: dict[str, int] = {}
        dialect_plan_users = 0
        for p in (fixture_plans + hot_func_plans):
            seen_tf = {num_to_test[t]["file"] for t in p.candidates}
            if any(tf.startswith("tests/dialects/") for tf in seen_tf):
                dialect_plan_users += 1
            for tf in seen_tf:
                if not tf.startswith("tests/dialects/"):
                    file_freq[tf] = file_freq.get(tf, 0) + 1
        if dialect_plan_users >= 2:
            try:
                importlib.import_module("tests.dialects.test_dialect")
            except Exception:
                pass
        for tf, cnt in file_freq.items():
            if cnt >= 2:
                try:
                    mod_name = tf.removesuffix(".py").replace("/", ".")
                    _import_test_module_lazy(mod_name, False)
                except Exception:
                    pass

        mod_cnt: dict[str, int] = {}
        for p in hot_func_plans:
            for rel in p.py_analyses:
                if rel.startswith("sqlglot/") and not rel.endswith("/__init__.py"):
                    m = rel.removesuffix(".py").replace("/", ".")
                    mod_cnt[m] = mod_cnt.get(m, 0) + 1
        for m, cnt in mod_cnt.items():
            if cnt >= 2:
                try:
                    importlib.import_module(m)
                except Exception:
                    pass

        gc.freeze()

        # 2A. Run hot_func commits FIRST in clean 1ms CoW forks from untouched Warm Zygote!
        def _run_hot_child(plan: CommitPlan) -> tuple[bool, list[int]]:
            ok = _apply_hot_func_patches(plan, sqlglot_root)
            if not ok:
                return False, []
            restore_open = _install_open_overlay(plan.patched_files, sqlglot_root)
            try:
                failing = _execute_commit_verification(
                    plan, maps, num_to_test, dur_by_num, sqlglot_root, {}, set(), engine=engine
                )
                return True, failing
            finally:
                restore_open()

        for plan in hot_func_plans:
            try:
                ok, failing = _run_in_fork(_run_hot_child, plan)
                if ok:
                    warm_results[plan.commit] = failing
                else:
                    fallback_commits.append(plan.commit)
            except Exception:
                fallback_commits.append(plan.commit)

        # 2B. Run fixture commits AFTER ALSO in clean 1ms CoW forks (zero cross-commit pollution!)
        def _run_fixture_child(plan: CommitPlan) -> list[int]:
            restore_open = _install_open_overlay(plan.patched_files, sqlglot_root, plan.deleted_files)
            try:
                return _execute_commit_verification(
                    plan, maps, num_to_test, dur_by_num, sqlglot_root, {}, set(), engine=engine
                )
            finally:
                restore_open()

        for plan in fixture_plans:
            try:
                warm_results[plan.commit] = _run_in_fork(_run_fixture_child, plan)
            except Exception:
                fallback_commits.append(plan.commit)

        return warm_results, fallback_commits

    try:
        warm_results, fallback_commits = _run_in_fork(_run_warm_zygote, timeout=float("inf"))
    except Exception:
        warm_results, fallback_commits = {}, []
        for plan in lazy_fork_plans + fixture_plans + hot_func_plans:
            results[plan.commit] = _safe_fallback_tests(plan)
    results.update(warm_results)

    # 3. Any lazy_fork / hot_func / fixture commit that needed fallback runs in a clean Pre-SQLGlot full_fork!
    if fallback_commits:
        plan_by_id = {p.commit: p for p in (lazy_fork_plans + fixture_plans + hot_func_plans)}
        for cid in fallback_commits:
            plan = plan_by_id[cid]
            try:
                results[cid] = _run_in_fork(_run_full_fork_child, plan)
            except Exception:
                results[cid] = _safe_fallback_tests(plan)

    for plan in plans:
        if plan.commit not in results:
            results[plan.commit] = _safe_fallback_tests(plan)

    return results

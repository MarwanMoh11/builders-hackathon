"""Ultra-fast, isolated, fail-fast, slack-aware in-process test verification engine."""

from __future__ import annotations

import ast
import builtins
import copy
import csv
import gzip
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import io
import json
import os
import pickle
import shutil
import signal
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from candidate_engine import CommitPlan
from packer import fast_cost_ms, optimal_cost_ms


class _FailFast(BaseException):
    """Raised immediately on first failing subTest so loops abort in milliseconds."""


class _TestTimeout(BaseException):
    """Raised by SIGALRM when a test exceeds its per-test timeout (infinite loops)."""


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
        code = compile(self.source, self.filepath, "exec")
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


def _hash_df(df) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(list(df.columns)).encode())
    h.update(str(len(df)).encode())
    for row in df.itertuples(index=False, name=None):
        h.update(repr(row).encode())
    return h.hexdigest()


def _hash_plan(plan) -> str:
    return hashlib.blake2b(repr(plan).encode(), digest_size=16).hexdigest()


class _LazyExecutorTables(dict):
    """Lazily parses TPC-H / TPC-DS .csv.gz files ONLY for tables actually referenced by executed queries.
    Uses schema column types directly (40x faster than ast.literal_eval)."""
    def __init__(self, directory: str, schema: dict[str, dict[str, str]]) -> None:
        super().__init__()
        self._dir = directory
        self._schema = schema

    def __getitem__(self, table: str) -> Any:
        if super().__contains__(table):
            return super().__getitem__(table)
        from sqlglot.executor.table import Table
        file_name = f"{self._dir}{table}.csv.gz"
        col_types = self._schema[table]
        parsers = []
        for dtype in col_types.values():
            dl = dtype.lower()
            if dl in ("bigint", "int", "integer", "smallint", "tinyint"):
                parsers.append(int)
            elif dl in ("double", "float", "real"):
                parsers.append(float)
            else:
                parsers.append(str)

        with open(file_name, "rb") as f:
            is_gz = f.read(2) == b"\x1f\x8b"
        fh = gzip.open(file_name, "rt", newline="") if is_gz else open(file_name, encoding="utf-8", newline="")
        with fh:
            reader = csv.reader(fh, delimiter="|")
            next(reader)
            rows = [
                tuple(None if (p is not str and v == "") else p(v) for p, v in zip(parsers, row))
                for row in reader
            ]
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


def _patch_optimizer_module(maps: dict[str, Any]) -> None:
    topt = sys.modules.get("tests.test_optimizer")
    if topt is None or getattr(topt, "_blast_patched", False):
        return
    import tests.helpers as th
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
    valid_duck_pairs: set[tuple[str, str]] = {tuple(v) for v in duck_hashes.values()}

    def _fast_check_file(self, file, func, pretty=False, execute=None, only=None, **kwargs):
        check_pickle = getattr(topt, "_blast_check_pickle", False)
        fixture_rel = f"tests/fixtures/optimizer/{file}.sql"
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

            with self.subTest(title):
                if check_pickle:
                    func_kwargs = pickle.loads(pickle.dumps(func_kwargs))
                optimized = topt.parse_and_optimize(func, sql, dialect, **func_kwargs)
                if check_pickle:
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
                        if (dhashes[0], dhashes[1]) in valid_duck_pairs:
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


def _patch_executor_module(maps: dict[str, Any]) -> None:
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
        from sqlglot.optimizer import optimize
        from sqlglot.planner import Plan
        from sqlglot.executor.python import PythonExecutor
        from sqlglot.executor.table import ensure_tables
        from sqlglot.schema import ensure_schema

        tag = "tpc-h" if tpch else "tpc-ds"
        schema = th.TPCH_SCHEMA if tpch else th.TPCDS_SCHEMA
        tables = self.tpch_tables if tpch else self.tpcds_tables
        sqls = _load_sqls(self, tpch)
        qmeta_map = maps["tpch_meta"] if tpch else maps["tpcds_meta"]
        cp = ACTIVE_COMMIT_PLAN
        assert cp is not None
        fixture_touched = cp.tpch_fixture_touched if tpch else cp.tpcds_fixture_touched

        allowed = ACTIVE_FIXTURE_FILTERS.get(tag, None)
        indices = list(range(1, len(sqls) + 1)) if allowed is None else sorted(i for i in allowed if 1 <= i <= len(sqls))

        # Sort candidate queries fastest-first so failures abort in ~0.03s!
        # When exec_engine_touched is False, optimize() is the only work until a query changes!
        if not cp.exec_engine_touched:
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
        for qidx in indices:
            args, sql, _ = sqls[qidx - 1]
            with self.subTest(f"{tag} {qidx}"):
                expr = parse_one(sql)
                if check_pickle:
                    expr = pickle.loads(pickle.dumps(expr))
                if not args.get("execute"):
                    continue
                optimized = optimize(expr, schema_obj, leave_tables_isolated=True)
                opt_sql_hash = hashlib.blake2b(optimized.sql().encode(), digest_size=12).hexdigest()
                duck_sql = transpile(sql, write="duckdb")[0]
                duck_hash = hashlib.blake2b(duck_sql.encode(), digest_size=12).hexdigest()
                qm = qmeta_map.get(qidx)

                # EXACT Opt-SQL Diff Shortcut: if optimized SQL, executor, planner, DuckDB SQL, and CSV
                # fixtures are all identical to baseline, Plan + PythonExecutor + DuckDB are 100% identical!
                if (
                    qm is not None
                    and qm.get("execute")
                    and not cp.exec_engine_touched
                    and not fixture_touched
                    and opt_sql_hash == qm.get("opt_sql_hash")
                    and duck_hash == qm.get("duck_sql_hash")
                ):
                    continue

                if np_mod is None:
                    import numpy as np_mod
                    import pandas as pd_mod
                    from pandas.testing import assert_frame_equal as afe_fn

                plan = Plan(optimized)

                # Need real PythonExecutor run on ONLY this query's tables!
                q_tables = {}
                t_names = []
                for tbl in find_tables(expr):
                    name = tbl.name
                    q_tables[name] = tables[name]
                    t_names.append(name)
                tables_ = ensure_tables(q_tables)
                table = PythonExecutor(tables=tables_).execute(plan)
                b = pd_mod.DataFrame(
                    ((np_mod.nan if c is None else c for c in r) for r in table.rows),
                    columns=table.columns,
                )

                # Check fast baseline result hash first before touching DuckDB!
                if (
                    qm is not None
                    and qm.get("execute")
                    and not fixture_touched
                    and duck_hash == qm.get("duck_sql_hash")
                    and _hash_df(b) == qm.get("result_hash")
                ):
                    continue

                # Full DuckDB comparison (respects assert_frame_equal float tolerances)
                conn = _get_duck_conn(self, tpch, t_names)
                if sql not in self.cache:
                    self.cache[sql] = conn.execute(duck_sql).fetchdf()
                a = self.cache[sql]
                afe_fn(a, b, check_dtype=False, check_index_type=False)

    def _fast_optimized_tpch(self):
        from sqlglot import transpile
        cp = ACTIVE_COMMIT_PLAN
        assert cp is not None
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
                if not cp.tpch_fixture_touched and base_map.get(qidx) == (h1, h2):
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


def _patch_dialect_validator(maps: dict[str, Any]) -> None:
    """Wrap Validator.validate_identity / validate_all / validate_transpile to skip
    untouched call indices (for tests >= 0.04s) whenever the return value is unused
    and we are not inside an assertLogs/assertRaises/assertWarns context!"""
    tdial = sys.modules.get("tests.dialects.test_dialect")
    if tdial is None or getattr(tdial, "_blast_val_patched", False):
        return

    val_ret_used: dict[str, set[int]] = {
        k: set(v) for k, v in maps.get("val_ret_used", {}).items()
    }

    # Wrap assertLogs/assertNoLogs/assertRaises/assertRaisesRegex/assertWarns so calls inside them never skip!
    if not getattr(unittest.TestCase, "_blast_ctx_patched", False):
        for cname in ("assertLogs", "assertNoLogs", "assertRaises", "assertRaisesRegex", "assertWarns", "assertWarnsRegex"):
            orig_cm = getattr(unittest.TestCase, cname, None)
            if orig_cm is None:
                continue

            def _make_cm(orig):
                def _wrapped(self, *args, **kwargs):
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

            setattr(unittest.TestCase, cname, _make_cm(orig_cm))
        unittest.TestCase._blast_ctx_patched = True

    for vmeth in ("validate_identity", "validate_all", "validate_transpile"):
        orig_v = getattr(tdial.Validator, vmeth)

        def _make_vwrap(orig):
            def _vwrap(self, *args, **kwargs):
                global ACTIVE_VAL_COUNTER, ACTIVE_VAL_DEPTH, VAL_FAILED_INSIDE
                allowed = ACTIVE_VAL_CALLS
                if allowed is not None and ACTIVE_VAL_DEPTH == 0:
                    ACTIVE_VAL_COUNTER += 1
                    if ACTIVE_VAL_COUNTER not in allowed and IN_ASSERT_CTX == 0:
                        used_lines = val_ret_used.get(ACTIVE_VAL_FILE)
                        if not used_lines or sys._getframe(1).f_lineno not in used_lines:
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

        for unit in ana.units:
            if unit.kind != "func_body" or not unit.symbol_name:
                return False
            qname = unit.symbol_name
            fn_node, cls_chain = _find_node(qname)
            if fn_node is None:
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

            # Build minimal AST to compile fn_node (wrap in class stub if __class__ freevar needed)
            stmts: list[ast.stmt] = []
            if has_future_ann:
                stmts.append(ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))

            # Strip decorators for clean extraction of raw function object (preserve full line/end_lineno!)
            fn_clone = copy.copy(fn_node)
            fn_clone.decorator_list = []

            if cls_chain and "__class__" in old_code.co_freevars:
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
            else:
                stmts.append(fn_clone)
                mod_ast = ast.Module(body=stmts, type_ignores=[])
                ast.fix_missing_locations(mod_ast)
                ns = dict(mod.__dict__)
                exec(compile(mod_ast, str(sqlglot_root / rel), "exec"), ns)
                new_fn = ns[fn_node.name]

            new_code = new_fn.__code__
            if new_code.co_freevars != old_code.co_freevars or new_code.co_cellvars != old_code.co_cellvars:
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
        if not commit_plan.exec_engine_touched and not commit_plan.tpcds_fixture_touched:
            qset = (sub_f or {}).get("tpc-ds")
            n = len(qset) if qset is not None else (15 if has_deferred else 60)
            return min(1500, max(30, n * 20))
        qset = (sub_f or {}).get("tpc-ds")
        if qset is not None:
            qmeta = maps["tpcds_meta"]
            return int(sum((qmeta.get(q, {}).get("sec_opt", 0.02) + qmeta.get(q, {}).get("sec_exec", 0.2)) * 1000 for q in qset))
        return 12000
    if tid_str.endswith("::test_execute_tpch"):
        if not commit_plan.exec_engine_touched and not commit_plan.tpch_fixture_touched:
            qset = (sub_f or {}).get("tpc-h")
            n = len(qset) if qset is not None else (8 if has_deferred else 22)
            return min(500, max(15, n * 15))
        return 1500
    if tid_str.endswith("::test_generator_memory"):
        dset = (sub_f or {}).get("gen-mem")
        n = len(dset) if dset is not None else (3 if has_deferred else 34)
        return n * 120
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
                return min(recorded, 400)
            total_cases += len(v)
        if total_cases == 0:
            return 2
        return max(3, min(recorded, int(total_cases * 0.8) + 4))
    if has_deferred:
        return max(5, min(recorded, 60))

    return max(2, min(recorded, 800))


def _execute_commit_verification(
    commit_plan: CommitPlan,
    maps: dict[str, Any],
    num_to_test: dict[int, dict],
    dur_by_num: dict[int, int],
    sqlglot_root: Path,
    setup_classes_done: dict[str, Any],
    broken_import_files: set[str],
    engine: Any = None,
) -> list[int]:
    """Run the Slack-Absorption + Fail-Fast verification loop inside the current (isolated) process."""
    global ACTIVE_GEN_MEM_DIALECTS, ACTIVE_COMMIT_PLAN
    ACTIVE_COMMIT_PLAN = commit_plan

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

    def _run_one(tnum: int) -> bool:
        """Returns True if test PASSED, False if test FAILED/ERRORED/TIMED OUT."""
        global ACTIVE_GEN_MEM_DIALECTS
        tinfo = num_to_test[tnum]
        tf, cls_name, meth = tinfo["id"].split("::")
        if tf in failed_test_files:
            return False

        # Set active subtest filters for this test
        global ACTIVE_GEN_MEM_DIALECTS, ACTIVE_VAL_CALLS, ACTIVE_VAL_COUNTER, ACTIVE_VAL_FILE
        ACTIVE_FIXTURE_FILTERS.clear()
        ACTIVE_GEN_MEM_DIALECTS = None
        ACTIVE_VAL_CALLS = None
        ACTIVE_VAL_COUNTER = 0
        ACTIVE_VAL_FILE = tf
        helpers_patched = "tests/helpers.py" in commit_plan.patched_files
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
                # Apply module-specific fast patches ONLY when the test module & tests/helpers.py are unpatched!
                if not helpers_patched:
                    if mod_name.startswith("tests.dialects.") and "tests/dialects/test_dialect.py" not in commit_plan.patched_files:
                        _patch_dialect_validator(maps)
                    if mod_name == "tests.test_optimizer" and "tests/test_optimizer.py" not in commit_plan.patched_files:
                        _patch_optimizer_module(maps)
                        mod._blast_check_pickle = any(
                            rel in ("sqlglot/serde.py", "sqlglot/expressions/core.py") or rel.startswith("sqlglot/expressions/")
                            for rel in commit_plan.patched_files
                        )
                        mod._blast_fast_scope = not any(
                            rel in ("sqlglot/optimizer/scope.py", "sqlglot/parser.py", "sqlglot/tokens.py", "sqlglot/tokenizer_core.py")
                            or rel.startswith("sqlglot/expressions/")
                            for rel in commit_plan.patched_files
                        )
                    elif mod_name == "tests.test_executor" and "tests/test_executor.py" not in commit_plan.patched_files:
                        _patch_executor_module(maps)
                    elif mod_name == "tests.test_generator" and "tests/test_generator.py" not in commit_plan.patched_files:
                        _patch_generator_module()
                    elif mod_name == "tests.test_parser" and "tests/test_parser.py" not in commit_plan.patched_files:
                        _patch_parser_max_nodes()
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

        # Run the single test method with SIGALRM timeout + Fail-Fast subTest!
        global VAL_FAILED_INSIDE
        VAL_FAILED_INSIDE = False
        used_val_filter = ACTIVE_VAL_CALLS is not None
        timeout_sec = max(3.5, min(65.0, 2.5 * tinfo["seconds"] + 3.5))
        prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout_sec)
        passed = False
        try:
            inst = cls(meth)
            res = unittest.TestResult()
            inst.run(res)
            passed = res.wasSuccessful()
        except (_FailFast, _TestTimeout):
            passed = False
        except BaseException:
            passed = False
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, prev_handler)
            ACTIVE_FIXTURE_FILTERS.clear()
            ACTIVE_GEN_MEM_DIALECTS = None
            ACTIVE_VAL_CALLS = None
            ACTIVE_VAL_COUNTER = 0

        # Safety net: if a val-calls filtered test failed OUTSIDE a validate_* call (e.g. a mock
        # assertion expecting a skipped call to have run), re-run unfiltered to avoid false positive!
        if not passed and used_val_filter and not VAL_FAILED_INSIDE:
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout_sec)
            try:
                inst = cls(meth)
                res = unittest.TestResult()
                inst.run(res)
                passed = res.wasSuccessful()
            except (_FailFast, _TestTimeout):
                passed = False
            except BaseException:
                passed = False
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, prev_handler)

        return passed

    # Special fast path: if commit has a known syntax error (or import-time crash in patched .py),
    # probe-import patched modules first!
    if commit_plan.has_syntax_error:
        # Every test file whose import closure includes a syntax-error file WILL fail to import.
        # Still verify remaining candidates (e.g. dynamic lazy imports).
        pass

    # Precompute estimated verification costs
    est_cpu = {t: _est_verify_ms(t, num_to_test, dur_by_num, commit_plan, maps) for t in remaining}

    while remaining:
        c_fail = fast_cost_ms(confirmed_fail, dur_by_num)
        c_all = fast_cost_ms(confirmed_fail | remaining, dur_by_num)

        # SLACK-ABSORPTION THEOREM: if adding ALL remaining candidates costs 0 extra simulated ms,
        # absorb them immediately for 0ms CPU and 0% miss risk!
        if c_all == c_fail:
            # Double-check with exact pack_optimal when set is moderate so we never leave even 1ms on the table
            if len(confirmed_fail) + len(remaining) <= 100:
                if optimal_cost_ms(confirmed_fail | remaining, dur_by_num) == optimal_cost_ms(confirmed_fail, dur_by_num):
                    confirmed_fail.update(remaining)
                    break
            else:
                confirmed_fail.update(remaining)
                break

        # Choose next test to verify:
        # Prefer tests with high recorded runtime R(t) (to immediately unlock huge free slack on fail!)
        # while deprioritizing tests whose est_cpu > 800ms until fast tests have run.
        def _priority(t: int) -> tuple[int, int, int]:
            e = est_cpu[t]
            r = dur_by_num[t]
            if num_to_test[t]["id"].endswith("::test_lazy_load"):
                return (-1, 0, e)
            bucket = 0 if (e <= 600 or r >= 3 * e) else 1
            return (bucket, -r, e)

        t_next = min(remaining, key=_priority)

        # Per-test economic check when confirmed_fail is already non-empty:
        # Only absorb directly if verifying t_next costs MORE CPU than the TRUE worst-case simulated
        # cost of adding t_next (max of its marginal cost on c_all and on confirmed_fail alone!).
        if confirmed_fail:
            c_without = fast_cost_ms(confirmed_fail | (remaining - {t_next}), dur_by_num)
            marg = c_all - c_without
            if marg == 0:
                # Only probe top-8 longest candidates (O(1) instead of O(N^2)!)
                top_k = sorted(remaining, key=lambda x: (-dur_by_num[x], est_cpu[x]))[:8]
                useful = [
                    x for x in top_k
                    if fast_cost_ms(confirmed_fail | (remaining - {x}), dur_by_num) < c_all
                ]
                if not useful:
                    t_next = top_k[0]
                else:
                    t_next = min(useful, key=_priority)
                    c_without = fast_cost_ms(confirmed_fail | (remaining - {t_next}), dur_by_num)
                    marg = c_all - c_without
            if marg > 0 and est_cpu[t_next] > 150:
                m_confirmed = fast_cost_ms(confirmed_fail | {t_next}, dur_by_num) - c_fail
                if est_cpu[t_next] > max(marg, m_confirmed):
                    remaining.remove(t_next)
                    confirmed_fail.add(t_next)
                    continue

        remaining.remove(t_next)
        passed = _run_one(t_next)
        if not passed:
            confirmed_fail.add(t_next)
            # If the test's file failed to import or setUpClass crashed, immediately mark ALL siblings as failing!
            tf = num_to_test[t_next]["file"]
            ck_str = "::".join(num_to_test[t_next]["id"].split("::")[:2])
            if tf in failed_test_files:
                # Pytest fails collection for EVERY test in an unimportable test file (even tests not in candidates!)
                broken_files = {tf} | set(maps.get("import_file_to_test_files", {}).get(tf, ()))
                for bf in broken_files:
                    failed_test_files.add(bf)
                    for x in file_to_all_tnums.get(bf, ()):
                        remaining.discard(x)
                        confirmed_fail.add(x)
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
                            remaining.add(x)
                            est_cpu[x] = _est_verify_ms(x, num_to_test, dur_by_num, commit_plan, maps)
            elif setup_classes_done.get(ck_str) is False:
                # If setUpClass crashed, EVERY test in that class fails (even if not originally in candidates!)
                for x in file_to_all_tnums.get(tf, ()):
                    if "::".join(num_to_test[x]["id"].split("::")[:2]) == ck_str:
                        remaining.discard(x)
                        confirmed_fail.add(x)

    return sorted(confirmed_fail)


# ---------------------------------------------------------------------------
# Dual-Fork Orchestrator (Pre-SQLGlot Zygote + Warm-SQLGlot Zygote)
# ---------------------------------------------------------------------------

def _run_in_fork(fn, *args) -> Any:
    """Fork a clean CoW child process, redirect stdout->stderr in child, return pickled result."""
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
                plan, maps, num_to_test, dur_by_num, sqlglot_root, {}, set(plan.syntax_error_files), engine=engine
            )
        finally:
            restore_open()
            if ov_dir is not None:
                shutil.rmtree(ov_dir, ignore_errors=True)

    # 1. Split full_fork_plans into:
    #    - eager_fork_plans: touch core modules loaded by `import sqlglot` (or run test_lazy_load)
    #    - lazy_fork_plans: ONLY touch lazy dialect/parser/generator/typing/optimizer/executor/test modules
    #      that are NOT loaded by `import sqlglot` -> can fork from Warm Zygote right after `import sqlglot`,
    #      skipping 0.25s of redundant sqlglot core re-import per commit!
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

    for plan in eager_fork_plans:
        try:
            results[plan.commit] = _run_in_fork(_run_full_fork_child, plan)
        except Exception:
            results[plan.commit] = sorted(plan.candidates)

    if not (lazy_fork_plans or fixture_plans or hot_func_plans):
        return results

    def _run_warm_zygote() -> tuple[dict[str, list[int]], list[str]]:
        """Runs inside ONE forked child that imports sqlglot core ONCE:
        Phase A: runs all lazy_fork_plans in 1ms CoW forks BEFORE loading any dialect/test modules!
        Phase B: pre-imports shared test modules, then runs fixture_plans & hot_func_plans!"""
        import sqlglot  # noqa: F401
        import tests.helpers  # noqa: F401

        warm_results: dict[str, list[int]] = {}
        fallback_commits: list[str] = []

        # Phase A: Run lazy_fork_plans while ZERO dialect/test modules are loaded!
        # Sub-phase A1: plans touching tests/test_optimizer.py or sqlglot/optimizer/ run before importing test_optimizer.
        # Sub-phase A2: pre-import tests.test_optimizer lazily ONCE (if needed by >= 2 remaining commits!),
        # then run dialect/typing/parser/generator lazy_fork_plans inheriting it in 1ms CoW fork!
        def _snapshot_loaded() -> set[str]:
            lf: set[str] = set()
            root_prefix = str(sqlglot_root.resolve()) + "/"
            for m in list(sys.modules.values()):
                mf = getattr(m, "__file__", None)
                if mf and mf.startswith(root_prefix):
                    lf.add(mf[len(root_prefix):])
            return lf

        loaded_files = _snapshot_loaded()
        opt_conflict_plans: list[CommitPlan] = []
        opt_safe_plans: list[CommitPlan] = []
        for plan in lazy_fork_plans:
            if any(
                rel == "tests/test_optimizer.py" or rel.startswith("sqlglot/optimizer/")
                for rel in plan.patched_files
                if rel.endswith(".py")
            ):
                opt_conflict_plans.append(plan)
            else:
                opt_safe_plans.append(plan)

        for plan in opt_conflict_plans:
            if any(rel in loaded_files for rel in plan.patched_files if rel.endswith(".py")):
                fallback_commits.append(plan.commit)
                continue
            try:
                warm_results[plan.commit] = _run_in_fork(_run_full_fork_child, plan)
            except Exception:
                fallback_commits.append(plan.commit)

        # Pre-import tests.test_optimizer lazily if >=2 remaining plans (lazy_safe + fixture + hot_func) touch it
        opt_users = sum(
            1
            for p in (opt_safe_plans + fixture_plans + hot_func_plans)
            if any(num_to_test[t]["file"] == "tests/test_optimizer.py" for t in p.candidates)
        )
        if opt_users >= 2:
            try:
                _import_test_module_lazy("tests.test_optimizer", False)
                loaded_files = _snapshot_loaded()
            except Exception:
                pass

        for plan in opt_safe_plans:
            if any(rel in loaded_files for rel in plan.patched_files if rel.endswith(".py")):
                fallback_commits.append(plan.commit)
                continue
            try:
                warm_results[plan.commit] = _run_in_fork(_run_full_fork_child, plan)
            except Exception:
                fallback_commits.append(plan.commit)

        if not (fixture_plans or hot_func_plans):
            return warm_results, fallback_commits

        # Phase B: Pre-import frequently-used test modules ONCE before forking hot_func children
        # so every CoW child inherits them in 0.001s instead of re-importing!
        file_freq: dict[str, int] = {}
        for p in (fixture_plans + hot_func_plans):
            seen_tf = {num_to_test[t]["file"] for t in p.candidates}
            for tf in seen_tf:
                file_freq[tf] = file_freq.get(tf, 0) + 1
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

        # 2A. Run pure fixture commits (in-process for simple .sql/.json, 1ms fork if tpc-h/tpc-ds touched!)
        def _run_fixture_child(plan: CommitPlan) -> list[int]:
            restore_open = _install_open_overlay(plan.patched_files, sqlglot_root, plan.deleted_files)
            try:
                return _execute_commit_verification(
                    plan, maps, num_to_test, dur_by_num, sqlglot_root, {}, set(), engine=engine
                )
            finally:
                restore_open()

        shared_setup: dict[str, Any] = {}
        for plan in fixture_plans:
            if plan.tpch_fixture_touched or plan.tpcds_fixture_touched or plan.deleted_files:
                try:
                    warm_results[plan.commit] = _run_in_fork(_run_fixture_child, plan)
                except Exception:
                    fallback_commits.append(plan.commit)
                continue
            restore_open = _install_open_overlay(plan.patched_files, sqlglot_root, plan.deleted_files)
            try:
                warm_results[plan.commit] = _execute_commit_verification(
                    plan, maps, num_to_test, dur_by_num, sqlglot_root, shared_setup, set(), engine=engine
                )
            except Exception:
                fallback_commits.append(plan.commit)
            finally:
                restore_open()

        # 2B. Run each hot_func commit in a 1ms CoW fork from this Warm Zygote!
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

        return warm_results, fallback_commits

    warm_results, fallback_commits = _run_in_fork(_run_warm_zygote)
    results.update(warm_results)

    # 3. Any lazy_fork / hot_func / fixture commit that needed fallback runs in a clean Pre-SQLGlot full_fork!
    if fallback_commits:
        plan_by_id = {p.commit: p for p in (lazy_fork_plans + fixture_plans + hot_func_plans)}
        for cid in fallback_commits:
            plan = plan_by_id[cid]
            try:
                results[cid] = _run_in_fork(_run_full_fork_child, plan)
            except Exception:
                results[cid] = sorted(plan.candidates)

    return results

"""Build-time exact (receiver_cls, attr_name), (receiver_cls, TRANSFORMS, expr_cls),
and (receiver_cls, dict_attr, key_with_misses) dependency index (Wave 2A / v5).

Generated inside Docker at build time from pinned sqlglot@eb11aec (0 bytes in Git).
Stored uncompressed at /app/dep_index.marshal (~1-2 MB, loads in ~8 ms in parent before os.fork()).

Soundness invariant (first-divergence proof):
  - Covers ALL 60 SQLGlot test files (all 1,631 tests), including setUpClass/setUp/tearDown!
  - Wraps UPPERCASE class attributes across Dialect, Tokenizer, Parser, Generator + all subclasses
    with non-data descriptors (_AttrSpy) recording exact runtime receiver class names.
  - Wraps tracked dict tables via receiver-bound _BoundSpyDict recording BOTH hits AND misses
    for __getitem__, get, __contains__, plus "*" wildcard for __iter__, keys, values, items,
    copy, __len__, __bool__, __eq__, __ne__, __repr__.
  - Instruments Generator.sql to record exact (GenCls, ExprCls) dispatch (BOTH hits AND misses!)
    plus direct TRANSFORMS reads.
  - Always preserves reflection tests, subprocess tests (test_lazy_load), entry-point tests,
    and any test not cleanly observed during build collection in `always_keep_mtids`.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import marshal
import os
import sys
import time
import unittest
import zlib
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any

DEP_INDEX_PATH = Path("/app/dep_index.marshal")

# Tracked dict attributes on Generator and Parser (runtime lookups via _BoundSpyDict including misses!)
TRACKED_GEN_DICTS: frozenset[str] = frozenset({
    "TRANSFORMS",
    "TYPE_MAPPING",
    "PROPERTIES_LOCATION",
    "TOKEN_MAPPING",
    "TYPE_PARAM_SETTINGS",
    "TIME_PART_SINGULARS",
    "AFTER_HAVING_MODIFIER_TRANSFORMS",
})

TRACKED_PAR_DICTS: frozenset[str] = frozenset({
    "FUNCTIONS",
    "FUNCTION_PARSERS",
    "STATEMENT_PARSERS",
    "EXPRESSION_PARSERS",
    "PROPERTY_PARSERS",
    "CONSTRAINT_PARSERS",
    "PRIMARY_PARSERS",
    "NO_PAREN_FUNCTIONS",
    "NO_PAREN_FUNCTION_PARSERS",
    "TYPE_CONVERTERS",
    "TYPE_LITERAL_PARSERS",
    "UNARY_PARSERS",
    "CAST_ACTIONS",
    "COLUMN_OPERATORS",
    "COMPARISON",
    "CONFLICT_ACTIONS",
    "CONJUNCTION",
    "DISJUNCTION",
    "EQUALITY",
    "EXPONENT",
    "FACTOR",
    "TERM",
    "BITWISE",
    "JSON_OPERATORS",
    "SUBQUERY_PREDICATES",
    "QUERY_MODIFIER_PARSERS",
    "RANGE_PARSERS",
    "SHOW_PARSERS",
    "SET_PARSERS",
    "ALTER_PARSERS",
    "ALTER_ALTER_PARSERS",
    "ANALYZE_EXPRESSION_PARSERS",
    "ARRAY_CONSTRUCTORS",
    "ASSIGNMENT",
    "NUMERIC_PARSERS",
    "PLACEHOLDER_PARSERS",
    "STRING_PARSERS",
    "KEY_CONSTRAINT_OPTIONS",
    "WINDOW_EXCLUDE_OPTIONS",
    "TRANSACTION_CHARACTERISTICS",
    "TRIGGER_DEFERRABLE",
    "TRIGGER_TIMING",
    "EXECUTE_AS_OPTIONS",
    "CREATE_SEQUENCE",
    "ISOLATED_LOADING_OPTIONS",
    "PROCEDURE_OPTIONS",
    "SCHEMA_BINDING_OPTIONS",
    "ODBC_DATETIME_LITERALS",
    "PIPE_SYNTAX_TRANSFORM_PARSERS",
    "SIGNED_TO_UNSIGNED_TYPE_TOKEN",
    "USABLES",
})

ALL_TRACKED_DICTS: frozenset[str] = TRACKED_GEN_DICTS | TRACKED_PAR_DICTS

# Dialect/Tokenizer metaclass attributes baked at class-definition time (handled via dialect_meta_attrs)
META_SKIP_ATTRS: frozenset[str] = frozenset({
    "BIT_STRINGS", "BYTE_STRINGS", "COMMENTS", "HEREDOC_STRING", "HEREDOC_STRINGS",
    "HEX_STRINGS", "HINT_START", "IDENTIFIERS", "NATIONAL_RAW_STRING",
    "NATIONAL_RAW_STRINGS", "NATIONAL_STRING", "QUOTES", "RAW_STRING", "RAW_STRINGS",
    "SINGLE_TOKENS", "UNICODE_STRINGS", "NUMERIC_LITERALS", "NUMERIC_ESCAPES",
    "KEYWORDS", "VAR_SINGLE_TOKENS", "COMMANDS", "COMMAND_PREFIX_TOKENS",
    "STRING_ESCAPES", "BYTE_STRING_ESCAPES", "IDENTIFIER_ESCAPES",
    "SUPPORTED_JSON_PATH_PARTS", "DATE_PART_MAPPING", "TIME_MAPPING",
    "FORMAT_MAPPING", "INVERSE_TIME_MAPPING", "INVERSE_FORMAT_MAPPING",
    "CREATABLE_KIND_MAPPING", "UNESCAPED_SEQUENCES", "ESCAPED_SEQUENCES",
    "VALID_INTERVAL_UNITS", "TIME_TRIE", "FORMAT_TRIE", "INVERSE_TIME_TRIE",
    "INVERSE_FORMAT_TRIE", "SET_TRIE", "SHOW_TRIE", "COERCES_TO",
})

# Reflection / subprocess / meta test files always kept by dep_index
ALWAYS_KEEP_TEST_FILES: frozenset[str] = frozenset({
    "tests/test_dialect_imports.py",
    "tests/test_dialect_entry_points.py",
    "tests/test_integration_loader.py",
    "tests/test_sqlglotc.py",
})

ALWAYS_KEEP_TEST_IDS: frozenset[str] = frozenset({
    "tests/dialects/test_dialect.py::TestDialect::test_enum",
    "tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata",
})
_SEPARATELY_HANDLED_TEST_IDS: frozenset[str] = frozenset({
    "tests/dialects/test_dialect.py::TestDialect::test_lazy_load",
})


def canon_runtime_key(k: Any, exp_expr_cls: type) -> str | None:
    """Canonicalize a runtime dict key into a stable marshal-safe string."""
    tk = type(k)
    if tk is str:
        return f"S:{k}"
    if tk is int:
        return f"I:{k}"
    if tk is bool:
        return f"B:{int(k)}"
    if isinstance(k, type) and issubclass(k, exp_expr_cls):
        return f"E:{k.__name__}"
    if isinstance(k, Enum):
        # Unify DType and DataType.Type under M:DType.<NAME>
        en = "DType" if tk.__name__ in ("Type", "DType") else tk.__name__
        return f"M:{en}.{k.name}"
    return None


def canon_source_key(kr: str, all_expr_classes: set[str]) -> str | None:
    """Canonicalize a patch_analyzer AST dict-key source repr (kr) to match canon_runtime_key."""
    s = kr.strip()
    if not s:
        return None
    if (s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"'):
        try:
            val = ast.literal_eval(s)
            if type(val) is str:
                return f"S:{val}"
        except Exception:
            return None
    if s in ("True", "False"):
        return f"B:{1 if s == 'True' else 0}"
    if s.isdigit() or (s[0] == "-" and s[1:].isdigit()):
        return f"I:{int(s)}"
    # exp.Foo or Foo
    parts = s.split(".")
    last = parts[-1]
    if len(parts) == 2 and parts[0] == "exp" and last in all_expr_classes:
        return f"E:{last}"
    if len(parts) == 1 and last in all_expr_classes:
        return f"E:{last}"
    # TokenType.FOO, exp.DType.INT, exp.DataType.Type.INT, etc.
    if len(parts) >= 2 and last.isupper():
        en = parts[-2]
        if en in ("Type", "DType") and ("DataType" in parts or "DType" in parts):
            en = "DType"
        return f"M:{en}.{last}"
    return None


# ---------------------------------------------------------------------------
# Runtime Spy Machinery (shared by dep_index builder & fine-grained replay!)
# ---------------------------------------------------------------------------

class _BoundSpyDict(dict):
    """Lightweight dict wrapper recording BOTH hits & misses + wildcard iteration."""

    __slots__ = ("_raw", "_recv", "_attr", "_on_key", "_exp_cls")

    def __init__(self, raw: dict, recv: str, attr: str, on_key: Any, exp_cls: type) -> None:
        super().__init__(raw)
        self._raw = raw
        self._recv = recv
        self._attr = attr
        self._on_key = on_key
        self._exp_cls = exp_cls

    def _rec(self, k: Any) -> None:
        ck = canon_runtime_key(k, self._exp_cls)
        self._on_key(self._recv, self._attr, ck if ck is not None else "*")

    def _rec_all(self) -> None:
        self._on_key(self._recv, self._attr, "*")

    def __getitem__(self, k: Any) -> Any:
        self._rec(k)
        return self._raw[k]

    def __setitem__(self, k: Any, v: Any) -> None:
        self._rec(k)
        self._raw[k] = v
        super().__setitem__(k, v)

    def __delitem__(self, k: Any) -> None:
        self._rec(k)
        del self._raw[k]
        super().pop(k, None)

    def pop(self, k: Any, *args: Any) -> Any:
        self._rec(k)
        super().pop(k, None)
        return self._raw.pop(k, *args)

    def popitem(self) -> tuple[Any, Any]:
        self._rec_all()
        k, v = self._raw.popitem()
        super().pop(k, None)
        return k, v

    def setdefault(self, k: Any, default: Any = None) -> Any:
        self._rec(k)
        if k not in self._raw:
            self._raw[k] = default
            super().__setitem__(k, default)
        return self._raw[k]

    def update(self, *args: Any, **kwargs: Any) -> None:
        self._rec_all()
        self._raw.update(*args, **kwargs)
        super().update(*args, **kwargs)

    def clear(self) -> None:
        self._rec_all()
        self._raw.clear()
        super().clear()

    def get(self, k: Any, default: Any = None) -> Any:
        self._rec(k)
        return self._raw.get(k, default)

    def __contains__(self, k: Any) -> bool:
        self._rec(k)
        return k in self._raw

    def __iter__(self):
        self._rec_all()
        return iter(self._raw)

    def keys(self):
        self._rec_all()
        return self._raw.keys()

    def values(self):
        self._rec_all()
        return self._raw.values()

    def items(self):
        self._rec_all()
        return self._raw.items()

    def copy(self) -> dict:
        self._rec_all()
        return self._raw.copy()

    def __len__(self) -> int:
        self._rec_all()
        return len(self._raw)

    def __bool__(self) -> bool:
        self._rec_all()
        return bool(self._raw)

    def __eq__(self, other: Any) -> bool:
        self._rec_all()
        return self._raw == (other._raw if isinstance(other, _BoundSpyDict) else other)

    def __ne__(self, other: Any) -> bool:
        self._rec_all()
        return self._raw != (other._raw if isinstance(other, _BoundSpyDict) else other)

    def __repr__(self) -> str:
        self._rec_all()
        return repr(self._raw)

    def __or__(self, other: Any) -> dict:
        self._rec_all()
        return self._raw | (other._raw if isinstance(other, _BoundSpyDict) else other)

    def __ror__(self, other: Any) -> dict:
        self._rec_all()
        return (other._raw if isinstance(other, _BoundSpyDict) else other) | self._raw


class _AttrSpy:
    """Non-data descriptor recording exact (receiver_cls, attr_name) reads!"""

    __slots__ = ("_attr", "_val", "_is_dict", "_on_attr", "_on_key", "_exp_cls", "_bound_cache")

    def __init__(self, attr: str, val: Any, is_dict: bool, on_attr: Any, on_key: Any, exp_cls: type) -> None:
        self._attr = attr
        self._val = val
        self._is_dict = is_dict
        self._on_attr = on_attr
        self._on_key = on_key
        self._exp_cls = exp_cls
        self._bound_cache: dict[str, _BoundSpyDict] | None = {} if is_dict else None

    def __get__(self, obj: Any, owner: type | None = None) -> Any:
        recv = obj.__class__.__name__ if obj is not None else (owner.__name__ if owner is not None else "")
        if self._is_dict:
            bc = self._bound_cache
            bd = bc.get(recv)  # type: ignore[union-attr]
            if bd is None:
                bd = _BoundSpyDict(self._val, recv, self._attr, self._on_key, self._exp_cls)
                bc[recv] = bd  # type: ignore[index]
            return bd
        self._on_attr(recv, self._attr)
        return self._val


def install_spies(
    on_attr: Any,
    on_key: Any,
    on_gen_dispatch: Any,
    on_method: Any | None = None,
) -> tuple[set[str], set[str], Any]:
    """Install _AttrSpy, _BoundSpyDict, Generator.sql dispatch spy, and optional dialect method spies
    on Pure-Python /sqlglot. Returns (tracked_attrs, tracked_dicts, restore_fn)."""
    from sqlglot import Dialect, Dialects, exp
    from sqlglot.generator import Generator, _DISPATCH_CACHE, _build_dispatch
    from sqlglot.parser import Parser
    from sqlglot.tokens import Tokenizer

    # Preload ALL dialects and pre-populate Generator _DISPATCH_CACHE BEFORE spying!
    all_classes: set[type] = set()
    for d in Dialects:
        d_inst = Dialect.get_or_raise(d)
        dc = type(d_inst)
        for base_cls in (dc, dc.parser_class, dc.generator_class, dc.tokenizer_class):
            for c in base_cls.__mro__:
                if c.__module__.startswith("sqlglot."):
                    all_classes.add(c)
        gcls = dc.generator_class
        if gcls not in _DISPATCH_CACHE:
            _DISPATCH_CACHE[gcls] = _build_dispatch(gcls)
    if Generator not in _DISPATCH_CACHE:
        _DISPATCH_CACHE[Generator] = _build_dispatch(Generator)

    # Also ensure PythonGenerator is included!
    try:
        from sqlglot.generators.python import PythonGenerator
        for c in PythonGenerator.__mro__:
            if c.__module__.startswith("sqlglot."):
                all_classes.add(c)
        if PythonGenerator not in _DISPATCH_CACHE:
            _DISPATCH_CACHE[PythonGenerator] = _build_dispatch(PythonGenerator)
    except Exception:
        pass

    saved_attrs: list[tuple[type, str, Any]] = []
    tracked_attrs: set[str] = set()
    tracked_dicts: set[str] = set()

    for cls in all_classes:
        is_gen = issubclass(cls, Generator)
        is_par = issubclass(cls, Parser)
        for name, val in list(cls.__dict__.items()):
            if not name.isupper() or name.startswith("_") or name in META_SKIP_ATTRS:
                continue
            if callable(val) or isinstance(val, (classmethod, staticmethod, property, type)):
                continue
            if isinstance(val, dict):
                if (is_gen and name in TRACKED_GEN_DICTS) or (is_par and name in TRACKED_PAR_DICTS):
                    saved_attrs.append((cls, name, val))
                    setattr(cls, name, _AttrSpy(name, val, True, on_attr, on_key, exp.Expr))
                    tracked_dicts.add(name)
            else:
                saved_attrs.append((cls, name, val))
                setattr(cls, name, _AttrSpy(name, val, False, on_attr, on_key, exp.Expr))
                tracked_attrs.add(name)

    # Optional: wrap non-dunder methods defined in sqlglot/parsers/*.py and sqlglot/generators/*.py
    wrapped_methods: set[tuple[str, str]] = set()
    if on_method is not None:
        for cls in all_classes:
            mod = cls.__module__
            if not (mod.startswith("sqlglot.parsers.") or mod.startswith("sqlglot.generators.")):
                continue
            if mod in ("sqlglot.parsers.base",):
                continue
            cname = cls.__name__
            for mname, mval in list(cls.__dict__.items()):
                if mname.startswith("__") and mname.endswith("__"):
                    continue
                if inspect.isfunction(mval):
                    saved_attrs.append((cls, mname, mval))
                    wrapped_methods.add((cname, mname))

                    def _make_mwrap(orig_fn, _cn, _mn):
                        def _mwrap(*args, **kwargs):
                            on_method(_cn, _mn)
                            return orig_fn(*args, **kwargs)
                        _mwrap.__name__ = orig_fn.__name__
                        _mwrap.__qualname__ = orig_fn.__qualname__
                        return _mwrap

                    wrapped = _make_mwrap(mval, cname, mname)
                    setattr(cls, mname, wrapped)
                    # Also update _DISPATCH_CACHE if this method is referenced anywhere in dispatch!
                    if issubclass(cls, Generator):
                        for gcls_c, dmap in _DISPATCH_CACHE.items():
                            if issubclass(gcls_c, cls):
                                for ek, fn in list(dmap.items()):
                                    if fn is mval:
                                        dmap[ek] = wrapped

    # Also compute wrapped_methods set even when on_method is None (so dep_index exports it!)
    else:
        for cls in all_classes:
            mod = cls.__module__
            if not (mod.startswith("sqlglot.parsers.") or mod.startswith("sqlglot.generators.")):
                continue
            if mod in ("sqlglot.parsers.base",):
                continue
            cname = cls.__name__
            for mname, mval in cls.__dict__.items():
                if mname.startswith("__") and mname.endswith("__"):
                    continue
                if inspect.isfunction(mval):
                    wrapped_methods.add((cname, mname))

    # Wrap Generator.sql for exact (GenCls, ExprCls) dispatch (hits + misses!)
    orig_gen_sql = Generator.sql

    def _spy_gen_sql(self, expression, key=None, comment=True):
        if not expression:
            return ""
        if isinstance(expression, str):
            return expression
        if key:
            value = expression.args.get(key)
            if value:
                return self.sql(value)
            return ""
        on_gen_dispatch(self.__class__.__name__, expression.__class__.__name__)
        return orig_gen_sql(self, expression, None, comment)

    Generator.sql = _spy_gen_sql  # type: ignore[method-assign]

    def _restore() -> None:
        Generator.sql = orig_gen_sql  # type: ignore[method-assign]
        for cls, name, val in saved_attrs:
            setattr(cls, name, val)
        _DISPATCH_CACHE.clear()

    return tracked_attrs, tracked_dicts, wrapped_methods, _restore


# ---------------------------------------------------------------------------
# Build-Time Parallel Collector (all 60 test files -> /app/dep_index.marshal)
# ---------------------------------------------------------------------------

def _collect_dep_worker(args: tuple[str, list[str], dict[str, int]]) -> dict[str, Any]:
    sg_root, test_files, tid_to_mtid = args
    sys.path.insert(0, sg_root)
    sys.path.insert(0, "/app")

    cur_mtid = [-1]
    setup_attr: set[tuple[str, str]] = set()
    setup_key: set[tuple[str, str, str]] = set()
    setup_disp: set[tuple[str, str]] = set()

    attr_reads: dict[tuple[str, str], set[int]] = {}
    dict_reads: dict[tuple[str, str, str], set[int]] = {}
    gen_dispatch: dict[tuple[str, str], set[int]] = {}
    seen_mtids: set[int] = set()
    failed_mtids: set[int] = set()

    def _on_attr(recv: str, attr: str) -> None:
        t = cur_mtid[0]
        if t >= 0:
            s = attr_reads.get((recv, attr))
            if s is None:
                attr_reads[(recv, attr)] = {t}
            else:
                s.add(t)
        elif t == -2:
            setup_attr.add((recv, attr))

    def _on_key(recv: str, attr: str, ck: str) -> None:
        t = cur_mtid[0]
        if t >= 0:
            k = (recv, attr, ck)
            s = dict_reads.get(k)
            if s is None:
                dict_reads[k] = {t}
            else:
                s.add(t)
        elif t == -2:
            setup_key.add((recv, attr, ck))

    def _on_disp(recv: str, ecls: str) -> None:
        t = cur_mtid[0]
        if t >= 0:
            k = (recv, ecls)
            s = gen_dispatch.get(k)
            if s is None:
                gen_dispatch[k] = {t}
            else:
                s.add(t)
        elif t == -2:
            setup_disp.add((recv, ecls))

    tracked_attrs, tracked_dicts, wrapped_methods, restore = install_spies(_on_attr, _on_key, _on_disp)

    # Preload fast helpers for special heavy tests
    import duckdb
    import tests.helpers as th
    from sqlglot import Dialect, Dialects, exp, parse_one, transpile
    from sqlglot.optimizer import optimize
    from sqlglot.planner import Plan
    from sqlglot.schema import MappingSchema, ensure_schema

    try:
        for tf in test_files:
            if tf in ALWAYS_KEEP_TEST_FILES:
                continue
            mod_name = tf.removesuffix(".py").replace("/", ".")
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue

            # Patch heavy TestOptimizer / TestExecutor / TestParser / TestGenerator methods to run fast
            # while still exercising 100% of their Dialect/Parser/Generator/PythonGenerator attribute & dispatch reads!
            if tf == "tests/test_optimizer.py":
                import tests.test_optimizer as topt
                opt_db = Path("/app/duckdb_opt.db")

                @classmethod  # type: ignore[misc]
                def _fast_opt_su(cls):
                    import sqlglot
                    cls.pool = None
                    sqlglot.schema = MappingSchema()
                    cls.conn = duckdb.connect(str(opt_db), read_only=True, config={"threads": 1})

                topt.TestOptimizer.setUpClass = _fast_opt_su

                def _fast_cf(self, file, func, pretty=False, execute=None, only=None, **kwargs):
                    import json
                    for i, (meta, sql, expected) in enumerate(th.load_sql_fixture_pairs(f"optimizer/{file}.sql"), start=1):
                        title = meta.get("title") or f"{i}, {sql}"
                        if only and title != only:
                            continue
                        dialect = meta.get("dialect")
                        fkw = kwargs.copy()
                        if schema := meta.get("schema"):
                            fkw["schema"] = json.loads(schema)
                        if (v := meta.get("leave_tables_isolated")) is not None:
                            fkw["leave_tables_isolated"] = th.string_to_bool(v)
                        if (v := meta.get("validate_qualify_columns")) is not None:
                            fkw["validate_qualify_columns"] = th.string_to_bool(v)
                        if dialect:
                            fkw["dialect"] = dialect
                        if (v := meta.get("canonicalize_table_aliases")) is not None:
                            fkw["canonicalize_table_aliases"] = th.string_to_bool(v)
                        optimized = func(parse_one(sql, read=dialect), **fkw)
                        optimized.sql(pretty=pretty, dialect=dialect)
                        fixture_execute = False if execute is False else meta.get("execute", execute)
                        if th.string_to_bool(fixture_execute):
                            for ex in (parse_one(sql, read=dialect), optimized):
                                if not isinstance(ex, (exp.Query, exp.Values)):
                                    ex = exp.select(exp.alias_(ex, "_col_0"))
                                ex.sql(dialect="duckdb")

                topt.TestOptimizer.check_file = _fast_cf
                # Skip slow DuckDB execution in test_optimize_in_nulls / correlated_all (still parses + optimizes + generates duckdb SQL!)
                orig_in_nulls = topt.TestOptimizer.test_optimize_in_nulls

                def _fast_in_nulls_dep(self):
                    for where in ("", " WHERE y.b IS NOT NULL", " WHERE y.b IS NULL", " WHERE FALSE", " WHERE y.c = x.a", " WHERE y.b = x.a"):
                        for pred in (f"x.a IN (SELECT y.b FROM y{where})", f"x.a = ANY (SELECT y.b FROM y{where})"):
                            for sql in (
                                f"SELECT x.a, {pred} AS r FROM x",
                                f"SELECT x.a, NOT ({pred}) AS r FROM x",
                                f"SELECT x.a, ({pred}) AND x.a > 1 AS r FROM x",
                                f"SELECT x.a, ({pred}) OR x.a > 1 AS r FROM x",
                                f"SELECT x.a FROM x WHERE NOT ({pred})",
                                f"SELECT x.a FROM x WHERE ({pred}) IS NULL",
                                f"SELECT x.a FROM x WHERE COALESCE(({pred}), TRUE)",
                                f"SELECT x.a FROM x WHERE ({pred}) = FALSE",
                                f"SELECT x.a FROM x JOIN x AS x2 ON x.b = x2.b AND ({pred}) = FALSE",
                                f"SELECT x.a FROM x QUALIFY ROW_NUMBER() OVER (ORDER BY x.a) > 0 AND ({pred}) = FALSE",
                                f"SELECT x.a FROM x GROUP BY x.a HAVING ({pred}) IS NULL",
                                f"SELECT x.a FROM x ORDER BY ({pred}), x.a",
                            ):
                                topt.optimizer.optimize(sql, schema=self.schema, dialect="duckdb").sql("duckdb")

                topt.TestOptimizer.test_optimize_in_nulls = _fast_in_nulls_dep

                def _fast_corr_dep(self):
                    for col in ("b", "c"):
                        for op in ("=", "<>", ">", ">=", "<", "<="):
                            pred = f"x.a {op} ALL (SELECT y.{col} FROM y WHERE y.c = x.b)"
                            for sql in (
                                f"SELECT x.a, x.b, {pred} AS r FROM x",
                                f"SELECT x.a, x.b FROM x WHERE {pred}",
                                f"SELECT x.a, x.b FROM x WHERE NOT ({pred})",
                            ):
                                topt.optimizer.optimize(sql, schema=self.schema, dialect="duckdb").sql("duckdb")

                topt.TestOptimizer.test_optimize_correlated_all = _fast_corr_dep

                # Slim 10000-UNION scope stress test
                orig_scope = topt.TestOptimizer.test_scope

                def _slim_scope(self):
                    op = topt.parse_one
                    topt.parse_one = lambda s, *a, **kw: op(" UNION ALL ".join(["SELECT x FROM t"] * 20), *a, **kw) if isinstance(s, str) and len(s) > 50_000 else op(s, *a, **kw)
                    try:
                        orig_scope(self)
                    finally:
                        topt.parse_one = op

                topt.TestOptimizer.test_scope = _slim_scope

            elif tf == "tests/test_executor.py":
                import tests.test_executor as texec
                from sqlglot.generators.python import PythonGenerator

                @classmethod  # type: ignore[misc]
                def _fast_exec_su(cls):
                    cls.conn = duckdb.connect(":memory:", config={"threads": 1})
                    cls.tpch_tables = {}
                    cls.tpcds_tables = {}
                    cls.cache = {}

                texec.TestExecutor.setUpClass = _fast_exec_su
                texec.TestExecutor.tearDownClass = classmethod(lambda cls: cls.conn.close())

                def _fast_tpcds_dep(self):
                    s_ds = ensure_schema(th.TPCDS_SCHEMA)
                    pg = PythonGenerator()
                    for args, sql, _ in th.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"):
                        expr = parse_one(sql)
                        if args.get("execute"):
                            opt = optimize(expr, s_ds, leave_tables_isolated=True)
                            plan = Plan(opt)
                            # Traverse all expressions in plan via PythonGenerator
                            for step in plan.dag:
                                for attr_v in step.__dict__.values():
                                    if isinstance(attr_v, exp.Expr):
                                        pg.generate(attr_v)
                                    elif isinstance(attr_v, list):
                                        for item in attr_v:
                                            if isinstance(item, exp.Expr):
                                                pg.generate(item)
                                    elif isinstance(attr_v, dict):
                                        for item in attr_v.values():
                                            if isinstance(item, exp.Expr):
                                                pg.generate(item)
                            transpile(sql, write="duckdb")

                def _fast_tpch_dep(self):
                    s_h = ensure_schema(th.TPCH_SCHEMA)
                    for args, sql, _ in th.load_sql_fixture_pairs("optimizer/tpc-h/tpc-h.sql"):
                        expr = parse_one(sql)
                        optimize(expr, s_h)
                        transpile(sql, write="duckdb")

                texec.TestExecutor.test_execute_tpcds = _fast_tpcds_dep
                texec.TestExecutor.test_execute_tpch = _fast_tpch_dep
                texec.TestExecutor.test_optimized_tpch = _fast_tpch_dep

            elif tf == "tests/test_parser.py":
                import tests.test_parser as tpar

                def _fast_max_nodes_dep(self):
                    parse_one("SELECT 1", max_nodes=1_000_000)
                    try:
                        parse_one("SELECT " + ",".join("x=x" for _ in range(100)), max_nodes=5)
                    except Exception:
                        pass

                tpar.TestParser.test_max_nodes = _fast_max_nodes_dep

            elif tf == "tests/test_generator.py":
                import tests.test_generator as tgen

                def _fast_gen_mem_dep(self):
                    for name in Dialects:
                        Dialect.get_or_raise(name).generator()

                tgen.TestGenerator.test_generator_memory = _fast_gen_mem_dep

            # Collect all unittest.TestCase subclasses defined in this module
            loader = unittest.defaultTestLoader
            suite = loader.loadTestsFromModule(mod)
            by_cls: dict[type, list[unittest.TestCase]] = {}
            for grp in suite:
                for tc in grp:
                    by_cls.setdefault(type(tc), []).append(tc)

            for cls, tcs in by_cls.items():
                setup_attr.clear()
                setup_key.clear()
                setup_disp.clear()
                cur_mtid[0] = -2
                su_ok = True
                try:
                    cls.setUpClass()
                except Exception:
                    su_ok = False
                cur_mtid[0] = -1

                cls_mtids: list[int] = []
                for tc in tcs:
                    mname = getattr(tc, "_testMethodName", "")
                    tid = f"{tf}::{cls.__name__}::{mname}"
                    mtid = tid_to_mtid.get(tid)
                    if mtid is None:
                        continue
                    if tid in _SEPARATELY_HANDLED_TEST_IDS:
                        seen_mtids.add(mtid)
                        continue
                    if tid in ALWAYS_KEEP_TEST_IDS:
                        failed_mtids.add(mtid)
                        continue
                    if not su_ok:
                        failed_mtids.add(mtid)
                        continue
                    cls_mtids.append(mtid)
                    cur_mtid[0] = mtid
                    res = unittest.TestResult()
                    try:
                        tc.run(res)
                        if res.wasSuccessful():
                            seen_mtids.add(mtid)
                        else:
                            failed_mtids.add(mtid)
                    except Exception:
                        failed_mtids.add(mtid)
                    finally:
                        cur_mtid[0] = -1

                # Union setUpClass reads into every test of cls!
                for mtid in cls_mtids:
                    for k in setup_attr:
                        attr_reads.setdefault(k, set()).add(mtid)
                    for k in setup_key:
                        dict_reads.setdefault(k, set()).add(mtid)
                    for k in setup_disp:
                        gen_dispatch.setdefault(k, set()).add(mtid)

                try:
                    cls.tearDownClass()
                except Exception:
                    pass
    finally:
        restore()

    return {
        "attr_reads": {k: sorted(v) for k, v in attr_reads.items()},
        "dict_reads": {k: sorted(v) for k, v in dict_reads.items()},
        "gen_dispatch": {k: sorted(v) for k, v in gen_dispatch.items()},
        "seen_mtids": sorted(seen_mtids),
        "failed_mtids": sorted(failed_mtids),
        "tracked_attrs": sorted(tracked_attrs),
        "tracked_dicts": sorted(tracked_dicts),
        "wrapped_methods": sorted(wrapped_methods),
    }


def build_dep_index(sg_root: Path, core_maps_path: Path = Path("/app/maps/core_maps.zlib"), out_path: Path = DEP_INDEX_PATH) -> tuple[int, int, int, int, int]:
    """Build /app/dep_index.marshal across all 60 test files in 4 parallel processes."""
    t0 = time.perf_counter()
    maps = marshal.loads(zlib.decompress(core_maps_path.read_bytes()))
    test_ids: list[str] = maps["test_ids"]
    tid_to_mtid = {tid: i for i, tid in enumerate(test_ids)}

    all_tfiles = sorted(
        {tid.split("::")[0] for tid in test_ids}
    )
    # Balance test files across 4 workers (put test_optimizer & test_executor on separate workers!)
    n_workers = min(4, os.cpu_count() or 2)
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    heavy_first = [
        "tests/test_optimizer.py",
        "tests/test_executor.py",
        "tests/dialects/test_dialect.py",
        "tests/dialects/test_snowflake.py",
    ]
    ordered = [f for f in heavy_first if f in all_tfiles] + [f for f in all_tfiles if f not in heavy_first]
    for i, tf in enumerate(ordered):
        buckets[i % n_workers].append(tf)

    merged_attr: dict[tuple[str, str], set[int]] = {}
    merged_dict: dict[tuple[str, str, str], set[int]] = {}
    merged_disp: dict[tuple[str, str], set[int]] = {}
    seen_all: set[int] = set()
    failed_all: set[int] = set()
    tracked_attrs: set[str] = set()
    tracked_dicts: set[str] = set()
    wrapped_methods: set[tuple[str, str]] = set()

    tasks = [(str(sg_root), b, tid_to_mtid) for b in buckets if b]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for res in pool.map(_collect_dep_worker, tasks):
            for k, v in res["attr_reads"].items():
                merged_attr.setdefault(k, set()).update(v)
            for k, v in res["dict_reads"].items():
                merged_dict.setdefault(k, set()).update(v)
            for k, v in res["gen_dispatch"].items():
                merged_disp.setdefault(k, set()).update(v)
            seen_all.update(res["seen_mtids"])
            failed_all.update(res["failed_mtids"])
            tracked_attrs.update(res["tracked_attrs"])
            tracked_dicts.update(res["tracked_dicts"])
            wrapped_methods.update(res["wrapped_methods"])

    # Any test in maps["test_ids"] that was NOT cleanly observed stays in always_keep_mtids!
    always_keep: set[int] = set(failed_all)
    for mtid, tid in enumerate(test_ids):
        if tid in _SEPARATELY_HANDLED_TEST_IDS:
            continue
        tf = tid.split("::")[0]
        if tf in ALWAYS_KEEP_TEST_FILES or tid in ALWAYS_KEEP_TEST_IDS or mtid not in seen_all:
            always_keep.add(mtid)

    # Group dict_reads by (recv, attr) -> {key: list[mtid]} for fast O(1) lookup
    dict_by_ra: dict[tuple[str, str], dict[str, list[int]]] = {}
    for (r, a, ck), mtids in merged_dict.items():
        dict_by_ra.setdefault((r, a), {})[ck] = sorted(mtids)

    payload = {
        "attr_reads": {k: sorted(v) for k, v in merged_attr.items()},
        "dict_by_ra": dict_by_ra,
        "gen_dispatch": {k: sorted(v) for k, v in merged_disp.items()},
        "tracked_attrs": sorted(tracked_attrs),
        "tracked_dicts": sorted(tracked_dicts),
        "wrapped_methods": sorted(wrapped_methods),
        "always_keep_mtids": sorted(always_keep),
    }
    blob = marshal.dumps(payload)
    out_path.write_bytes(blob)
    dt = time.perf_counter() - t0
    print(
        f"[dep_index] seen={len(seen_all)}/{len(test_ids)} always_keep={len(always_keep)} "
        f"attrs={len(merged_attr)} dicts={len(merged_dict)} disp={len(merged_disp)} "
        f"({len(blob)/1024:.0f} KB) in {dt:.1f}s",
        flush=True,
    )
    return len(seen_all), len(always_keep), len(merged_attr), len(merged_dict), len(blob)


_DEP_INDEX_CACHE: dict[str, Any] | None = None


def load_dep_index(path: Path = DEP_INDEX_PATH) -> dict[str, Any] | None:
    global _DEP_INDEX_CACHE
    if _DEP_INDEX_CACHE is None:
        if path.is_file():
            try:
                _DEP_INDEX_CACHE = marshal.loads(path.read_bytes())
            except Exception:
                _DEP_INDEX_CACHE = None
    return _DEP_INDEX_CACHE

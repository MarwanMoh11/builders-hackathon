"""Dependency-checked stage-reuse replay & lazy DuckDB query cache (Wave 1 / v4).

All index data is generated at Docker build time from pinned sqlglot@eb11aec (0 bytes in Git):
  1. /app/val_replay/<stem>.idx + <stem>.ast  -- per-dialect-test Validator operation index
  2. /app/opt_replay/<file>.idx + <file>.ast  -- per-optimizer-fixture stage index + annotate_funcs/types
  3. /app/duckdb_query_cache.pkl.zlib         -- read-only query-result cache for TestOptimizer DuckDB tests

Soundness invariants:
  - Replay activates ONLY when every patched file is a safe body/dict/attr edit inside
    sqlglot/{generators,parsers,dialects}/*.py (excluding base/core files), with zero
    deleted files, zero syntax errors, and verified clean import of all patched modules.
  - Every dependency set records ALL Dialect, Parser, and Generator classes observed at
    baseline during each stage (including indirect calls inside parse/optimize/generate).
  - Structural AST identity (ast_struct_sig) verifies class name, ordered args, parent
    pointers, arg_key, list index, comments, _type, _meta, and exact primitive/Enum types.
  - Any missing cache record, non-deterministic observation, unsupported object type,
    consumed return value, or active assert/mock context falls back to exact live execution.
"""

from __future__ import annotations

import hashlib
import marshal
import mmap
import os
import pickle
import subprocess
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

VAL_REPLAY_DIR = Path("/app/val_replay")
OPT_REPLAY_DIR = Path("/app/opt_replay")
DUCKDB_CACHE_PATH = Path("/app/duckdb_query_cache.pkl.zlib")

_UNPACKABLE = object()

# Cross-module non-class imports from sqlglot/parsers/* or sqlglot/generators/* into sqlglot/dialects/*
_EXTRA_PARSER_TO_DIALECT: dict[str, tuple[str, ...]] = {
    "sqlglot/parsers/exasol.py": ("Exasol",),
    "sqlglot/parsers/singlestore.py": ("SingleStore",),
}
_EXTRA_GEN_ATTR_TO_DIALECT: dict[tuple[str, str], tuple[str, ...]] = {
    ("sqlglot/generators/duckdb.py", "WS_CONTROL_CHARS_TO_DUCK"): ("DuckDB",),
}


@dataclass(frozen=True)
class ReplayScope:
    dirty_dials: frozenset[str]
    dirty_pars: frozenset[str]
    dirty_gens: frozenset[str]
    dirty_files: frozenset[str]
    is_gen_only: bool


# ---------------------------------------------------------------------------
# Exact Structural AST Signature + Root-Agnostic Marshal-Safe AST Pack/Unpack
# ---------------------------------------------------------------------------

_EXP_MAP_CACHE: dict[int, dict[str, type]] = {}


def _get_exp_map(exp_mod: Any) -> dict[str, type]:
    mid = id(exp_mod)
    m = _EXP_MAP_CACHE.get(mid)
    if m is None:
        Expr = exp_mod.Expr
        m = {
            name: obj
            for name, obj in vars(exp_mod).items()
            if isinstance(obj, type) and issubclass(obj, Expr)
        }
        _EXP_MAP_CACHE[mid] = m
    return m


def _hash_scalar(upd: Any, v: Any) -> bool:
    """Hash a non-Expr scalar/container into blake2b. Returns False if unsupported."""
    if v is None:
        upd(b"N;")
        return True
    t = type(v)
    if t is str:
        b = v.encode("utf-8", "surrogatepass")
        upd(b"S"); upd(str(len(b)).encode()); upd(b":"); upd(b); upd(b";")
        return True
    if t is bool:
        upd(b"B1;" if v else b"B0;")
        return True
    if t is int:
        upd(b"I"); upd(str(v).encode()); upd(b";")
        return True
    if t is float:
        upd(b"F"); upd(repr(v).encode()); upd(b";")
        return True
    if isinstance(v, Enum):
        upd(b"M"); upd(t.__qualname__.encode()); upd(b"."); upd(v.name.encode()); upd(b";")
        return True
    if t is tuple:
        upd(b"("); upd(str(len(v)).encode()); upd(b":")
        for item in v:
            if not _hash_scalar(upd, item):
                return False
        upd(b")")
        return True
    if t is list:
        upd(b"["); upd(str(len(v)).encode()); upd(b":")
        for item in v:
            if not _hash_scalar(upd, item):
                return False
        upd(b"]")
        return True
    if t is dict:
        try:
            items = sorted(v.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0])))
        except Exception:
            return False
        upd(b"{"); upd(str(len(items)).encode()); upd(b":")
        for dk, dv in items:
            if not _hash_scalar(upd, dk) or not _hash_scalar(upd, dv):
                return False
        upd(b"}")
        return True
    return False


def ast_struct_sig(root: Any, exp_mod: Any) -> bytes | None:
    """Compute a 16-byte exact structural AST fingerprint including parent/arg_key/index integrity.
    Returns None if any AST invariant fails or an unsupported object type is encountered."""
    Expr = exp_mod.Expr
    if not isinstance(root, Expr):
        return None
    if root.parent is not None or root.arg_key is not None or root.index is not None:
        return None
    h = hashlib.blake2b(digest_size=16)
    upd = h.update

    def _visit(node: Any) -> bool:
        upd(b"E"); upd(type(node).__name__.encode()); upd(b"\0")
        comments = node.comments
        if comments is None:
            upd(b"c0")
        else:
            upd(b"c1"); upd(str(len(comments)).encode()); upd(b":")
            for c in comments:
                if type(c) is not str:
                    return False
                cb = c.encode("utf-8", "surrogatepass")
                upd(str(len(cb)).encode()); upd(b":"); upd(cb)
        meta = node._meta
        if meta is None:
            upd(b"m0")
        else:
            upd(b"m1")
            if not _hash_scalar(upd, meta):
                return False
        nt = node._type
        if nt is None:
            upd(b"t0")
        elif isinstance(nt, Expr):
            upd(b"t1")
            if not _visit(nt):
                return False
        else:
            upd(b"t2")
            if not _hash_scalar(upd, nt):
                return False
        args = node.args
        upd(b"a"); upd(str(len(args)).encode()); upd(b":")
        for k, v in args.items():
            kb = k.encode("utf-8")
            upd(str(len(kb)).encode()); upd(b":"); upd(kb); upd(b"=")
            if isinstance(v, Expr):
                if v.parent is not node or v.arg_key != k or v.index is not None:
                    return False
                if not _visit(v):
                    return False
            elif type(v) is list:
                upd(b"L"); upd(str(len(v)).encode()); upd(b":")
                for idx, item in enumerate(v):
                    if isinstance(item, Expr):
                        if item.parent is not node or item.arg_key != k or item.index != idx:
                            return False
                        if not _visit(item):
                            return False
                    else:
                        if not _hash_scalar(upd, item):
                            return False
                upd(b"]")
            else:
                if not _hash_scalar(upd, v):
                    return False
        upd(b";")
        return True

    if not _visit(root):
        return None
    return h.digest()


def pack_expr(root: Any, exp_mod: Any) -> Any:
    """Pack an Expr tree into marshal-safe primitives portable across /sqlglot and /sqlglot-c.
    Returns _UNPACKABLE if any node/value cannot be faithfully reconstructed."""
    Expr = exp_mod.Expr
    exp_map = _get_exp_map(exp_mod)

    def _p(x: Any) -> Any:
        if x is None:
            return None
        if isinstance(x, Expr):
            cname = type(x).__name__
            if cname not in exp_map:
                return _UNPACKABLE
            p_args = []
            for k, v in x.args.items():
                if type(k) is not str:
                    return _UNPACKABLE
                pv = _p(v)
                if pv is _UNPACKABLE:
                    return _UNPACKABLE
                p_args.append((k, pv))
            pt = _p(x._type) if x._type is not None else None
            if pt is _UNPACKABLE:
                return _UNPACKABLE
            pm = _p(x._meta) if x._meta is not None else None
            if pm is _UNPACKABLE:
                return _UNPACKABLE
            pc = x.comments
            if pc is not None:
                if type(pc) is not list or any(type(c) is not str for c in pc):
                    return _UNPACKABLE
                pc = tuple(pc)
            return ("E", cname, tuple(p_args), pt, pm, pc)
        t = type(x)
        if t in (str, int, float, bool):
            return x
        if t is list:
            out = []
            for item in x:
                pi = _p(item)
                if pi is _UNPACKABLE:
                    return _UNPACKABLE
                out.append(pi)
            return out
        if t is tuple:
            out_t = []
            for item in x:
                pi = _p(item)
                if pi is _UNPACKABLE:
                    return _UNPACKABLE
                out_t.append(pi)
            return ("T", tuple(out_t))
        if t is dict:
            out_d = []
            for dk, dv in x.items():
                pk = _p(dk)
                pv = _p(dv)
                if pk is _UNPACKABLE or pv is _UNPACKABLE:
                    return _UNPACKABLE
                out_d.append((pk, pv))
            return ("D", tuple(out_d))
        if isinstance(x, Enum):
            if getattr(exp_mod, t.__name__, None) is t:
                return ("M1", t.__name__, x.name)
            qn = t.__qualname__.split(".")
            if len(qn) == 2:
                outer = getattr(exp_mod, qn[0], None)
                if outer is not None and getattr(outer, qn[1], None) is t:
                    return ("M2", qn[0], qn[1], x.name)
            return _UNPACKABLE
        return _UNPACKABLE

    return _p(root)


def unpack_expr(packed: Any, exp_mod: Any) -> Any:
    """Reconstruct a fresh Expr tree (with exact parent/arg_key/index/meta/type/comments)
    using exp_mod (works identically for /sqlglot and /sqlglot-c!)."""
    Expr = exp_mod.Expr
    exp_map = _get_exp_map(exp_mod)

    def _u(x: Any) -> Any:
        tx = type(x)
        if tx is tuple:
            tag = x[0]
            if tag == "E":
                cls = exp_map[x[1]]
                obj = cls.__new__(cls)
                args: dict[str, Any] = {}
                for k, pv in x[2]:
                    uv = _u(pv)
                    args[k] = uv
                    if isinstance(uv, Expr):
                        uv.parent = obj
                        uv.arg_key = k
                        uv.index = None
                    elif type(uv) is list:
                        for idx, item in enumerate(uv):
                            if isinstance(item, Expr):
                                item.parent = obj
                                item.arg_key = k
                                item.index = idx
                obj.args = args
                obj.parent = None
                obj.arg_key = None
                obj.index = None
                obj._type = _u(x[3]) if x[3] is not None else None
                obj._meta = _u(x[4]) if x[4] is not None else None
                obj.comments = list(x[5]) if x[5] is not None else None
                obj._hash = None
                return obj
            if tag == "M1":
                return getattr(exp_mod, x[1])[x[2]]
            if tag == "M2":
                return getattr(getattr(exp_mod, x[1]), x[2])[x[3]]
            if tag == "T":
                return tuple(_u(v) for v in x[1])
            if tag == "D":
                return {_u(k): _u(v) for k, v in x[1]}
            return x
        if tx is list:
            return [_u(v) for v in x]
        return x

    return _u(packed)


def norm_dialect_key(d: Any) -> str | None:
    if d is None:
        return ""
    if type(d) is str:
        return d
    val = getattr(d, "value", None)
    if type(val) is str:
        return val
    return None


# ---------------------------------------------------------------------------
# Soundness Gate: Classify Replay Scope from CommitPlan + Static Graphs
# ---------------------------------------------------------------------------

def classify_replay_scope(
    commit_plan: Any,
    maps: dict[str, Any],
    engine: Any = None,
) -> ReplayScope | None:
    """Return ReplayScope if the commit is 100% eligible for dependency-checked stage replay,
    else None (disables replay -> full live verification)."""
    if (
        commit_plan is None
        or commit_plan.is_noop
        or commit_plan.deleted_files
        or commit_plan.has_syntax_error
        or commit_plan.syntax_error_files
        or not commit_plan.patched_files
    ):
        return None

    static_meta = maps.get("static_ast_meta")
    if not static_meta:
        return None
    rev_imp: dict[str, Any] = static_meta.get("rev_import_graph", {})
    file_classes: dict[str, list[str]] = static_meta.get("file_classes", {})
    class_subs: dict[str, list[str]] = maps.get("class_subclasses", {})

    all_dials = set(class_subs.get("Dialect", []))
    all_pars = set(class_subs.get("Parser", []))
    all_gens = set(class_subs.get("Generator", []))

    _SAFE_KINDS = frozenset({"func_body", "dict_key", "dict_lambda_body", "class_attr"})

    gen_files: set[str] = set()
    par_files: set[str] = set()
    dial_files: set[str] = set()

    for rel in commit_plan.patched_files:
        if rel.startswith("sqlglot/generators/") and rel.endswith(".py") and rel != "sqlglot/generators/__init__.py":
            gen_files.add(rel)
        elif rel.startswith("sqlglot/parsers/") and rel.endswith(".py") and rel not in ("sqlglot/parsers/__init__.py", "sqlglot/parsers/base.py"):
            par_files.add(rel)
        elif rel.startswith("sqlglot/dialects/") and rel.endswith(".py") and rel not in ("sqlglot/dialects/__init__.py", "sqlglot/dialects/dialect.py"):
            dial_files.add(rel)
        else:
            return None

        ana = commit_plan.py_analyses.get(rel)
        if ana is None or ana.syntax_error or not ana.units:
            return None
        for u in ana.units:
            if u.kind not in _SAFE_KINDS:
                return None

    def _expand_subclasses(seed_classes: set[str]) -> set[str]:
        out = set(seed_classes)
        for c in list(seed_classes):
            out.update(class_subs.get(c, ()))
        return out

    dirty_dials: set[str] = set()
    dirty_pars: set[str] = set()
    dirty_gens: set[str] = set()
    dirty_files: set[str] = set(commit_plan.patched_files)

    # 1. Propagate generator files (follow generator->generator rev-imports; reject unexpected core/parser importers)
    q = list(gen_files)
    seen_g = set(gen_files)
    while q:
        f = q.pop()
        dirty_files.add(f)
        for cname in file_classes.get(f, ()):
            if cname in all_gens:
                dirty_gens.add(cname)
        # Check special generator attr re-exports into dialects
        ana = commit_plan.py_analyses.get(f)
        if ana is not None:
            for u in ana.units:
                if u.symbol_name and (f, u.symbol_name) in _EXTRA_GEN_ATTR_TO_DIALECT:
                    dirty_dials.update(_EXTRA_GEN_ATTR_TO_DIALECT[(f, u.symbol_name)])
        for imp_f in rev_imp.get(f, ()):
            if imp_f.startswith("sqlglot/generators/"):
                if imp_f not in seen_g:
                    seen_g.add(imp_f)
                    q.append(imp_f)
            elif imp_f.startswith("sqlglot/dialects/") or imp_f == "sqlglot/executor/python.py" or imp_f.startswith("tests/"):
                continue
            else:
                return None

    # 2. Propagate parser files (follow parser->parser AND parser->generator rev-imports)
    q = list(par_files)
    seen_p = set(par_files)
    extra_gen_from_par: set[str] = set()
    while q:
        f = q.pop()
        dirty_files.add(f)
        for cname in file_classes.get(f, ()):
            if cname in all_pars:
                dirty_pars.add(cname)
        if f in _EXTRA_PARSER_TO_DIALECT:
            dirty_dials.update(_EXTRA_PARSER_TO_DIALECT[f])
        for imp_f in rev_imp.get(f, ()):
            if imp_f.startswith("sqlglot/parsers/"):
                if imp_f not in seen_p:
                    seen_p.add(imp_f)
                    q.append(imp_f)
            elif imp_f.startswith("sqlglot/generators/"):
                extra_gen_from_par.add(imp_f)
            elif imp_f.startswith("sqlglot/dialects/") or imp_f.startswith("tests/"):
                continue
            else:
                return None

    # Propagate any generators that import edited parsers (e.g. exasol/snowflake/teradata/tsql)
    q = list(extra_gen_from_par)
    seen_eg = set(extra_gen_from_par)
    while q:
        f = q.pop()
        dirty_files.add(f)
        for cname in file_classes.get(f, ()):
            if cname in all_gens:
                dirty_gens.add(cname)
        for imp_f in rev_imp.get(f, ()):
            if imp_f.startswith("sqlglot/generators/") and imp_f not in seen_eg:
                seen_eg.add(imp_f)
                q.append(imp_f)

    # 3. Propagate dialect files (marks Dialect + Parser + Generator + subclasses dirty!)
    q = list(dial_files)
    seen_d = set(dial_files)
    while q:
        f = q.pop()
        dirty_files.add(f)
        for cname in file_classes.get(f, ()):
            if cname in all_dials:
                dirty_dials.add(cname)
        for imp_f in rev_imp.get(f, ()):
            if imp_f.startswith("sqlglot/dialects/"):
                if imp_f not in seen_d:
                    seen_d.add(imp_f)
                    q.append(imp_f)
            elif imp_f.startswith("sqlglot/parsers/"):
                for cname in file_classes.get(imp_f, ()):
                    if cname in all_pars:
                        dirty_pars.add(cname)
            elif imp_f.startswith("sqlglot/generators/"):
                for cname in file_classes.get(imp_f, ()):
                    if cname in all_gens:
                        dirty_gens.add(cname)
            elif imp_f.startswith("tests/"):
                continue
            else:
                return None

    # Expand all subclasses
    dirty_dials = _expand_subclasses(dirty_dials)
    dirty_pars = _expand_subclasses(dirty_pars)
    dirty_gens = _expand_subclasses(dirty_gens)

    # For any dirty dialect, also mark its corresponding parser & generator classes dirty
    if dirty_dials:
        for f, classes in file_classes.items():
            if f.startswith("sqlglot/dialects/") and any(c in dirty_dials for c in classes):
                stem = Path(f).stem
                for cname in file_classes.get(f"sqlglot/parsers/{stem}.py", ()):
                    if cname in all_pars:
                        dirty_pars.add(cname)
                for cname in file_classes.get(f"sqlglot/generators/{stem}.py", ()):
                    if cname in all_gens:
                        dirty_gens.add(cname)
        dirty_pars = _expand_subclasses(dirty_pars)
        dirty_gens = _expand_subclasses(dirty_gens)

    # Safety fallback: if a patched file defines no tracked class (e.g. only helper function),
    # ensure its dialect stem's class family is marked dirty!
    for f in gen_files:
        if not any(c in dirty_gens for c in file_classes.get(f, ())):
            return None
    for f in par_files:
        if not any(c in dirty_pars for c in file_classes.get(f, ())):
            return None
    for f in dial_files:
        if not any(c in dirty_dials for c in file_classes.get(f, ())):
            return None

    return ReplayScope(
        dirty_dials=frozenset(dirty_dials),
        dirty_pars=frozenset(dirty_pars),
        dirty_gens=frozenset(dirty_gens),
        dirty_files=frozenset(dirty_files),
        is_gen_only=bool(gen_files and not par_files and not dial_files and not dirty_dials and not dirty_pars),
    )


# ---------------------------------------------------------------------------
# Lazy Runtime Shard Loaders (CoW-preloaded .idx + O(1) mmap byte-sliced .ast!)
# ---------------------------------------------------------------------------

def _write_ast_table(asts: list[Any], ast_path: Path) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    pos = 0
    with open(ast_path, "wb") as f:
        for a in asts:
            b = marshal.dumps(a)
            ln = len(b)
            f.write(b)
            offsets.append((pos, ln))
            pos += ln
    return offsets


class _MmapAstStore:
    __slots__ = ("_path", "_offsets", "_fd", "_mm")

    def __init__(self, path: Path, offsets: list[tuple[int, int]]) -> None:
        self._path = path
        self._offsets = offsets
        self._fd: int = -1
        self._mm: mmap.mmap | None = None

    def get_expr(self, ast_idx: int, exp_mod: Any) -> Any:
        mm = self._mm
        if mm is None:
            fd = os.open(self._path, os.O_RDONLY)
            try:
                mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            except Exception:
                os.close(fd)
                raise
            self._fd = fd
            self._mm = mm
        off, ln = self._offsets[ast_idx]
        return unpack_expr(marshal.loads(mm[off : off + ln]), exp_mod)

    def close(self) -> None:
        if self._mm is not None:
            try:
                self._mm.close()
            except Exception:
                pass
            self._mm = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except Exception:
                pass
            self._fd = -1


_RAW_VAL_IDX: dict[str, tuple[list, dict, dict, list[tuple[int, int]]] | None] = {}
_RAW_OPT_IDX: dict[str, tuple[list, dict, list[tuple[int, int]]] | None] = {}


def preload_all_replay_indices() -> None:
    """Preload all uncompressed .idx marshal tables ONCE in parent before os.fork()
    so every forked child shares them read-only via CoW with 0.00ms load overhead!"""
    if VAL_REPLAY_DIR.is_dir():
        for p in VAL_REPLAY_DIR.glob("*.idx"):
            stem = p.stem
            if stem not in _RAW_VAL_IDX:
                try:
                    _RAW_VAL_IDX[stem] = marshal.loads(p.read_bytes())
                except Exception:
                    _RAW_VAL_IDX[stem] = None
    if OPT_REPLAY_DIR.is_dir():
        for p in OPT_REPLAY_DIR.glob("*.idx"):
            name = p.stem
            if name not in _RAW_OPT_IDX:
                try:
                    _RAW_OPT_IDX[name] = marshal.loads(p.read_bytes())
                except Exception:
                    _RAW_OPT_IDX[name] = None


class ValReplayShard:
    __slots__ = ("stem", "deps_table", "parses", "edges", "dirty_flags", "_store")

    def __init__(
        self,
        stem: str,
        deps_table: list,
        parses: dict,
        edges: dict,
        ast_offsets: list[tuple[int, int]],
        ast_path: Path,
        scope: ReplayScope,
    ) -> None:
        self.stem = stem
        self.deps_table = deps_table
        self.parses = parses
        self.edges = edges
        self._store = _MmapAstStore(ast_path, ast_offsets)
        dd = scope.dirty_dials
        dp = scope.dirty_pars
        dg = scope.dirty_gens
        self.dirty_flags: list[bool] = [
            bool(
                (dd and any(x in dd for x in ds))
                or (dp and any(x in dp for x in ps))
                or (dg and any(x in dg for x in gs))
            )
            for ds, ps, gs in deps_table
        ]

    def get_expr(self, ast_idx: int, exp_mod: Any) -> Any:
        return self._store.get_expr(ast_idx, exp_mod)

    def close(self) -> None:
        self._store.close()


_VAL_SHARD_CACHE: dict[str, ValReplayShard | None] = {}
_OPT_SHARD_CACHE: dict[str, OptReplayShard | None] = {}


def clear_replay_shard_caches() -> None:
    for sh in _VAL_SHARD_CACHE.values():
        if sh is not None:
            sh.close()
    for sh in _OPT_SHARD_CACHE.values():
        if sh is not None:
            sh.close()
    _VAL_SHARD_CACHE.clear()
    _OPT_SHARD_CACHE.clear()


def get_val_shard(test_file: str, scope: ReplayScope) -> ValReplayShard | None:
    shard = _VAL_SHARD_CACHE.get(test_file)
    if shard is not None or test_file in _VAL_SHARD_CACHE:
        return shard
    stem = Path(test_file).stem
    raw = _RAW_VAL_IDX.get(stem)
    if raw is None and stem not in _RAW_VAL_IDX:
        idx_path = VAL_REPLAY_DIR / f"{stem}.idx"
        if idx_path.is_file():
            try:
                raw = marshal.loads(idx_path.read_bytes())
            except Exception:
                raw = None
        _RAW_VAL_IDX[stem] = raw
    ast_path = VAL_REPLAY_DIR / f"{stem}.ast"
    if raw is None or not ast_path.is_file():
        _VAL_SHARD_CACHE[test_file] = None
        return None
    deps_table, parses, edges, ast_offsets = raw
    shard = ValReplayShard(stem, deps_table, parses, edges, ast_offsets, ast_path, scope)
    _VAL_SHARD_CACHE[test_file] = shard
    return shard


class OptReplayShard:
    __slots__ = ("name", "deps_table", "rows", "dirty_flags", "_store")

    def __init__(
        self,
        name: str,
        deps_table: list,
        rows: dict,
        ast_offsets: list[tuple[int, int]],
        ast_path: Path,
        scope: ReplayScope,
    ) -> None:
        self.name = name
        self.deps_table = deps_table
        self.rows = rows
        self._store = _MmapAstStore(ast_path, ast_offsets)
        dd = scope.dirty_dials
        dp = scope.dirty_pars
        dg = scope.dirty_gens
        self.dirty_flags: list[bool] = [
            bool(
                (dd and any(x in dd for x in ds))
                or (dp and any(x in dp for x in ps))
                or (dg and any(x in dg for x in gs))
            )
            for ds, ps, gs in deps_table
        ]

    def get_expr(self, ast_idx: int, exp_mod: Any) -> Any:
        return self._store.get_expr(ast_idx, exp_mod)

    def close(self) -> None:
        self._store.close()


def get_opt_shard(name: str, scope: ReplayScope) -> OptReplayShard | None:
    shard = _OPT_SHARD_CACHE.get(name)
    if shard is not None or name in _OPT_SHARD_CACHE:
        return shard
    safe_name = name.replace("/", "__")
    raw = _RAW_OPT_IDX.get(safe_name)
    if raw is None and safe_name not in _RAW_OPT_IDX:
        idx_path = OPT_REPLAY_DIR / f"{safe_name}.idx"
        if idx_path.is_file():
            try:
                raw = marshal.loads(idx_path.read_bytes())
            except Exception:
                raw = None
        _RAW_OPT_IDX[safe_name] = raw
    ast_path = OPT_REPLAY_DIR / f"{safe_name}.ast"
    if raw is None or not ast_path.is_file():
        _OPT_SHARD_CACHE[name] = None
        return None
    deps_table, rows, ast_offsets = raw
    shard = OptReplayShard(name, deps_table, rows, ast_offsets, ast_path, scope)
    _OPT_SHARD_CACHE[name] = shard
    return shard


# ---------------------------------------------------------------------------
# Lazy DuckDB Query-Result Cache Proxy (Zero Engine Startup on Cache Hit!)
# ---------------------------------------------------------------------------

_DUCKDB_CACHE_DATA: dict[str, Any] | None = None
_READ_SQL_PREFIXES = ("SELECT", "WITH", "VALUES", "FROM", "TABLE", "(", "SUMMARIZE", "DESCRIBE", "EXPLAIN", "SHOW")


def load_duckdb_query_cache() -> dict[str, Any]:
    global _DUCKDB_CACHE_DATA
    if _DUCKDB_CACHE_DATA is None:
        if DUCKDB_CACHE_PATH.is_file():
            try:
                _DUCKDB_CACHE_DATA = pickle.loads(zlib.decompress(DUCKDB_CACHE_PATH.read_bytes()))
            except Exception:
                _DUCKDB_CACHE_DATA = {"queries": {}, "ddls": {}}
        else:
            _DUCKDB_CACHE_DATA = {"queries": {}, "ddls": {}}
    return _DUCKDB_CACHE_DATA


def _is_pure_read_sql(sql: str) -> bool:
    s = sql.lstrip()
    if not s:
        return False
    u = s[:16].upper()
    if not u.startswith(_READ_SQL_PREFIXES):
        return False
    su = sql.upper()
    if any(k in su for k in ("READ_PARQUET", "READ_CSV", "READ_JSON", "COPY ", "ATTACH ", "INSTALL ", "LOAD ")):
        return False
    return True


def _ddl_step_sig(prev_sig: bytes, sql: str) -> bytes:
    h = hashlib.blake2b(digest_size=12)
    h.update(prev_sig)
    h.update(b"|")
    h.update(sql.encode("utf-8", "surrogatepass"))
    return h.digest()


class _CachedDuckCursor:
    __slots__ = ("description", "_rows", "_pos", "_owner", "_sql")

    def __init__(self, cols: tuple[str, ...], rows: list[tuple[Any, ...]], owner: LazyCachedDuckConn, sql: str) -> None:
        self.description = tuple((c, None, None, None, None, None, None) for c in cols)
        self._rows = rows
        self._pos = 0
        self._owner = owner
        self._sql = sql

    def fetchall(self) -> list[tuple[Any, ...]]:
        res = list(self._rows[self._pos:])
        self._pos = len(self._rows)
        return res

    def fetchone(self) -> tuple[Any, ...] | None:
        if self._pos >= len(self._rows):
            return None
        r = self._rows[self._pos]
        self._pos += 1
        return r

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        res = list(self._rows[self._pos : self._pos + size])
        self._pos += len(res)
        return res

    def df(self):
        return self._owner._materialize().execute(self._sql).df()

    def fetchdf(self):
        return self._owner._materialize().execute(self._sql).fetchdf()


class LazyCachedDuckConn:
    """Drop-in lazy DuckDB connection proxy backed by build-time query cache.
    Never imports or starts DuckDB on cache hits; replays DDL and executes real DuckDB on any miss."""

    def __init__(self, init_tag: bytes, real_factory: Any, cache_enabled: bool = True) -> None:
        self._sig: bytes = init_tag
        self._real_factory = real_factory
        self._cache_enabled = cache_enabled
        self._real_conn: Any = None
        self._pending_ddl: list[str] = []
        self._closed = False

    def _materialize(self) -> Any:
        if self._real_conn is None:
            conn = self._real_factory()
            for stmt in self._pending_ddl:
                conn.execute(stmt)
            self._pending_ddl.clear()
            self._real_conn = conn
        return self._real_conn

    def execute(self, query: str, parameters: Any = None, *args: Any, **kwargs: Any) -> Any:
        if (
            self._real_conn is not None
            or not self._cache_enabled
            or parameters is not None
            or args
            or kwargs
            or type(query) is not str
        ):
            return self._materialize().execute(query, parameters, *args, **kwargs) if parameters is not None else self._materialize().execute(query, *args, **kwargs)

        cache = load_duckdb_query_cache()
        if _is_pure_read_sql(query):
            hit = cache["queries"].get((self._sig, query))
            if hit is not None:
                cols, rows = hit
                return _CachedDuckCursor(cols, rows, self, query)
            return self._materialize().execute(query)

        # DDL / DML statement
        next_sig = _ddl_step_sig(self._sig, query)
        if next_sig in cache["ddls"]:
            self._sig = next_sig
            self._pending_ddl.append(query)
            return _CachedDuckCursor((), [], self, query)
        return self._materialize().execute(query)

    def close(self) -> None:
        self._closed = True
        if self._real_conn is not None:
            self._real_conn.close()
            self._real_conn = None

    def __enter__(self) -> LazyCachedDuckConn:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._materialize(), name)


# ---------------------------------------------------------------------------
# Build-Time Collectors (run inside Dockerfile, 0 bytes in Git!)
# ---------------------------------------------------------------------------

def _install_stage_recorders() -> tuple[list[tuple[set[str], set[str], set[str]]], Any]:
    """Instrument Dialect.get_or_raise, Tokenizer.tokenize, Parser.parse/parse_into,
    and Generator.generate on Pure-Python /sqlglot to record exact class dependencies."""
    from sqlglot.dialects.dialect import Dialect
    from sqlglot.generator import Generator
    from sqlglot.parser import Parser
    from sqlglot.tokens import Tokenizer

    stack: list[tuple[set[str], set[str], set[str]]] = []

    orig_get = Dialect.get_or_raise.__func__
    orig_tok = Tokenizer.tokenize
    orig_parse = Parser.parse
    orig_parse_into = Parser.parse_into
    orig_gen = Generator.generate

    @classmethod  # type: ignore[misc]
    def _rec_get(cls, dialect=None):
        res = orig_get(cls, dialect)
        if stack:
            dn = type(res).__name__
            for dset, _, _ in stack:
                dset.add(dn)
        return res

    def _rec_tok(self, sql, *args, **kwargs):
        if stack:
            dn = type(self.dialect).__name__
            for dset, _, _ in stack:
                dset.add(dn)
        return orig_tok(self, sql, *args, **kwargs)

    def _rec_parse(self, raw_tokens, sql=None):
        if stack:
            dn = type(self.dialect).__name__
            pn = type(self).__name__
            for dset, pset, _ in stack:
                dset.add(dn)
                pset.add(pn)
        return orig_parse(self, raw_tokens, sql=sql)

    def _rec_parse_into(self, expression_types, raw_tokens, sql=None):
        if stack:
            dn = type(self.dialect).__name__
            pn = type(self).__name__
            for dset, pset, _ in stack:
                dset.add(dn)
                pset.add(pn)
        return orig_parse_into(self, expression_types, raw_tokens, sql=sql)

    def _rec_gen(self, expression, copy=True):
        if stack:
            dn = type(self.dialect).__name__
            gn = type(self).__name__
            for dset, _, gset in stack:
                dset.add(dn)
                gset.add(gn)
        return orig_gen(self, expression, copy=copy)

    Dialect.get_or_raise = _rec_get
    Tokenizer.tokenize = _rec_tok
    Parser.parse = _rec_parse
    Parser.parse_into = _rec_parse_into
    Generator.generate = _rec_gen

    def _restore() -> None:
        Dialect.get_or_raise = classmethod(orig_get)
        Tokenizer.tokenize = orig_tok
        Parser.parse = orig_parse
        Parser.parse_into = orig_parse_into
        Generator.generate = orig_gen

    return stack, _restore


def _collect_one_dialect_file(args: tuple[str, str, str]) -> tuple[str, int, int, int]:
    sg_root, test_file, out_dir_str = args
    sys.path.insert(0, sg_root)
    import importlib
    import unittest
    from sqlglot import ErrorLevel, UnsupportedError, exp, parse_one as sg_parse_one
    import tests.dialects.test_dialect as tdial

    rec_stack, restore_rec = _install_stage_recorders()

    deps_list: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    deps_to_id: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], int] = {}
    asts: list[Any] = []
    parses: dict[tuple[str, str], tuple[int, bytes, int]] = {}
    edges: dict[tuple[str, str, str, str | None, bool, bool, int], int] = {}
    bad_parses: set[tuple[str, str]] = set()

    def _intern_deps(dset: set[str], pset: set[str], gset: set[str]) -> int:
        key = (tuple(sorted(dset)), tuple(sorted(pset)), tuple(sorted(gset)))
        idx = deps_to_id.get(key)
        if idx is None:
            idx = len(deps_list)
            deps_list.append(key)
            deps_to_id[key] = idx
        return idx

    def _merge_dep_ids(a: int, b: int) -> int:
        if a == b:
            return a
        da, pa, ga = deps_list[a]
        db, pb, gb = deps_list[b]
        return _intern_deps(set(da) | set(db), set(pa) | set(pb), set(ga) | set(gb))

    def _record_parse(d_key: str, sql: str, expr: Any, dset: set[str], pset: set[str], gset: set[str]) -> bool:
        pk = (d_key, sql)
        if pk in bad_parses:
            return False
        sig = ast_struct_sig(expr, exp)
        if sig is None:
            bad_parses.add(pk)
            parses.pop(pk, None)
            return False
        dep_id = _intern_deps(dset, pset, gset)
        prev = parses.get(pk)
        if prev is not None:
            if prev[1] != sig:
                bad_parses.add(pk)
                parses.pop(pk, None)
                return False
            merged = _merge_dep_ids(prev[0], dep_id)
            if merged != prev[0]:
                parses[pk] = (merged, sig, prev[2])
            return True
        packed = pack_expr(expr, exp)
        if packed is _UNPACKABLE:
            ast_idx = -1
        else:
            ast_idx = len(asts)
            asts.append(packed)
        parses[pk] = (dep_id, sig, ast_idx)
        return True

    def _record_edge(
        rd: str, rsql: str, wd: str, expected: str | None, pretty: bool, identify: bool, ulevel: int,
        dset: set[str], pset: set[str], gset: set[str],
    ) -> None:
        ek = (rd, rsql, wd, expected, bool(pretty), bool(identify), ulevel)
        dep_id = _intern_deps(dset, pset, gset)
        prev = edges.get(ek)
        edges[ek] = dep_id if prev is None else _merge_dep_ids(prev, dep_id)

    # Track assertLogs/assertRaises/mock.patch contexts via a real class proxy (Python `with` resolves __enter__/__exit__ on type(cm)!)
    in_ctx = [0]

    class _RecCtxProxy:
        __slots__ = ("_cm",)

        def __init__(self, cm: Any) -> None:
            self._cm = cm

        def __enter__(self) -> Any:
            in_ctx[0] += 1
            try:
                return self._cm.__enter__()
            except BaseException:
                in_ctx[0] -= 1
                raise

        def __exit__(self, *ea: Any) -> Any:
            try:
                return self._cm.__exit__(*ea)
            finally:
                in_ctx[0] -= 1

        def __getattr__(self, name: str) -> Any:
            return getattr(self._cm, name)

    for cname in ("assertLogs", "assertNoLogs", "assertRaises", "assertRaisesRegex", "assertWarns", "assertWarnsRegex"):
        orig_cm = getattr(unittest.TestCase, cname, None)
        if orig_cm is None:
            continue
        has_callable = cname not in ("assertLogs", "assertNoLogs")
        min_a = 3 if "Regex" in cname else 2

        def _make_cm(orig, hc, ma):
            def _w(self, *a, **kw):
                if hc and len(a) >= ma and a[ma - 1] is not None:
                    in_ctx[0] += 1
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        in_ctx[0] -= 1
                cm = orig(self, *a, **kw)
                if not hasattr(cm, "__enter__"):
                    return cm
                return _RecCtxProxy(cm)
            return _w

        setattr(unittest.TestCase, cname, _make_cm(orig_cm, has_callable, min_a))

    from unittest import mock as _umock
    _ope, _opx = _umock._patch.__enter__, _umock._patch.__exit__

    def _wpe(self):
        r = _ope(self)
        in_ctx[0] += 1
        return r

    def _wpx(self, *ei):
        try:
            return _opx(self, *ei)
        finally:
            in_ctx[0] -= 1

    _umock._patch.__enter__ = _wpe  # type: ignore[method-assign]
    _umock._patch.__exit__ = _wpx  # type: ignore[method-assign]

    orig_vi = tdial.Validator.validate_identity
    orig_vt = tdial.Validator.validate_transpile
    orig_va = tdial.Validator.validate_all
    orig_po = tdial.Validator.parse_one

    def _rec_vi(self, sql, write_sql=None, pretty=False, check_command_warning=False, identify=False):
        dk = norm_dialect_key(self.dialect)
        if (
            in_ctx[0] > 0
            or check_command_warning
            or type(self).parse_one is not orig_po
            or type(sql) is not str
            or dk is None
            or (write_sql is not None and type(write_sql) is not str)
        ):
            return orig_vi(self, sql, write_sql=write_sql, pretty=pretty, check_command_warning=check_command_warning, identify=identify)
        p_d, p_p, p_g = set(), set(), set()
        rec_stack.append((p_d, p_p, p_g))
        try:
            expression = self.parse_one(sql)
        finally:
            rec_stack.pop()
        g_d, g_p, g_g = set(), set(), set()
        rec_stack.append((g_d, g_p, g_g))
        try:
            actual = expression.sql(dialect=self.dialect, pretty=pretty, identify=identify)
        finally:
            rec_stack.pop()
        expected = write_sql or sql
        self.assertEqual(expected, actual)
        if _record_parse(dk, sql, expression, p_d, p_p, p_g):
            _record_edge(dk, sql, dk, expected, bool(pretty), bool(identify), 0, g_d, g_p, g_g)
        return expression

    def _rec_vt(self, sql, write_sql, write_dialect=None):
        rd = norm_dialect_key(self.dialect)
        wd = norm_dialect_key(write_dialect)
        if (
            in_ctx[0] > 0
            or type(self).parse_one is not orig_po
            or type(sql) is not str
            or type(write_sql) is not str
            or rd is None
            or wd is None
        ):
            return orig_vt(self, sql, write_sql, write_dialect=write_dialect)
        p_d, p_p, p_g = set(), set(), set()
        rec_stack.append((p_d, p_p, p_g))
        try:
            expression = self.parse_one(sql)
        finally:
            rec_stack.pop()
        g_d, g_p, g_g = set(), set(), set()
        rec_stack.append((g_d, g_p, g_g))
        try:
            actual = expression.sql(dialect=write_dialect)
        finally:
            rec_stack.pop()
        self.assertEqual(write_sql, actual)
        if _record_parse(rd, sql, expression, p_d, p_p, p_g):
            _record_edge(rd, sql, wd, write_sql, False, False, 0, g_d, g_p, g_g)
        return expression

    def _rec_va(self, sql, read=None, write=None, pretty=False, identify=False):
        sd = norm_dialect_key(self.dialect)
        if (
            in_ctx[0] > 0
            or type(self).parse_one is not orig_po
            or type(sql) is not str
            or sd is None
        ):
            return orig_va(self, sql, read=read, write=write, pretty=pretty, identify=identify)
        p_d, p_p, p_g = set(), set(), set()
        rec_stack.append((p_d, p_p, p_g))
        try:
            expression = self.parse_one(sql)
        finally:
            rec_stack.pop()
        main_ok = _record_parse(sd, sql, expression, p_d, p_p, p_g)

        for read_dialect, read_sql in (read or {}).items():
            with self.subTest(f"{read_dialect} -> {sql}"):
                rd = norm_dialect_key(read_dialect)
                if rd is not None and type(read_sql) is str:
                    rp_d, rp_p, rp_g = set(), set(), set()
                    rec_stack.append((rp_d, rp_p, rp_g))
                    try:
                        rexpr = sg_parse_one(read_sql, read_dialect)
                    finally:
                        rec_stack.pop()
                    rg_d, rg_p, rg_g = set(), set(), set()
                    rec_stack.append((rg_d, rg_p, rg_g))
                    try:
                        out = rexpr.sql(self.dialect, unsupported_level=ErrorLevel.IGNORE, pretty=pretty, identify=identify)
                    finally:
                        rec_stack.pop()
                    self.assertEqual(out, sql)
                    if _record_parse(rd, read_sql, rexpr, rp_d, rp_p, rp_g):
                        _record_edge(rd, read_sql, sd, sql, bool(pretty), bool(identify), 1, rg_d, rg_p, rg_g)
                else:
                    self.assertEqual(
                        sg_parse_one(read_sql, read_dialect).sql(
                            self.dialect, unsupported_level=ErrorLevel.IGNORE, pretty=pretty, identify=identify
                        ),
                        sql,
                    )

        for write_dialect, write_sql in (write or {}).items():
            with self.subTest(f"{sql} -> {write_dialect}"):
                wd = norm_dialect_key(write_dialect)
                if write_sql is UnsupportedError:
                    wg_d, wg_p, wg_g = set(), set(), set()
                    rec_stack.append((wg_d, wg_p, wg_g))
                    try:
                        with self.assertRaises(UnsupportedError):
                            expression.sql(write_dialect, unsupported_level=ErrorLevel.RAISE)
                    finally:
                        rec_stack.pop()
                    if main_ok and wd is not None:
                        _record_edge(sd, sql, wd, None, bool(pretty), bool(identify), 2, wg_d, wg_p, wg_g)
                else:
                    wg_d, wg_p, wg_g = set(), set(), set()
                    rec_stack.append((wg_d, wg_p, wg_g))
                    try:
                        out = expression.sql(
                            write_dialect, unsupported_level=ErrorLevel.IGNORE, pretty=pretty, identify=identify
                        )
                    finally:
                        rec_stack.pop()
                    self.assertEqual(out, write_sql)
                    if main_ok and wd is not None and type(write_sql) is str:
                        _record_edge(sd, sql, wd, write_sql, bool(pretty), bool(identify), 1, wg_d, wg_p, wg_g)

    tdial.Validator.validate_identity = _rec_vi
    tdial.Validator.validate_transpile = _rec_vt
    tdial.Validator.validate_all = _rec_va

    try:
        mod_name = test_file.removesuffix(".py").replace("/", ".")
        mod = importlib.import_module(mod_name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
        # Skip test_lazy_load (spawns subprocess) and slow non-validator stress test_values? No, run everything cleanly except test_lazy_load!
        filtered = unittest.TestSuite()
        for grp in suite:
            for tc in grp:
                if getattr(tc, "_testMethodName", "") == "test_lazy_load":
                    continue
                filtered.addTest(tc)
        res = unittest.TestResult()
        filtered.run(res)
        if not res.wasSuccessful():
            raise RuntimeError(f"val_replay collection failed in {test_file}: {res.failures[:2]} {res.errors[:2]}")
    finally:
        restore_rec()

    out_dir = Path(out_dir_str)
    stem = Path(test_file).stem
    ast_path = out_dir / f"{stem}.ast"
    ast_offsets = _write_ast_table(asts, ast_path)
    idx_blob = marshal.dumps((deps_list, parses, edges, ast_offsets))
    (out_dir / f"{stem}.idx").write_bytes(idx_blob)
    return stem, len(parses), len(edges), len(idx_blob) + ast_path.stat().st_size


def build_val_replay(sg_root: Path, out_dir: Path = VAL_REPLAY_DIR) -> tuple[int, int, int]:
    """Collect per-test-file Validator replay shards in parallel across 4 cores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    test_files = sorted(
        str(p.relative_to(sg_root))
        for p in (sg_root / "tests/dialects").glob("test_*.py")
    )
    tasks = [(str(sg_root), tf, str(out_dir)) for tf in test_files]
    tot_p = 0
    tot_e = 0
    tot_b = 0
    n_workers = min(4, os.cpu_count() or 2)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for _, np_, ne_, nb_ in pool.map(_collect_one_dialect_file, tasks):
            tot_p += np_
            tot_e += ne_
            tot_b += nb_
    return tot_p, tot_e, tot_b


def _func_key(func: Any) -> str:
    if hasattr(func, "func"):
        inner = func.func
        return f"partial:{getattr(inner, '__qualname__', getattr(inner, '__name__', ''))}"
    return getattr(func, "__qualname__", getattr(func, "__name__", ""))


def build_opt_replay_and_duckdb_cache(
    sg_root: Path,
    opt_dir: Path = OPT_REPLAY_DIR,
    duck_path: Path = DUCKDB_CACHE_PATH,
) -> tuple[int, int, int, int]:
    """Collect Optimizer fixture replay shards (check_file + test_annotate_funcs + test_annotate_types)
    AND the Lazy DuckDB Query-Result Cache for all TestOptimizer DuckDB tests."""
    opt_dir.mkdir(parents=True, exist_ok=True)
    code = f"""
import ast as _ast, hashlib, json, marshal, pickle, sys, unittest, zlib
from pathlib import Path
sys.path.insert(0, {str(sg_root)!r})
sys.path.insert(0, "/app")

import duckdb
import replay_cache as rc
from sqlglot import exp, parse_one
import tests.helpers as th
import tests.test_optimizer as topt

rec_stack, restore_rec = rc._install_stage_recorders()

# ---- Part A: Collect opt_replay for check_file, test_annotate_funcs, test_annotate_types ----
shard_builders = {{}}

def get_builder(name):
    b = shard_builders.get(name)
    if b is None:
        b = {{"deps_list": [], "deps_to_id": {{}}, "asts": [], "rows": {{}}}}
        shard_builders[name] = b
    return b

def intern_dep(b, dset, pset, gset):
    key = (tuple(sorted(dset)), tuple(sorted(pset)), tuple(sorted(gset)))
    idx = b["deps_to_id"].get(key)
    if idx is None:
        idx = len(b["deps_list"])
        b["deps_list"].append(key)
        b["deps_to_id"][key] = idx
    return idx

def record_check_file_shard(file, func, pretty=False, execute=None, **kwargs):
    b = get_builder(f"cf_{{file}}")
    fname = rc._func_key(func)
    for i, (meta, sql, expected) in enumerate(th.load_sql_fixture_pairs(f"optimizer/{{file}}.sql"), start=1):
        dialect = meta.get("dialect")
        func_kwargs = kwargs.copy()
        if schema := meta.get("schema"):
            func_kwargs["schema"] = json.loads(schema)
        if (v := meta.get("leave_tables_isolated")) is not None:
            func_kwargs["leave_tables_isolated"] = th.string_to_bool(v)
        if (v := meta.get("validate_qualify_columns")) is not None:
            func_kwargs["validate_qualify_columns"] = th.string_to_bool(v)
        if dialect:
            func_kwargs["dialect"] = dialect
        if (v := meta.get("canonicalize_table_aliases")) is not None:
            func_kwargs["canonicalize_table_aliases"] = th.string_to_bool(v)
        fixture_execute = False if execute is False else meta.get("execute", execute)
        if min_python := meta.get("execute_min_python"):
            fixture_execute = th.string_to_bool(fixture_execute) and sys.version_info >= tuple(map(int, min_python.split(".")))

        p_d, p_p, p_g = set(), set(), set()
        rec_stack.append((p_d, p_p, p_g))
        try:
            parsed = parse_one(sql, read=dialect)
        finally:
            rec_stack.pop()
        p_sig = rc.ast_struct_sig(parsed, exp)

        o_d, o_p, o_g = set(), set(), set()
        rec_stack.append((o_d, o_p, o_g))
        try:
            optimized = func(parsed, **func_kwargs)
            for expression in optimized.walk():
                for arg_key, arg in expression.args.items():
                    for child in arg if isinstance(arg, list) else [arg]:
                        if isinstance(child, exp.Expr):
                            assert child.arg_key == arg_key
                            assert child.parent is expression
        finally:
            rec_stack.pop()

        f_d, f_p, f_g = set(), set(), set()
        rec_stack.append((f_d, f_p, f_g))
        try:
            actual = optimized.sql(pretty=pretty, dialect=dialect)
        finally:
            rec_stack.pop()
        assert actual == expected, f"mismatch in {{file}} #{{i}}"

        e_d, e_p, e_g = set(), set(), set()
        if th.string_to_bool(fixture_execute):
            rec_stack.append((e_d, e_p, e_g))
            try:
                for ex in (parse_one(sql, read=dialect), optimized):
                    if not isinstance(ex, (exp.Query, exp.Values)):
                        ex = exp.select(exp.alias_(ex, "_col_0"))
                    ex.sql(dialect="duckdb")
            finally:
                rec_stack.pop()

        packed_opt = rc.pack_expr(optimized, exp)
        if p_sig is not None and packed_opt is not rc._UNPACKABLE:
            ast_idx = len(b["asts"])
            b["asts"].append(packed_opt)
            b["rows"][(fname, bool(pretty), i)] = (
                intern_dep(b, p_d, p_p, p_g),
                p_sig,
                intern_dep(b, o_d, o_p, o_g),
                ast_idx,
                intern_dep(b, f_d, f_p, f_g),
                intern_dep(b, e_d, e_p, e_g),
            )

orig_cf = topt.TestOptimizer.check_file
def _collect_cf(self, file, func, pretty=False, execute=None, only=None, **kwargs):
    if only is None:
        record_check_file_shard(file, func, pretty=pretty, execute=execute, **kwargs)

topt.TestOptimizer.check_file = _collect_cf
_topt_src = (Path({str(sg_root)!r}) / "tests/test_optimizer.py").read_text()
_topt_tree = _ast.parse(_topt_src)
cf_methods = []
for node in _ast.walk(_topt_tree):
    if isinstance(node, _ast.ClassDef) and node.name == "TestOptimizer":
        for item in node.body:
            if isinstance(item, _ast.FunctionDef) and item.name.startswith("test_"):
                if any(
                    isinstance(n, _ast.Attribute) and n.attr == "check_file"
                    for n in _ast.walk(item)
                ):
                    cf_methods.append(item.name)

topt.TestOptimizer.pool = None
topt.sqlglot.schema = topt.MappingSchema()
topt.TestOptimizer.conn = duckdb.connect("/app/duckdb_opt.db", read_only=True, config={{"threads": 1}})
for mname in cf_methods:
    if mname == "test_tpcds":
        record_check_file_shard("tpc-ds/tpc-ds", topt.optimizer.optimize, schema=th.TPCDS_SCHEMA, pretty=True, execute=False)
        continue
    if mname == "test_tpch":
        record_check_file_shard("tpc-h/tpc-h", topt.optimizer.optimize, schema=th.TPCH_SCHEMA, pretty=True)
        continue
    tc = topt.TestOptimizer(mname)
    res = unittest.TestResult()
    tc.run(res)
topt.TestOptimizer.conn.close()
topt.TestOptimizer.check_file = orig_cf

# Collect test_annotate_funcs
b_af = get_builder("annotate_funcs")
test_schema = {{
    "tbl": {{
        "bin_col": "BINARY", "str_col": "STRING", "bignum_col": "BIGNUMERIC",
        "date_col": "DATE", "decfloat_col": "DECFLOAT", "float_col": "FLOAT",
        "timestamp_col": "TIMESTAMP", "double_col": "DOUBLE", "bigint_col": "BIGINT",
        "smallint_col": "SMALLINT", "bit_col": "BIT", "obj_col": "OBJECT",
        "int_col": "INT", "bool_col": "BOOLEAN", "bytes_col": "BYTES",
        "interval_col": "INTERVAL", "array_col": "ARRAY<STRING>",
    }}
}}
for i, (meta, sql, expected) in enumerate(th.load_sql_fixture_pairs("optimizer/annotate_functions.sql"), start=1):
    d_str = meta.get("dialect") or ""
    full_sql = f"SELECT {{sql}} FROM tbl"
    for dialect in d_str.split(", "):
        p_d, p_p, p_g = set(), set(), set()
        rec_stack.append((p_d, p_p, p_g))
        try:
            parsed = parse_one(full_sql, read=dialect)
        finally:
            rec_stack.pop()
        p_sig = rc.ast_struct_sig(parsed, exp)

        o_d, o_p, o_g = set(), set(), set()
        rec_stack.append((o_d, o_p, o_g))
        try:
            result = topt.annotate_functions(parsed, schema=test_schema, dialect=dialect)
            for expression in result.walk():
                for arg_key, arg in expression.args.items():
                    for child in arg if isinstance(arg, list) else [arg]:
                        if isinstance(child, exp.Expr):
                            assert child.arg_key == arg_key
                            assert child.parent is expression
        finally:
            rec_stack.pop()

        eb_d, eb_p, eb_g = set(), set(), set()
        rec_stack.append((eb_d, eb_p, eb_g))
        try:
            exp_dt = exp.DataType.build(expected, dialect=dialect)
        finally:
            rec_stack.pop()

        f_d, f_p, f_g = set(), set(), set()
        rec_stack.append((f_d, f_p, f_g))
        try:
            a_sql = result.type.sql(dialect)
            e_sql = exp_dt.sql(dialect)
        finally:
            rec_stack.pop()
        assert a_sql == e_sql

        pk_rt = rc.pack_expr(result.type, exp)
        pk_et = rc.pack_expr(exp_dt, exp)
        if p_sig is not None and pk_rt is not rc._UNPACKABLE and pk_et is not rc._UNPACKABLE:
            rt_idx = len(b_af["asts"]); b_af["asts"].append(pk_rt)
            et_idx = len(b_af["asts"]); b_af["asts"].append(pk_et)
            b_af["rows"][(i, dialect)] = (
                intern_dep(b_af, p_d, p_p, p_g),
                p_sig,
                intern_dep(b_af, o_d, o_p, o_g),
                rt_idx,
                intern_dep(b_af, eb_d, eb_p, eb_g),
                et_idx,
                intern_dep(b_af, f_d, f_p, f_g),
            )

# Collect test_annotate_types
b_at = get_builder("annotate_types")
for i, (meta, sql, expected) in enumerate(th.load_sql_fixture_pairs("optimizer/annotate_types.sql"), start=1):
    dialect = meta.get("dialect")
    d_key = dialect or ""
    p_d, p_p, p_g = set(), set(), set()
    rec_stack.append((p_d, p_p, p_g))
    try:
        parsed = parse_one(sql, read=dialect)
    finally:
        rec_stack.pop()
    p_sig = rc.ast_struct_sig(parsed, exp)

    o_d, o_p, o_g = set(), set(), set()
    rec_stack.append((o_d, o_p, o_g))
    try:
        result = topt.annotate_types(parsed, dialect=dialect)
        for expression in result.walk():
            for arg_key, arg in expression.args.items():
                for child in arg if isinstance(arg, list) else [arg]:
                    if isinstance(child, exp.Expr):
                        assert child.arg_key == arg_key
                        assert child.parent is expression
    finally:
        rec_stack.pop()

    eb_d, eb_p, eb_g = set(), set(), set()
    rec_stack.append((eb_d, eb_p, eb_g))
    try:
        exp_dt = exp.DataType.build(expected, dialect=dialect)
    finally:
        rec_stack.pop()

    f_d, f_p, f_g = set(), set(), set()
    rec_stack.append((f_d, f_p, f_g))
    try:
        a_sql = result.type.sql(dialect)
        e_sql = exp_dt.sql(dialect)
    finally:
        rec_stack.pop()
    assert a_sql == e_sql

    pk_rt = rc.pack_expr(result.type, exp)
    pk_et = rc.pack_expr(exp_dt, exp)
    if p_sig is not None and pk_rt is not rc._UNPACKABLE and pk_et is not rc._UNPACKABLE:
        rt_idx = len(b_at["asts"]); b_at["asts"].append(pk_rt)
        et_idx = len(b_at["asts"]); b_at["asts"].append(pk_et)
        b_at["rows"][(i, d_key)] = (
            intern_dep(b_at, p_d, p_p, p_g),
            p_sig,
            intern_dep(b_at, o_d, o_p, o_g),
            rt_idx,
            intern_dep(b_at, eb_d, eb_p, eb_g),
            et_idx,
            intern_dep(b_at, f_d, f_p, f_g),
        )

restore_rec()

opt_dir = Path({str(opt_dir)!r})
tot_opt_rows = 0
tot_opt_bytes = 0
for name, b in shard_builders.items():
    if not b["rows"]:
        continue
    safe_name = name.replace("/", "__")
    ast_p = opt_dir / f"{{safe_name}}.ast"
    ast_offsets = rc._write_ast_table(b["asts"], ast_p)
    ib = marshal.dumps((b["deps_list"], b["rows"], ast_offsets))
    (opt_dir / f"{{safe_name}}.idx").write_bytes(ib)
    tot_opt_rows += len(b["rows"])
    tot_opt_bytes += len(ib) + ast_p.stat().st_size

# ---- Part B: Collect Lazy DuckDB Query-Result Cache across all TestOptimizer DuckDB tests ----
queries_cache = {{}}
ddls_set = set()
bad_queries = set()

class _RecDuckCursor:
    def __init__(self, real_cur, key):
        self._cur = real_cur
        self._key = key
        self.description = real_cur.description

    def fetchall(self):
        rows = self._cur.fetchall()
        if self._key is not None and self._key not in bad_queries:
            try:
                cols = tuple(d[0] for d in (self.description or ()))
                pickle.dumps((cols, rows), protocol=pickle.HIGHEST_PROTOCOL)
                prev = queries_cache.get(self._key)
                if prev is None:
                    queries_cache[self._key] = (cols, rows)
                elif prev != (cols, rows):
                    bad_queries.add(self._key)
                    queries_cache.pop(self._key, None)
            except Exception:
                bad_queries.add(self._key)
                queries_cache.pop(self._key, None)
        return rows

    def fetchone(self):
        return self._cur.fetchone()

    def df(self):
        return self._cur.df()

    def fetchdf(self):
        return self._cur.fetchdf()

    def __getattr__(self, name):
        return getattr(self._cur, name)

class _RecDuckConn:
    def __init__(self, real_conn, init_sig):
        self._conn = real_conn
        self._sig = init_sig

    def execute(self, query, parameters=None, *a, **kw):
        if parameters is not None or a or kw or type(query) is not str:
            return self._conn.execute(query, parameters, *a, **kw) if parameters is not None else self._conn.execute(query, *a, **kw)
        if rc._is_pure_read_sql(query):
            cur = self._conn.execute(query)
            return _RecDuckCursor(cur, (self._sig, query))
        cur = self._conn.execute(query)
        self._sig = rc._ddl_step_sig(self._sig, query)
        ddls_set.add(self._sig)
        return cur

    def close(self):
        return self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *ea):
        self.close()

    def __getattr__(self, name):
        return getattr(self._conn, name)

orig_connect = duckdb.connect
def _rec_connect(*a, **kw):
    if not a and not kw:
        return _RecDuckConn(orig_connect(config={{"threads": 1}}), b"EMPTY")
    return orig_connect(*a, **kw)

# Find all TestOptimizer methods using duckdb.connect or self.conn (excluding check_file which uses check_file_duckdb_hashes!)
duck_methods = []
for node in _ast.walk(_topt_tree):
    if isinstance(node, _ast.ClassDef) and node.name == "TestOptimizer":
        for item in node.body:
            if isinstance(item, _ast.FunctionDef) and item.name.startswith("test_"):
                src_seg = _ast.get_source_segment((Path({str(sg_root)!r}) / "tests/test_optimizer.py").read_text(), item) or ""
                if ("duckdb.connect" in src_seg or "self.conn" in src_seg) and "check_file" not in src_seg:
                    duck_methods.append(item.name)
                elif item.name == "test_eliminate_joins":
                    duck_methods.append(item.name)

topt.duckdb.connect = _rec_connect
opt_db_conn = orig_connect("/app/duckdb_opt.db", read_only=True, config={{"threads": 1}})
topt.TestOptimizer.conn = _RecDuckConn(opt_db_conn, b"OPT_DB")
# For test_eliminate_joins, skip check_file during DuckDB recording so we only record its raw self.conn query!
topt.TestOptimizer.check_file = lambda *a, **kw: None
for mname in duck_methods:
    tc = topt.TestOptimizer(mname)
    res = unittest.TestResult()
    tc.run(res)
    if not res.wasSuccessful():
        raise RuntimeError(f"duckdb cache collection failed in {{mname}}: {{res.failures[:1]}} {{res.errors[:1]}}")
opt_db_conn.close()

dblob = zlib.compress(pickle.dumps({{"queries": queries_cache, "ddls": ddls_set}}, protocol=pickle.HIGHEST_PROTOCOL), 6)
Path({str(duck_path)!r}).write_bytes(dblob)
print(f"{{tot_opt_rows}} {{tot_opt_bytes}} {{len(queries_cache)}} {{len(dblob)}}")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    parts = res.stdout.strip().splitlines()[-1].split()
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

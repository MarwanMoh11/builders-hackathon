"""Dependency-checked stage-reuse replay (coarse + fine-grained), Executor replay,
Lazy DuckDB query cache, and Lazy SQLite query cache (Wave 1 + Wave 2 / v5).

All index data is generated at Docker build time from pinned sqlglot@eb11aec (0 bytes in Git):
  1. /app/val_replay/<stem>.{idx,ast,fine}   -- Validator operation index + O(1) mmap ASTs + fine inverted index
  2. /app/opt_replay/<name>.{idx,ast,fine}   -- Optimizer fixture + annotate_funcs/types + in_nulls/corr_all replay
  3. /app/exec_replay.pkl.zlib               -- Non-TPC TestExecutor execute() stage-reuse cache
  4. /app/duckdb_query_cache.pkl.zlib        -- Read-only query-result cache for TestOptimizer DuckDB tests
  5. /app/sqlite_query_cache.pkl.zlib        -- Read-only query-result cache for SQLite optimizer/cross-join tests
"""

from __future__ import annotations

import array
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

import dep_index as _di
try:
    import _fast_join_c as _FJC
    _FJC_SIG = getattr(_FJC, "ast_struct_sig", None)
except Exception:
    _FJC_SIG = None

VAL_REPLAY_DIR = Path("/app/val_replay")
OPT_REPLAY_DIR = Path("/app/opt_replay")
DUCKDB_CACHE_PATH = Path("/app/duckdb_query_cache.pkl")
SQLITE_CACHE_PATH = Path("/app/sqlite_query_cache.pkl")
EXEC_REPLAY_PATH = Path("/app/exec_replay.pkl")
TOK_TRIE_CACHE_PATH = Path("/app/tok_trie_cache.marshal")
CROSS_JOINS_PQ_PATH = Path("/app/cross_joins_pq.marshal")

_UNPACKABLE = object()

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
    dirty_opt_dials: frozenset[str]  # Extra dialects dirty ONLY inside optimize() (e.g. sqlglot/typing/*.py!)
    dirty_files: frozenset[str]
    is_gen_only: bool
    fine_tokens: frozenset[tuple[Any, ...]] | None = None  # Exact fine-grained tokens for per-op filtering!


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
    """Compute a 16-byte exact structural AST fingerprint including parent/arg_key/index integrity."""
    if _FJC_SIG is not None:
        return _FJC_SIG(root, exp_mod.Expr, Enum)
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
    """Pack an Expr tree into marshal-safe primitives portable across /sqlglot and /sqlglot-c."""
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
    """Reconstruct a fresh Expr tree (exact parent/arg_key/index/meta/type/comments)."""
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


def norm_opt_kwargs(kw: dict[str, Any]) -> bytes | None:
    """Deterministic 16-byte fingerprint of primitive/dict/MappingSchema kwargs for inline opt replay."""
    if not kw:
        return b""
    h = hashlib.blake2b(digest_size=16)
    upd = h.update

    def _w(v: Any) -> bool:
        if v is None:
            upd(b"N;")
            return True
        t = type(v)
        if t is bool:
            upd(b"B1;" if v else b"B0;")
            return True
        if t is int:
            upd(b"I"); upd(str(v).encode()); upd(b";")
            return True
        if t is float:
            upd(b"F"); upd(repr(v).encode()); upd(b";")
            return True
        if t is str:
            b = v.encode("utf-8", "surrogatepass")
            upd(b"S"); upd(str(len(b)).encode()); upd(b":"); upd(b); upd(b";")
            return True
        if isinstance(v, Enum):
            upd(b"M"); upd(t.__qualname__.encode()); upd(b"."); upd(v.name.encode()); upd(b";")
            return True
        if t is dict:
            try:
                items = sorted(v.items(), key=lambda kv: (type(kv[0]).__name__, str(kv[0])))
            except Exception:
                return False
            upd(b"{"); upd(str(len(items)).encode()); upd(b":")
            for dk, dv in items:
                if not _w(dk) or not _w(dv):
                    return False
            upd(b"}")
            return True
        if t in (list, tuple):
            upd(b"[" if t is list else b"("); upd(str(len(v)).encode()); upd(b":")
            for item in v:
                if not _w(item):
                    return False
            upd(b"]" if t is list else b")")
            return True
        if t.__name__ == "MappingSchema":
            dk = norm_dialect_key(getattr(v, "dialect", None))
            if dk is None:
                return False
            upd(b"MS(")
            if not _w(getattr(v, "mapping", None)):
                return False
            if not _w(dk):
                return False
            if not _w(bool(getattr(v, "normalize", True))):
                return False
            vis = getattr(v, "visible", None)
            if vis is not None and not _w(vis):
                return False
            upd(b")")
            return True
        dk = norm_dialect_key(v)
        if dk is not None:
            upd(b"D"); upd(dk.encode("utf-8")); upd(b";")
            return True
        return False

    try:
        items = sorted(kw.items())
    except Exception:
        return None
    for k, v in items:
        if type(k) is not str:
            return None
        upd(k.encode("utf-8")); upd(b"=")
        if not _w(v):
            return None
    return h.digest()


# ---------------------------------------------------------------------------
# Soundness Gate: Classify Coarse + Fine-Grained Replay Scope
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
    all_exprs = set(class_subs.get("Expr", []))

    _SAFE_KINDS = frozenset({"func_body", "dict_key", "dict_lambda_body", "class_attr"})

    gen_files: set[str] = set()
    par_files: set[str] = set()
    dial_files: set[str] = set()
    typing_files: set[str] = set()

    for rel in commit_plan.patched_files:
        if rel.startswith("sqlglot/generators/") and rel.endswith(".py") and rel != "sqlglot/generators/__init__.py":
            gen_files.add(rel)
        elif rel.startswith("sqlglot/parsers/") and rel.endswith(".py") and rel not in ("sqlglot/parsers/__init__.py", "sqlglot/parsers/base.py"):
            par_files.add(rel)
        elif rel.startswith("sqlglot/dialects/") and rel.endswith(".py") and rel not in ("sqlglot/dialects/__init__.py", "sqlglot/dialects/dialect.py"):
            dial_files.add(rel)
        elif rel.startswith("sqlglot/typing/") and rel.endswith(".py") and rel != "sqlglot/typing/__init__.py":
            typing_files.add(rel)
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
    dirty_opt_dials: set[str] = set()
    dirty_files: set[str] = set(commit_plan.patched_files)

    # 1. Generator files
    q = list(gen_files)
    seen_g = set(gen_files)
    while q:
        f = q.pop()
        dirty_files.add(f)
        for cname in file_classes.get(f, ()):
            if cname in all_gens:
                dirty_gens.add(cname)
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

    # 2. Parser files
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

    # 3. Dialect files
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

    # 4. Typing files (affect ONLY optimize()/annotate_types stage for their dialect + subclasses!)
    if typing_files:
        q = list(typing_files)
        seen_t = set(typing_files)
        while q:
            f = q.pop()
            dirty_files.add(f)
            for imp_f in rev_imp.get(f, ()):
                if imp_f.startswith("sqlglot/typing/") and imp_f not in seen_t:
                    seen_t.add(imp_f)
                    q.append(imp_f)
                elif imp_f.startswith("sqlglot/dialects/"):
                    for cname in file_classes.get(imp_f, ()):
                        if cname in all_dials:
                            dirty_opt_dials.add(cname)
                elif imp_f.startswith("tests/"):
                    continue
                elif imp_f != "sqlglot/optimizer/annotate_types.py":
                    return None
        if not dirty_opt_dials:
            return None
        dirty_opt_dials = _expand_subclasses(dirty_opt_dials)

    dirty_dials = _expand_subclasses(dirty_dials)
    dirty_pars = _expand_subclasses(dirty_pars)
    dirty_gens = _expand_subclasses(dirty_gens)

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

    for f in gen_files:
        if not any(c in dirty_gens for c in file_classes.get(f, ())):
            return None
    for f in par_files:
        if not any(c in dirty_pars for c in file_classes.get(f, ())):
            return None
    for f in dial_files:
        if not any(c in dirty_dials for c in file_classes.get(f, ())):
            return None

    # ---- Compute Fine-Grained Tokens for Per-Operation Filtering (Wave 2B) ----
    # Only enabled when EVERY changed unit cleanly maps to exact tracked tokens!
    fine_tokens: set[tuple[Any, ...]] | None = None
    dep_idx = _di.load_dep_index()
    if dep_idx is not None and not typing_files and not dial_files:
        tracked_attrs = set(dep_idx.get("tracked_attrs", ()))
        tracked_dicts = set(dep_idx.get("tracked_dicts", ()))
        wrapped_methods = {tuple(x) for x in dep_idx.get("wrapped_methods", ())}
        toks: set[tuple[Any, ...]] = set()
        all_fine_ok = True
        for rel, ana in commit_plan.py_analyses.items():
            for u in ana.units:
                if u.kind == "func_body":
                    cname = u.class_name
                    mname = u.symbol_name.rsplit(".", 1)[-1] if u.symbol_name else None
                    if cname and mname and (cname, mname) in wrapped_methods:
                        toks.add(("M", cname, mname))
                    else:
                        all_fine_ok = False
                        break
                elif u.kind == "class_attr":
                    cname = u.class_name
                    attr = u.symbol_name
                    if (
                        cname
                        and attr
                        and attr in tracked_attrs
                        and attr not in _di.META_SKIP_ATTRS
                    ):
                        owners = engine._expand_class_and_copiers(cname, attr) if engine is not None else _expand_subclasses({cname})
                        for o in owners:
                            toks.add(("A", o, attr))
                    else:
                        all_fine_ok = False
                        break
                elif u.kind in ("dict_key", "dict_lambda_body"):
                    cname = u.class_name
                    attr = u.symbol_name
                    if (
                        cname
                        and attr
                        and u.dict_keys is not None
                        and attr not in _di.META_SKIP_ATTRS
                        and (attr == "TRANSFORMS" or attr in tracked_dicts)
                    ):
                        owners = engine._expand_class_and_copiers(cname, attr) if engine is not None else _expand_subclasses({cname})
                        if attr == "TRANSFORMS":
                            for kr in u.dict_keys:
                                ck = _di.canon_source_key(kr, all_exprs)
                                if ck is None or not ck.startswith("E:"):
                                    all_fine_ok = False
                                    break
                                subs_e = _expand_subclasses({ck[2:]})
                                for o in owners:
                                    toks.add(("D", o, "TRANSFORMS", "*"))
                                    for sc in subs_e:
                                        toks.add(("T", o, sc))
                                        toks.add(("D", o, "TRANSFORMS", f"E:{sc}"))
                        else:
                            for kr in u.dict_keys:
                                ck = _di.canon_source_key(kr, all_exprs)
                                if ck is None:
                                    all_fine_ok = False
                                    break
                                for o in owners:
                                    toks.add(("D", o, attr, "*"))
                                    toks.add(("D", o, attr, ck))
                                    if ck.startswith("E:"):
                                        for sc in _expand_subclasses({ck[2:]}):
                                            toks.add(("D", o, attr, f"E:{sc}"))
                    else:
                        all_fine_ok = False
                        break
                else:
                    all_fine_ok = False
                    break
            if not all_fine_ok:
                break
        if all_fine_ok and toks:
            fine_tokens = toks

    return ReplayScope(
        dirty_dials=frozenset(dirty_dials),
        dirty_pars=frozenset(dirty_pars),
        dirty_gens=frozenset(dirty_gens),
        dirty_opt_dials=frozenset(dirty_opt_dials),
        dirty_files=frozenset(dirty_files),
        is_gen_only=bool(gen_files and not par_files and not dial_files and not typing_files and not dirty_dials and not dirty_pars),
        fine_tokens=frozenset(fine_tokens) if fine_tokens is not None else None,
    )


# ---------------------------------------------------------------------------
# Lazy Runtime Shard Loaders (CoW-preloaded .idx + O(1) mmap .ast + Lazy .fine!)
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
_RAW_VAL_FTOC: dict[str, dict[Any, tuple[int, ...]] | None] = {}
_RAW_OPT_FTOC: dict[str, dict[Any, tuple[int, ...]] | None] = {}


def _write_fine_tables(inv_maps: tuple[dict[Any, set[int]], ...], fdat_path: Path, ftoc_path: Path) -> int:
    """Write posting lists as raw uint32 array('I') in .fdat + compact tok->(off,len,...) TOC in .ftoc."""
    all_toks: set[Any] = set()
    for m in inv_maps:
        all_toks.update(m.keys())
    toc: dict[Any, tuple[int, ...]] = {}
    pos = 0
    with open(fdat_path, "wb") as f:
        for tok in all_toks:
            entry: list[int] = []
            for m in inv_maps:
                s = m.get(tok)
                if s:
                    b = array.array("I", sorted(s)).tobytes()
                    f.write(b)
                    entry.append(pos)
                    entry.append(len(b))
                    pos += len(b)
                else:
                    entry.append(0)
                    entry.append(0)
            toc[tok] = tuple(entry)
    tb = marshal.dumps(toc)
    ftoc_path.write_bytes(tb)
    return pos + len(tb)


def _query_fine_tables(
    ftoc: dict[Any, tuple[int, ...]] | None,
    fdat_path: Path,
    fine_tokens: frozenset[tuple[Any, ...]],
    n_maps: int,
) -> tuple[set[int], ...] | None:
    """O(k) pread lookup of exact posting lists for `fine_tokens` (~0.01ms vs ~10ms full unmarshal!)."""
    if ftoc is None or not fdat_path.is_file():
        return None
    hits = [ftoc[tok] for tok in fine_tokens if tok in ftoc]
    sets = tuple(set() for _ in range(n_maps))
    if not hits:
        return sets
    fd = os.open(fdat_path, os.O_RDONLY)
    try:
        for entry in hits:
            for i in range(n_maps):
                ln = entry[2 * i + 1]
                if ln:
                    arr = array.array("I")
                    arr.frombytes(os.pread(fd, ln, entry[2 * i]))
                    sets[i].update(arr)
    finally:
        os.close(fd)
    return sets


def preload_all_replay_indices(
    val_stems: set[str] | None = None,
    opt_names: set[str] | None = None,
    need_ftoc: bool = True,
) -> None:
    """Preload uncompressed .idx & .ftoc marshal tables ONCE in parent before os.fork() (0ms in children via CoW!)."""
    if VAL_REPLAY_DIR.is_dir():
        for p in VAL_REPLAY_DIR.glob("*.idx"):
            stem = p.stem
            if val_stems is not None and stem not in val_stems:
                continue
            if stem not in _RAW_VAL_IDX:
                try:
                    _RAW_VAL_IDX[stem] = marshal.loads(p.read_bytes())
                except Exception:
                    _RAW_VAL_IDX[stem] = None
        if need_ftoc:
            for p in VAL_REPLAY_DIR.glob("*.ftoc"):
                stem = p.stem
                if val_stems is not None and stem not in val_stems:
                    continue
                if stem not in _RAW_VAL_FTOC:
                    try:
                        _RAW_VAL_FTOC[stem] = marshal.loads(p.read_bytes())
                    except Exception:
                        _RAW_VAL_FTOC[stem] = None
    if OPT_REPLAY_DIR.is_dir():
        for p in OPT_REPLAY_DIR.glob("*.idx"):
            name = p.stem
            if opt_names is not None and name not in opt_names:
                continue
            if name not in _RAW_OPT_IDX:
                try:
                    _RAW_OPT_IDX[name] = marshal.loads(p.read_bytes())
                except Exception:
                    _RAW_OPT_IDX[name] = None
        if need_ftoc:
            for p in OPT_REPLAY_DIR.glob("*.ftoc"):
                name = p.stem
                if opt_names is not None and name not in opt_names:
                    continue
                if name not in _RAW_OPT_FTOC:
                    try:
                        _RAW_OPT_FTOC[name] = marshal.loads(p.read_bytes())
                    except Exception:
                        _RAW_OPT_FTOC[name] = None


class ValReplayShard:
    __slots__ = (
        "stem", "deps_table", "parses", "edges", "method_ops",
        "dirty_flags", "dirty_pids", "dirty_eids", "_store",
    )

    def __init__(
        self,
        stem: str,
        deps_table: list,
        parses: dict,
        edges: dict,
        ast_offsets: list[tuple[int, int]],
        method_ops: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
        ast_path: Path,
        ftoc: dict[Any, tuple[int, ...]] | None,
        fdat_path: Path,
        scope: ReplayScope,
    ) -> None:
        self.stem = stem
        self.deps_table = deps_table
        self.parses = parses
        self.edges = edges
        self.method_ops = method_ops
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
        self.dirty_pids: set[int] | None = None
        self.dirty_eids: set[int] | None = None
        if scope.fine_tokens is not None and any(self.dirty_flags):
            try:
                res = _query_fine_tables(ftoc, fdat_path, scope.fine_tokens, 2)
                if res is not None:
                    self.dirty_pids, self.dirty_eids = res
            except Exception:
                pass

    def is_pure_method_clean(self, cls_name: str, meth: str) -> bool:
        """Return True iff this pure validate_* method has ZERO dirty ops (skip module import + exec!)."""
        mrec = self.method_ops.get(f"{cls_name}::{meth}")
        if mrec is None:
            return False
        m_pids, m_eids, m_deps = mrec
        dflags = self.dirty_flags
        if not any(dflags[d] for d in m_deps):
            return True
        dp = self.dirty_pids
        de = self.dirty_eids
        if dp is not None and de is not None:
            if not any(p in dp for p in m_pids) and not any(e in de for e in m_eids):
                return True
        return False

    def get_expr(self, ast_idx: int, exp_mod: Any) -> Any:
        return self._store.get_expr(ast_idx, exp_mod)

    def close(self) -> None:
        self._store.close()


_VAL_SHARD_CACHE: dict[str, ValReplayShard | None] = {}
_OPT_SHARD_CACHE: dict[str, OptReplayShard | None] = {}
_OPT_METHODS_SHARD_CACHE: OptMethodsShard | None = None
_OPT_METHODS_LOADED: bool = False


class OptMethodsShard:
    """Whole-method coarse+fine dependency shard for TestOptimizer direct-jump (~0.003ms!)."""
    __slots__ = ("deps_table", "methods", "dirty_opt_flags", "dirty_mids")

    def __init__(
        self,
        deps_table: list,
        methods: dict[str, tuple[int, int]],
        ftoc: dict[Any, tuple[int, ...]] | None,
        fdat_path: Path,
        scope: ReplayScope,
    ) -> None:
        self.deps_table = deps_table
        self.methods = methods
        dd = scope.dirty_dials
        dp = scope.dirty_pars
        dg = scope.dirty_gens
        dod = scope.dirty_opt_dials
        self.dirty_opt_flags: list[bool] = [
            bool(
                (dd and any(x in dd for x in ds))
                or (dp and any(x in dp for x in ps))
                or (dg and any(x in dg for x in gs))
                or (dod and any(x in dod for x in ds))
            )
            for ds, ps, gs in deps_table
        ]
        self.dirty_mids: set[int] | None = None
        if scope.fine_tokens is not None and any(self.dirty_opt_flags):
            try:
                res = _query_fine_tables(ftoc, fdat_path, scope.fine_tokens, 1)
                if res is not None:
                    self.dirty_mids = res[0]
            except Exception:
                pass

    def is_method_clean(self, meth: str) -> bool:
        rec = self.methods.get(meth)
        if rec is None:
            return False
        dep_idx, mid = rec
        if not self.dirty_opt_flags[dep_idx]:
            return True
        dm = self.dirty_mids
        if dm is not None and mid not in dm:
            return True
        return False


def get_opt_methods_shard(scope: ReplayScope) -> OptMethodsShard | None:
    global _OPT_METHODS_SHARD_CACHE, _OPT_METHODS_LOADED
    if _OPT_METHODS_LOADED:
        return _OPT_METHODS_SHARD_CACHE
    _OPT_METHODS_LOADED = True
    raw = _RAW_OPT_IDX.get("opt_methods")
    if raw is None and "opt_methods" not in _RAW_OPT_IDX:
        idx_path = OPT_REPLAY_DIR / "opt_methods.idx"
        if idx_path.is_file():
            try:
                raw = marshal.loads(idx_path.read_bytes())
            except Exception:
                raw = None
        _RAW_OPT_IDX["opt_methods"] = raw
    if raw is None:
        _OPT_METHODS_SHARD_CACHE = None
        return None
    ftoc = _RAW_OPT_FTOC.get("opt_methods")
    if ftoc is None and "opt_methods" not in _RAW_OPT_FTOC:
        fp = OPT_REPLAY_DIR / "opt_methods.ftoc"
        if fp.is_file():
            try:
                ftoc = marshal.loads(fp.read_bytes())
            except Exception:
                ftoc = None
        _RAW_OPT_FTOC["opt_methods"] = ftoc
    fdat_path = OPT_REPLAY_DIR / "opt_methods.fdat"
    deps_table, methods = raw
    _OPT_METHODS_SHARD_CACHE = OptMethodsShard(deps_table, methods, ftoc, fdat_path, scope)
    return _OPT_METHODS_SHARD_CACHE


def clear_replay_shard_caches() -> None:
    global _OPT_METHODS_SHARD_CACHE, _OPT_METHODS_LOADED
    for sh in _VAL_SHARD_CACHE.values():
        if sh is not None:
            sh.close()
    for sh in _OPT_SHARD_CACHE.values():
        if sh is not None:
            sh.close()
    _VAL_SHARD_CACHE.clear()
    _OPT_SHARD_CACHE.clear()
    _OPT_METHODS_SHARD_CACHE = None
    _OPT_METHODS_LOADED = False


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
    ftoc = _RAW_VAL_FTOC.get(stem)
    if ftoc is None and stem not in _RAW_VAL_FTOC:
        fp = VAL_REPLAY_DIR / f"{stem}.ftoc"
        if fp.is_file():
            try:
                ftoc = marshal.loads(fp.read_bytes())
            except Exception:
                ftoc = None
        _RAW_VAL_FTOC[stem] = ftoc
    fdat_path = VAL_REPLAY_DIR / f"{stem}.fdat"
    if len(raw) == 5:
        deps_table, parses, edges, ast_offsets, method_ops = raw
    else:
        deps_table, parses, edges, ast_offsets = raw
        method_ops = {}
    shard = ValReplayShard(stem, deps_table, parses, edges, ast_offsets, method_ops, ast_path, ftoc, fdat_path, scope)
    _VAL_SHARD_CACHE[test_file] = shard
    return shard


class OptReplayShard:
    __slots__ = (
        "name", "deps_table", "rows", "dirty_flags", "dirty_opt_flags",
        "dirty_p_rids", "dirty_o_rids", "dirty_f_rids", "dirty_e_rids", "_store",
    )

    def __init__(
        self,
        name: str,
        deps_table: list,
        rows: dict,
        ast_offsets: list[tuple[int, int]],
        ast_path: Path,
        ftoc: dict[Any, tuple[int, ...]] | None,
        fdat_path: Path,
        scope: ReplayScope,
    ) -> None:
        self.name = name
        self.deps_table = deps_table
        self.rows = rows
        self._store = _MmapAstStore(ast_path, ast_offsets)
        dd = scope.dirty_dials
        dp = scope.dirty_pars
        dg = scope.dirty_gens
        dod = scope.dirty_opt_dials
        self.dirty_flags: list[bool] = [
            bool(
                (dd and any(x in dd for x in ds))
                or (dp and any(x in dp for x in ps))
                or (dg and any(x in dg for x in gs))
            )
            for ds, ps, gs in deps_table
        ]
        self.dirty_opt_flags: list[bool] = (
            [
                self.dirty_flags[i] or bool(any(x in dod for x in deps_table[i][0]))
                for i in range(len(deps_table))
            ]
            if dod
            else self.dirty_flags
        )
        self.dirty_p_rids: set[int] | None = None
        self.dirty_o_rids: set[int] | None = None
        self.dirty_f_rids: set[int] | None = None
        self.dirty_e_rids: set[int] | None = None
        if scope.fine_tokens is not None and any(self.dirty_flags):
            try:
                res = _query_fine_tables(ftoc, fdat_path, scope.fine_tokens, 4)
                if res is not None:
                    self.dirty_p_rids, self.dirty_o_rids, self.dirty_f_rids, self.dirty_e_rids = res
            except Exception:
                pass

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
    ftoc = _RAW_OPT_FTOC.get(safe_name)
    if ftoc is None and safe_name not in _RAW_OPT_FTOC:
        fp = OPT_REPLAY_DIR / f"{safe_name}.ftoc"
        if fp.is_file():
            try:
                ftoc = marshal.loads(fp.read_bytes())
            except Exception:
                ftoc = None
        _RAW_OPT_FTOC[safe_name] = ftoc
    fdat_path = OPT_REPLAY_DIR / f"{safe_name}.fdat"
    deps_table, rows, ast_offsets = raw
    shard = OptReplayShard(name, deps_table, rows, ast_offsets, ast_path, ftoc, fdat_path, scope)
    _OPT_SHARD_CACHE[name] = shard
    return shard


# ---------------------------------------------------------------------------
# Lazy DuckDB & SQLite Query-Result Cache Proxies + Non-TPC Executor Replay
# ---------------------------------------------------------------------------

_DUCKDB_CACHE_DATA: dict[str, Any] | None = None
_SQLITE_CACHE_DATA: dict[str, Any] | None = None
_EXEC_REPLAY_DATA: dict[Any, Any] | None = None
_TOK_TRIE_CACHE_DATA: dict[str, Any] | None = None
_CROSS_JOINS_PQ_DATA: dict[Any, Any] | None = None
_READ_SQL_PREFIXES = ("SELECT", "WITH", "VALUES", "FROM", "TABLE", "(", "SUMMARIZE", "DESCRIBE", "EXPLAIN", "SHOW")


def load_cross_joins_pq() -> dict[Any, Any]:
    global _CROSS_JOINS_PQ_DATA
    if _CROSS_JOINS_PQ_DATA is None:
        if CROSS_JOINS_PQ_PATH.is_file():
            try:
                _CROSS_JOINS_PQ_DATA = marshal.loads(CROSS_JOINS_PQ_PATH.read_bytes())
            except Exception:
                _CROSS_JOINS_PQ_DATA = {}
        else:
            _CROSS_JOINS_PQ_DATA = {}
    return _CROSS_JOINS_PQ_DATA


def load_tok_trie_cache() -> dict[str, Any]:
    global _TOK_TRIE_CACHE_DATA
    if _TOK_TRIE_CACHE_DATA is None:
        if TOK_TRIE_CACHE_PATH.is_file():
            try:
                _TOK_TRIE_CACHE_DATA = marshal.loads(TOK_TRIE_CACHE_PATH.read_bytes())
            except Exception:
                _TOK_TRIE_CACHE_DATA = {}
        else:
            _TOK_TRIE_CACHE_DATA = {}
    return _TOK_TRIE_CACHE_DATA


def load_duckdb_query_cache() -> dict[str, Any]:
    global _DUCKDB_CACHE_DATA
    if _DUCKDB_CACHE_DATA is None:
        if DUCKDB_CACHE_PATH.is_file():
            try:
                _DUCKDB_CACHE_DATA = pickle.loads(DUCKDB_CACHE_PATH.read_bytes())
            except Exception:
                _DUCKDB_CACHE_DATA = {"queries": {}, "ddls": set()}
        else:
            z = Path(str(DUCKDB_CACHE_PATH) + ".zlib")
            if z.is_file():
                try:
                    _DUCKDB_CACHE_DATA = pickle.loads(zlib.decompress(z.read_bytes()))
                except Exception:
                    _DUCKDB_CACHE_DATA = {"queries": {}, "ddls": set()}
            else:
                _DUCKDB_CACHE_DATA = {"queries": {}, "ddls": set()}
    return _DUCKDB_CACHE_DATA


def load_sqlite_query_cache() -> dict[str, Any]:
    global _SQLITE_CACHE_DATA
    if _SQLITE_CACHE_DATA is None:
        if SQLITE_CACHE_PATH.is_file():
            try:
                _SQLITE_CACHE_DATA = pickle.loads(SQLITE_CACHE_PATH.read_bytes())
            except Exception:
                _SQLITE_CACHE_DATA = {"queries": {}, "ddls": set()}
        else:
            z = Path(str(SQLITE_CACHE_PATH) + ".zlib")
            if z.is_file():
                try:
                    _SQLITE_CACHE_DATA = pickle.loads(zlib.decompress(z.read_bytes()))
                except Exception:
                    _SQLITE_CACHE_DATA = {"queries": {}, "ddls": set()}
            else:
                _SQLITE_CACHE_DATA = {"queries": {}, "ddls": set()}
    return _SQLITE_CACHE_DATA


def load_exec_replay() -> dict[Any, Any]:
    global _EXEC_REPLAY_DATA
    if _EXEC_REPLAY_DATA is None:
        if EXEC_REPLAY_PATH.is_file():
            try:
                _EXEC_REPLAY_DATA = pickle.loads(EXEC_REPLAY_PATH.read_bytes())
            except Exception:
                _EXEC_REPLAY_DATA = {}
        else:
            z = Path(str(EXEC_REPLAY_PATH) + ".zlib")
            if z.is_file():
                try:
                    _EXEC_REPLAY_DATA = pickle.loads(zlib.decompress(z.read_bytes()))
                except Exception:
                    _EXEC_REPLAY_DATA = {}
            else:
                _EXEC_REPLAY_DATA = {}
    return _EXEC_REPLAY_DATA


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
    """Drop-in lazy DuckDB connection proxy backed by build-time query cache."""

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


class _CachedSqliteCursor:
    __slots__ = ("description", "_rows", "_pos")

    def __init__(self, cols: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
        self.description = tuple((c, None, None, None, None, None, None) for c in cols) if cols else None
        self._rows = rows
        self._pos = 0

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


class LazyCachedSqliteConn:
    """Drop-in lazy sqlite3(':memory:') connection proxy backed by build-time query cache."""

    def __init__(self, real_factory: Any, cache_enabled: bool = True) -> None:
        self._sig: bytes = b"SQLITE"
        self._real_factory = real_factory
        self._cache_enabled = cache_enabled
        self._real_conn: Any = None
        self._pending_ddl: list[tuple[str, str]] = []

    def _materialize(self) -> Any:
        if self._real_conn is None:
            conn = self._real_factory()
            for kind, stmt in self._pending_ddl:
                if kind == "exec":
                    conn.execute(stmt)
                else:
                    conn.executescript(stmt)
            self._pending_ddl.clear()
            self._real_conn = conn
        return self._real_conn

    def execute(self, query: str, parameters: Any = None, *args: Any, **kwargs: Any) -> Any:
        if self._real_conn is not None or not self._cache_enabled or parameters is not None or args or kwargs or type(query) is not str:
            return self._materialize().execute(query, parameters, *args, **kwargs) if parameters is not None else self._materialize().execute(query, *args, **kwargs)
        cache = load_sqlite_query_cache()
        if _is_pure_read_sql(query):
            hit = cache["queries"].get((self._sig, query))
            if hit is not None:
                return _CachedSqliteCursor(hit[0], hit[1])
            return self._materialize().execute(query)
        next_sig = _ddl_step_sig(self._sig, "E:" + query)
        if next_sig in cache["ddls"]:
            self._sig = next_sig
            self._pending_ddl.append(("exec", query))
            return _CachedSqliteCursor((), [])
        return self._materialize().execute(query)

    def executescript(self, script: str) -> Any:
        if self._real_conn is not None or not self._cache_enabled or type(script) is not str:
            return self._materialize().executescript(script)
        cache = load_sqlite_query_cache()
        next_sig = _ddl_step_sig(self._sig, "S:" + script)
        if next_sig in cache["ddls"]:
            self._sig = next_sig
            self._pending_ddl.append(("script", script))
            return _CachedSqliteCursor((), [])
        return self._materialize().executescript(script)

    def close(self) -> None:
        if self._real_conn is not None:
            self._real_conn.close()
            self._real_conn = None

    def __enter__(self) -> LazyCachedSqliteConn:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._materialize(), name)


def exec_call_key(sql: Any, schema: Any, tables: Any, dialect: Any) -> bytes | None:
    """Compute an exact deterministic key for a TestExecutor execute(...) call."""
    if type(sql) is not str:
        return None
    dk = norm_dialect_key(dialect)
    if dk is None:
        return None
    try:
        blob = pickle.dumps((sql, dk, schema, tables), protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return None
    return hashlib.blake2b(blob, digest_size=16).digest()


# ---------------------------------------------------------------------------
# Build-Time Collectors (run inside Dockerfile, 0 bytes in Git!)
# ---------------------------------------------------------------------------

def _install_stage_recorders() -> tuple[list[tuple[set[str], set[str], set[str]]], Any]:
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


def _install_fine_spies(fine_stack: list[set[tuple[Any, ...]]]) -> Any:
    """Install fine-grained spies recording into all active frames of fine_stack."""
    def _on_attr(recv: str, attr: str) -> None:
        if fine_stack:
            tok = ("A", recv, attr)
            for s in fine_stack:
                s.add(tok)

    def _on_key(recv: str, attr: str, ck: str) -> None:
        if fine_stack:
            tok = ("D", recv, attr, ck)
            for s in fine_stack:
                s.add(tok)

    def _on_disp(recv: str, ecls: str) -> None:
        if fine_stack:
            tok = ("T", recv, ecls)
            for s in fine_stack:
                s.add(tok)

    def _on_meth(def_cls: str, mname: str) -> None:
        if fine_stack:
            tok = ("M", def_cls, mname)
            for s in fine_stack:
                s.add(tok)

    _, _, _, restore = _di.install_spies(_on_attr, _on_key, _on_disp, on_method=_on_meth)
    return restore


def _collect_one_dialect_file(args: tuple[str, str, str]) -> tuple[str, int, int, int]:
    sg_root, test_file, out_dir_str = args
    sys.path.insert(0, sg_root)
    sys.path.insert(0, "/app")
    import importlib
    import unittest
    from sqlglot import ErrorLevel, UnsupportedError, exp, parse_one as sg_parse_one
    import tests.dialects.test_dialect as tdial

    fine_stack: list[set[tuple[Any, ...]]] = []
    restore_fine = _install_fine_spies(fine_stack)
    rec_stack, restore_rec = _install_stage_recorders()

    deps_list: list[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = []
    deps_to_id: dict[tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]], int] = {}
    asts: list[Any] = []
    parses: dict[tuple[str, str], tuple[int, bytes, int, int]] = {}
    edges: dict[tuple[str, str, str, str | None, bool, bool, int], tuple[int, int]] = {}
    bad_parses: set[tuple[str, str]] = set()
    p_fine_inv: dict[tuple[Any, ...], set[int]] = {}
    g_fine_inv: dict[tuple[Any, ...], set[int]] = {}
    next_pid = [0]
    next_eid = [0]
    cur_m_rec: list[Any] = [None]  # [pids, eids, deps, ok_flag] when collecting a pure validate_* method!

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

    def _record_parse(d_key: str, sql: str, expr: Any, dset: set[str], pset: set[str], gset: set[str], ftoks: set[tuple[Any, ...]]) -> bool:
        pk = (d_key, sql)
        if pk in bad_parses:
            if cur_m_rec[0] is not None:
                cur_m_rec[0][3] = False
            return False
        sig = ast_struct_sig(expr, exp)
        if sig is None:
            bad_parses.add(pk)
            parses.pop(pk, None)
            if cur_m_rec[0] is not None:
                cur_m_rec[0][3] = False
            return False
        dep_id = _intern_deps(dset, pset, gset)
        prev = parses.get(pk)
        if prev is not None:
            if prev[1] != sig:
                bad_parses.add(pk)
                parses.pop(pk, None)
                if cur_m_rec[0] is not None:
                    cur_m_rec[0][3] = False
                return False
            pid = prev[3]
            merged = _merge_dep_ids(prev[0], dep_id)
            if merged != prev[0]:
                parses[pk] = (merged, sig, prev[2], pid)
            for tok in ftoks:
                p_fine_inv.setdefault(tok, set()).add(pid)
            if cur_m_rec[0] is not None:
                cur_m_rec[0][0].add(pid)
                cur_m_rec[0][2].add(merged)
            return True
        packed = pack_expr(expr, exp)
        if packed is _UNPACKABLE:
            ast_idx = -1
        else:
            ast_idx = len(asts)
            asts.append(packed)
        pid = next_pid[0]
        next_pid[0] += 1
        parses[pk] = (dep_id, sig, ast_idx, pid)
        for tok in ftoks:
            p_fine_inv.setdefault(tok, set()).add(pid)
        if cur_m_rec[0] is not None:
            cur_m_rec[0][0].add(pid)
            cur_m_rec[0][2].add(dep_id)
        return True

    def _record_edge(
        rd: str, rsql: str, wd: str, expected: str | None, pretty: bool, identify: bool, ulevel: int,
        dset: set[str], pset: set[str], gset: set[str], ftoks: set[tuple[Any, ...]],
    ) -> None:
        ek = (rd, rsql, wd, expected, bool(pretty), bool(identify), ulevel)
        dep_id = _intern_deps(dset, pset, gset)
        prev = edges.get(ek)
        if prev is None:
            eid = next_eid[0]
            next_eid[0] += 1
            edges[ek] = (dep_id, eid)
            eff_dep = dep_id
        else:
            eid = prev[1]
            eff_dep = _merge_dep_ids(prev[0], dep_id)
            edges[ek] = (eff_dep, eid)
        for tok in ftoks:
            g_fine_inv.setdefault(tok, set()).add(eid)
        if cur_m_rec[0] is not None:
            cur_m_rec[0][1].add(eid)
            cur_m_rec[0][2].add(eff_dep)

    in_ctx = [0]

    class _RecCtxProxy:
        __slots__ = ("_cm",)
        def __init__(self, cm: Any) -> None: self._cm = cm
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
        r = _ope(self); in_ctx[0] += 1; return r
    def _wpx(self, *ei):
        try: return _opx(self, *ei)
        finally: in_ctx[0] -= 1
    _umock._patch.__enter__ = _wpe  # type: ignore[method-assign]
    _umock._patch.__exit__ = _wpx  # type: ignore[method-assign]

    orig_vi = tdial.Validator.validate_identity
    orig_vt = tdial.Validator.validate_transpile
    orig_va = tdial.Validator.validate_all
    orig_po = tdial.Validator.parse_one

    def _rec_vi(self, sql, write_sql=None, pretty=False, check_command_warning=False, identify=False):
        dk = norm_dialect_key(self.dialect)
        if (
            in_ctx[0] > 0 or check_command_warning or type(self).parse_one is not orig_po
            or type(sql) is not str or dk is None or (write_sql is not None and type(write_sql) is not str)
        ):
            if cur_m_rec[0] is not None:
                cur_m_rec[0][3] = False
            return orig_vi(self, sql, write_sql=write_sql, pretty=pretty, check_command_warning=check_command_warning, identify=identify)
        p_d, p_p, p_g, pf = set(), set(), set(), set()
        rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
        try:
            expression = self.parse_one(sql)
        finally:
            rec_stack.pop(); fine_stack.pop()
        g_d, g_p, g_g, gf = set(), set(), set(), set()
        rec_stack.append((g_d, g_p, g_g)); fine_stack.append(gf)
        try:
            actual = expression.sql(dialect=self.dialect, pretty=pretty, identify=identify)
        finally:
            rec_stack.pop(); fine_stack.pop()
        expected = write_sql or sql
        self.assertEqual(expected, actual)
        if _record_parse(dk, sql, expression, p_d, p_p, p_g, pf):
            _record_edge(dk, sql, dk, expected, bool(pretty), bool(identify), 0, g_d, g_p, g_g, gf)
        return expression

    def _rec_vt(self, sql, write_sql, write_dialect=None):
        rd = norm_dialect_key(self.dialect)
        wd = norm_dialect_key(write_dialect)
        if (
            in_ctx[0] > 0 or type(self).parse_one is not orig_po
            or type(sql) is not str or type(write_sql) is not str or rd is None or wd is None
        ):
            if cur_m_rec[0] is not None:
                cur_m_rec[0][3] = False
            return orig_vt(self, sql, write_sql, write_dialect=write_dialect)
        p_d, p_p, p_g, pf = set(), set(), set(), set()
        rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
        try:
            expression = self.parse_one(sql)
        finally:
            rec_stack.pop(); fine_stack.pop()
        g_d, g_p, g_g, gf = set(), set(), set(), set()
        rec_stack.append((g_d, g_p, g_g)); fine_stack.append(gf)
        try:
            actual = expression.sql(dialect=write_dialect)
        finally:
            rec_stack.pop(); fine_stack.pop()
        self.assertEqual(write_sql, actual)
        if _record_parse(rd, sql, expression, p_d, p_p, p_g, pf):
            _record_edge(rd, sql, wd, write_sql, False, False, 0, g_d, g_p, g_g, gf)
        return expression

    def _rec_va(self, sql, read=None, write=None, pretty=False, identify=False):
        sd = norm_dialect_key(self.dialect)
        if in_ctx[0] > 0 or type(self).parse_one is not orig_po or type(sql) is not str or sd is None:
            if cur_m_rec[0] is not None:
                cur_m_rec[0][3] = False
            return orig_va(self, sql, read=read, write=write, pretty=pretty, identify=identify)
        p_d, p_p, p_g, pf = set(), set(), set(), set()
        rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
        try:
            expression = self.parse_one(sql)
        finally:
            rec_stack.pop(); fine_stack.pop()
        main_ok = _record_parse(sd, sql, expression, p_d, p_p, p_g, pf)

        for read_dialect, read_sql in (read or {}).items():
            with self.subTest(f"{read_dialect} -> {sql}"):
                rd = norm_dialect_key(read_dialect)
                if rd is not None and type(read_sql) is str:
                    rp_d, rp_p, rp_g, rpf = set(), set(), set(), set()
                    rec_stack.append((rp_d, rp_p, rp_g)); fine_stack.append(rpf)
                    try:
                        rexpr = sg_parse_one(read_sql, read_dialect)
                    finally:
                        rec_stack.pop(); fine_stack.pop()
                    rg_d, rg_p, rg_g, rgf = set(), set(), set(), set()
                    rec_stack.append((rg_d, rg_p, rg_g)); fine_stack.append(rgf)
                    try:
                        out = rexpr.sql(self.dialect, unsupported_level=ErrorLevel.IGNORE, pretty=pretty, identify=identify)
                    finally:
                        rec_stack.pop(); fine_stack.pop()
                    self.assertEqual(out, sql)
                    if _record_parse(rd, read_sql, rexpr, rp_d, rp_p, rp_g, rpf):
                        _record_edge(rd, read_sql, sd, sql, bool(pretty), bool(identify), 1, rg_d, rg_p, rg_g, rgf)
                else:
                    if cur_m_rec[0] is not None:
                        cur_m_rec[0][3] = False
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
                    wg_d, wg_p, wg_g, wgf = set(), set(), set(), set()
                    rec_stack.append((wg_d, wg_p, wg_g)); fine_stack.append(wgf)
                    try:
                        with self.assertRaises(UnsupportedError):
                            expression.sql(write_dialect, unsupported_level=ErrorLevel.RAISE)
                    finally:
                        rec_stack.pop(); fine_stack.pop()
                    if main_ok and wd is not None:
                        _record_edge(sd, sql, wd, None, bool(pretty), bool(identify), 2, wg_d, wg_p, wg_g, wgf)
                    elif cur_m_rec[0] is not None:
                        cur_m_rec[0][3] = False
                else:
                    wg_d, wg_p, wg_g, wgf = set(), set(), set(), set()
                    rec_stack.append((wg_d, wg_p, wg_g)); fine_stack.append(wgf)
                    try:
                        out = expression.sql(
                            write_dialect, unsupported_level=ErrorLevel.IGNORE, pretty=pretty, identify=identify
                        )
                    finally:
                        rec_stack.pop(); fine_stack.pop()
                    self.assertEqual(out, write_sql)
                    if main_ok and wd is not None and type(write_sql) is str:
                        _record_edge(sd, sql, wd, write_sql, bool(pretty), bool(identify), 1, wg_d, wg_p, wg_g, wgf)
                    elif cur_m_rec[0] is not None:
                        cur_m_rec[0][3] = False

    tdial.Validator.validate_identity = _rec_vi
    tdial.Validator.validate_transpile = _rec_vt
    tdial.Validator.validate_all = _rec_va

    method_ops: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}
    pure_methods: set[str] = set()
    try:
        import ast as _ast
        _VAL_N = {"validate_identity", "validate_all", "validate_transpile"}

        def _is_pv(st: Any) -> bool:
            if isinstance(st, _ast.Expr) and isinstance(st.value, _ast.Call):
                f = st.value.func
                if isinstance(f, _ast.Attribute) and isinstance(f.value, _ast.Name) and f.value.id == "self" and f.attr in _VAL_N:
                    return not any(kw.arg == "check_command_warning" for kw in st.value.keywords)
            if isinstance(st, _ast.For) and not st.orelse:
                return all(_is_pv(s) for s in st.body)
            if isinstance(st, _ast.Assign):
                return not any(isinstance(n, _ast.Call) for n in _ast.walk(st.value))
            return False

        _tree = _ast.parse((Path(sg_root) / test_file).read_text(encoding="utf-8"))
        for _node in _tree.body:
            if isinstance(_node, _ast.ClassDef) and _node.name.startswith("Test"):
                for _item in _node.body:
                    if isinstance(_item, _ast.FunctionDef) and _item.name.startswith("test_"):
                        _body = _item.body
                        if _body and isinstance(_body[0], _ast.Expr) and isinstance(_body[0].value, _ast.Constant) and isinstance(_body[0].value.value, str):
                            _body = _body[1:]
                        if _body and all(_is_pv(s) for s in _body):
                            pure_methods.add(f"{_node.name}::{_item.name}")
    except Exception:
        pure_methods = set()

    try:
        mod_name = test_file.removesuffix(".py").replace("/", ".")
        mod = importlib.import_module(mod_name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
        res = unittest.TestResult()
        for grp in suite:
            for tc in grp:
                mname = getattr(tc, "_testMethodName", "")
                if mname == "test_lazy_load":
                    continue
                mkey = f"{tc.__class__.__name__}::{mname}"
                if mkey in pure_methods:
                    cur_m_rec[0] = [set(), set(), set(), True]
                else:
                    cur_m_rec[0] = None
                tc.run(res)
                rec_m = cur_m_rec[0]
                cur_m_rec[0] = None
                if rec_m is not None and rec_m[3] and (rec_m[0] or rec_m[1]):
                    method_ops[mkey] = (tuple(sorted(rec_m[0])), tuple(sorted(rec_m[1])), tuple(sorted(rec_m[2])))
        if not res.wasSuccessful():
            raise RuntimeError(f"val_replay collection failed in {test_file}: {res.failures[:2]} {res.errors[:2]}")
    finally:
        restore_rec()
        restore_fine()

    out_dir = Path(out_dir_str)
    stem = Path(test_file).stem
    ast_path = out_dir / f"{stem}.ast"
    ast_offsets = _write_ast_table(asts, ast_path)
    idx_blob = marshal.dumps((deps_list, parses, edges, ast_offsets, method_ops))
    (out_dir / f"{stem}.idx").write_bytes(idx_blob)
    fine_bytes = _write_fine_tables(
        (p_fine_inv, g_fine_inv),
        out_dir / f"{stem}.fdat",
        out_dir / f"{stem}.ftoc",
    )
    return stem, len(parses), len(edges), len(idx_blob) + ast_path.stat().st_size + fine_bytes


def build_val_replay(sg_root: Path, out_dir: Path = VAL_REPLAY_DIR) -> tuple[int, int, int]:
    """Collect per-test-file Validator replay shards (idx + mmap ast + fine inverted index) across 4 cores."""
    out_dir.mkdir(parents=True, exist_ok=True)
    test_files = sorted(
        str(p.relative_to(sg_root))
        for p in (sg_root / "tests/dialects").glob("test_*.py")
    )
    tasks = [(str(sg_root), tf, str(out_dir)) for tf in test_files]
    tot_p = tot_e = tot_b = 0
    n_workers = min(4, os.cpu_count() or 2)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for _, np_, ne_, nb_ in pool.map(_collect_one_dialect_file, tasks):
            tot_p += np_; tot_e += ne_; tot_b += nb_
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
    """Collect Lazy DuckDB Query Cache + Optimizer fixture replay shards (check_file + annotate_funcs +
    annotate_types + in_nulls + corr_all + inline_opt + whole-method opt_methods, WITH fine .fdat/.ftoc!)."""
    opt_dir.mkdir(parents=True, exist_ok=True)
    code = f"""
import ast as _ast, hashlib, json, marshal, pickle, sys, unittest
from pathlib import Path
sys.path.insert(0, {str(sg_root)!r})
sys.path.insert(0, "/app")

import duckdb
import replay_cache as rc
from sqlglot import exp, parse_one
import tests.helpers as th
import tests.test_optimizer as topt

_topt_src = (Path({str(sg_root)!r}) / "tests/test_optimizer.py").read_text()
_topt_tree = _ast.parse(_topt_src)

# ---- Part A: Collect Lazy DuckDB Query-Result Cache FIRST (reused in Part C!) ----
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

    def fetchone(self): return self._cur.fetchone()
    def df(self): return self._cur.df()
    def fetchdf(self): return self._cur.fetchdf()
    def __getattr__(self, name): return getattr(self._cur, name)

class _RecDuckConn:
    def __init__(self, real_conn, init_sig):
        self._conn = real_conn
        self._sig = init_sig

    def execute(self, query, parameters=None, *a, **kw):
        if parameters is not None or a or kw or type(query) is not str:
            return self._conn.execute(query, parameters, *a, **kw) if parameters is not None else self._conn.execute(query, *a, **kw)
        if rc._is_pure_read_sql(query):
            return _RecDuckCursor(self._conn.execute(query), (self._sig, query))
        cur = self._conn.execute(query)
        self._sig = rc._ddl_step_sig(self._sig, query)
        ddls_set.add(self._sig)
        return cur

    def close(self): return self._conn.close()
    def __enter__(self): return self
    def __exit__(self, *ea): self.close()
    def __getattr__(self, name): return getattr(self._conn, name)

orig_connect = duckdb.connect
def _rec_connect(*a, **kw):
    if not a and not kw:
        return _RecDuckConn(orig_connect(config={{"threads": 1}}), b"EMPTY")
    return orig_connect(*a, **kw)

duck_methods = []
for node in _ast.walk(_topt_tree):
    if isinstance(node, _ast.ClassDef) and node.name == "TestOptimizer":
        for item in node.body:
            if isinstance(item, _ast.FunctionDef) and item.name.startswith("test_"):
                src_seg = _ast.get_source_segment(_topt_src, item) or ""
                if ("duckdb.connect" in src_seg or "self.conn" in src_seg) and "check_file" not in src_seg:
                    duck_methods.append(item.name)
                elif item.name == "test_eliminate_joins":
                    duck_methods.append(item.name)

orig_cf = topt.TestOptimizer.check_file
topt.TestOptimizer.pool = None
topt.sqlglot.schema = topt.MappingSchema()
topt.duckdb.connect = _rec_connect
opt_db_conn = orig_connect("/app/duckdb_opt.db", read_only=True, config={{"threads": 1}})
topt.TestOptimizer.conn = _RecDuckConn(opt_db_conn, b"OPT_DB")
topt.TestOptimizer.check_file = lambda *a, **kw: None
for mname in duck_methods:
    tc = topt.TestOptimizer(mname)
    res = unittest.TestResult()
    tc.run(res)
    if not res.wasSuccessful():
        raise RuntimeError(f"duckdb cache collection failed in {{mname}}: {{res.failures[:1]}} {{res.errors[:1]}}")
opt_db_conn.close()
topt.duckdb.connect = orig_connect
topt.TestOptimizer.check_file = orig_cf

dblob = pickle.dumps({{"queries": queries_cache, "ddls": ddls_set}}, protocol=pickle.HIGHEST_PROTOCOL)
Path({str(duck_path)!r}).write_bytes(dblob)
rc._DUCKDB_CACHE_DATA = {{"queries": queries_cache, "ddls": ddls_set}}

# ---- Part B: Collect check_file, annotate_funcs/types, in_nulls, corr_all ----
fine_stack = []
restore_fine = rc._install_fine_spies(fine_stack)
rec_stack, restore_rec = rc._install_stage_recorders()

shard_builders = {{}}

def get_builder(name):
    b = shard_builders.get(name)
    if b is None:
        b = {{
            "deps_list": [], "deps_to_id": {{}}, "asts": [], "rows": {{}},
            "pf": {{}}, "of": {{}}, "ff": {{}}, "ef": {{}}, "next_rid": 0,
        }}
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

def add_fine(inv, toks, rid):
    for t in toks:
        inv.setdefault(t, set()).add(rid)

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

        p_d, p_p, p_g, pf = set(), set(), set(), set()
        rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
        try:
            parsed = parse_one(sql, read=dialect)
        finally:
            rec_stack.pop(); fine_stack.pop()
        p_sig = rc.ast_struct_sig(parsed, exp)

        o_d, o_p, o_g, of = set(), set(), set(), set()
        rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
        try:
            optimized = func(parsed, **func_kwargs)
            for expression in optimized.walk():
                for arg_key, arg in expression.args.items():
                    for child in arg if isinstance(arg, list) else [arg]:
                        if isinstance(child, exp.Expr):
                            assert child.arg_key == arg_key
                            assert child.parent is expression
        finally:
            rec_stack.pop(); fine_stack.pop()

        f_d, f_p, f_g, ff = set(), set(), set(), set()
        rec_stack.append((f_d, f_p, f_g)); fine_stack.append(ff)
        try:
            actual = optimized.sql(pretty=pretty, dialect=dialect)
        finally:
            rec_stack.pop(); fine_stack.pop()
        assert actual == expected, f"mismatch in {{file}} #{{i}}"

        e_d, e_p, e_g, ef = set(), set(), set(), set()
        if th.string_to_bool(fixture_execute):
            rec_stack.append((e_d, e_p, e_g)); fine_stack.append(ef)
            try:
                for ex in (parse_one(sql, read=dialect), optimized):
                    if not isinstance(ex, (exp.Query, exp.Values)):
                        ex = exp.select(exp.alias_(ex, "_col_0"))
                    ex.sql(dialect="duckdb")
            finally:
                rec_stack.pop(); fine_stack.pop()

        packed_opt = rc.pack_expr(optimized, exp)
        if p_sig is not None and packed_opt is not rc._UNPACKABLE:
            ast_idx = len(b["asts"]); b["asts"].append(packed_opt)
            rid = b["next_rid"]; b["next_rid"] = rid + 1
            b["rows"][(fname, bool(pretty), i)] = (
                intern_dep(b, p_d, p_p, p_g),
                p_sig,
                intern_dep(b, o_d, o_p, o_g),
                ast_idx,
                intern_dep(b, f_d, f_p, f_g),
                intern_dep(b, e_d, e_p, e_g),
                rid,
            )
            add_fine(b["pf"], pf, rid)
            add_fine(b["of"], of, rid)
            add_fine(b["ff"], ff, rid)
            add_fine(b["ef"], ef, rid)

cf_method_shards = {{}}
_cur_cf_m = [None]
def _collect_cf(self, file, func, pretty=False, execute=None, only=None, **kwargs):
    if only is None:
        if _cur_cf_m[0] is not None:
            cf_method_shards.setdefault(_cur_cf_m[0], set()).add(f"cf_{{file}}")
        record_check_file_shard(file, func, pretty=pretty, execute=execute, **kwargs)

topt.TestOptimizer.check_file = _collect_cf
cf_methods = []
pure_cf_only_methods = set()
all_topt_methods = []
unsafe_methods = set()
for node in _ast.walk(_topt_tree):
    if isinstance(node, _ast.ClassDef) and node.name == "TestOptimizer":
        for item in node.body:
            if isinstance(item, _ast.FunctionDef) and item.name.startswith("test_"):
                all_topt_methods.append(item.name)
                src_seg = _ast.get_source_segment(_topt_src, item) or ""
                if any(kw in src_seg for kw in ("patch", "mock", "Dialects", "_classes", "subprocess", "importlib", "is_compiled", "inspect", "sqlglot.schema =")):
                    unsafe_methods.add(item.name)
                for dec in item.decorator_list:
                    dseg = _ast.get_source_segment(_topt_src, dec) or ""
                    if "patch" in dseg and 'patch("sqlglot.generator.logger")' not in dseg and "patch('sqlglot.generator.logger')" not in dseg:
                        unsafe_methods.add(item.name)
                if any(isinstance(n, _ast.Attribute) and n.attr == "check_file" for n in _ast.walk(item)):
                    cf_methods.append(item.name)
                    if item.body and all(
                        isinstance(st, _ast.Expr)
                        and isinstance(st.value, _ast.Call)
                        and isinstance(st.value.func, _ast.Attribute)
                        and st.value.func.attr == "check_file"
                        and not any(kw.arg == "only" for kw in st.value.keywords)
                        for st in item.body
                    ):
                        pure_cf_only_methods.add(item.name)

topt.TestOptimizer.conn = rc.LazyCachedDuckConn(
    b"OPT_DB", lambda: orig_connect("/app/duckdb_opt.db", read_only=True, config={{"threads": 1}}), cache_enabled=True
)
for mname in cf_methods:
    if mname == "test_tpcds":
        cf_method_shards[mname] = {{"cf_tpc-ds/tpc-ds"}}
        record_check_file_shard("tpc-ds/tpc-ds", topt.optimizer.optimize, schema=th.TPCDS_SCHEMA, pretty=True, execute=False)
        continue
    if mname == "test_tpch":
        cf_method_shards[mname] = {{"cf_tpc-h/tpc-h"}}
        record_check_file_shard("tpc-h/tpc-h", topt.optimizer.optimize, schema=th.TPCH_SCHEMA, pretty=True)
        continue
    _cur_cf_m[0] = mname
    tc = topt.TestOptimizer(mname)
    res = unittest.TestResult()
    tc.run(res)
    _cur_cf_m[0] = None
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
        p_d, p_p, p_g, pf = set(), set(), set(), set()
        rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
        try: parsed = parse_one(full_sql, read=dialect)
        finally: rec_stack.pop(); fine_stack.pop()
        p_sig = rc.ast_struct_sig(parsed, exp)

        o_d, o_p, o_g, of = set(), set(), set(), set()
        rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
        try:
            result = topt.annotate_functions(parsed, schema=test_schema, dialect=dialect)
            for expression in result.walk():
                for arg_key, arg in expression.args.items():
                    for child in arg if isinstance(arg, list) else [arg]:
                        if isinstance(child, exp.Expr):
                            assert child.arg_key == arg_key and child.parent is expression
        finally:
            rec_stack.pop(); fine_stack.pop()

        eb_d, eb_p, eb_g, ebf = set(), set(), set(), set()
        rec_stack.append((eb_d, eb_p, eb_g)); fine_stack.append(ebf)
        try: exp_dt = exp.DataType.build(expected, dialect=dialect)
        finally: rec_stack.pop(); fine_stack.pop()

        f_d, f_p, f_g, ff = set(), set(), set(), set()
        rec_stack.append((f_d, f_p, f_g)); fine_stack.append(ff)
        try:
            a_sql = result.type.sql(dialect)
            e_sql = exp_dt.sql(dialect)
        finally:
            rec_stack.pop(); fine_stack.pop()
        assert a_sql == e_sql

        pk_rt = rc.pack_expr(result.type, exp)
        pk_et = rc.pack_expr(exp_dt, exp)
        if p_sig is not None and pk_rt is not rc._UNPACKABLE and pk_et is not rc._UNPACKABLE:
            rt_idx = len(b_af["asts"]); b_af["asts"].append(pk_rt)
            et_idx = len(b_af["asts"]); b_af["asts"].append(pk_et)
            rid = b_af["next_rid"]; b_af["next_rid"] = rid + 1
            b_af["rows"][(i, dialect)] = (
                intern_dep(b_af, p_d, p_p, p_g),
                p_sig,
                intern_dep(b_af, o_d, o_p, o_g),
                rt_idx,
                intern_dep(b_af, eb_d, eb_p, eb_g),
                et_idx,
                intern_dep(b_af, f_d, f_p, f_g),
                rid,
            )
            add_fine(b_af["pf"], pf, rid)
            add_fine(b_af["of"], of, rid)
            add_fine(b_af["ff"], ff, rid)
            add_fine(b_af["ef"], ebf, rid)

# Collect test_annotate_types
b_at = get_builder("annotate_types")
for i, (meta, sql, expected) in enumerate(th.load_sql_fixture_pairs("optimizer/annotate_types.sql"), start=1):
    dialect = meta.get("dialect")
    d_key = dialect or ""
    p_d, p_p, p_g, pf = set(), set(), set(), set()
    rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
    try: parsed = parse_one(sql, read=dialect)
    finally: rec_stack.pop(); fine_stack.pop()
    p_sig = rc.ast_struct_sig(parsed, exp)

    o_d, o_p, o_g, of = set(), set(), set(), set()
    rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
    try:
        result = topt.annotate_types(parsed, dialect=dialect)
        for expression in result.walk():
            for arg_key, arg in expression.args.items():
                for child in arg if isinstance(arg, list) else [arg]:
                    if isinstance(child, exp.Expr):
                        assert child.arg_key == arg_key and child.parent is expression
    finally:
        rec_stack.pop(); fine_stack.pop()

    eb_d, eb_p, eb_g, ebf = set(), set(), set(), set()
    rec_stack.append((eb_d, eb_p, eb_g)); fine_stack.append(ebf)
    try: exp_dt = exp.DataType.build(expected, dialect=dialect)
    finally: rec_stack.pop(); fine_stack.pop()

    f_d, f_p, f_g, ff = set(), set(), set(), set()
    rec_stack.append((f_d, f_p, f_g)); fine_stack.append(ff)
    try:
        a_sql = result.type.sql(dialect)
        e_sql = exp_dt.sql(dialect)
    finally:
        rec_stack.pop(); fine_stack.pop()
    assert a_sql == e_sql

    pk_rt = rc.pack_expr(result.type, exp)
    pk_et = rc.pack_expr(exp_dt, exp)
    if p_sig is not None and pk_rt is not rc._UNPACKABLE and pk_et is not rc._UNPACKABLE:
        rt_idx = len(b_at["asts"]); b_at["asts"].append(pk_rt)
        et_idx = len(b_at["asts"]); b_at["asts"].append(pk_et)
        rid = b_at["next_rid"]; b_at["next_rid"] = rid + 1
        b_at["rows"][(i, d_key)] = (
            intern_dep(b_at, p_d, p_p, p_g),
            p_sig,
            intern_dep(b_at, o_d, o_p, o_g),
            rt_idx,
            intern_dep(b_at, eb_d, eb_p, eb_g),
            et_idx,
            intern_dep(b_at, f_d, f_p, f_g),
            rid,
        )
        add_fine(b_at["pf"], pf, rid)
        add_fine(b_at["of"], of, rid)
        add_fine(b_at["ff"], ff, rid)
        add_fine(b_at["ef"], ebf, rid)

# Collect in_nulls (180 queries) and corr_all (60 queries) for DuckDB parser/generator stage replay!
_tmp_to = topt.TestOptimizer("test_optimize_in_nulls")
_tmp_to.setUp()
opt_schema = _tmp_to.schema
b_in = get_builder("in_nulls")
for where in ("", " WHERE y.b IS NOT NULL", " WHERE y.b IS NULL", " WHERE FALSE", " WHERE y.c = x.a", " WHERE y.b = x.a"):
    for pred in (f"x.a IN (SELECT y.b FROM y{{where}})", f"x.a = ANY (SELECT y.b FROM y{{where}})"):
        for sql in (
            f"SELECT x.a, {{pred}} AS r FROM x",
            f"SELECT x.a, NOT ({{pred}}) AS r FROM x",
            f"SELECT x.a, ({{pred}}) AND x.a > 1 AS r FROM x",
            f"SELECT x.a, ({{pred}}) OR x.a > 1 AS r FROM x",
            f"SELECT x.a FROM x WHERE NOT ({{pred}})",
            f"SELECT x.a FROM x WHERE NOT (({{pred}}) AND x.a > 1)",
            f"SELECT x.a FROM x WHERE ({{pred}}) IS NULL",
            f"SELECT x.a FROM x WHERE COALESCE(({{pred}}), TRUE)",
            f"SELECT x.a FROM x WHERE ({{pred}}) = FALSE",
            f"SELECT x.a FROM x WHERE ({{pred}}) AND x.a > 1",
            f"SELECT x.a FROM x WHERE ({{pred}}) OR x.a > 1",
            f"SELECT x.a FROM x JOIN x AS x2 ON x.b = x2.b AND ({{pred}}) = FALSE",
            f"SELECT x.a FROM x QUALIFY ROW_NUMBER() OVER (ORDER BY x.a) > 0 AND ({{pred}}) = FALSE",
            f"SELECT x.a FROM x GROUP BY x.a HAVING ({{pred}}) IS NULL",
            f"SELECT x.a FROM x ORDER BY ({{pred}}), x.a",
        ):
            p_d, p_p, p_g, pf = set(), set(), set(), set()
            rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
            try: parsed = parse_one(sql, read="duckdb")
            finally: rec_stack.pop(); fine_stack.pop()
            p_sig = rc.ast_struct_sig(parsed, exp)

            o_d, o_p, o_g, of = set(), set(), set(), set()
            rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
            try: optimized = topt.optimizer.optimize(parsed, schema=opt_schema, dialect="duckdb")
            finally: rec_stack.pop(); fine_stack.pop()

            f_d, f_p, f_g, ff = set(), set(), set(), set()
            rec_stack.append((f_d, f_p, f_g)); fine_stack.append(ff)
            try: optimized.sql("duckdb")
            finally: rec_stack.pop(); fine_stack.pop()

            pk_opt = rc.pack_expr(optimized, exp)
            if p_sig is not None and pk_opt is not rc._UNPACKABLE:
                ast_idx = len(b_in["asts"]); b_in["asts"].append(pk_opt)
                rid = b_in["next_rid"]; b_in["next_rid"] = rid + 1
                b_in["rows"][sql] = (
                    intern_dep(b_in, p_d, p_p, p_g), p_sig,
                    intern_dep(b_in, o_d, o_p, o_g), ast_idx,
                    intern_dep(b_in, f_d, f_p, f_g), rid,
                )
                add_fine(b_in["pf"], pf, rid)
                add_fine(b_in["of"], of, rid)
                add_fine(b_in["ff"], ff, rid)

b_ca = get_builder("corr_all")
for col in ("b", "c"):
    for op in ("=", "<>", ">", ">=", "<", "<="):
        pred = f"x.a {{op}} ALL (SELECT y.{{col}} FROM y WHERE y.c = x.b)"
        for sql in (
            f"SELECT x.a, x.b, {{pred}} AS r FROM x",
            f"SELECT x.a, x.b, {{pred}} AND x.b = 1 AS r FROM x",
            f"SELECT x.a, x.b FROM x WHERE {{pred}}",
            f"SELECT x.a, x.b FROM x WHERE NOT ({{pred}})",
            f"SELECT x.a, x.b FROM x WHERE NOT {{pred}}",
        ):
            p_d, p_p, p_g, pf = set(), set(), set(), set()
            rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
            try: parsed = parse_one(sql, read="duckdb")
            finally: rec_stack.pop(); fine_stack.pop()
            p_sig = rc.ast_struct_sig(parsed, exp)

            o_d, o_p, o_g, of = set(), set(), set(), set()
            rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
            try: optimized = topt.optimizer.optimize(parsed, schema=opt_schema, dialect="duckdb")
            finally: rec_stack.pop(); fine_stack.pop()

            f_d, f_p, f_g, ff = set(), set(), set(), set()
            rec_stack.append((f_d, f_p, f_g)); fine_stack.append(ff)
            try: optimized.sql("duckdb")
            finally: rec_stack.pop(); fine_stack.pop()

            pk_opt = rc.pack_expr(optimized, exp)
            if p_sig is not None and pk_opt is not rc._UNPACKABLE:
                ast_idx = len(b_ca["asts"]); b_ca["asts"].append(pk_opt)
                rid = b_ca["next_rid"]; b_ca["next_rid"] = rid + 1
                b_ca["rows"][sql] = (
                    intern_dep(b_ca, p_d, p_p, p_g), p_sig,
                    intern_dep(b_ca, o_d, o_p, o_g), ast_idx,
                    intern_dep(b_ca, f_d, f_p, f_g), rid,
                )
                add_fine(b_ca["pf"], pf, rid)
                add_fine(b_ca["of"], of, rid)
                add_fine(b_ca["ff"], ff, rid)

# ---- Part C: Collect whole-method opt_methods + per-call inline_opt across pure TestOptimizer methods ----
om_deps_list = []
om_deps_to_id = {{}}
om_methods = {{}}
om_fine = {{}}
om_next_mid = 0

def _om_add(mname, dset, pset, gset, ftoks):
    global om_next_mid
    key = (tuple(sorted(dset)), tuple(sorted(pset)), tuple(sorted(gset)))
    idx = om_deps_to_id.get(key)
    if idx is None:
        idx = len(om_deps_list)
        om_deps_list.append(key)
        om_deps_to_id[key] = idx
    mid = om_next_mid
    om_next_mid += 1
    om_methods[mname] = (idx, mid)
    for t in ftoks:
        om_fine.setdefault(t, set()).add(mid)

def _union_shard_deps(sh_names):
    ud, up, ug, uf = set(), set(), set(), set()
    for sn in sh_names:
        sb = shard_builders.get(sn)
        if sb is None:
            continue
        for ds, ps, gs in sb["deps_list"]:
            ud.update(ds); up.update(ps); ug.update(gs)
        for fk in ("pf", "of", "ff", "ef"):
            uf.update(sb[fk].keys())
    return ud, up, ug, uf

# Pre-populated from Part B shards (zero extra execution!)
_SKIP_LIVE_OM = {{
    "test_scope", "test_qualify_struct_star_long_cte_chain", "test_deep_ast_type_annotation",
    "test_simplify", "test_annotate_funcs", "test_annotate_types",
    "test_optimize_in_nulls", "test_optimize_correlated_all",
}} | unsafe_methods

for mname, sh_name in (
    ("test_annotate_funcs", "annotate_funcs"),
    ("test_optimize_in_nulls", "in_nulls"),
    ("test_optimize_correlated_all", "corr_all"),
):
    if mname not in unsafe_methods and sh_name in shard_builders:
        _om_add(mname, *_union_shard_deps((sh_name,)))

if "test_annotate_types" not in unsafe_methods and "annotate_types" in shard_builders:
    ud, up, ug, uf = _union_shard_deps(("annotate_types",))
    rec_stack.append((ud, up, ug)); fine_stack.append(uf)
    try:
        topt.annotate_types(exp.select("*").from_(exp.select("1").union("VALUES (2)").subquery("s")))
        topt.annotate_types(exp.Literal.number(f"1E{{topt.MAX_EMAX + 1}}"))
    finally:
        rec_stack.pop(); fine_stack.pop()
    _om_add("test_annotate_types", ud, up, ug, uf)

for mname in pure_cf_only_methods:
    if mname not in unsafe_methods and mname in cf_method_shards:
        _om_add(mname, *_union_shard_deps(cf_method_shards[mname]))

# Record inline_opt + remaining pure TestOptimizer methods (including test_qualify_columns!)
b_inl = get_builder("inline_opt")
inl_bad = set()
inl_sigs = {{}}
inl_depth = [0]

orig_topt_po = topt.parse_one
orig_topt_opt = topt.optimizer.optimize
orig_topt_q = topt.qualify
orig_mod_q = topt.optimizer.qualify.qualify
orig_mod_qc = topt.optimizer.qualify_columns.qualify_columns

def _inl_record(key, p_deps, p_sig, o_deps, out_expr, pf, of):
    if key in inl_bad or not isinstance(out_expr, exp.Expr):
        return
    osig = rc.ast_struct_sig(out_expr, exp)
    if osig is None:
        inl_bad.add(key); b_inl["rows"].pop(key, None); return
    prev = inl_sigs.get(key)
    if prev is not None:
        if prev != osig:
            inl_bad.add(key); b_inl["rows"].pop(key, None)
        return
    pk = rc.pack_expr(out_expr, exp)
    if pk is rc._UNPACKABLE:
        inl_bad.add(key); return
    ast_idx = len(b_inl["asts"]); b_inl["asts"].append(pk)
    rid = b_inl["next_rid"]; b_inl["next_rid"] = rid + 1
    p_idx = intern_dep(b_inl, *p_deps) if p_deps is not None else -1
    o_idx = intern_dep(b_inl, *o_deps) if o_deps is not None else -1
    b_inl["rows"][key] = (p_idx, p_sig, o_idx, ast_idx, rid)
    inl_sigs[key] = osig
    if pf: add_fine(b_inl["pf"], pf, rid)
    if of: add_fine(b_inl["of"], of, rid)

def _rec_topt_po(sql, read=None, dialect=None, **kw):
    if inl_depth[0] != 0 or type(sql) is not str or kw:
        return orig_topt_po(sql, read=read, dialect=dialect, **kw)
    dk = rc.norm_dialect_key(read if read is not None else dialect)
    if dk is None:
        return orig_topt_po(sql, read=read, dialect=dialect)
    p_d, p_p, p_g, pf = set(), set(), set(), set()
    inl_depth[0] += 1
    rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
    try:
        res = orig_topt_po(sql, read=read, dialect=dialect)
    finally:
        rec_stack.pop(); fine_stack.pop(); inl_depth[0] -= 1
    ps = rc.ast_struct_sig(res, exp)
    if ps is not None:
        _inl_record(("P", sql, dk), (p_d, p_p, p_g), ps, None, res, pf, None)
    return res

def _rec_topt_opt(expression, *args, **kwargs):
    if inl_depth[0] != 0 or args or "rules" in kwargs:
        return orig_topt_opt(expression, *args, **kwargs)
    kw_h = rc.norm_opt_kwargs(kwargs)
    if kw_h is None:
        return orig_topt_opt(expression, *args, **kwargs)
    if type(expression) is str:
        d_arg = kwargs.get("dialect")
        p_d, p_p, p_g, pf = set(), set(), set(), set()
        inl_depth[0] += 1
        rec_stack.append((p_d, p_p, p_g)); fine_stack.append(pf)
        try:
            parsed = orig_topt_po(expression, read=d_arg)
        finally:
            rec_stack.pop(); fine_stack.pop()
        ps = rc.ast_struct_sig(parsed, exp)
        o_d, o_p, o_g, of = set(), set(), set(), set()
        rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
        try:
            res = orig_topt_opt(parsed, **kwargs)
        finally:
            rec_stack.pop(); fine_stack.pop(); inl_depth[0] -= 1
        if ps is not None:
            _inl_record(("OS", expression, kw_h), (p_d, p_p, p_g), ps, (o_d, o_p, o_g), res, pf, of)
        return res
    if isinstance(expression, exp.Expr) and kwargs.get("copy", True):
        isig = rc.ast_struct_sig(expression, exp)
        o_d, o_p, o_g, of = set(), set(), set(), set()
        inl_depth[0] += 1
        rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
        try:
            res = orig_topt_opt(expression, **kwargs)
        finally:
            rec_stack.pop(); fine_stack.pop(); inl_depth[0] -= 1
        if isig is not None:
            _inl_record(("OE", isig, kw_h), None, None, (o_d, o_p, o_g), res, None, of)
        return res
    return orig_topt_opt(expression, *args, **kwargs)

def _make_rec_q(orig_fn, tag):
    import functools as _ft
    @_ft.wraps(orig_fn)
    def _rec_q(expression, *args, **kwargs):
        if inl_depth[0] != 0 or args or not isinstance(expression, exp.Expr):
            return orig_fn(expression, *args, **kwargs)
        kw_h = rc.norm_opt_kwargs(kwargs)
        if kw_h is None:
            return orig_fn(expression, *args, **kwargs)
        isig = rc.ast_struct_sig(expression, exp)
        o_d, o_p, o_g, of = set(), set(), set(), set()
        inl_depth[0] += 1
        rec_stack.append((o_d, o_p, o_g)); fine_stack.append(of)
        try:
            res = orig_fn(expression, **kwargs)
        finally:
            rec_stack.pop(); fine_stack.pop(); inl_depth[0] -= 1
        if isig is not None:
            _inl_record((tag, isig, kw_h), None, None, (o_d, o_p, o_g), res, None, of)
        return res
    return _rec_q

_rec_q_fn = _make_rec_q(orig_mod_q, "QE")
_rec_qc_fn = _make_rec_q(orig_mod_qc, "QCE")

def _stub_cf_union(self, file, func, pretty=False, execute=None, only=None, **kwargs):
    if only is None and _cur_cf_m[0] is not None:
        cf_method_shards.setdefault(_cur_cf_m[0], set()).add(f"cf_{{file}}")
    elif only is not None:
        orig_cf(self, file, func, pretty=pretty, execute=execute, only=only, **kwargs)

topt.parse_one = _rec_topt_po
topt.optimizer.optimize = _rec_topt_opt
topt.qualify = _rec_q_fn
topt.optimizer.qualify.qualify = _rec_q_fn
topt.optimizer.qualify_columns.qualify_columns = _rec_qc_fn
topt.TestOptimizer.check_file = _stub_cf_union
topt.duckdb.connect = lambda *a, **kw: rc.LazyCachedDuckConn(b"EMPTY", lambda: orig_connect(config={{"threads": 1}}), cache_enabled=True) if (not a and not kw) else orig_connect(*a, **kw)

try:
    for mname in all_topt_methods:
        if mname in _SKIP_LIVE_OM or mname in pure_cf_only_methods or mname in om_methods:
            continue
        _cur_cf_m[0] = mname
        m_d, m_p, m_g, m_f = set(), set(), set(), set()
        rec_stack.append((m_d, m_p, m_g)); fine_stack.append(m_f)
        try:
            tc = topt.TestOptimizer(mname)
            res = unittest.TestResult()
            tc.run(res)
        finally:
            rec_stack.pop(); fine_stack.pop()
            _cur_cf_m[0] = None
        if res.wasSuccessful():
            if mname in cf_method_shards:
                ud, up, ug, uf = _union_shard_deps(cf_method_shards[mname])
                m_d.update(ud); m_p.update(up); m_g.update(ug); m_f.update(uf)
            _om_add(mname, m_d, m_p, m_g, m_f)
finally:
    topt.parse_one = orig_topt_po
    topt.optimizer.optimize = orig_topt_opt
    topt.qualify = orig_topt_q
    topt.optimizer.qualify.qualify = orig_mod_q
    topt.optimizer.qualify_columns.qualify_columns = orig_mod_qc
    topt.TestOptimizer.check_file = orig_cf
    topt.duckdb.connect = orig_connect
    topt.TestOptimizer.conn.close()

restore_rec()
restore_fine()

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
    fb_sz = rc._write_fine_tables(
        (b["pf"], b["of"], b["ff"], b["ef"]),
        opt_dir / f"{{safe_name}}.fdat",
        opt_dir / f"{{safe_name}}.ftoc",
    )
    tot_opt_rows += len(b["rows"])
    tot_opt_bytes += len(ib) + ast_p.stat().st_size + fb_sz

# Write opt_methods shard (whole-method direct-jump for TestOptimizer!)
if om_methods:
    om_ib = marshal.dumps((om_deps_list, om_methods))
    (opt_dir / "opt_methods.idx").write_bytes(om_ib)
    om_fb = rc._write_fine_tables((om_fine,), opt_dir / "opt_methods.fdat", opt_dir / "opt_methods.ftoc")
    tot_opt_bytes += len(om_ib) + om_fb

print(f"{{tot_opt_rows}} {{tot_opt_bytes}} {{len(queries_cache)}} {{len(dblob)}}")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"build_opt_replay_and_duckdb_cache failed ({res.returncode}):\n{res.stderr}\n{res.stdout}")
    parts = res.stdout.strip().splitlines()[-1].split()
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


def build_exec_and_sqlite_cache(
    sg_root: Path,
    exec_path: Path = EXEC_REPLAY_PATH,
    sqlite_path: Path = SQLITE_CACHE_PATH,
) -> tuple[int, int, int, int]:
    """Collect (1) Non-TPC TestExecutor execute() stage-reuse cache and (2) Lazy SQLite Query Cache."""
    code = f"""
import importlib, pickle, sqlite3, sys, unittest, zlib
from pathlib import Path
sys.path.insert(0, {str(sg_root)!r})
sys.path.insert(0, "/app")

import replay_cache as rc
import runner
from sqlglot import exp, parse_one
from sqlglot.executor.python import PythonExecutor
from sqlglot.executor.table import ensure_tables
from sqlglot.helper import dict_depth
from sqlglot.optimizer import optimize
from sqlglot.optimizer.annotate_types import annotate_types
from sqlglot.planner import Plan
from sqlglot.schema import ensure_schema, flatten_schema, nested_get, nested_set
import tests.test_executor as texec

# ---- Part A: Collect non-TPC TestExecutor execute() stage-reuse cache ----
exec_cache = {{}}
bad_exec = set()
orig_exec = texec.execute

def _rec_exec(sql, schema=None, dialect=None, tables=None):
    res_tbl = orig_exec(sql, schema=schema, dialect=dialect, tables=tables)
    k = rc.exec_call_key(sql, schema, tables, dialect)
    if k is not None and k not in bad_exec:
        try:
            tables_ = ensure_tables(tables, dialect=dialect)
            s_eff = schema
            if not s_eff:
                s_eff = {{}}
                flattened_tables = flatten_schema(tables_.mapping, depth=dict_depth(tables_.mapping))
                for keys in flattened_tables:
                    tbl = nested_get(tables_.mapping, *zip(keys, keys))
                    assert tbl is not None
                    for col in tbl.columns:
                        val = tbl[0][col]
                        ct = annotate_types(exp.convert(val), dialect=dialect).type or type(val).__name__
                        nested_set(s_eff, [*keys, col], ct)
            schema_obj = ensure_schema(s_eff, dialect=dialect)
            if tables_.supported_table_args and tables_.supported_table_args != schema_obj.supported_table_args:
                raise ValueError("table args mismatch")
            opt = optimize(sql, schema_obj, leave_tables_isolated=True, dialect=dialect)
            plan = Plan(opt)
            det_h = runner._hash_plan_det(plan)
            py_h = runner._hash_plan_py_code(plan)
            pk_opt = rc.pack_expr(opt, exp)
            if pk_opt is not rc._UNPACKABLE:
                val = (pk_opt, det_h, py_h, tuple(res_tbl.columns), list(res_tbl.rows))
                pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)
                if k in exec_cache and exec_cache[k][1:] != val[1:]:
                    bad_exec.add(k)
                    exec_cache.pop(k, None)
                else:
                    exec_cache[k] = val
        except Exception:
            bad_exec.add(k)
            exec_cache.pop(k, None)
    return res_tbl

texec.execute = _rec_exec
texec.TestExecutor.setUpClass = classmethod(lambda cls: setattr(cls, "cache", {{}}))
texec.TestExecutor.tearDownClass = classmethod(lambda cls: None)
suite = unittest.defaultTestLoader.loadTestsFromTestCase(texec.TestExecutor)
filt = unittest.TestSuite()
skip_e = {{"test_execute_tpcds", "test_execute_tpch", "test_optimized_tpch", "test_agg_order"}}
for tc in suite:
    if getattr(tc, "_testMethodName", "") not in skip_e:
        filt.addTest(tc)
res = unittest.TestResult()
filt.run(res)
if not res.wasSuccessful():
    raise RuntimeError(f"exec_replay collection failed: {{res.failures[:1]}} {{res.errors[:1]}}")
eblob = pickle.dumps(exec_cache, protocol=pickle.HIGHEST_PROTOCOL)
Path({str(exec_path)!r}).write_bytes(eblob)

# ---- Part B: Collect Lazy SQLite Query Cache (test_optimize_cross_joins & test_simplify_between & test_optimizer) ----
sq_queries = {{}}
sq_ddls = set()
sq_bad = set()

class _RecSqCursor:
    def __init__(self, real_cur, key):
        self._cur = real_cur
        self._key = key
        self.description = real_cur.description
    def fetchall(self):
        rows = self._cur.fetchall()
        if self._key is not None and self._key not in sq_bad:
            try:
                cols = tuple(d[0] for d in (self.description or ()))
                pickle.dumps((cols, rows), protocol=pickle.HIGHEST_PROTOCOL)
                prev = sq_queries.get(self._key)
                if prev is None:
                    sq_queries[self._key] = (cols, rows)
                elif prev != (cols, rows):
                    sq_bad.add(self._key)
                    sq_queries.pop(self._key, None)
            except Exception:
                sq_bad.add(self._key)
                sq_queries.pop(self._key, None)
        return rows
    def fetchone(self): return self._cur.fetchone()
    def __getattr__(self, name): return getattr(self._cur, name)

class _RecSqConn:
    def __init__(self, real_conn):
        self._conn = real_conn
        self._sig = b"SQLITE"
    def execute(self, query, parameters=None, *a, **kw):
        if parameters is not None or a or kw or type(query) is not str:
            return self._conn.execute(query, parameters, *a, **kw) if parameters is not None else self._conn.execute(query, *a, **kw)
        if rc._is_pure_read_sql(query):
            return _RecSqCursor(self._conn.execute(query), (self._sig, query))
        cur = self._conn.execute(query)
        self._sig = rc._ddl_step_sig(self._sig, "E:" + query)
        sq_ddls.add(self._sig)
        return cur
    def executescript(self, script):
        cur = self._conn.executescript(script)
        if type(script) is str:
            self._sig = rc._ddl_step_sig(self._sig, "S:" + script)
            sq_ddls.add(self._sig)
        return cur
    def close(self): return self._conn.close()
    def __enter__(self): return self
    def __exit__(self, *ea): self.close()
    def __getattr__(self, name): return getattr(self._conn, name)

orig_sq_connect = sqlite3.connect
def _rec_sq_connect(db, *a, **kw):
    if db == ":memory:" and not a and not kw:
        return _RecSqConn(orig_sq_connect(":memory:"))
    return orig_sq_connect(db, *a, **kw)

for mod_n in ("tests.test_optimize_cross_joins", "tests.test_simplify_between"):
    m = importlib.import_module(mod_n)
    m.sqlite3 = type("_SqMod", (), {{"connect": staticmethod(_rec_sq_connect), "sqlite_version_info": sqlite3.sqlite_version_info}})()
    s = unittest.defaultTestLoader.loadTestsFromModule(m)
    r = unittest.TestResult()
    s.run(r)
    if not r.wasSuccessful():
        raise RuntimeError(f"sqlite cache failed in {{mod_n}}: {{r.failures[:1]}} {{r.errors[:1]}}")

sblob = pickle.dumps({{"queries": sq_queries, "ddls": sq_ddls}}, protocol=pickle.HIGHEST_PROTOCOL)
Path({str(sqlite_path)!r}).write_bytes(sblob)
print(f"{{len(exec_cache)}} {{len(eblob)}} {{len(sq_queries)}} {{len(sblob)}}")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"build_opt_replay_and_duckdb_cache failed ({res.returncode}):\n{res.stderr}\n{res.stdout}")
    parts = res.stdout.strip().splitlines()[-1].split()
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])

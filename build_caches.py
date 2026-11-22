"""Build-time parallel cache generator (runs inside Dockerfile, adds 0 bytes to git repo).

Precomputes across 4 cores:
Phase 1 (all 4 cores parallel, ~4.5s):
  - Core 1: Read-only DuckDB DBs (/app/duckdb_{tpcds,tpch,opt}.db)
  - Core 2: Preconverted /app/maps_raw/*, /app/compiled_roots.marshal, /app/fixture_cache.marshal,
            /app/base_ast_index/*, /app/guarded_isinstance.marshal, /app/tpc_tables/*
  - Core 3: Shared root-agnostic TPC-DS optimized ASTs (/app/tpc_opt_ast/*.ast) + master Plan pickles
            built ONCE in C-accelerated /sqlglot-c (eliminates 3x redundant optimize() runs!)
  - Core 4: 14-stage deduplicated TPC-DS optimizer checkpoints (/app/tpc_pq/{i}.ckp + {i}.ast),
            cross_joins_pq (/app/cross_joins_pq.marshal), and tok_trie_cache (/app/tok_trie_cache.marshal)
            built in C-accelerated /sqlglot-c

Phase 2 (4 cores parallel via isolated subprocesses — zero multi-thread fork hazards!):
  - Core 1 (dedicated critical path): opt_replay (with 14-stage ckps, p_ast_idx, om_opt_files) + duckdb_query_cache
  - Cores 2-4:
      2a. Parallel: exec_replay (Pure /sqlglot w/ PythonGenerator fine tokens) + sqlite_query_cache,
                    tpc_rows (/sqlglot-c), per-root TPC-DS Plans ({py,noopt,cb}_plan) -> tpc_pygen_fine.marshal
      2b. dep_index (n_workers=n_cpus-1, test_optimizer.py solo bucket + LPT remainder)
      2c. val_replay (n_workers=n_cpus, LPT-bucketed 1 spy install per worker + whole_methods)
"""

from __future__ import annotations

import ast
import concurrent.futures
import csv
import gzip
import marshal
import os
import pickle
import subprocess
import sys
import threading
import time
import zlib
from pathlib import Path


def _pick_fast_root() -> tuple[str, str]:
    if Path("/sqlglot-c/sqlglot").is_dir():
        return "/sqlglot-c", "c"
    return "/sqlglot", "py"


def _build_tpc_opt_master(sg_root: str, plan_out_dir: str) -> int:
    """Optimize all 57 execute TPC-DS queries ONCE in `sg_root`, saving root-agnostic
    marshal ASTs to /app/tpc_opt_ast/{i}.ast and Plans to `plan_out_dir/{i}.pkl`."""
    code = f"""
import marshal, pickle, sys
from pathlib import Path
sys.path.insert(0, {sg_root!r})
sys.path.insert(0, "/app")
import replay_cache as rc
import tests.helpers as th
from sqlglot import exp, parse_one
from sqlglot.optimizer import optimize
from sqlglot.planner import Plan
from sqlglot.schema import ensure_schema

ast_out = Path("/app/tpc_opt_ast")
ast_out.mkdir(parents=True, exist_ok=True)
plan_out = Path({plan_out_dir!r})
plan_out.mkdir(parents=True, exist_ok=True)
tpcds_sqls = list(th.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
s_ds = ensure_schema(th.TPCDS_SCHEMA)
cnt = 0
for i, (args, sql, _) in enumerate(tpcds_sqls, 1):
    if args.get("execute"):
        opt = optimize(parse_one(sql), s_ds, leave_tables_isolated=True)
        pk = rc.pack_expr(opt, exp)
        if pk is not rc._UNPACKABLE:
            (ast_out / f"{{i}}.ast").write_bytes(marshal.dumps(pk))
        pl = Plan(opt)
        (plan_out / f"{{i}}.pkl").write_bytes(pickle.dumps(pl, protocol=pickle.HIGHEST_PROTOCOL))
        cnt += 1
print(cnt)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"_build_tpc_opt_master({sg_root}) failed:\n{res.stderr}\n{res.stdout}")
    return int(res.stdout.strip().splitlines()[-1])


def _build_tpc_plans_from_ast(sg_root: str, plan_out_dir: str) -> int:
    """Reconstruct Plans for `sg_root` from root-agnostic /app/tpc_opt_ast/*.ast (~0.4s!)."""
    code = f"""
import marshal, pickle, sys
from pathlib import Path
sys.path.insert(0, {sg_root!r})
sys.path.insert(0, "/app")
import replay_cache as rc
from sqlglot import exp
from sqlglot.planner import Plan

plan_out = Path({plan_out_dir!r})
plan_out.mkdir(parents=True, exist_ok=True)
cnt = 0
for p in sorted(Path("/app/tpc_opt_ast").glob("*.ast"), key=lambda x: int(x.stem)):
    opt = rc.unpack_expr(marshal.loads(p.read_bytes()), exp)
    pl = Plan(opt)
    (plan_out / f"{{p.stem}}.pkl").write_bytes(pickle.dumps(pl, protocol=pickle.HIGHEST_PROTOCOL))
    cnt += 1
print(cnt)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"_build_tpc_plans_from_ast({sg_root}) failed:\n{res.stderr}\n{res.stdout}")
    return int(res.stdout.strip().splitlines()[-1])


def _build_tpc_pygen_fine(sg_root: Path, py_plan_dir: str) -> tuple[int, int, int]:
    """Precompute exact PythonGenerator fine tokens per TPC-DS execute query (/app/tpc_pygen_fine.marshal).
    MUST run on Pure-Python /sqlglot so _install_fine_spies intercepts PythonGenerator!"""
    code = f"""
import marshal, pickle, sys
from pathlib import Path
sys.path.insert(0, {str(sg_root)!r})
sys.path.insert(0, "/app")
import replay_cache as rc, runner

fine_stack = []
restore_fine = rc._install_fine_spies(fine_stack)
inv = {{}}
cnt = 0
try:
    for p in sorted(Path({py_plan_dir!r}).glob("*.pkl"), key=lambda x: int(x.stem)):
        qi = int(p.stem)
        plan = pickle.loads(p.read_bytes())
        qf = set()
        fine_stack.append(qf)
        try:
            runner._hash_plan_py_code(plan)
        finally:
            fine_stack.pop()
        for tok in qf:
            inv.setdefault(tok, set()).add(qi)
        cnt += 1
finally:
    restore_fine()

payload = {{tok: tuple(sorted(qs)) for tok, qs in inv.items()}}
blob = marshal.dumps(payload)
Path("/app/tpc_pygen_fine.marshal").write_bytes(blob)
print(f"{{cnt}} {{len(payload)}} {{len(blob)}}")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"_build_tpc_pygen_fine failed:\n{res.stderr}\n{res.stdout}")
    parts = res.stdout.strip().splitlines()[-1].split()
    return int(parts[0]), int(parts[1]), int(parts[2])


def _build_pq_cj_tok(sg_root: str) -> tuple[int, int, int, int, int]:
    """Build 14-stage TPC-DS checkpoints (tpc_pq/{i}.ckp + {i}.ast), cross_joins_pq,
    and tok_trie_cache in ONE C-accelerated subprocess."""
    code = f"""
import marshal, sqlite3, sys
from pathlib import Path
sys.path.insert(0, {sg_root!r})
sys.path.insert(0, "/app")
import replay_cache as rc, tests.helpers as th
from sqlglot import Dialect, Dialects, exp, parse_one
from sqlglot.optimizer import RULES, optimize
from sqlglot.schema import ensure_schema
from sqlglot.tokens import Tokenizer

# 1. tpc_pq (deduplicated 14-stage optimizer checkpoints for all 57 TPC-DS execute queries)
s_ds = ensure_schema(th.TPCDS_SCHEMA)
out_pq = Path("/app/tpc_pq")
out_pq.mkdir(parents=True, exist_ok=True)
n_pq = 0
for i, (args, sql, _) in enumerate(th.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"), 1):
    if args.get("execute"):
        cur = parse_one(sql)
        uniq_asts = []
        sig_to_idx = {{}}
        before_idxs = []
        stage_sigs = []
        ok = True
        for rule in RULES:
            bsig = rc.ast_struct_sig(cur, exp)
            if bsig is None:
                ok = False
                break
            bidx = sig_to_idx.get(bsig)
            if bidx is None:
                pk = rc.pack_expr(cur, exp)
                if pk is rc._UNPACKABLE:
                    ok = False
                    break
                bidx = len(uniq_asts)
                uniq_asts.append(pk)
                sig_to_idx[bsig] = bidx
            before_idxs.append(bidx)
            cur = optimize(cur, s_ds, rules=(rule,), leave_tables_isolated=True, copy=False)
            asig = rc.ast_struct_sig(cur, exp)
            if asig is None:
                ok = False
                break
            stage_sigs.append(asig)
        if ok:
            fsig = stage_sigs[-1]
            fidx = sig_to_idx.get(fsig)
            if fidx is None:
                pk = rc.pack_expr(cur, exp)
                if pk is not rc._UNPACKABLE:
                    fidx = len(uniq_asts)
                    uniq_asts.append(pk)
                    sig_to_idx[fsig] = fidx
                else:
                    ok = False
        if ok:
            (out_pq / f"{{i}}.ckp").write_bytes(
                marshal.dumps((uniq_asts, tuple(before_idxs), tuple(stage_sigs), fidx))
            )
            (out_pq / f"{{i}}.ast").write_bytes(marshal.dumps(uniq_asts[before_idxs[2]]))
            n_pq += 1

# 2. cross_joins_pq (post-RULES[:6] ASTs for 48 test_conditional_cross_joins queries)
schema = {{name: {{"a": "INT"}} for name in ("x", "y", "z")}}
out_cj = {{}}
for criteria in (
    "ON x.a = y.a",
    "ON x.a < y.a",
    "ON y.a = 3",
    "ON x.a = y.a OR y.a IS NULL",
    "ON FALSE",
    "ON NULL",
    "ON TRUE",
    "USING(a)",
):
    queries = [
        f"SELECT x.a, y.a FROM x CROSS JOIN y {{criteria}}",
        f"SELECT x.a, y.a, z.a FROM x CROSS JOIN y {{criteria}} LEFT JOIN z ON y.a = z.a",
    ]
    if sqlite3.sqlite_version_info >= (3, 39, 0):
        queries.append(
            f"SELECT x.a, y.a, z.a FROM x CROSS JOIN y {{criteria}} RIGHT JOIN z ON y.a = z.a"
        )
    for sql in queries:
        for dialect in ("sqlite", "mysql"):
            ast6 = optimize(sql, schema=schema, dialect=dialect, rules=RULES[:6])
            pk = rc.pack_expr(ast6, exp)
            if pk is not rc._UNPACKABLE:
                out_cj[(sql, dialect)] = pk
cj_blob = marshal.dumps(out_cj)
Path("/app/cross_joins_pq.marshal").write_bytes(cj_blob)

# 3. tok_trie_cache (Tokenizer._KEYWORD_TRIE for all 34 baseline dialects)
cache = {{}}
all_tok_cls = {{Tokenizer}}
for d in Dialects:
    di = Dialect.get_or_raise(d)
    for c in di.tokenizer_class.__mro__:
        if c.__module__.startswith("sqlglot.") and issubclass(c, Tokenizer):
            all_tok_cls.add(c)
for c in all_tok_cls:
    key = f"{{c.__module__}}.{{c.__qualname__}}"
    trie = getattr(c, "_KEYWORD_TRIE", None)
    if isinstance(trie, dict):
        try:
            marshal.dumps(trie)
            cache[key] = trie
        except Exception:
            pass
tok_blob = marshal.dumps(cache)
Path("/app/tok_trie_cache.marshal").write_bytes(tok_blob)
print(f"{{n_pq}} {{len(out_cj)}} {{len(cj_blob)}} {{len(cache)}} {{len(tok_blob)}}")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"_build_pq_cj_tok failed:\n{res.stderr}\n{res.stdout}")
    parts = res.stdout.strip().splitlines()[-1].split()
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])


def _build_tpc_rows(sg_root: str, plan_dir: str) -> int:
    """Precompute exact baseline PythonExecutor (columns, rows) for all 57 TPC-DS queries."""
    code = f"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, {sg_root!r})
sys.path.insert(0, "/app")
import runner, candidate_engine as ce, tests.helpers as th
from sqlglot.executor.python import PythonExecutor
from sqlglot.executor.table import Table, ensure_tables

runner._install_fast_python_executor(ce.CommitPlan(commit="cache", is_noop=False))
tables = {{}}
for tname in th.TPCDS_SCHEMA:
    ct, rs = pickle.loads(Path(f"/app/tpc_tables/tpcds/{{tname}}.pkl").read_bytes())
    tables[tname] = Table(columns=ct, rows=list(rs))
out = Path("/app/tpc_rows")
out.mkdir(parents=True, exist_ok=True)
cnt = 0
for p in sorted(Path({plan_dir!r}).glob("*.pkl"), key=lambda x: int(x.stem)):
    qi = int(p.stem)
    plan = pickle.loads(p.read_bytes())
    tbl = PythonExecutor(tables=ensure_tables(tables)).execute(plan)
    (out / f"{{qi}}.pkl").write_bytes(pickle.dumps((tuple(tbl.columns), list(tbl.rows)), protocol=pickle.HIGHEST_PROTOCOL))
    cnt += 1
print(cnt)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"_build_tpc_rows failed:\n{res.stderr}\n{res.stdout}")
    return int(res.stdout.strip().splitlines()[-1])


def _build_tpc_tables(sg_root: Path, th) -> tuple[int, int]:
    total_files = 0
    total_bytes = 0
    for tag, sub, schema in [
        ("tpcds", "tests/fixtures/optimizer/tpc-ds/", th.TPCDS_SCHEMA),
        ("tpch", "tests/fixtures/optimizer/tpc-h/", th.TPCH_SCHEMA),
    ]:
        out_dir = Path("/app/tpc_tables") / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        for table, col_types in schema.items():
            fn = sg_root / f"{sub}{table}.csv.gz"
            if not fn.exists():
                continue
            with open(fn, "rb") as f:
                is_gz = f.read(2) == b"\x1f\x8b"
            fh = gzip.open(fn, "rt", newline="") if is_gz else open(fn, encoding="utf-8", newline="")
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
            blob = pickle.dumps((col_types, rows), protocol=pickle.HIGHEST_PROTOCOL)
            (out_dir / f"{table}.pkl").write_bytes(blob)
            total_files += 1
            total_bytes += len(blob)
    return total_files, total_bytes


def _build_duckdb_files_subproc(sg_root: Path) -> None:
    """Build persistent read-only DuckDB databases in an isolated subprocess."""
    code = f"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, {str(sg_root)!r})
import duckdb
import tests.helpers as th
from sqlglot import transpile

sg_root = Path({str(sg_root)!r})
duck_res_dir = Path("/app/tpc_duck")
duck_res_dir.mkdir(parents=True, exist_ok=True)
for db_name, tag, sub, schema, fix_rel in [
    ("/app/duckdb_tpcds.db", "tpcds", "tests/fixtures/optimizer/tpc-ds/", th.TPCDS_SCHEMA, "optimizer/tpc-ds/tpc-ds.sql"),
    ("/app/duckdb_tpch.db", "tpch", "tests/fixtures/optimizer/tpc-h/", th.TPCH_SCHEMA, "optimizer/tpc-h/tpc-h.sql"),
]:
    p = Path(db_name)
    if p.exists():
        p.unlink()
    conn = duckdb.connect(str(p), config={{"threads": 1}})
    try:
        for table, col_types in schema.items():
            fn = sg_root / f"{{sub}}{{table}}.csv.gz"
            if fn.exists():
                conn.execute(
                    f"CREATE TABLE {{table}} AS SELECT * FROM READ_CSV('{{fn}}', delim='|', header=True, columns={{col_types}})"
                )
        conn.execute("CHECKPOINT")
        for qi, (qargs, qsql, _) in enumerate(th.load_sql_fixture_pairs(fix_rel), start=1):
            if tag == "tpcds" and not qargs.get("execute"):
                continue
            dsql = transpile(qsql, write="duckdb")[0]
            cur = conn.execute(dsql)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            (duck_res_dir / f"{{tag}}_{{qi}}.pkl").write_bytes(
                pickle.dumps((cols, rows), protocol=pickle.HIGHEST_PROTOCOL)
            )
    finally:
        conn.close()

opt_p = Path("/app/duckdb_opt.db")
if opt_p.exists():
    opt_p.unlink()
conn = duckdb.connect(str(opt_p), config={{"threads": 1}})
try:
    conn.execute(
        '''
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
        CHECKPOINT;
        '''
    )
finally:
    conn.close()
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"_build_duckdb_files_subproc failed:\n{res.stderr}\n{res.stdout}")


def _build_local_io_bundle(sg_root: Path) -> tuple[int, int, int, int, int, int, int, int, int, int, int]:
    """Preconvert maps_raw, cache compiled_roots, build base_ast_index + guarded_isinstance,
    pre-parse SQL fixtures (marshal), and pre-parse TPC CSV tables."""
    # 0. Pre-decompress & preconvert maps/*.zlib into /app/maps_raw/
    raw_root = Path("/app/maps_raw")
    raw_root.mkdir(parents=True, exist_ok=True)
    maps_src = Path("/app/maps")
    n_raw = 0
    if (maps_src / "core_maps.zlib").is_file():
        core_obj = marshal.loads(zlib.decompress((maps_src / "core_maps.zlib").read_bytes()))
        sm = core_obj.get("static_ast_meta")
        if isinstance(sm, dict) and not sm.get("_preconverted"):
            sm["rev_import_graph"] = {k: frozenset(v) for k, v in sm["rev_import_graph"].items()}
            cdg = sm["class_dep_graph"]
            sm["class_dep_graph"] = {
                "class_children": {k: frozenset(v) for k, v in cdg["class_children"].items()},
                "attr_copiers": {tuple(k): frozenset(tuple(x) for x in v) for k, v in cdg["attr_copiers"].items()},
                "class_aliases": {k: frozenset(v) for k, v in cdg["class_aliases"].items()},
                "class_copiers": {k: frozenset(v) for k, v in cdg["class_copiers"].items()},
                "attr_mutators": {k: frozenset(v) for k, v in cdg["attr_mutators"].items()},
                "all_class_body_refs": frozenset(cdg["all_class_body_refs"]),
            }
            sm["non_runtime_info"] = {
                k: (frozenset(v[0]), frozenset(v[1])) for k, v in sm["non_runtime_info"].items()
            }
            sm["import_time_called"] = {k: frozenset(v) for k, v in sm["import_time_called"].items()}
            sm["import_time_refs"] = {k: frozenset(v) for k, v in sm["import_time_refs"].items()}
            sm["_preconverted"] = True
        for _sk in ("dialect_meta_attrs", "expr_meta_attrs", "pure_enums"):
            if _sk in core_obj and not isinstance(core_obj[_sk], frozenset):
                core_obj[_sk] = frozenset(core_obj[_sk])
        if "strict_enum_attrs" in core_obj and isinstance(core_obj["strict_enum_attrs"], dict):
            core_obj["strict_enum_attrs"] = {k: frozenset(v) for k, v in core_obj["strict_enum_attrs"].items()}
        (raw_root / "core_maps.marshal").write_bytes(marshal.dumps(core_obj))
        n_raw += 1
    if (maps_src / "dict_maps.zlib").is_file():
        (raw_root / "dict_maps.marshal").write_bytes(zlib.decompress((maps_src / "dict_maps.zlib").read_bytes()))
        n_raw += 1
    if (maps_src / "cmp_agree.zlib").is_file():
        cmp_obj = frozenset(marshal.loads(zlib.decompress((maps_src / "cmp_agree.zlib").read_bytes())))
        (raw_root / "cmp_agree.marshal").write_bytes(marshal.dumps(cmp_obj))
        n_raw += 1
    for sub in ("subtests", "line_cov"):
        sdir = maps_src / sub
        if sdir.is_dir():
            ddir = raw_root / sub
            ddir.mkdir(parents=True, exist_ok=True)
            for zp in sdir.glob("*.zlib"):
                (ddir / (zp.stem + ".marshal")).write_bytes(zlib.decompress(zp.read_bytes()))
                n_raw += 1

    # Precompute compiled .so -> .py manifest for zero-rglob container startup!
    compiled_roots: dict[str, list[str]] = {}
    for root_str in ("/sqlglot-c", "/sqlglot-c-noopt", "/sqlglot-c-base"):
        r_p = Path(root_str)
        sg_c = r_p / "sqlglot"
        if sg_c.is_dir():
            out_py: list[str] = []
            for p in sg_c.rglob("*.so"):
                rel = p.relative_to(r_p).as_posix()
                stem_part = rel.split(".cpython-", 1)[0]
                if stem_part.endswith("__mypyc"):
                    continue
                out_py.append(stem_part + ".py")
            compiled_roots[root_str] = sorted(out_py)
    Path("/app/compiled_roots.marshal").write_bytes(marshal.dumps(compiled_roots))

    # 0a. Per-file Base AST Index for hunk-scoped incremental patch analysis
    import patch_analyzer as pa
    n_ast_f, n_ast_u, ast_bytes = pa.build_base_ast_index(sg_root)

    # 0c. Precompute _guarded_isinstance_exprs sidecar (/app/guarded_isinstance.marshal -> 53% planning CPU win!)
    import candidate_engine as ce
    core_maps_loaded = marshal.loads((raw_root / "core_maps.marshal").read_bytes())
    eng = ce.CandidateEngine(core_maps_loaded, [], sg_root)
    gi_out: dict[str, Any] = {}
    for attr, readers in core_maps_loaded.get("attr_readers", {}).items():
        if 1 <= len(readers) <= 4:
            gi_out[attr] = eng._guarded_isinstance_exprs(attr, readers)
    gi_blob = marshal.dumps(gi_out)
    Path("/app/guarded_isinstance.marshal").write_bytes(gi_blob)

    # 1. Pure-Python pre-parsed SQL fixtures (marshal)
    sys.path.insert(0, str(sg_root))
    import tests.helpers as th

    fixtures_root = sg_root / "tests/fixtures"
    fixture_pairs: dict[str, list[tuple[dict, str, str]]] = {}
    fixture_singles: dict[str, list[str]] = {}
    for sql_file in sorted(fixtures_root.rglob("*.sql")):
        rel_under = str(sql_file.relative_to(fixtures_root))
        try:
            singles = list(th.load_sql_fixtures(rel_under))
            if singles:
                fixture_singles[rel_under] = singles
        except Exception:
            pass
        try:
            pairs = list(th.load_sql_fixture_pairs(rel_under))
            if pairs:
                fixture_pairs[rel_under] = pairs
        except Exception:
            pass

    fix_blob = marshal.dumps({"fixture_pairs": fixture_pairs, "fixture_singles": fixture_singles})
    Path("/app/fixture_cache.marshal").write_bytes(fix_blob)

    # 2. Pure-Python pre-parsed TPC-DS & TPC-H CSV tables
    n_tbl, tbl_bytes = _build_tpc_tables(sg_root, th)
    return (
        n_raw, len(fixture_pairs), len(fixture_singles), len(fix_blob),
        n_tbl, tbl_bytes, n_ast_f, n_ast_u, ast_bytes, len(gi_out), len(gi_blob),
    )


def _run_dep_index_subproc(sg_root: Path, n_workers: int) -> tuple[int, int, int, int, int]:
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, "/app")
import dep_index as di
res = di.build_dep_index(Path({str(sg_root)!r}), n_workers={n_workers})
print(" ".join(str(x) for x in res))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"build_dep_index failed:\n{res.stderr}\n{res.stdout}")
    for ln in res.stdout.strip().splitlines()[:-1]:
        print(ln, flush=True)
    parts = res.stdout.strip().splitlines()[-1].split()
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])


def _run_val_replay_subproc(sg_root: Path, n_workers: int) -> tuple[int, int, int]:
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, "/app")
import replay_cache as rc
res = rc.build_val_replay(Path({str(sg_root)!r}), n_workers={n_workers})
print(" ".join(str(x) for x in res))
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"build_val_replay failed:\n{res.stderr}\n{res.stdout}")
    parts = res.stdout.strip().splitlines()[-1].split()
    return int(parts[0]), int(parts[1]), int(parts[2])


def main() -> None:
    t0 = time.perf_counter()
    sg_root = Path("/sqlglot")
    fast_root_str, master_tag = _pick_fast_root()
    master_plan_dir = f"/app/tpc_cache_{master_tag}_plan"
    n_cpus = min(4, max(1, os.cpu_count() or 4))

    # ---- PHASE 1: 4-core parallel independent prerequisites (~4.5s wall!) ----
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        f_duck = pool.submit(_build_duckdb_files_subproc, sg_root)
        f_master = pool.submit(_build_tpc_opt_master, fast_root_str, master_plan_dir)
        f_pq = pool.submit(_build_pq_cj_tok, fast_root_str)
        io_res = _build_local_io_bundle(sg_root)
        f_duck.result()
        n_master = f_master.result()
        n_pq, n_cj, cj_bytes, n_tok, tok_bytes = f_pq.result()

    (
        n_raw, n_fix_p, n_fix_s, fix_bytes,
        n_tbl, tbl_bytes, n_ast_f, n_ast_u, ast_bytes, n_gi, gi_bytes,
    ) = io_res
    t_p1 = time.perf_counter() - t0

    # ---- PHASE 2: Overlap 1-core critical-path opt_replay with multi-core dep_index + val_replay + fast caches ----
    import replay_cache as rc

    opt_done = threading.Event()

    def _run_opt_replay():
        try:
            return rc.build_opt_replay_and_duckdb_cache(sg_root)
        finally:
            opt_done.set()

    def _run_secondary_pipeline():
        other_roots: list[tuple[str, str, str]] = []
        for tag, r_str in (
            ("py", "/sqlglot"),
            ("c", "/sqlglot-c"),
            ("noopt", "/sqlglot-c-noopt"),
            ("cb", "/sqlglot-c-base"),
        ):
            if tag == master_tag:
                continue
            if Path(r_str, "sqlglot").is_dir():
                other_roots.append((tag, r_str, f"/app/tpc_cache_{tag}_plan"))

        def _build_other_plans_and_pygen() -> tuple[dict[str, int], tuple[int, int, int]]:
            out_cnt: dict[str, int] = {master_tag: n_master}
            for tag, r_str, pdir in other_roots:
                out_cnt[tag] = _build_tpc_plans_from_ast(r_str, pdir)
            pygen_res = _build_tpc_pygen_fine(sg_root, "/app/tpc_cache_py_plan")
            return out_cnt, pygen_res

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as sub_pool:
            # MUST run exec_replay on Pure-Python sg_root so _install_fine_spies intercepts PythonGenerator!
            f_exec = sub_pool.submit(rc.build_exec_and_sqlite_cache, sg_root)
            f_rows = sub_pool.submit(_build_tpc_rows, fast_root_str, master_plan_dir)
            f_plans = sub_pool.submit(_build_other_plans_and_pygen)
            exec_res = f_exec.result()
            n_rows_r = f_rows.result()
            plan_counts, pygen_res = f_plans.result()

        # Step 2b: dep_index (n_cpus-1 while opt_replay runs, or n_cpus if opt_replay already finished)
        dep_w = n_cpus if opt_done.is_set() else max(1, n_cpus - 1)
        dep_res = _run_dep_index_subproc(sg_root, dep_w)

        # Step 2c: val_replay (LPT-bucketed 1 spy install per worker, grabs all 4 cores seamlessly)
        val_res = _run_val_replay_subproc(sg_root, n_cpus)
        return exec_res, n_rows_r, plan_counts, pygen_res, dep_res, val_res

    t_p2 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p2_pool:
        f_opt = p2_pool.submit(_run_opt_replay)
        f_sec = p2_pool.submit(_run_secondary_pipeline)
        orows, obytes, dq, dbytes = f_opt.result()
        exec_res, n_rows, plan_counts, pygen_res, dep_res, val_res = f_sec.result()

    erows, ebytes, sq, sbytes = exec_res
    n_pyg_q, n_pyg_toks, pyg_bytes = pygen_res
    d_seen, d_keep, d_attrs, d_dicts, d_bytes = dep_res
    vp, ve, vb = val_res

    print(
        f"[build_caches] raw_maps={n_raw}, base_ast={n_ast_f}f/{n_ast_u}u ({ast_bytes/1024:.0f} KB), "
        f"guarded_is={n_gi} ({gi_bytes/1024:.1f} KB), tok_tries={n_tok} ({tok_bytes/1024:.0f} KB), "
        f"fixtures={n_fix_p}p/{n_fix_s}s ({fix_bytes/1024:.1f} KB), "
        f"tpc_tables={n_tbl} ({tbl_bytes/1024:.1f} KB), "
        f"tpc_plans py={plan_counts.get('py',0)} c={plan_counts.get('c',0)} noopt={plan_counts.get('noopt',0)} cb={plan_counts.get('cb',0)} "
        f"rows={n_rows} pq={n_pq} cj={n_cj} ({cj_bytes/1024:.0f} KB) pygen={n_pyg_q}q/{n_pyg_toks}t ({pyg_bytes/1024:.1f} KB), "
        f"dep_index={d_seen}t/{d_keep}k ({d_bytes/1024:.0f} KB), "
        f"val_replay={vp}p/{ve}e ({vb/1024:.0f} KB), opt_replay={orows}r ({obytes/1024:.0f} KB), "
        f"duckdb={dq}q ({dbytes/1024:.1f} KB), exec_replay={erows}r ({ebytes/1024:.0f} KB), sqlite={sq}q ({sbytes/1024:.1f} KB) "
        f"[p1={t_p1:.1f}s p2={time.perf_counter()-t_p2:.1f}s] total={time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

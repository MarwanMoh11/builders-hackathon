"""Build-time cache generator (runs inside Dockerfile, adds 0 bytes to git repo).

Precomputes:
1. /app/fixture_cache.pkl.zlib: pure-Python pre-parsed SQL fixtures (zero sqlglot imports;
   loaded safely in parent BEFORE os.fork() so every CoW child reads fixtures in 0ms).
2. /app/tpc_tables/{tpcds,tpch}/<table>.pkl: pure-Python pre-parsed CSV rows for all
   TPC-DS (24) and TPC-H (8) tables (eliminates gzip + csv.reader + ast.literal_eval!).
3. /app/tpc_cache_py/{qi}.pkl (and /app/tpc_cache_c/{qi}.pkl if /sqlglot-c exists):
   per-query uncompressed baseline optimized ASTs + Plans so children only unpickle
   the exact queries they actually touch (~2ms/query vs ~950ms monolithic decompress!).
"""

from __future__ import annotations

import ast
import csv
import gzip
import pickle
import subprocess
import sys
import time
import zlib
from pathlib import Path


def _build_tpc_cache_for_root(sg_root: str, out_dir: str) -> int:
    code = f"""
import pickle, sys
from pathlib import Path
sys.path.insert(0, {sg_root!r})
from sqlglot import exp, parse_one
from sqlglot.optimizer import optimize
from sqlglot.planner import Plan
from sqlglot.schema import ensure_schema
import tests.helpers as th

def pack_expr(x):
    if isinstance(x, exp.Expression):
        return (
            "E",
            type(x),
            {{k: pack_expr(v) for k, v in x.args.items()}},
            pack_expr(x._type),
            x._meta,
            x.comments,
        )
    if isinstance(x, list):
        return [pack_expr(v) for v in x]
    if isinstance(x, tuple):
        return ("T", tuple(pack_expr(v) for v in x))
    return x

out = Path({out_dir!r})
out.mkdir(parents=True, exist_ok=True)
tpcds_sqls = list(th.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
s_ds = ensure_schema(th.TPCDS_SCHEMA)
cnt = 0
for i, (args, sql, _) in enumerate(tpcds_sqls, 1):
    if args.get("execute"):
        opt = optimize(parse_one(sql), s_ds, leave_tables_isolated=True)
        pl = Plan(opt.copy())
        (out / f"{{i}}.pkl").write_bytes(pickle.dumps((pack_expr(opt), pl), protocol=pickle.HIGHEST_PROTOCOL))
        cnt += 1
print(cnt)
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return int(res.stdout.strip().splitlines()[-1])


def _build_tpc_tables(sg_root: Path) -> tuple[int, int]:
    sys.path.insert(0, str(sg_root))
    import tests.helpers as th

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


def _build_duckdb_files(sg_root: Path) -> None:
    """Build persistent read-only DuckDB databases for TPC-DS, TPC-H, and TestOptimizer
    so tests connect in ~1ms with ZERO gzip CSV parsing at runtime!"""
    import duckdb
    import tests.helpers as th

    for db_name, sub, schema in [
        ("/app/duckdb_tpcds.db", "tests/fixtures/optimizer/tpc-ds/", th.TPCDS_SCHEMA),
        ("/app/duckdb_tpch.db", "tests/fixtures/optimizer/tpc-h/", th.TPCH_SCHEMA),
    ]:
        p = Path(db_name)
        if p.exists():
            p.unlink()
        conn = duckdb.connect(str(p), config={"threads": 1})
        try:
            for table, col_types in schema.items():
                fn = sg_root / f"{sub}{table}.csv.gz"
                if fn.exists():
                    conn.execute(
                        f"CREATE TABLE {table} AS SELECT * FROM READ_CSV('{fn}', delim='|', header=True, columns={col_types})"
                    )
            conn.execute("CHECKPOINT")
        finally:
            conn.close()

    opt_p = Path("/app/duckdb_opt.db")
    if opt_p.exists():
        opt_p.unlink()
    conn = duckdb.connect(str(opt_p), config={"threads": 1})
    try:
        conn.execute(
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
            CHECKPOINT;
            """
        )
    finally:
        conn.close()


def main() -> None:
    t0 = time.perf_counter()
    sg_root = Path("/sqlglot")
    sys.path.insert(0, str(sg_root))
    import tests.helpers as th

    # 1. Pure-Python pre-parsed SQL fixtures
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

    fix_blob = zlib.compress(
        pickle.dumps(
            {"fixture_pairs": fixture_pairs, "fixture_singles": fixture_singles},
            protocol=pickle.HIGHEST_PROTOCOL,
        ),
        6,
    )
    Path("/app/fixture_cache.pkl.zlib").write_bytes(fix_blob)

    # 2. Pure-Python pre-parsed TPC-DS & TPC-H CSV tables
    n_tbl, tbl_bytes = _build_tpc_tables(sg_root)

    # 3. Per-query uncompressed TPC-DS AST & Plan caches for /sqlglot, /sqlglot-c, and /sqlglot-c-base
    n_py = _build_tpc_cache_for_root("/sqlglot", "/app/tpc_cache_py")
    n_c = 0
    if Path("/sqlglot-c/sqlglot").is_dir():
        n_c = _build_tpc_cache_for_root("/sqlglot-c", "/app/tpc_cache_c")
    n_cb = 0
    if Path("/sqlglot-c-base/sqlglot").is_dir():
        n_cb = _build_tpc_cache_for_root("/sqlglot-c-base", "/app/tpc_cache_cb")

    # 4. Pre-built read-only DuckDB databases (~1ms connect, 0 CSV parsing at runtime!)
    _build_duckdb_files(sg_root)

    # 5. Precompute exact baseline PythonExecutor result (columns, rows) for all 57 TPC-DS queries
    #    so duckdb-only commits skip PythonExecutor completely (~300x faster)!
    rows_code = """
import pickle, sys
from pathlib import Path
sys.path.insert(0, "/sqlglot")
sys.path.insert(0, "/app")
import runner, candidate_engine as ce, tests.helpers as th
from sqlglot.executor.python import PythonExecutor
from sqlglot.executor.table import Table, ensure_tables

runner._install_fast_python_executor(ce.CommitPlan(commit="cache", is_noop=False))
tables = {}
for tname in th.TPCDS_SCHEMA:
    ct, rs = pickle.loads(Path(f"/app/tpc_tables/tpcds/{tname}.pkl").read_bytes())
    tables[tname] = Table(columns=ct, rows=list(rs))
out = Path("/app/tpc_rows")
out.mkdir(parents=True, exist_ok=True)
cnt = 0
for p in sorted(Path("/app/tpc_cache_py").glob("*.pkl")):
    qi = int(p.stem)
    _, plan = pickle.loads(p.read_bytes())
    tbl = PythonExecutor(tables=ensure_tables(tables)).execute(plan)
    (out / f"{qi}.pkl").write_bytes(pickle.dumps((tuple(tbl.columns), list(tbl.rows)), protocol=pickle.HIGHEST_PROTOCOL))
    cnt += 1
print(cnt)
"""
    res_r = subprocess.run([sys.executable, "-c", rows_code], capture_output=True, text=True, check=True)
    n_rows = int(res_r.stdout.strip().splitlines()[-1])

    # 5b. Precompute post-qualify+pushdown_projections ASTs for all 57 TPC-DS execute queries
    #     so late-stage optimizer commits (RULES[2:] & typing/*) skip ~70% of optimize() CPU!
    pq_code = """
import marshal, sys
from pathlib import Path
sys.path.insert(0, "/sqlglot")
sys.path.insert(0, "/app")
import replay_cache as rc, tests.helpers as th
from sqlglot import exp, parse_one
from sqlglot.optimizer import RULES, optimize
from sqlglot.schema import ensure_schema

s_ds = ensure_schema(th.TPCDS_SCHEMA)
out = Path("/app/tpc_pq")
out.mkdir(parents=True, exist_ok=True)
cnt = 0
for i, (args, sql, _) in enumerate(th.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"), 1):
    if args.get("execute"):
        pq = optimize(parse_one(sql), s_ds, rules=RULES[:2], leave_tables_isolated=True)
        pk = rc.pack_expr(pq, exp)
        if pk is not rc._UNPACKABLE:
            (out / f"{i}.ast").write_bytes(marshal.dumps(pk))
            cnt += 1
print(cnt)
"""
    res_pq = subprocess.run([sys.executable, "-c", pq_code], capture_output=True, text=True, check=True)
    n_pq = int(res_pq.stdout.strip().splitlines()[-1])

    # 6. Build-time Exact Dep Index (Wave 2A), Fine-Grained Validator/Optimizer Replay (Wave 1+2B),
    #    Executor Replay, Lazy DuckDB Cache, and Lazy SQLite Cache (Wave 2C)!
    t_rc = time.perf_counter()
    import dep_index as di
    import replay_cache as rc
    d_seen, d_keep, d_attrs, d_dicts, d_bytes = di.build_dep_index(sg_root)
    vp, ve, vb = rc.build_val_replay(sg_root)
    orows, obytes, dq, dbytes = rc.build_opt_replay_and_duckdb_cache(sg_root)
    erows, ebytes, sq, sbytes = rc.build_exec_and_sqlite_cache(sg_root)

    print(
        f"[build_caches] fixtures={len(fixture_pairs)}p/{len(fixture_singles)}s ({len(fix_blob)/1024:.1f} KB), "
        f"tpc_tables={n_tbl} ({tbl_bytes/1024:.1f} KB), tpc_cache py={n_py} c={n_c} cb={n_cb} rows={n_rows} pq={n_pq}, "
        f"dep_index={d_seen}t/{d_keep}k ({d_bytes/1024:.0f} KB), "
        f"val_replay={vp}p/{ve}e ({vb/1024:.0f} KB), opt_replay={orows}r ({obytes/1024:.0f} KB), "
        f"duckdb={dq}q ({dbytes/1024:.1f} KB), exec_replay={erows}r ({ebytes/1024:.0f} KB), sqlite={sq}q ({sbytes/1024:.1f} KB) "
        f"[w1+w2={time.perf_counter()-t_rc:.1f}s] total={time.perf_counter() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

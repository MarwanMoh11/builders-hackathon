"""Build-time cache generator (runs inside Dockerfile, adds 0 bytes to git repo).

Precomputes TWO separate caches:
1. /app/fixture_cache.pkl.zlib: pure-Python pre-parsed SQL fixtures (zero sqlglot imports!
   Loaded safely in parent BEFORE os.fork() so every CoW child reads fixtures in 0ms).
2. /app/tpc_cache.pkl.zlib: bit-exact baseline optimized ASTs and execution Plans for all
   57 executable TPC-DS queries (loaded lazily inside children AFTER /sqlglot is on sys.path).
"""

from __future__ import annotations

import pickle
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, "/sqlglot")

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
            {k: pack_expr(v) for k, v in x.args.items()},
            pack_expr(x._type),
            x._meta,
            x.comments,
        )
    if isinstance(x, list):
        return [pack_expr(v) for v in x]
    if isinstance(x, tuple):
        return ("T", tuple(pack_expr(v) for v in x))
    return x


def main() -> None:
    t0 = time.perf_counter()
    tpcds_sqls = list(th.load_sql_fixture_pairs("optimizer/tpc-ds/tpc-ds.sql"))
    s_ds = ensure_schema(th.TPCDS_SCHEMA)
    tpcds_cache: dict[int, tuple] = {}
    for i, (args, sql, _) in enumerate(tpcds_sqls, 1):
        if args.get("execute"):
            opt = optimize(parse_one(sql), s_ds, leave_tables_isolated=True)
            pl = Plan(opt.copy())
            tpcds_cache[i] = (pack_expr(opt), pl)

    tpc_blob = zlib.compress(pickle.dumps({"tpcds": tpcds_cache}, protocol=pickle.HIGHEST_PROTOCOL), 6)
    Path("/app/tpc_cache.pkl.zlib").write_bytes(tpc_blob)

    fixtures_root = Path("/sqlglot/tests/fixtures")
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
    print(
        f"[build_caches] cached {len(tpcds_cache)} TPC-DS ({len(tpc_blob)/1024:.1f} KB) + "
        f"{len(fixture_pairs)} pair / {len(fixture_singles)} single fixtures ({len(fix_blob)/1024:.1f} KB) "
        f"in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Build-time cache generator (runs inside Dockerfile, adds 0 bytes to git repo).

Precomputes bit-exact baseline optimized ASTs and execution Plans for all executable
TPC-DS queries so planner/executor commits skip redundant parse_one()/optimize()/Plan().
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

    blob = zlib.compress(pickle.dumps({"tpcds": tpcds_cache}, protocol=pickle.HIGHEST_PROTOCOL), 6)
    out = Path("/app/tpc_cache.pkl.zlib")
    out.write_bytes(blob)
    print(
        f"[build_caches] cached {len(tpcds_cache)} TPC-DS queries in {time.perf_counter() - t0:.2f}s "
        f"({len(blob) / 1024:.1f} KB)",
        flush=True,
    )


if __name__ == "__main__":
    main()

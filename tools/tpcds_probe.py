import sys, os, time
sys.path.insert(0, "/opt/sqlglot"); os.chdir("/opt/sqlglot")
import concurrent.futures
sys.path.insert(0, "/app/br")
import engine
concurrent.futures.ProcessPoolExecutor = engine.InlineExecutor
from tests import test_executor as te
from sqlglot import parse_one, transpile
watch = {("sqlglot/dialects/dialect.py", 1280), ("sqlglot/expressions/datatypes.py", 435),
         ("sqlglot/executor/context.py", 53), ("sqlglot/executor/context.py", 54)}
watch |= {(a, b) for a, b in [tuple(x.split(":")) for x in sys.argv[1:]] for b in [int(b)]} if len(sys.argv) > 1 else set()
mon = sys.monitoring; T = mon.COVERAGE_ID; mon.use_tool_id(T, "p")
hit = set()
root = "/opt/sqlglot/"
def cb(code, line):
    f = code.co_filename
    if f.startswith(root) and (f[len(root):], line) in watch: hit.add((f[len(root):], line))
    return mon.DISABLE
mon.register_callback(T, mon.events.LINE, cb)
t0 = time.process_time()
te.TestExecutor.setUpClass()
print("setUpClass", round(time.process_time() - t0, 2), file=sys.stderr)
cls = te.TestExecutor
counts = {w: 0 for w in watch}
t0 = time.process_time()
for i, (args, sql, _) in enumerate(cls.tpcds_sqls):
    hit.clear(); mon.restart_events(); mon.set_events(T, mon.events.LINE)
    expr = parse_one(sql)
    te.initializer(te.TPCDS_SCHEMA, cls.tpcds_tables)
    table = te.mp_execute(expr, args)
    transpile(sql, write="duckdb")
    mon.set_events(T, 0)
    for w in hit: counts[w] += 1
print("queries", len(cls.tpcds_sqls), "cpu", round(time.process_time() - t0, 2), file=sys.stderr)
for w, c in sorted(counts.items()): print(w, c, file=sys.stderr)

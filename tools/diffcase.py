import json, sys, os
sys.path.insert(0, "/app/br")
import engine
from candidates import CoverageMap, plan_commit
from valdiff import diff_modules
tests = json.load(open("/data/tests.json"))["tests"]
cov = CoverageMap("/app/br/map.json.gz")
plan = plan_commit("x", open(sys.argv[1], encoding="utf-8").read(), "/opt/sqlglot", cov, {})
engine.setup_parent(tests)
sources = {rel: new for rel, (_o, new, _a) in plan.py_files.items()}
old, new, failed = engine.reimport(sources)
print("purged", len(old), "failed", failed, file=sys.stderr)
attrs, globs, missing = diff_modules(old, new)
print("attrs", sorted(attrs), file=sys.stderr)
for mod, g in sorted(globs):
    o = vars(old[mod]).get(g, "<missing>"); n = vars(new[mod]).get(g, "<missing>")
    print("global", mod, g, type(o).__name__, repr(o)[:100], "|", type(n).__name__, repr(n)[:100], file=sys.stderr)
    if isinstance(o, dict) and isinstance(n, dict):
        from valdiff import Comparer
        c = Comparer()
        print("   only old:", sorted(set(map(str,o))-set(map(str,n)))[:10], "only new:", sorted(set(map(str,n))-set(map(str,o)))[:10], len(o), len(n), file=sys.stderr)
        for k in o:
            if k not in n: continue
            if not Comparer().eq(o[k], n[k]): print("   diff key", k, repr(o[k])[:80], repr(n[k])[:80], file=sys.stderr)

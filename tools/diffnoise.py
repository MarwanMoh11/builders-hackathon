import json, sys, os, time
sys.path.insert(0, "/app/br")
import engine
from valdiff import diff_modules, Comparer
tests = json.load(open("/data/tests.json"))["tests"]
engine.setup_parent(tests, import_tests=True)
t=time.process_time()
old, new, failed = engine.reimport({})
t1=time.process_time()
attrs, globs, missing = diff_modules(old, new)
t2=time.process_time()
print(f"reimport {t1-t:.2f}s diff {t2-t1:.2f}s failed={failed} attrs={len(attrs)} globals={len(globs)}", file=sys.stderr)
print(sorted(attrs)[:10], sorted(globs)[:20], file=sys.stderr)
# drill into one
k = sorted(attrs)[0] if attrs else None
if k:
    mod, _, qa = k.partition(":"); q, _, a = qa.rpartition(".")
    o = old[mod]; n = new[mod]
    for p in q.split("."): o = getattr(o, p); n = getattr(n, p)
    ov, nv = vars(o)[a], vars(n)[a]
    c = Comparer()
    def find(x, y, path, d=0):
        if d > 12: return
        if c.eq(x, y): return
        print("  diff at", path, type(x).__name__, repr(x)[:150], "|", repr(y)[:150], file=sys.stderr)
        if isinstance(x, dict):
            cy = {c.canon_key(kk): vv for kk, vv in y.items()}
            for kk, vv in x.items():
                ck = c.canon_key(kk)
                if ck in cy and not Comparer().eq(vv, cy[ck]):
                    find(vv, cy[ck], path + [repr(kk)[:60]], d + 1); return
            print("  keys differ", set(map(c.canon_key, x)) ^ set(cy), file=sys.stderr)
        elif isinstance(x, (list, tuple)):
            for i, (xx, yy) in enumerate(zip(x, y)):
                if not Comparer().eq(xx, yy): find(xx, yy, path + [i], d + 1); return
        else:
            import valdiff
            sx, sy = valdiff._state(x), valdiff._state(y)
            if sx is not None: find(sx, sy, path + ["<state>"], d + 1)
            elif hasattr(x, "__closure__"):
                print("  func", x.__code__.co_qualname, file=sys.stderr)
                for i,(cx,cy2) in enumerate(zip(x.__closure__ or (), y.__closure__ or ())):
                    if not Comparer().eq(cx.cell_contents, cy2.cell_contents): find(cx.cell_contents, cy2.cell_contents, path+[f"cell{i}"], d+1); return
    find(ov, nv, [k])

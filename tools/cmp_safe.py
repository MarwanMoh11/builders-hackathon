"""Decide, per comparison site, whether `is` and `==` (and `is not` / `!=`) must agree for every
operand type pair the traced test suite produced there. Writes the agreed sites."""
import enum, importlib, inspect, json, marshal, sys, zlib, collections
sys.path.insert(0, "/opt/sqlglot")
import sqlglot  # noqa: F401
import pkgutil
for i in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
    try: importlib.import_module(i.name)
    except Exception: pass

def resolve(name):
    mod, _, qual = name.rpartition(".")
    # qualname may be nested: walk from the longest importable module prefix
    parts = name.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except Exception:
            continue
        try:
            for p in parts[cut:]:
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    return None

IDENTITY_EQ = (object.__eq__, type.__eq__)
PRIMITIVE = {int, float, complex, str, bytes, bool, tuple, list, dict, set, frozenset, range, type(None)}

def identity_eq(t):
    return t.__eq__ in IDENTITY_EQ or (isinstance(t, type) and issubclass(t, enum.Enum) and t.__eq__ is enum.Enum.__eq__)

def none_safe(t):
    # does `x == None` give False (or fall back to identity) for every x of type t?
    if t is type(None) or identity_eq(t) or t in PRIMITIVE:
        return True
    if isinstance(t, type) and issubclass(t, enum.Enum):
        return all(c.__eq__ in IDENTITY_EQ or c in PRIMITIVE or c is enum.Enum or issubclass(c, enum.Enum)
                   for c in t.__mro__ if "__eq__" in vars(c))
    owner = next((c for c in t.__mro__ if "__eq__" in vars(c)), None)
    return owner is not None and f"{owner.__module__}.{owner.__qualname__}" in NONE_SAFE_EQ

# __eq__ implementations read by hand: each returns False / NotImplemented for None
NONE_SAFE_EQ = set(sys.argv[2:])

cache = {}
import builtins as _b, types as _types
_BUILTIN_TYPES = {f"builtins.{t.__name__}": t for t in vars(_types).values() if isinstance(t, type)}
_BUILTIN_TYPES.update({f"builtins.{n}": t for n, t in vars(_b).items() if isinstance(t, type)})
_BUILTIN_TYPES["builtins.NoneType"] = type(None)


def T(name):
    if name not in cache:
        cache[name] = _BUILTIN_TYPES.get(name) or resolve(name)
    return cache[name]

sites = json.load(open(sys.argv[1]))
agreed, unknown = [], collections.Counter()
for site, pairs in sites.items():
    ok = True
    for l, r in pairs:
        tl, tr = T(l), T(r)
        if tl is None or tr is None:
            ok = False; unknown[l if tl is None else r] += 1; break
        if tl is type(None) and tr is type(None):
            continue
        if tl is type(None) or tr is type(None):
            other = tr if tl is type(None) else tl
            if none_safe(other):
                continue
            ok = False; unknown["None vs " + f"{other.__module__}.{other.__qualname__}"] += 1; break
        if tl is tr and isinstance(tl, type) and issubclass(tl, enum.Enum):
            continue
        if identity_eq(tl) and identity_eq(tr):
            continue
        ok = False; break
    if ok:
        agreed.append(site)
open("/out/cmp_agree.zlib", "wb").write(zlib.compress(marshal.dumps(sorted(agreed)), 9))
print(f"{len(agreed)}/{len(sites)} sites provably agree", file=sys.stderr)
for k, v in unknown.most_common(12):
    print("  unresolved:", v, k, file=sys.stderr)
for s in ["sqlglot/dialects/dialect.py:1280:15", "sqlglot/executor/context.py:53:11"]:
    print(" ", s, s in agreed, file=sys.stderr)

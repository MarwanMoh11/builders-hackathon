"""Structural comparison of module state before and after re-importing patched code.

Used for import-time changes: after the patched modules are imported again, every class attribute
and module global is compared with its pristine counterpart. Objects are compared by what they
hold, not by identity, since a re-import makes new objects for everything.
"""

from __future__ import annotations

import enum
import functools
import re
import types

from pyanalyze import own_signature

_PRIMITIVES = (int, float, complex, str, bytes, bool, type(None), range, type(Ellipsis))


def _type_name(t) -> str:
    return f"{getattr(t, '__module__', '?')}.{getattr(t, '__qualname__', getattr(t, '__name__', '?'))}"


class Comparer:
    def __init__(self, max_depth: int = 60):
        self.memo: dict[tuple[int, int], bool] = {}
        self.max_depth = max_depth
        self.keep: list = []  # keep compared objects alive so ids stay unique

    def code_eq(self, a: types.CodeType, b: types.CodeType) -> bool:
        if own_signature(a) != own_signature(b):
            return False
        ka = [c for c in a.co_consts if isinstance(c, types.CodeType)]
        kb = [c for c in b.co_consts if isinstance(c, types.CodeType)]
        return len(ka) == len(kb) and all(self.code_eq(x, y) for x, y in zip(ka, kb))

    def canon_key(self, k):
        if isinstance(k, _PRIMITIVES):
            if isinstance(k, float) and k != k:
                return ("nan",)
            return (type(k).__name__, k)
        if isinstance(k, type):
            return ("type", _type_name(k))
        if isinstance(k, enum.Enum):
            return ("enum", _type_name(type(k)), k.name)
        if isinstance(k, tuple):
            return ("tuple",) + tuple(self.canon_key(x) for x in k)
        if isinstance(k, frozenset):
            return ("frozenset",) + tuple(sorted((self.canon_key(x) for x in k), key=repr))
        if isinstance(k, (types.FunctionType, types.BuiltinFunctionType)):
            return ("func", getattr(k, "__module__", ""), getattr(k, "__qualname__", ""))
        return ("obj", _type_name(type(k)), repr(k))

    def eq(self, a, b, depth: int = 0) -> bool:
        if a is b:
            return True
        ta, tb = type(a), type(b)
        if _type_name(ta) != _type_name(tb):
            return False
        if isinstance(a, _PRIMITIVES):
            if isinstance(a, float):
                return repr(a) == repr(b)
            return a == b
        if depth > self.max_depth:
            return True
        key = (id(a), id(b))
        known = self.memo.get(key)
        if known is not None:
            return known
        self.memo[key] = True  # assume equal while comparing (cycles)
        self.keep.append((a, b))
        result = self._eq(a, b, ta, depth + 1)
        self.memo[key] = result
        return result

    def _eq(self, a, b, t, depth) -> bool:
        if isinstance(a, type):
            return _type_name(a) == _type_name(b)
        if isinstance(a, enum.Enum):
            return a.name == b.name
        if isinstance(a, types.ModuleType):
            return a.__name__ == b.__name__
        if isinstance(a, (list, tuple)):
            if len(a) != len(b):
                return False
            if all(self.eq(x, y, depth) for x, y in zip(a, b)):
                return True
            # tuple(set_of_classes) iterates in id order, which differs between imports
            if all(isinstance(x, type) for x in a) and all(isinstance(y, type) for y in b):
                return sorted(map(_type_name, a)) == sorted(map(_type_name, b))
            return False
        if isinstance(a, dict):
            if len(a) != len(b):
                return False
            cb = {}
            for k, v in b.items():
                cb[self.canon_key(k)] = v
            for k, v in a.items():
                ck = self.canon_key(k)
                if ck not in cb or not self.eq(v, cb[ck], depth):
                    return False
            # Insertion order is not compared: dicts built from sets iterate in hash (id) order,
            # which differs between two imports of the very same code.
            return True
        if isinstance(a, (set, frozenset)):
            return len(a) == len(b) and {self.canon_key(x) for x in a} == {self.canon_key(x) for x in b}
        if isinstance(a, types.FunctionType):
            if not self.code_eq(a.__code__, b.__code__):
                return False
            if not self.eq(a.__defaults__, b.__defaults__, depth):
                return False
            if not self.eq(a.__kwdefaults__, b.__kwdefaults__, depth):
                return False
            ca, cb = a.__closure__ or (), b.__closure__ or ()
            if len(ca) != len(cb):
                return False
            for x, y in zip(ca, cb):
                try:
                    vx = x.cell_contents
                except ValueError:
                    vx = _EMPTY
                try:
                    vy = y.cell_contents
                except ValueError:
                    vy = _EMPTY
                if not self.eq(vx, vy, depth):
                    return False
            return True
        if isinstance(a, (types.BuiltinFunctionType, types.MethodDescriptorType, types.WrapperDescriptorType)):
            return getattr(a, "__qualname__", None) == getattr(b, "__qualname__", None) and self.eq(
                getattr(a, "__self__", None), getattr(b, "__self__", None), depth
            ) if not isinstance(getattr(a, "__self__", None), types.ModuleType) else (
                getattr(a, "__qualname__", None) == getattr(b, "__qualname__", None)
            )
        if isinstance(a, types.MethodType):
            return self.eq(a.__func__, b.__func__, depth) and self.eq(a.__self__, b.__self__, depth)
        if isinstance(a, (staticmethod, classmethod)):
            return self.eq(a.__func__, b.__func__, depth)
        if isinstance(a, property):
            return all(self.eq(getattr(a, n), getattr(b, n), depth) for n in ("fget", "fset", "fdel"))
        if isinstance(a, functools.partial):
            return self.eq(a.func, b.func, depth) and self.eq(a.args, b.args, depth) and self.eq(
                a.keywords, b.keywords, depth
            )
        if isinstance(a, re.Pattern):
            return a.pattern == b.pattern and a.flags == b.flags
        if getattr(t, "__module__", "") in ("typing", "types", "typing_extensions"):
            return repr(a) == repr(b)
        state_a = _state(a)
        state_b = _state(b)
        if state_a is None or state_b is None:
            if t.__eq__ is object.__eq__:
                # An opaque object (a lock, a C handle): nothing comparable inside.
                return True
            try:
                return bool(a == b)
            except Exception:  # noqa: BLE001
                return repr(a) == repr(b)
        return self.eq(state_a, state_b, depth)


class _Empty:
    pass


_EMPTY = _Empty()


def _state(obj):
    state = {}
    has = False
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        state.update(d)
        has = True
    for klass in type(obj).__mro__:
        for slot in getattr(klass, "__slots__", ()) or ():
            if isinstance(slot, str) and slot not in ("__dict__", "__weakref__"):
                try:
                    state["@" + slot] = getattr(obj, slot)
                except AttributeError:
                    state["@" + slot] = _EMPTY
                has = True
    return state if has else None


def classes_of(module: types.ModuleType):
    """(qualname, class) for classes defined in the module, nested classes included."""
    out = []
    for value in list(vars(module).values()):
        if isinstance(value, type) and value.__module__ == module.__name__:
            stack = [value]
            while stack:
                klass = stack.pop()
                if not klass.__qualname__.startswith(value.__qualname__):
                    continue
                out.append((klass.__qualname__, klass))
                for v in list(vars(klass).values()):
                    if (
                        isinstance(v, type)
                        and v.__module__ == module.__name__
                        and v.__qualname__.startswith(klass.__qualname__ + ".")
                    ):
                        stack.append(v)
    seen = set()
    unique = []
    for q, k in out:
        if q not in seen:
            seen.add(q)
            unique.append((q, k))
    return unique


def diff_modules(old: dict[str, types.ModuleType], new: dict[str, types.ModuleType]):
    """Return (changed class attributes as 'module:Qual.attr', changed globals as (module, name),
    modules missing after the re-import)."""
    comparer = Comparer()
    changed_attrs: set[str] = set()
    changed_globals: set[tuple[str, str]] = set()
    missing: list[str] = []
    for name, old_module in old.items():
        new_module = new.get(name)
        if new_module is None:
            missing.append(name)
            continue
        old_vars, new_vars = vars(old_module), vars(new_module)
        for gname in set(old_vars) | set(new_vars):
            if gname in ("__builtins__", "__loader__", "__spec__", "__cached__", "__file__"):
                continue
            ov = old_vars.get(gname, _EMPTY)
            nv = new_vars.get(gname, _EMPTY)
            submodule = f"{name}.{gname}"
            if getattr(ov, "__name__", None) == submodule or getattr(nv, "__name__", None) == submodule:
                continue  # a package attribute that a same-named submodule import rebinds
            if isinstance(ov, type) and getattr(ov, "__module__", None) == name:
                continue  # classes are compared attribute by attribute below
            if not comparer.eq(ov, nv):
                changed_globals.add((name, gname))
        old_classes = dict(classes_of(old_module))
        new_classes = dict(classes_of(new_module))
        for qual in set(old_classes) | set(new_classes):
            oc, nc = old_classes.get(qual), new_classes.get(qual)
            if oc is None or nc is None:
                changed_globals.add((name, qual.split(".")[0]))
                continue
            if [_type_name(b) for b in oc.__bases__] != [_type_name(b) for b in nc.__bases__]:
                changed_attrs.add(f"{name}:{qual}.__bases__")
            od, nd = vars(oc), vars(nc)
            for attr in set(od) | set(nd):
                if attr in ("__dict__", "__weakref__", "__doc__", "__firstlineno__", "__static_attributes__"):
                    continue
                ov, nv = od.get(attr, _EMPTY), nd.get(attr, _EMPTY)
                if isinstance(ov, type) and isinstance(nv, type):
                    continue  # nested classes are walked on their own
                if not comparer.eq(ov, nv):
                    changed_attrs.add(f"{name}:{qual}.{attr}")
    return changed_attrs, changed_globals, missing

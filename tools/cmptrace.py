"""pytest plugin: record the runtime operand types at every identity/equality comparison in sqlglot.

Load with `-p cmptrace` (tools/ on PYTHONPATH). For each `a is b`, `a is not b`, `a == b` and
`a != b` (single-operator comparisons) in sqlglot's own source, the operands are wrapped in a
recorder at import time, and the suite runs as usual. The result maps each comparison site
(file, line, column) to the set of (left type, right type) pairs the test suite ever produced
there. It lets the selector prove an `is` <-> `==` swap changes nothing for any test: e.g. when
every value compared to None at that site has an `__eq__` that treats None like identity.

Pool work runs in-process (with pickling) so every comparison is observed.
"""

from __future__ import annotations

import ast
import builtins
import importlib.abc
import importlib.machinery
import json
import os
import sys

ROOT = os.path.realpath(os.environ.get("SQLGLOT_ROOT", "/opt/sqlglot")) + os.sep
OUT = os.environ.get("BR_OUT", "/tmp/cmptrace")
WORKER = os.environ.get("PYTEST_XDIST_WORKER", "main")

SITES: dict[str, set] = {}


def _type_name(value) -> str:
    t = type(value)
    return f"{t.__module__}.{t.__qualname__}"


def _record(site: str, left, right):
    seen = SITES.get(site)
    if seen is None:
        seen = SITES[site] = set()
    seen.add((_type_name(left), _type_name(right)))


class _Wrap(ast.NodeTransformer):
    """a OP b  ->  br_cmp_hook_(site, a, b, OPCODE) for OP in is / is not / == / !=."""

    OPS = {ast.Is: 0, ast.IsNot: 1, ast.Eq: 2, ast.NotEq: 3}

    def __init__(self, rel: str):
        self.rel = rel

    def visit_Compare(self, node: ast.Compare):
        self.generic_visit(node)
        if len(node.ops) != 1 or type(node.ops[0]) not in self.OPS:
            return node
        site = f"{self.rel}:{node.lineno}:{node.col_offset}"
        call = ast.Call(
            func=ast.Name(id="br_cmp_hook_", ctx=ast.Load()),
            args=[ast.Constant(site), node.left, node.comparators[0], ast.Constant(self.OPS[type(node.ops[0])])],
            keywords=[],
        )
        return ast.copy_location(call, node)


CHECK_TWIN = os.environ.get("BR_TWIN") == "1"  # slow: evaluates == on deep trees
AGREE: dict[str, list] = {}  # site -> [executions, is/== agreed every time]


def br_cmp_hook_(site, left, right, op):
    _record(site, left, right)
    if op == 0:
        result = left is right
    elif op == 1:
        result = left is not right
    elif op == 2:
        result = left == right
    else:
        result = left != right
    if not CHECK_TWIN:
        return result
    # Would the identity/equality twin of this operator have given the same answer?
    try:
        if op in (0, 1):
            twin = (left == right) if op == 0 else (left != right)
        else:
            twin = (left is right) if op == 2 else (left is not right)
        agree = type(twin) is bool and type(result) is bool and twin == result
    except BaseException:  # noqa: BLE001 - a raising __eq__ disagrees
        agree = False
    entry = AGREE.get(site)
    if entry is None:
        entry = AGREE[site] = [0, True]
    entry[0] += 1
    if not agree:
        entry[1] = False
    return result


class _Loader(importlib.machinery.SourceFileLoader):
    def source_to_code(self, data, path, *, _optimize=-1):
        rel = os.path.realpath(path)[len(ROOT):]
        tree = ast.parse(data, filename=path)
        tree = ast.fix_missing_locations(_Wrap(rel).visit(tree))
        return compile(tree, path, "exec", dont_inherit=True)

    def get_code(self, fullname):
        source = self.get_data(self.path)
        return self.source_to_code(source, self.path)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == "sqlglot" or fullname.startswith("sqlglot.")):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            return spec
        spec.loader = _Loader(fullname, spec.origin)
        return spec


class InlineExecutor:
    def __init__(self, *args, initializer=None, initargs=(), **kwargs):
        import pickle

        if initializer is not None:
            initializer(*pickle.loads(pickle.dumps(initargs)))

    def submit(self, fn, *args, **kwargs):
        import pickle
        from concurrent.futures import Future

        future = Future()
        try:
            fn, args, kwargs = pickle.loads(pickle.dumps((fn, args, kwargs)))
            future.set_result(pickle.loads(pickle.dumps(fn(*args, **kwargs))))
        except BaseException as error:  # noqa: BLE001
            future.set_exception(error)
        return future

    def shutdown(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def pytest_configure(config):
    import concurrent.futures

    concurrent.futures.ProcessPoolExecutor = InlineExecutor
    builtins.br_cmp_hook_ = br_cmp_hook_
    sys.meta_path.insert(0, _Finder())


def pytest_unconfigure(config):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"cmp-{WORKER}.json"), "w") as f:
        json.dump(
            {site: {"types": sorted(map(list, pairs)), "agree": AGREE.get(site, [0, None])[1]}
             for site, pairs in SITES.items()},
            f,
        )

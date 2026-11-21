"""Blast Radius selector: for each patch, print the tests that fail on it, split across workers.

Plan (static, from the coverage map) -> run the candidate tests against the patched code in a
forked child -> report the ones that fail, plus expensive candidates we chose not to run.
"""

from __future__ import annotations

import glob
import json
import os
import select as _select
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = os.environ.get("BR_DATA", "/data")
MAP_PATH = os.environ.get("BR_MAP", os.path.join(HERE, "map.json.gz"))
COSTS_PATH = os.environ.get("BR_COSTS", os.path.join(HERE, "costs.json"))
PARALLEL = int(os.environ.get("BR_PARALLEL", "2"))
WORKER_MS = 2000
MAX_WORKERS = 8
GIANT_SECONDS = float(os.environ.get("BR_GIANT", "2.5"))  # never run tests costlier than this


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def pack(numbers, dur_ms):
    """Split tests over 1..8 workers (longest first onto the lightest), cheapest total."""
    items = sorted(set(numbers), key=lambda n: (-dur_ms[n], n))
    if not items:
        return 0, []
    best = None
    for k in range(1, min(MAX_WORKERS, len(items)) + 1):
        loads = [0] * k
        bins = [[] for _ in range(k)]
        for n in items:
            i = min(range(k), key=loads.__getitem__)
            loads[i] += dur_ms[n]
            bins[i].append(n)
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, bins)
    return best


# ------------------------------------------------------------------ deciding what to run


def split_run_blind(candidates, run_cost, dur_ms, never_run):
    """Choose tests to run (pay CPU, learn the truth) and tests to report unrun (pay their time)."""
    candidates = set(candidates)
    giants = {n for n in candidates if n in never_run or run_cost.get(n, dur_ms[n] / 1000) > GIANT_SECONDS}
    rest = candidates - giants
    if not giants:
        return rest, set()
    # With a giant reported anyway, other tests ride along on spare workers almost for free.
    blind_all, _ = pack(candidates, dur_ms)
    giant_only, _ = pack(giants, dur_ms)
    run_cpu_ms = sum(run_cost.get(n, dur_ms[n] / 1000) for n in rest) * 1000
    if blind_all <= giant_only + WORKER_MS + run_cpu_ms:
        return set(), candidates
    return rest, giants


# ------------------------------------------------------------------ the child


def child_main(plan, cov, run_cost, dur_ms, never_run, write_fd):
    import engine
    from pyanalyze import all_codes, code_lines
    from valdiff import diff_modules

    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    out = {"failing": [], "blind": [], "everything": False, "note": ""}
    try:
        engine.install_overlays(plan.overlays)
        candidates = set(plan.candidates)
        do_reimport = plan.reimport
        if not do_reimport:
            for rel, (_old, _new, analysis) in plan.py_files.items():
                if not engine.hot_patch(os.path.join(engine.ROOT, rel), analysis):
                    do_reimport = True
                    out["note"] += f"hot patch failed {rel}; "
                    break
        if do_reimport:
            sources = {rel: new for rel, (_old, new, _a) in plan.py_files.items()}
            old, new, failed = engine.reimport(sources)
            if failed:
                out["everything"] = True
                out["note"] += f"import failed {failed[:3]}; "
                raise SystemExit
            attrs, globals_, missing = diff_modules(old, new)
            out["note"] += f"attrs={sorted(attrs)[:6]} globals={sorted(globals_)[:6]}; "
            candidates |= _readers(attrs, globals_, old, cov, code_lines, all_codes)
        run, blind = split_run_blind(candidates, run_cost, dur_ms, never_run)
        failing, _cpu = engine.run_tests(run)
        out["failing"] = sorted(failing)
        out["blind"] = sorted(blind)
        out["ran"] = len(run)
    except SystemExit:
        pass
    except BaseException as error:  # noqa: BLE001
        out["everything"] = True
        out["note"] += f"child error {error!r}"
    data = json.dumps(out).encode()
    view = memoryview(data)
    while view:
        written = os.write(write_fd, view)
        view = view[written:]
    os.close(write_fd)
    os._exit(0)


def _relpath(filename):
    import engine

    root = engine.ROOT.rstrip("/") + "/"
    return filename[len(root):] if filename.startswith(root) else None


def _functions_of(module):
    import types

    name = module.__name__
    out = []
    stack = [v for v in vars(module).values()]
    seen = set()
    while stack:
        v = stack.pop()
        if id(v) in seen:
            continue
        seen.add(id(v))
        if isinstance(v, types.FunctionType):
            if v.__module__ == name:
                out.append(v)
        elif isinstance(v, (staticmethod, classmethod)):
            stack.append(v.__func__)
        elif isinstance(v, property):
            stack.extend(f for f in (v.fget, v.fset, v.fdel) if f)
        elif isinstance(v, type) and v.__module__ == name:
            stack.extend(vars(v).values())
    return out


def _readers(attrs, globals_, old_modules, cov, code_lines, all_codes):
    import types

    tests: set[int] = set()
    for key in attrs:
        mod, _, qual_attr = key.partition(":")
        qual, _, attr = qual_attr.rpartition(".")
        if key in cov.attrs:
            tests |= cov.attrs[key]
            continue
        module = old_modules.get(mod)
        klass = module
        for part in qual.split("."):
            klass = getattr(klass, part, None) if klass is not None else None
        value = vars(klass).get(attr) if isinstance(klass, type) else None
        funcs = []
        if isinstance(value, types.FunctionType):
            funcs = [value]
        elif isinstance(value, (staticmethod, classmethod)):
            funcs = [value.__func__]
        elif isinstance(value, property):
            funcs = [f for f in (value.fget, value.fset, value.fdel) if f]
        if funcs:
            for f in funcs:
                rel = _relpath(f.__code__.co_filename)
                if rel:
                    tests |= cov.tests_for_lines(rel, code_lines(f.__code__, nested=True))
        elif value is None or attr == "__bases__":
            # added or rebased: whoever uses the class at all
            prefix = f"{mod}:{qual}."
            for k, readers in cov.attrs.items():
                if k.startswith(prefix):
                    tests |= readers
    for mod, gname in globals_:
        module = old_modules.get(mod)
        if module is None:
            continue
        for func in _functions_of(module):
            if any(gname in c.co_names for c in all_codes(func.__code__)):
                rel = _relpath(func.__code__.co_filename)
                if rel:
                    tests |= cov.tests_for_lines(rel, code_lines(func.__code__, nested=True))
        prefix = f"{mod}:{gname}."
        for k, readers in cov.attrs.items():
            if k.startswith(prefix):
                tests |= readers
    return tests


# ------------------------------------------------------------------ main


def main():
    started = time.time()
    real_stdout = os.dup(1)
    os.dup2(2, 1)  # anything printed while importing goes to stderr
    sys.stdout = sys.stderr

    from candidates import CoverageMap, plan_commit

    tests = json.load(open(os.path.join(DATA, "tests.json")))["tests"]
    dur_ms = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
    tests_by_file: dict[str, list[int]] = {}
    for t in tests:
        tests_by_file.setdefault(t["file"], []).append(t["number"])
    all_tests = [t["number"] for t in tests]
    cov = CoverageMap(MAP_PATH)
    run_cost: dict[int, float] = {}
    never_run: set[int] = set()
    if os.path.exists(COSTS_PATH):
        costs = json.load(open(COSTS_PATH))
        number_by_id = {t["id"]: t["number"] for t in tests}
        for test_id, seconds in costs.get("cpu", {}).items():
            if test_id in number_by_id:
                run_cost[number_by_id[test_id]] = seconds
        never_run = {number_by_id[i] for i in costs.get("never_run", []) if i in number_by_id}

    import engine

    root = engine.ROOT
    commits = sorted(glob.glob(os.path.join(DATA, "commits", "*.patch")))
    plans = []
    for path in commits:
        commit = os.path.basename(path)[: -len(".patch")]
        with open(path, encoding="utf-8", errors="surrogateescape", newline="") as f:
            text = f.read()
        plans.append(plan_commit(commit, text, root, cov, tests_by_file))

    answers: dict[str, list[list[int]]] = {}
    jobs = []
    for plan in plans:
        if plan.everything:
            answers[plan.commit] = pack(all_tests, dur_ms)[1]
        elif plan.data_only:
            answers[plan.commit] = pack(plan.candidates, dur_ms)[1]
        elif not plan.candidates and not plan.reimport:
            answers[plan.commit] = []
        else:
            jobs.append(plan)
        log(f"{plan.commit}: {'; '.join(plan.notes)}")

    if jobs:
        engine.setup_parent(tests)
        log(f"parent ready in {time.time() - started:.2f}s")
        pending = list(jobs)
        running = {}  # read fd -> (pid, plan, chunks, deadline)
        while pending or running:
            while pending and len(running) < PARALLEL:
                plan = pending.pop(0)
                r, w = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(r)
                    child_main(plan, cov, run_cost, dur_ms, never_run, w)
                os.close(w)
                running[r] = (pid, plan, [], time.time() + 60)
            ready, _, _ = _select.select(list(running), [], [], 1.0)
            now = time.time()
            for r in list(running):
                pid, plan, chunks, deadline = running[r]
                if r in ready:
                    chunk = os.read(r, 1 << 16)
                    if chunk:
                        chunks.append(chunk)
                        continue
                elif now < deadline:
                    continue
                else:
                    try:
                        os.kill(pid, 9)
                    except ProcessLookupError:
                        pass
                os.close(r)
                os.waitpid(pid, 0)
                del running[r]
                try:
                    result = json.loads(b"".join(chunks).decode())
                except ValueError:
                    result = {"everything": True, "note": "no result"}
                if result.get("everything"):
                    answers[plan.commit] = pack(all_tests, dur_ms)[1]
                else:
                    answers[plan.commit] = pack(set(result["failing"]) | set(result["blind"]), dur_ms)[1]
                log(
                    f"{plan.commit}: ran={result.get('ran')} failing={len(result.get('failing', []))} "
                    f"blind={len(result.get('blind', []))} {result.get('note', '')}"
                )

    with os.fdopen(real_stdout, "w") as out:
        for plan in plans:
            out.write(json.dumps({"commit": plan.commit, "workers": answers.get(plan.commit, [])}) + "\n")
    import resource

    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime
    log(f"done in {time.time() - started:.2f}s wall, {cpu:.2f}s cpu")
    sys.stderr.flush()
    os._exit(0)  # skip interpreter teardown: it only burns CPU


if __name__ == "__main__":
    main()

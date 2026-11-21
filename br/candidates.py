"""Work out, for one patch, which tests could fail and how the patch must be applied to check them."""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field

from patching import FileDiff, apply_hunks, changed_old_lines, parse_patch
from pyanalyze import StmtIndex, all_codes, analyze, code_lines, is_function


class CoverageMap:
    def __init__(self, path: str):
        with gzip.open(path, "rt") as f:
            data = json.load(f)
        self.sets = [frozenset(s) for s in data["sets"]]
        self.lines = {name: {int(l): i for l, i in per.items()} for name, per in data["lines"].items()}
        self.opens = {name: frozenset(nums) for name, nums in data["opens"].items()}
        self.collect = {name: frozenset(lines) for name, lines in data["collect"].items()}
        self.attrs = {k: frozenset(v) for k, v in data.get("attrs", {}).items()}

    def tests_for_lines(self, filename: str, lines) -> set[int]:
        per = self.lines.get(filename, {})
        out: set[int] = set()
        for line in lines:
            index = per.get(line)
            if index is not None:
                out |= self.sets[index]
        return out


@dataclass
class Plan:
    commit: str
    candidates: set[int] = field(default_factory=set)  # tests that may fail; to run or report
    everything: bool = False  # any test may fail (syntax error, deleted module ...)
    py_files: dict = field(default_factory=dict)  # path -> (old_text, new_text, analysis)
    reimport: bool = False  # an import-time change: modules must be re-executed
    overlays: dict = field(default_factory=dict)  # data file path -> new text
    data_only: bool = True  # only data files (fixtures) changed
    notes: list[str] = field(default_factory=list)


def plan_commit(commit: str, patch_text: str, root: str, cov: CoverageMap, tests_by_file) -> Plan:
    plan = Plan(commit)
    try:
        diffs = parse_patch(patch_text)
    except Exception as error:  # noqa: BLE001
        plan.everything = True
        plan.notes.append(f"unparsable patch: {error!r}")
        return plan
    for fd in diffs:
        path = fd.path
        try:
            _plan_file(plan, fd, root, cov, tests_by_file)
        except Exception as error:  # noqa: BLE001 - never crash on one odd file
            plan.everything = True
            plan.notes.append(f"error planning {path}: {error!r}")
    return plan


def _plan_file(plan, fd: FileDiff, root, cov, tests_by_file):
    path = fd.path
    if fd.binary:
        plan.notes.append(f"binary {path}")
        if path.startswith("tests/") or path.startswith("sqlglot/"):
            readers = cov.opens.get(path)
            if readers is not None:
                plan.candidates |= readers
                plan.data_only = plan.data_only and True
            else:
                plan.everything = True
        return
    old_text = ""
    if fd.old_path is not None:
        full = os.path.join(root, fd.old_path)
        if not os.path.exists(full):
            plan.notes.append(f"missing {fd.old_path}")
            return
        with open(full, encoding="utf-8", newline="") as f:
            old_text = f.read()
    new_text = apply_hunks(old_text, fd.hunks) if fd.new_path is not None else None

    if path.endswith(".py") and (path.startswith("sqlglot/") or path.startswith("tests/")):
        plan.data_only = False
        _plan_python(plan, fd, old_text, new_text, cov, tests_by_file, root)
    elif path.startswith("tests/") or path.startswith("sqlglot/"):
        _plan_data_file(plan, fd, old_text, new_text, cov)
    else:
        plan.notes.append(f"ignored {path}")


def _plan_data_file(plan: Plan, fd: FileDiff, old_text: str, new_text: str | None, cov: CoverageMap):
    path = fd.path
    if new_text == old_text:
        return
    readers = cov.opens.get(path)
    if readers is None:
        plan.notes.append(f"unread data file {path}")
        return
    plan.overlays[path] = new_text
    plan.candidates |= readers
    plan.notes.append(f"data {path}: {len(readers)} readers")


def _expand(index: StmtIndex, lines: set[int]) -> set[int]:
    out: set[int] = set()
    for line in lines:
        rng = index.own_range(line)
        if rng:
            out.update(range(rng[0], rng[1] + 1))
    return out


def _plan_python(plan, fd, old_text, new_text, cov, tests_by_file, root=""):
    path = fd.path
    if new_text is None or fd.old_path is None:
        # A module deleted or created: imports change, so be safe.
        plan.everything = new_text is None
        plan.reimport = True
        plan.notes.append(f"{'deleted' if new_text is None else 'new'} {path}")
        return
    analysis = analyze(old_text, new_text, os.path.join(root, path) if root else path)
    if analysis.syntax_error:
        plan.everything = True
        plan.notes.append(f"syntax error {path}")
        return
    if analysis.noop:
        plan.notes.append(f"noop {path}")
        return
    plan.py_files[path] = (old_text, new_text, analysis)

    index = StmtIndex.build(old_text)
    total = len(old_text.splitlines())
    touched: set[int] = set()  # old lines the patch touches, insertion neighbours included
    for hunk in fd.hunks:
        removed, inserted = changed_old_lines(hunk)
        touched.update(removed)
        for after in inserted:
            for line in (index.prev_code_line(after), index.next_code_line(after, total)):
                if line:
                    touched.add(line)

    collect = cov.collect.get(path, frozenset())
    body_lines: set[int] = set()
    for code in all_codes(analysis.old_module):
        if not is_function(code):
            body_lines |= code_lines(code)

    tests: set[int] = set()
    import_time = False
    kinds = []
    for change in analysis.changes:
        kinds.append(change.kind)
        if change.kind == "body":
            import_time = True
            continue
        own_all = code_lines(change.old, nested=True)
        own = code_lines(change.old)
        lines = _expand(index, touched & set(range(min(own_all, default=0), max(own_all, default=-1) + 1)))
        lines &= own_all
        if not lines:
            lines = own_all  # e.g. a signature change: everyone who calls it
        tests |= cov.tests_for_lines(path, lines)
        # A function that ran while modules were imported (a builder called from a class body)
        # has left results behind that swapping its code cannot reach.
        if (own - body_lines) & collect:
            import_time = True
    for _parents, removed in analysis.removed:
        tests |= cov.tests_for_lines(path, code_lines(removed, nested=True))
    if analysis.structural:
        import_time = True
    if path.startswith("tests/") and import_time:
        tests |= set(tests_by_file.get(path, ()))
    plan.candidates |= tests
    plan.reimport = plan.reimport or import_time
    plan.notes.append(
        f"py {path}: kinds={sorted(set(kinds))} structural={analysis.structural} "
        f"tests={len(tests)} import_time={import_time}"
    )

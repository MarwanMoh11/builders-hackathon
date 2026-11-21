"""Compare the compiled code of a Python file before and after a patch.

Line numbers are ignored, so a change that only adds or removes comments or blank lines compiles
to the same code and is reported as a no-op. Otherwise every code object whose own instructions
or constants changed is reported, together with what kind of code it is:

* "function": a def, lambda or comprehension body: its effect starts when it is called, so a
  live function can be hot-patched by swapping its __code__.
* "body": a module or class body, which only runs at import time.
"""

from __future__ import annotations

import ast
import types
from dataclasses import dataclass, field

CO_NEWLOCALS = 0x0002


def _const_key(value):
    if isinstance(value, types.CodeType):
        return ("<code>", value.co_name)
    if isinstance(value, (tuple, frozenset)):
        items = [_const_key(v) for v in value]
        if isinstance(value, frozenset):
            items.sort(key=repr)
        return (type(value).__name__, tuple(items))
    return (type(value).__name__, repr(value))


def _firstlineno_const_indexes(code: types.CodeType) -> set[int]:
    """Class bodies store their own line number (`__firstlineno__ = <n>`, Python 3.13+), so a
    line shift changes that constant without changing behavior."""
    if "__firstlineno__" not in code.co_names:
        return set()
    import dis

    out: set[int] = set()
    previous = None
    for ins in dis.get_instructions(code):
        if ins.opname in ("STORE_NAME", "STORE_FAST") and ins.argval == "__firstlineno__":
            if previous is not None and previous.opname in ("LOAD_CONST", "LOAD_SMALL_INT"):
                if previous.opname == "LOAD_CONST":
                    out.add(previous.arg)
        if ins.opname not in ("CACHE", "EXTENDED_ARG"):
            previous = ins
    return out


def own_signature(code: types.CodeType):
    masked = _firstlineno_const_indexes(code)
    return (
        code.co_code,
        tuple("<lineno>" if i in masked else _const_key(c) for i, c in enumerate(code.co_consts)),
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
        code.co_exceptiontable,
        code.co_name,
        code.co_qualname,
    )


def children(code: types.CodeType) -> list[types.CodeType]:
    return [c for c in code.co_consts if isinstance(c, types.CodeType)]


def is_function(code: types.CodeType) -> bool:
    return bool(code.co_flags & CO_NEWLOCALS)


@dataclass
class CodeChange:
    old: types.CodeType
    new: types.CodeType
    kind: str  # "function" or "body"
    path: list[types.CodeType]  # old ancestors from the module down to (not including) old
    new_path: list[types.CodeType]


@dataclass
class FileAnalysis:
    syntax_error: bool = False
    noop: bool = False
    structural: bool = False  # code objects appeared or disappeared somewhere
    changes: list[CodeChange] = field(default_factory=list)
    removed: list[tuple[list[types.CodeType], types.CodeType]] = field(default_factory=list)
    added: list[tuple[list[types.CodeType], types.CodeType]] = field(default_factory=list)
    old_module: types.CodeType | None = None
    new_module: types.CodeType | None = None


def compile_source(source: str, filename: str) -> types.CodeType:
    return compile(source, filename, "exec", dont_inherit=True, optimize=0)


def analyze(old_source: str, new_source: str, filename: str) -> FileAnalysis:
    result = FileAnalysis()
    old_module = compile_source(old_source, filename)
    try:
        new_module = compile_source(new_source, filename)
    except (SyntaxError, ValueError):
        result.syntax_error = True
        return result
    result.old_module, result.new_module = old_module, new_module

    import difflib

    def walk(old, new, path, new_path):
        if own_signature(old) != own_signature(new):
            result.changes.append(
                CodeChange(old, new, "function" if is_function(old) else "body", list(path), list(new_path))
            )
        old_kids, new_kids = children(old), children(new)
        old_names = [k.co_qualname for k in old_kids]
        new_names = [k.co_qualname for k in new_kids]
        if old_names == new_names:
            for o, n in zip(old_kids, new_kids):
                walk(o, n, path + [old], new_path + [new])
            return
        result.structural = True
        matcher = difflib.SequenceMatcher(a=old_names, b=new_names, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for o, n in zip(old_kids[i1:i2], new_kids[j1:j2]):
                    walk(o, n, path + [old], new_path + [new])
            else:
                for o in old_kids[i1:i2]:
                    result.removed.append((path + [old], o))
                for n in new_kids[j1:j2]:
                    result.added.append((new_path + [new], n))

    walk(old_module, new_module, [], [])
    result.noop = not result.changes and not result.structural
    return result


def code_lines(code: types.CodeType, nested: bool = False) -> set[int]:
    lines = {line for _, _, line in code.co_lines() if line is not None}
    if nested:
        for kid in children(code):
            lines |= code_lines(kid, True)
    return lines


def all_codes(code: types.CodeType):
    yield code
    for kid in children(code):
        yield from all_codes(kid)


# ---------------------------------------------------------------- statement ranges


@dataclass
class StmtIndex:
    """Statements of a source file with the line range that holds each one's own code."""

    stmts: list[tuple[int, int, int, ast.stmt]]  # (start, own_end, end, node)
    code_lines: set[int]

    @classmethod
    def build(cls, source: str) -> "StmtIndex":
        tree = ast.parse(source)
        stmts = []
        code_lines: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt):
                end = node.end_lineno or node.lineno
                own_end = end
                bodies = []
                for name in ("body", "orelse", "finalbody", "handlers", "cases"):
                    value = getattr(node, name, None)
                    if isinstance(value, list) and value and not isinstance(node, (ast.Lambda,)):
                        bodies.extend(value)
                if bodies:
                    first = min(getattr(b, "lineno", end) for b in bodies)
                    own_end = max(node.lineno, first - 1)
                start = node.lineno
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.decorator_list:
                        start = min(d.lineno for d in node.decorator_list)
                stmts.append((start, own_end, end, node))
                code_lines.update(range(start, own_end + 1))
        stmts.sort(key=lambda s: (s[0], -s[2]))
        return cls(stmts, code_lines)

    def innermost(self, line: int):
        best = None
        for start, own_end, end, node in self.stmts:
            if start <= line <= end:
                if best is None or (end - start) <= (best[2] - best[0]):
                    best = (start, own_end, end, node)
        return best

    def own_range(self, line: int) -> tuple[int, int] | None:
        found = self.innermost(line)
        if found is None:
            return None
        start, own_end, end, node = found
        if start <= line <= own_end:
            return start, own_end
        # A line inside a compound statement but outside any child statement (an `else:` line,
        # a comment, a blank line): the whole statement is the honest answer.
        return start, end

    def next_code_line(self, line: int, limit: int) -> int | None:
        for candidate in range(line + 1, limit + 1):
            if candidate in self.code_lines:
                return candidate
        return None

    def prev_code_line(self, line: int) -> int | None:
        for candidate in range(line, 0, -1):
            if candidate in self.code_lines:
                return candidate
        return None

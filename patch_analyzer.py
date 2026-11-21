"""Unified diff parser, in-memory patch applier, and AST/semantic/fixture diff analyzer."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_EMPTY_SET: set[Any] = set()

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
CODING_RE = re.compile(r"coding[:=]\s*([-\w.]+)")


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class FilePatch:
    old_path: str | None  # None for /dev/null (new file)
    new_path: str | None  # None for /dev/null (deleted file)
    hunks: list[Hunk] = field(default_factory=list)
    is_binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def is_new(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None


def _strip_prefix(raw: str) -> str | None:
    raw = raw.strip()
    if raw == "/dev/null":
        return None
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    if raw.startswith("a/") or raw.startswith("b/"):
        return raw[2:]
    return raw


def parse_unified_diff(patch_text: str) -> list[FilePatch]:
    """Parse a git unified diff into FilePatch objects."""
    lines = patch_text.splitlines(keepends=True)
    patches: list[FilePatch] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            # Extract fallback paths from header
            header_old = None
            header_new = None
            parts = line.strip().split(" ")
            if len(parts) >= 4:
                header_old = _strip_prefix(parts[2])
                header_new = _strip_prefix(parts[3])
            i += 1
            old_path = header_old
            new_path = header_new
            is_binary = False
            hunks: list[Hunk] = []
            while i < n and not lines[i].startswith("diff --git "):
                cur = lines[i]
                if cur.startswith("--- "):
                    old_path = _strip_prefix(cur[4:])
                    i += 1
                elif cur.startswith("+++ "):
                    new_path = _strip_prefix(cur[4:])
                    i += 1
                elif cur.startswith("Binary files ") or cur.startswith("GIT binary patch"):
                    is_binary = True
                    i += 1
                elif cur.startswith("@@ "):
                    m = HUNK_RE.match(cur)
                    if not m:
                        i += 1
                        continue
                    old_start = int(m.group(1))
                    old_count = int(m.group(2)) if m.group(2) is not None else 1
                    new_start = int(m.group(3))
                    new_count = int(m.group(4)) if m.group(4) is not None else 1
                    i += 1
                    hunk_lines: list[str] = []
                    while i < n and not lines[i].startswith(("@@ ", "diff --git ")):
                        hl = lines[i]
                        if hl.startswith((" ", "+", "-", "\\")):
                            hunk_lines.append(hl)
                            i += 1
                        else:
                            break
                    hunks.append(Hunk(old_start, old_count, new_start, new_count, hunk_lines))
                else:
                    i += 1
            patches.append(FilePatch(old_path, new_path, hunks, is_binary))
        elif line.startswith("--- "):
            old_path = _strip_prefix(line[4:])
            i += 1
            new_path = None
            if i < n and lines[i].startswith("+++ "):
                new_path = _strip_prefix(lines[i][4:])
                i += 1
            hunks = []
            while i < n and lines[i].startswith("@@ "):
                m = HUNK_RE.match(lines[i])
                if not m:
                    break
                old_start = int(m.group(1))
                old_count = int(m.group(2)) if m.group(2) is not None else 1
                new_start = int(m.group(3))
                new_count = int(m.group(4)) if m.group(4) is not None else 1
                i += 1
                hunk_lines = []
                while i < n and not lines[i].startswith(("@@ ", "diff --git ", "--- ")):
                    hl = lines[i]
                    if hl.startswith((" ", "+", "-", "\\")):
                        hunk_lines.append(hl)
                        i += 1
                    else:
                        break
                hunks.append(Hunk(old_start, old_count, new_start, new_count, hunk_lines))
            patches.append(FilePatch(old_path, new_path, hunks, False))
        else:
            i += 1

    return patches


def apply_file_patch(old_text: str, fp: FilePatch) -> str:
    """Apply parsed unified diff hunks in-memory to old_text."""
    if fp.is_deleted:
        return ""
    if not fp.hunks:
        return old_text

    old_lines = old_text.splitlines(keepends=True) if old_text else []
    out: list[str] = []
    cursor = 0  # 0-based index into old_lines

    for hunk in fp.hunks:
        target = max(0, hunk.old_start - 1) if hunk.old_start > 0 else 0
        if target > cursor:
            out.extend(old_lines[cursor:target])
            cursor = target

        for idx, hl in enumerate(hunk.lines):
            if hl.startswith("\\"):
                # "\ No newline at end of file" applies to the immediately preceding line
                if idx > 0 and hunk.lines[idx - 1].startswith("+") and out:
                    out[-1] = out[-1].rstrip("\r\n")
                continue
            tag = hl[0]
            body = hl[1:]
            if tag == " ":
                if cursor < len(old_lines):
                    out.append(old_lines[cursor])
                    cursor += 1
                else:
                    out.append(body)
            elif tag == "-":
                if cursor < len(old_lines):
                    cursor += 1
            elif tag == "+":
                out.append(body)

    if cursor < len(old_lines):
        out.extend(old_lines[cursor:])
    return "".join(out)


def extract_changed_old_lines(fp: FilePatch) -> tuple[set[int], set[int]]:
    """Return (modified_or_deleted_old_lines, pure_insertion_anchor_old_lines) 1-based."""
    changed: set[int] = set()
    anchors: set[int] = set()
    for hunk in fp.hunks:
        old_ln = hunk.old_start
        saw_del = False
        for hl in hunk.lines:
            if hl.startswith("\\"):
                continue
            tag = hl[0]
            if tag == " ":
                saw_del = False
                old_ln += 1
            elif tag == "-":
                saw_del = True
                changed.add(old_ln)
                old_ln += 1
            elif tag == "+":
                # Only record insertion anchors for PURE insertions (not replacements of '-' lines,
                # whose old lines are already in `changed`!).
                if not saw_del:
                    if old_ln > 1:
                        anchors.add(old_ln - 1)
                    anchors.add(old_ln)
    return changed, anchors


@dataclass
class ChangeUnit:
    """One fine-grained semantic change extracted from a modified Python file."""
    kind: str  # "func_body", "dict_key", "class_attr", "module_attr", "module_other", "syntax_error"
    file: str
    class_name: str | None = None
    symbol_name: str | None = None  # func qualname, attr name, or dict attr name
    dict_keys: set[str] | None = None  # None means wildcard "*"
    old_lines: set[int] = field(default_factory=set)
    enclosing_lines: set[int] = field(default_factory=set)


@dataclass
class PyFileAnalysis:
    file: str
    is_noop: bool  # True ONLY when AST is identical and no coding cookie changed
    syntax_error: bool = False
    units: list[ChangeUnit] = field(default_factory=list)
    all_changed_old_lines: set[int] = field(default_factory=set)
    # Functions whose body changed (qualname -> (start_line, end_line))
    changed_funcs: list[str] = field(default_factory=list)


def _strip_docstrings_and_type_checking(tree: ast.AST) -> str:
    """Exact AST dump (preserves docstrings and strings, ignores comments/whitespace)."""
    return ast.dump(tree, include_attributes=False)


def _is_sound_cmp_pair(
    left: ast.AST,
    right: ast.AST,
    allow_none: bool,
    pure_enums: set[str],
    strict_enum_attrs: set[tuple[str, str]],
) -> bool:
    if allow_none and (
        (isinstance(right, ast.Constant) and right.value is None)
        or (isinstance(left, ast.Constant) and left.value is None)
    ):
        return True
    for lhs, rhs in ((left, right), (right, left)):
        if isinstance(rhs, ast.Attribute) and isinstance(rhs.value, ast.Name):
            enum_name = rhs.value.id
            if enum_name in pure_enums:
                return True
            if isinstance(lhs, ast.Attribute) and (lhs.attr, enum_name) in strict_enum_attrs:
                return True
    return False


def _strip_leading_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    """Return body without a leading bare string-literal docstring statement."""
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _ast_nodes_equiv(
    a: Any,
    b: Any,
    allow_none: bool,
    pure_enums: set[str],
    strict_enum_attrs: set[tuple[str, str]],
    ignore_docstrings: bool = False,
) -> bool:
    """Zero-copy recursive AST equivalence checker allowing provably sound is/== swaps
    and (when no doctest runner exists in tests.json) pure docstring changes."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all(
            _ast_nodes_equiv(x, y, allow_none, pure_enums, strict_enum_attrs, ignore_docstrings)
            for x, y in zip(a, b)
        )
    if isinstance(a, ast.Compare):
        if len(a.ops) == 1 and len(b.ops) == 1 and len(a.comparators) == 1 and len(b.comparators) == 1:
            op_a, op_b = a.ops[0], b.ops[0]
            if type(op_a) is not type(op_b):
                eq_is_pair = (
                    (isinstance(op_a, ast.Is) and isinstance(op_b, ast.Eq))
                    or (isinstance(op_a, ast.Eq) and isinstance(op_b, ast.Is))
                    or (isinstance(op_a, ast.IsNot) and isinstance(op_b, ast.NotEq))
                    or (isinstance(op_a, ast.NotEq) and isinstance(op_b, ast.IsNot))
                )
                if not (
                    eq_is_pair
                    and _is_sound_cmp_pair(a.left, a.comparators[0], allow_none, pure_enums, strict_enum_attrs)
                    and _ast_nodes_equiv(a.left, b.left, allow_none, pure_enums, strict_enum_attrs, ignore_docstrings)
                    and _ast_nodes_equiv(a.comparators[0], b.comparators[0], allow_none, pure_enums, strict_enum_attrs, ignore_docstrings)
                ):
                    return False
                return True
        if len(a.ops) != len(b.ops) or len(a.comparators) != len(b.comparators):
            return False
        return (
            _ast_nodes_equiv(a.left, b.left, allow_none, pure_enums, strict_enum_attrs, ignore_docstrings)
            and all(type(x) is type(y) for x, y in zip(a.ops, b.ops))
            and all(_ast_nodes_equiv(x, y, allow_none, pure_enums, strict_enum_attrs, ignore_docstrings) for x, y in zip(a.comparators, b.comparators))
        )
    if isinstance(a, ast.AST):
        is_doc_owner = ignore_docstrings and isinstance(a, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for field_name in a._fields:
            va = getattr(a, field_name)
            vb = getattr(b, field_name)
            if is_doc_owner and field_name == "body":
                va = _strip_leading_docstring(va)
                vb = _strip_leading_docstring(vb)
            if not _ast_nodes_equiv(
                va,
                vb,
                allow_none,
                pure_enums,
                strict_enum_attrs,
                ignore_docstrings,
            ):
                return False
        return True
    return a == b


def _diff_has_identity_or_eq_change(fp: FilePatch) -> bool:
    for h in fp.hunks:
        for line in h.lines:
            if line.startswith(("+", "-")) and (" is " in line or "==" in line or "!=" in line or "None" in line):
                return True
    return False


def _is_sound_cmp_equiv(
    file: str,
    old_ast: ast.AST,
    new_ast: ast.AST,
    fp: FilePatch,
    pure_enums: set[str],
    strict_enum_attrs: set[tuple[str, str]],
    ignore_docstrings: bool = False,
) -> bool:
    if not file.startswith("sqlglot/") or not _diff_has_identity_or_eq_change(fp):
        return False
    allow_none = file != "sqlglot/executor/env.py"
    return _ast_nodes_equiv(old_ast, new_ast, allow_none, pure_enums, strict_enum_attrs, ignore_docstrings)


def _coding_cookie(src: str) -> str | None:
    for line in src.splitlines()[:2]:
        m = CODING_RE.search(line)
        if m:
            return m.group(1).lower()
    return None


def _const_key_repr(node: ast.AST | None) -> str | None:
    """Resolve a static dict key AST node to a canonical string key."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return repr(node.value)
        return None
    if isinstance(node, ast.Attribute):
        # e.g. exp.Array, exp.DType.INT, TokenType.VAR
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            parts.reverse()
            return ".".join(parts)
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Tuple):
        elts = [_const_key_repr(e) for e in node.elts]
        if any(e is None for e in elts):
            return None
        return "(" + ", ".join(elts) + ")"  # type: ignore[arg-type]
    return None


def _diff_dict_nodes(old_dict: ast.Dict, new_dict: ast.Dict) -> set[str] | None:
    """Return set of changed/added/removed canonical dict keys, or None for wildcard '*'."""
    old_unpacks: list[str] = []
    old_map: dict[str, str] = {}
    for k, v in zip(old_dict.keys, old_dict.values):
        if k is None:
            old_unpacks.append(ast.dump(v, include_attributes=False))
        else:
            kr = _const_key_repr(k)
            if kr is None:
                return None
            old_map[kr] = ast.dump(v, include_attributes=False)

    new_unpacks: list[str] = []
    new_map: dict[str, str] = {}
    for k, v in zip(new_dict.keys, new_dict.values):
        if k is None:
            new_unpacks.append(ast.dump(v, include_attributes=False))
        else:
            kr = _const_key_repr(k)
            if kr is None:
                return None
            new_map[kr] = ast.dump(v, include_attributes=False)

    if old_unpacks != new_unpacks:
        return None

    changed_keys: set[str] = set()
    for k in set(old_map) | set(new_map):
        if old_map.get(k) != new_map.get(k):
            changed_keys.add(k)
    return changed_keys


def _stmt_line_span(node: ast.AST) -> set[int]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", start)
    if start is None:
        return set()
    return set(range(start, (end or start) + 1))


def _func_signature_dump(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Dump everything about a function EXCEPT its body statements."""
    clone = ast.FunctionDef(
        name=fn.name,
        args=fn.args,
        body=[],
        decorator_list=fn.decorator_list,
        returns=fn.returns,
        type_params=getattr(fn, "type_params", []),
    )
    return ast.dump(clone, include_attributes=False)


def _extract_assign_target_and_value(stmt: ast.stmt) -> tuple[str | None, ast.AST | None]:
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.targets[0].id, stmt.value
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id, stmt.value
    return None, None


def _is_type_checking_if(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.If):
        return False
    test = stmt.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _stmt_src_equal(
    o_node: ast.AST,
    n_node: ast.AST,
    old_lines_arr: list[str],
    new_lines_arr: list[str],
) -> bool:
    o1 = getattr(o_node, "lineno", None)
    o2 = getattr(o_node, "end_lineno", None)
    n1 = getattr(n_node, "lineno", None)
    n2 = getattr(n_node, "end_lineno", None)
    if o1 is None or o2 is None or n1 is None or n2 is None:
        return False
    if (o2 - o1) != (n2 - n1):
        return False
    return old_lines_arr[o1 - 1 : o2] == new_lines_arr[n1 - 1 : n2]


def _compare_block(
    file: str,
    old_body: list[ast.stmt],
    new_body: list[ast.stmt],
    old_lines_arr: list[str],
    new_lines_arr: list[str],
    class_chain: list[str],
    changed_old_lines: set[int],
    anchor_old_lines: set[int],
    units: list[ChangeUnit],
    changed_funcs: list[str],
    ignore_docstrings: bool = False,
) -> None:
    """Compare statements in a module or class body and emit precise ChangeUnits."""
    if ignore_docstrings:
        old_body = _strip_leading_docstring(old_body)
        new_body = _strip_leading_docstring(new_body)

    def stmt_key(s: ast.stmt) -> tuple[str, str] | None:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return ("func", s.name)
        if isinstance(s, ast.ClassDef):
            return ("class", s.name)
        name, _ = _extract_assign_target_and_value(s)
        if name is not None:
            return ("assign", name)
        return None

    # Check if statement sequence structure matches cleanly by keyed defs/classes/assigns
    old_by_key: dict[tuple[str, str], list[ast.stmt]] = {}
    new_by_key: dict[tuple[str, str], list[ast.stmt]] = {}
    old_unkeyed: list[ast.stmt] = []
    new_unkeyed: list[ast.stmt] = []

    for s in old_body:
        if _is_type_checking_if(s):
            continue
        k = stmt_key(s)
        if k is None:
            old_unkeyed.append(s)
        else:
            old_by_key.setdefault(k, []).append(s)

    for s in new_body:
        if _is_type_checking_if(s):
            continue
        k = stmt_key(s)
        if k is None:
            new_unkeyed.append(s)
        else:
            new_by_key.setdefault(k, []).append(s)

    class_qual = ".".join(class_chain) if class_chain else None

    # Any change to unkeyed statements (imports, if-blocks, bare expressions, etc.)
    if not _ast_nodes_equiv(old_unkeyed, new_unkeyed, False, _EMPTY_SET, _EMPTY_SET, ignore_docstrings):
        units.append(
            ChangeUnit(
                kind="class_attr" if class_qual else "module_other",
                file=file,
                class_name=class_qual,
                symbol_name="*",
                old_lines=set(changed_old_lines),
            )
        )

    for key in set(old_by_key) | set(new_by_key):
        o_list = old_by_key.get(key, [])
        n_list = new_by_key.get(key, [])
        if len(o_list) == 1 and len(n_list) == 1 and _stmt_src_equal(o_list[0], n_list[0], old_lines_arr, new_lines_arr):
            continue
        if _ast_nodes_equiv(o_list, n_list, False, _EMPTY_SET, _EMPTY_SET, ignore_docstrings):
            continue

        kind_tag, name = key
        if len(o_list) != 1 or len(n_list) != 1:
            # Added, deleted, or duplicated definition
            span = set()
            for s in o_list:
                span |= _stmt_line_span(s)
            if kind_tag == "func":
                qname = f"{class_qual}.{name}" if class_qual else name
                changed_funcs.append(qname)
                units.append(
                    ChangeUnit(
                        kind="class_attr" if class_qual else "module_other",
                        file=file,
                        class_name=class_qual,
                        # Added/deleted methods (e.g. *_sql) may be dispatched dynamically via getattr!
                        symbol_name="*" if class_qual else name,
                        old_lines=span & (changed_old_lines | anchor_old_lines),
                        enclosing_lines=span,
                    )
                )
            elif kind_tag == "class":
                sub_qual = f"{class_qual}.{name}" if class_qual else name
                units.append(
                    ChangeUnit(
                        kind="class_attr",
                        file=file,
                        class_name=sub_qual,
                        symbol_name="*",
                        old_lines=span,
                        enclosing_lines=span,
                    )
                )
            else:
                units.append(
                    ChangeUnit(
                        kind="class_attr" if class_qual else "module_attr",
                        file=file,
                        class_name=class_qual,
                        symbol_name=name,
                        old_lines=span,
                        enclosing_lines=span,
                    )
                )
            continue

        os_node = o_list[0]
        ns_node = n_list[0]

        if kind_tag == "class":
            assert isinstance(os_node, ast.ClassDef) and isinstance(ns_node, ast.ClassDef)
            sub_chain = [*class_chain, name]
            sub_qual = ".".join(sub_chain)
            # Check bases/keywords/decorators
            if (
                not _ast_nodes_equiv(os_node.bases, ns_node.bases, False, _EMPTY_SET, _EMPTY_SET)
                or not _ast_nodes_equiv(os_node.keywords, ns_node.keywords, False, _EMPTY_SET, _EMPTY_SET)
                or not _ast_nodes_equiv(os_node.decorator_list, ns_node.decorator_list, False, _EMPTY_SET, _EMPTY_SET)
            ):
                units.append(
                    ChangeUnit(
                        kind="class_attr",
                        file=file,
                        class_name=sub_qual,
                        symbol_name="*",
                        old_lines=_stmt_line_span(os_node),
                    )
                )
            _compare_block(
                file,
                os_node.body,
                ns_node.body,
                old_lines_arr,
                new_lines_arr,
                sub_chain,
                changed_old_lines,
                anchor_old_lines,
                units,
                changed_funcs,
                ignore_docstrings,
            )

        elif kind_tag == "func":
            assert isinstance(os_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            assert isinstance(ns_node, (ast.FunctionDef, ast.AsyncFunctionDef))
            qname = f"{class_qual}.{name}" if class_qual else name
            changed_funcs.append(qname)
            fn_span = _stmt_line_span(os_node)
            sig_changed = _func_signature_dump(os_node) != _func_signature_dump(ns_node)
            # Find the tightest changed statement spans inside old function body
            hit_lines = fn_span & (changed_old_lines | anchor_old_lines)
            stmt_lines: set[int] = set()
            for sub in ast.walk(os_node):
                if isinstance(sub, ast.stmt) and sub is not os_node:
                    sp = _stmt_line_span(sub)
                    if sp & hit_lines:
                        if isinstance(sub, ast.If):
                            hdr = _stmt_line_span(sub.test) | {sub.lineno}
                            if hdr & hit_lines:
                                stmt_lines |= hdr
                        elif isinstance(sub, (ast.For, ast.AsyncFor)):
                            hdr = _stmt_line_span(sub.target) | _stmt_line_span(sub.iter) | {sub.lineno}
                            if hdr & hit_lines:
                                stmt_lines |= hdr
                        elif isinstance(sub, ast.While):
                            hdr = _stmt_line_span(sub.test) | {sub.lineno}
                            if hdr & hit_lines:
                                stmt_lines |= hdr
                        elif isinstance(sub, (ast.With, ast.AsyncWith)):
                            hdr = {sub.lineno}
                            for item in sub.items:
                                hdr |= _stmt_line_span(item)
                            if hdr & hit_lines:
                                stmt_lines |= hdr
                        elif isinstance(sub, ast.Try):
                            if sub.lineno in hit_lines:
                                stmt_lines.add(sub.lineno)
                            for h in sub.handlers:
                                h_hdr = {h.lineno} | (_stmt_line_span(h.type) if h.type is not None else set())
                                if h_hdr & hit_lines:
                                    stmt_lines |= h_hdr
                        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            if sub.lineno in hit_lines:
                                stmt_lines.add(sub.lineno)
                        else:
                            stmt_lines |= sp
                # Block-aware insertion anchors: if an insertion falls between statements
                # (including before/after/inside a leading docstring!), bind to neighboring executable statements.
                for attr in ("body", "orelse", "finalbody"):
                    raw_blk = getattr(sub, attr, None)
                    if isinstance(raw_blk, list) and raw_blk:
                        blk = _strip_leading_docstring(raw_blk) if attr == "body" else raw_blk
                        if not blk:
                            continue
                        for idx_s, s_item in enumerate(blk):
                            s_start = getattr(s_item, "lineno", 0)
                            s_end = getattr(s_item, "end_lineno", s_start)
                            prev_end = getattr(blk[idx_s - 1], "end_lineno", 0) if idx_s > 0 else getattr(sub, "lineno", 0)
                            # Insertion between previous executable statement (or def/class/if header) and s_item
                            if any(prev_end <= al <= s_start for al in anchor_old_lines):
                                stmt_lines.add(s_start)
                                if idx_s > 0:
                                    stmt_lines.add(getattr(blk[idx_s - 1], "lineno", s_start))
                            # Insertion right after the last executable statement in block
                            if idx_s == len(blk) - 1:
                                block_end = getattr(sub, "end_lineno", s_end)
                                if any(s_end <= al <= block_end + 1 for al in anchor_old_lines):
                                    stmt_lines.add(s_start)
            if not stmt_lines or sig_changed:
                stmt_lines = set(fn_span)
            units.append(
                ChangeUnit(
                    kind="class_attr" if (sig_changed and class_qual) else ("module_other" if sig_changed else "func_body"),
                    file=file,
                    class_name=class_qual,
                    symbol_name="*" if (sig_changed and class_qual) else qname,
                    old_lines=hit_lines | stmt_lines,
                    enclosing_lines=fn_span,
                )
            )

        elif kind_tag == "assign":
            _, o_val = _extract_assign_target_and_value(os_node)
            _, n_val = _extract_assign_target_and_value(ns_node)
            span = _stmt_line_span(os_node)
            if isinstance(o_val, ast.Dict) and isinstance(n_val, ast.Dict):
                diff_keys = _diff_dict_nodes(o_val, n_val)
                units.append(
                    ChangeUnit(
                        kind="dict_key",
                        file=file,
                        class_name=class_qual,
                        symbol_name=name,
                        dict_keys=diff_keys,
                        old_lines=span & (changed_old_lines | anchor_old_lines),
                        enclosing_lines=span,
                    )
                )
            else:
                units.append(
                    ChangeUnit(
                        kind="class_attr" if class_qual else "module_attr",
                        file=file,
                        class_name=class_qual,
                        symbol_name=name,
                        old_lines=span & (changed_old_lines | anchor_old_lines),
                        enclosing_lines=span,
                    )
                )


def analyze_python_change(
    file: str,
    old_src: str,
    new_src: str,
    fp: FilePatch,
    *,
    old_ast_cache: dict[str, ast.Module] | None = None,
    pure_enums: set[str] | None = None,
    strict_enum_attrs: set[tuple[str, str]] | None = None,
    ignore_docstrings: bool = False,
) -> PyFileAnalysis:
    """Analyze a modified Python file into semantic ChangeUnits."""
    changed_old, anchor_old = extract_changed_old_lines(fp)
    if fp.is_new or fp.is_deleted:
        try:
            if not fp.is_deleted:
                ast.parse(new_src, filename=file)
        except SyntaxError:
            return PyFileAnalysis(
                file=file,
                is_noop=False,
                syntax_error=True,
                units=[ChangeUnit(kind="syntax_error", file=file)],
            )
        return PyFileAnalysis(
            file=file,
            is_noop=False,
            units=[ChangeUnit(kind="module_other", file=file, symbol_name="*")],
        )

    try:
        new_ast = ast.parse(new_src, filename=file)
    except SyntaxError:
        return PyFileAnalysis(
            file=file,
            is_noop=False,
            syntax_error=True,
            units=[ChangeUnit(kind="syntax_error", file=file)],
            all_changed_old_lines=changed_old,
        )

    if old_ast_cache is not None and file in old_ast_cache:
        old_ast = old_ast_cache[file]
    else:
        old_ast = ast.parse(old_src, filename=file)
        if old_ast_cache is not None:
            old_ast_cache[file] = old_ast

    # Check if AST is 100% identical (or provably sound None/Enum comparison equivalent)
    # Zero-copy recursive comparator short-circuits on the first differing node (no 500KB ast.dump string!)
    if _coding_cookie(old_src) == _coding_cookie(new_src):
        if _ast_nodes_equiv(old_ast, new_ast, False, _EMPTY_SET, _EMPTY_SET, ignore_docstrings):
            return PyFileAnalysis(file=file, is_noop=True)
        if _is_sound_cmp_equiv(
            file, old_ast, new_ast, fp, pure_enums or _EMPTY_SET, strict_enum_attrs or _EMPTY_SET, ignore_docstrings
        ):
            return PyFileAnalysis(file=file, is_noop=True)

    units: list[ChangeUnit] = []
    changed_funcs: list[str] = []
    _compare_block(
        file,
        old_ast.body,
        new_ast.body,
        old_src.splitlines(),
        new_src.splitlines(),
        [],
        changed_old,
        anchor_old,
        units,
        changed_funcs,
        ignore_docstrings,
    )

    # If only TYPE_CHECKING blocks changed (units is empty), check if file is in sqlglot/typing/
    # (test_typing.py parses AST of sqlglot/typing/*.py!)
    if not units and not file.startswith("sqlglot/typing/"):
        return PyFileAnalysis(file=file, is_noop=True)
    if not units:
        units.append(ChangeUnit(kind="module_other", file=file, symbol_name="*", old_lines=changed_old))

    return PyFileAnalysis(
        file=file,
        is_noop=False,
        units=units,
        all_changed_old_lines=changed_old | anchor_old,
        changed_funcs=changed_funcs,
    )


def diff_sql_fixture_pairs(old_text: str, new_text: str) -> tuple[bool, list[int]]:
    """Compare two load_sql_fixture_pairs .sql files.

    Returns (parse_error_or_shifted, changed_1based_case_indices).
    If parse_error_or_shifted is True, all cases in the fixture should be checked (or test immediately fails).
    """
    def _parse(text: str) -> list[tuple[dict[str, str], str, str]]:
        filtered = "\n".join(line for line in text.splitlines() if line and not line.startswith("--"))
        stmts = filtered.split(";")
        out = []
        for i in range(0, len(stmts), 2):
            if i + 1 < len(stmts):
                sql = stmts[i].strip()
                meta: dict[str, str] = {}
                sql_lines = sql.split("\n")
                j = 0
                while j < len(sql_lines) and sql_lines[j].startswith("#"):
                    key, val = sql_lines[j].split(":", maxsplit=1)
                    meta[key.lstrip("#").strip()] = val.strip()
                    j += 1
                sql = "\n".join(sql_lines[j:])
                expected = stmts[i + 1].strip()
                out.append((meta, sql, expected))
        return out

    old_cases = _parse(old_text)
    try:
        new_cases = _parse(new_text)
    except Exception:
        return True, []

    if len(old_cases) != len(new_cases):
        return True, []

    changed = [idx for idx, (o, n) in enumerate(zip(old_cases, new_cases), start=1) if o != n]
    return False, changed


def diff_sql_fixture_lines(old_text: str, new_text: str) -> tuple[bool, list[int]]:
    """Compare two load_sql_fixtures .sql files (identity.sql, pretty.sql, etc.)."""
    def _parse(text: str) -> list[str]:
        filtered = "\n".join(line for line in text.splitlines() if line and not line.startswith("--"))
        out = []
        buf: list[str] = []
        for line in filtered.splitlines():
            buf.append(line)
            if not line.startswith("#"):
                out.append("\n".join(buf))
                buf.clear()
        return out

    old_cases = _parse(old_text)
    new_cases = _parse(new_text)
    if len(old_cases) != len(new_cases):
        return True, []
    changed = [idx for idx, (o, n) in enumerate(zip(old_cases, new_cases), start=1) if o != n]
    return False, changed

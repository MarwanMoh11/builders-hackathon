"""Candidate test & subtest selector using precomputed sqlglot@eb11aec dependency maps."""

from __future__ import annotations

import ast
import marshal
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CLASS_DEF_RE = re.compile(r"^class\s+(\w+)\s*[\(:]", re.MULTILINE)

from patch_analyzer import (
    ChangeUnit,
    FilePatch,
    PyFileAnalysis,
    analyze_python_change,
    apply_file_patch,
    diff_sql_fixture_lines,
    diff_sql_fixture_pairs,
    parse_unified_diff,
)


@dataclass
class CommitPlan:
    commit: str
    is_noop: bool
    has_syntax_error: bool = False
    syntax_error_files: list[str] = field(default_factory=list)
    patched_files: dict[str, str] = field(default_factory=dict)
    deleted_files: set[str] = field(default_factory=set)
    py_analyses: dict[str, PyFileAnalysis] = field(default_factory=dict)
    candidates: set[int] = field(default_factory=set)
    subtest_filters: dict[int, dict[str, set[Any] | None]] = field(default_factory=dict)
    deferred_subtests: dict[int, list[tuple[str, Any]]] = field(default_factory=dict)
    group: str = "full_fork"
    exec_engine_touched: bool = False
    tpcds_fixture_touched: bool = False
    tpch_fixture_touched: bool = False
    has_max_nodes_edit: bool = False


class CandidateEngine:
    def __init__(
        self,
        maps: dict[str, Any],
        tests_list: list[dict],
        sqlglot_root: Path,
        subtests_dir: Path | None = None,
    ) -> None:
        self.maps = maps
        self.sqlglot_root = sqlglot_root
        self.tests_list = tests_list
        self.subtests_dir = subtests_dir

        self.id_to_num: dict[str, int] = {t["id"]: t["number"] for t in tests_list}
        self.num_to_test: dict[int, dict] = {t["number"]: t for t in tests_list}

        map_test_ids: list[str] = maps["test_ids"]
        self.map_tid_to_num: dict[int, int] = {}
        self.num_to_map_tid: dict[int, int] = {}
        for mtid, tid_str in enumerate(map_test_ids):
            num = self.id_to_num.get(tid_str)
            if num is not None:
                self.map_tid_to_num[mtid] = num
                self.num_to_map_tid[num] = mtid

        self.file_to_test_nums: dict[str, list[int]] = {}
        for t in tests_list:
            self.file_to_test_nums.setdefault(t["file"], []).append(t["number"])

        # Lazy subtest group cache (v2 maps store maps/subtests/<mtid>.zlib separately!)
        self._subtest_tags_by_num: dict[int, list[str]] = {}
        self._loaded_subtests: dict[int, list[dict[str, Any]]] = {}
        if "subtest_index" in maps:
            for mtid, tags in maps["subtest_index"].items():
                rnum = self.map_tid_to_num.get(mtid)
                if rnum is not None:
                    self._subtest_tags_by_num[rnum] = tags
        elif "subtests" in maps:
            # Backward compat with v1 monolithic map
            for sg in maps["subtests"].values():
                rnum = self.map_tid_to_num.get(sg["tid"])
                if rnum is not None:
                    self._loaded_subtests.setdefault(rnum, []).append(sg)
                    self._subtest_tags_by_num.setdefault(rnum, []).append(sg["tag"])

        # Pre-grouped dict_keys_by_oa: owner -> attr -> key -> list[mtid]
        if "dict_keys_by_oa" in maps:
            self.dict_keys_by_oa: dict[str, dict[str, dict[str, list[int]]]] = maps["dict_keys_by_oa"]
        else:
            dk_oa: dict[str, dict[str, dict[str, list[int]]]] = {}
            for full_k, mtids in maps["dict_key_to_tests"].items():
                o, a, k = full_k.split("|", 2)
                dk_oa.setdefault(o, {}).setdefault(a, {})[k] = mtids
            self.dict_keys_by_oa = dk_oa

        self.typing_test_num = self.id_to_num.get(
            "tests/test_typing.py::TestTyping::test_no_redundant_expression_metadata"
        )
        self.lazy_load_test_num = self.id_to_num.get(
            "tests/dialects/test_dialect.py::TestDialect::test_lazy_load"
        )
        self.dialect_meta_attrs: set[str] = set(maps["dialect_meta_attrs"])
        self.expr_meta_attrs: set[str] = set(maps["expr_meta_attrs"])
        self.all_expr_classes: set[str] = set(maps["class_subclasses"].get("Expr", []))
        self.all_dialect_family: set[str] = (
            set(maps["class_subclasses"].get("Dialect", []))
            | set(maps["class_subclasses"].get("Tokenizer", []))
            | set(maps["class_subclasses"].get("Parser", []))
            | set(maps["class_subclasses"].get("Generator", []))
        )
        self.pure_enums: set[str] = set(maps.get("pure_enums", []))
        self.strict_enum_attrs: set[tuple[str, str]] = {
            tuple(x) for x in maps.get("strict_enum_attrs", [])
        }
        self._file_cache: dict[str, str] = {}
        self._old_ast_cache: dict[str, Any] = {}
        self._rev_import_graph: dict[str, set[str]] | None = None
        self._file_classes_cache: dict[str, list[str]] = {}
        # Ignore pure docstring edits ONLY when tests/test_docs.py is absent from runtime tests.json
        self.ignore_docstrings: bool = "tests/test_docs.py" not in self.file_to_test_nums

    def _get_rev_import_graph(self) -> dict[str, set[str]]:
        """Lazily build EXACT top-level static reverse-import graph across sqlglot/**/*.py
        (ignoring TYPE_CHECKING blocks, docstrings, comments, and function-local imports)."""
        if self._rev_import_graph is not None:
            return self._rev_import_graph
        mod_to_file: dict[str, str] = {}
        sg_dir = self.sqlglot_root / "sqlglot"
        py_files = list(sg_dir.rglob("*.py")) if sg_dir.exists() else []
        for p in py_files:
            rel = p.relative_to(self.sqlglot_root).as_posix()
            if rel.endswith("/__init__.py"):
                mod = rel.removesuffix("/__init__.py").replace("/", ".")
            else:
                mod = rel.removesuffix(".py").replace("/", ".")
            mod_to_file[mod] = rel

        def _scan_stmts(stmts: list[ast.stmt], cur_pkg: str, targets: list[str]) -> None:
            for s in stmts:
                if isinstance(s, ast.Import):
                    for alias in s.names:
                        targets.append(alias.name)
                elif isinstance(s, ast.ImportFrom):
                    if s.level and s.level > 0:
                        parts = cur_pkg.split(".")
                        base = ".".join(parts[: max(0, len(parts) - (s.level - 1))])
                        mod_name = f"{base}.{s.module}" if s.module else base
                    else:
                        mod_name = s.module or ""
                    if mod_name:
                        targets.append(mod_name)
                        for alias in s.names:
                            if alias.name != "*":
                                targets.append(f"{mod_name}.{alias.name}")
                elif isinstance(s, ast.If):
                    # Skip `if TYPE_CHECKING:` blocks (not executed at runtime!)
                    test = s.test
                    is_tc = (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
                        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
                    )
                    if not is_tc:
                        _scan_stmts(s.body, cur_pkg, targets)
                    _scan_stmts(s.orelse, cur_pkg, targets)
                elif isinstance(s, ast.Try):
                    _scan_stmts(s.body, cur_pkg, targets)
                    for h in s.handlers:
                        _scan_stmts(h.body, cur_pkg, targets)
                    _scan_stmts(s.orelse, cur_pkg, targets)
                    _scan_stmts(s.finalbody, cur_pkg, targets)

        rev: dict[str, set[str]] = {}
        for p in py_files:
            rel = p.relative_to(self.sqlglot_root).as_posix()
            cur_pkg = (
                rel.removesuffix("/__init__.py").replace("/", ".")
                if rel.endswith("/__init__.py")
                else rel.rsplit("/", 1)[0].replace("/", ".")
            )
            txt = self._read_base_file(rel)
            try:
                tree = ast.parse(txt, filename=rel)
            except SyntaxError:
                continue
            self._file_classes_cache[rel] = [s.name for s in tree.body if isinstance(s, ast.ClassDef)]
            targets: list[str] = []
            _scan_stmts(tree.body, cur_pkg, targets)
            for t in targets:
                tf = mod_to_file.get(t)
                if tf and tf != rel:
                    rev.setdefault(tf, set()).add(rel)
        self._rev_import_graph = rev
        return rev

    def _get_subtests(self, rnum: int) -> list[dict[str, Any]]:
        if rnum not in self._subtest_tags_by_num:
            return []
        loaded = self._loaded_subtests.get(rnum)
        if loaded is not None:
            return loaded
        mtid = self.num_to_map_tid.get(rnum)
        if mtid is None or self.subtests_dir is None:
            return []
        p = self.subtests_dir / f"{mtid}.zlib"
        if not p.exists():
            self._loaded_subtests[rnum] = []
            return []
        groups = marshal.loads(zlib.decompress(p.read_bytes()))
        self._loaded_subtests[rnum] = groups
        return groups

    def _read_base_file(self, rel_path: str) -> str:
        txt = self._file_cache.get(rel_path)
        if txt is None:
            p = self.sqlglot_root / rel_path
            txt = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
            self._file_cache[rel_path] = txt
        return txt

    def _to_rnums(self, mtids: list[int] | set[int]) -> set[int]:
        m = self.map_tid_to_num
        out: set[int] = set()
        for t in mtids:
            r = m.get(t)
            if r is not None:
                out.add(r)
        return out

    def _match_dict_key_in_bucket(self, bucket: dict[str, list[Any]], ast_key_repr: str) -> list[Any]:
        hits: list[Any] = []
        if "*" in bucket:
            hits.extend(bucket["*"])
        if ast_key_repr in bucket:
            hits.extend(bucket[ast_key_repr])
            return hits
        if not ast_key_repr.startswith(("'", '"')):
            last = ast_key_repr.rsplit(".", 1)[-1]
            t_key = f"TYPE:{last}"
            if t_key in bucket:
                hits.extend(bucket[t_key])
            suffix = f".{last}"
            angle_mid = f".{last}:"
            for rk, vals in bucket.items():
                if (rk.startswith("ENUM:") and rk.endswith(suffix)) or (rk.startswith("<") and angle_mid in rk):
                    hits.extend(vals)
        return hits

    def _match_dict_key(self, owner: str, attr: str, ast_key_repr: str) -> list[int]:
        o_map = self.dict_keys_by_oa.get(owner)
        if not o_map:
            return []
        bucket = o_map.get(attr)
        if not bucket:
            return []
        return self._match_dict_key_in_bucket(bucket, ast_key_repr)

    def _subtest_match_lines(self, sg: dict[str, Any], rel: str, lines: set[int]) -> set[Any]:
        lm = sg["lines"].get(rel)
        if not lm:
            return set()
        out: set[Any] = set()
        for ln in lines:
            cids = lm.get(ln)
            if cids:
                out.update(cids)
        return out

    def _add_subtest_filter(
        self,
        filters: dict[int, dict[str, set[Any] | None]],
        rnum: int,
        tag: str,
        cases: set[Any] | None,
    ) -> None:
        tf = filters.setdefault(rnum, {})
        if tag not in tf:
            tf[tag] = None if cases is None else set(cases)
        else:
            if tf[tag] is None or cases is None:
                tf[tag] = None
            else:
                tf[tag].update(cases)  # type: ignore[union-attr]

    def _subclasses_of(self, class_name: str | None) -> list[str]:
        if not class_name:
            return []
        subs = set(self.maps["class_subclasses"].get(class_name, []))
        if not subs and "." in class_name:
            leaf = class_name.rsplit(".", 1)[-1]
            parent_d = class_name.split(".", 1)[0]
            for dsub in self.maps["class_subclasses"].get(parent_d, [parent_d]):
                subs.add(f"{dsub}.{leaf}")
                subs.add(dsub)
        if not subs:
            subs.add(class_name)
        return sorted(subs)

    def _tests_covering_lines(self, rel: str, lines: set[int]) -> set[int]:
        lm = self.maps["line_cov"].get(rel)
        if not lm:
            return set()
        out: set[int] = set()
        for ln in lines:
            mtids = lm.get(ln)
            if mtids:
                out.update(self._to_rnums(mtids))
        return out

    def _tests_for_broken_import(self, rel: str) -> set[int]:
        out: set[int] = set()
        if rel.startswith("tests/"):
            if rel in self.file_to_test_nums:
                out.update(self.file_to_test_nums[rel])
            for tf in self.maps["import_file_to_test_files"].get(rel, []):
                out.update(self.file_to_test_nums.get(tf, []))
            out.update(self._to_rnums(self.maps["file_cov"].get(rel, [])))
            return out

        # Transitive static reverse-import closure across sqlglot/
        rev_graph = self._get_rev_import_graph()
        closure: set[str] = {rel}
        stack = [rel]
        while stack:
            cur = stack.pop()
            for nxt in rev_graph.get(cur, ()):
                if nxt not in closure:
                    closure.add(nxt)
                    stack.append(nxt)

        touches_dialect_pkg = False
        for f in closure:
            for tf in self.maps["import_file_to_test_files"].get(f, []):
                out.update(self.file_to_test_nums.get(tf, []))
            out.update(self._to_rnums(self.maps["dyn_import_to_tests"].get(f, [])))
            out.update(self._to_rnums(self.maps["file_cov"].get(f, [])))

            if f.startswith(("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/")):
                touches_dialect_pkg = True

            # Any dialect/parser/generator/expr class defined in a broken file cascades to all tests using it or its subclasses!
            classes = self._file_classes_cache.get(f)
            if classes is None:
                classes = _CLASS_DEF_RE.findall(self._read_base_file(f))
                self._file_classes_cache[f] = classes
            for cls_name in classes:
                for sc in self._subclasses_of(cls_name):
                    out.update(self._to_rnums(self.maps["dialect_to_tests"].get(sc, [])))
                    out.update(self._to_rnums(self.maps["expr_to_tests"].get(sc, [])))
                out.update(self._to_rnums(self.maps["dialect_to_tests"].get(cls_name, [])))
                out.update(self._to_rnums(self.maps["expr_to_tests"].get(cls_name, [])))

        if touches_dialect_pkg:
            # Tests that iterate over all Dialects (test_enum, test_get_or_raise, test_generator_memory, etc.)
            out.update(self.file_to_test_nums.get("tests/dialects/test_dialect.py", []))
            gmem = self.id_to_num.get("tests/test_generator.py::TestGenerator::test_generator_memory")
            if gmem is not None:
                out.add(gmem)
        return out

    def resolve_subtest_filter(self, commit_plan: CommitPlan, rnum: int) -> dict[str, set[Any] | None] | None:
        """Lazily decompress maps/subtests/<mtid>.zlib and compute subtest filters ONLY when
        runner.py actually decides to verify test rnum!"""
        specs = commit_plan.deferred_subtests.pop(rnum, None)
        if specs:
            sgs = self._get_subtests(rnum)
            if sgs:
                for kind, payload in specs:
                    if kind == "lines":
                        rel, old_lines = payload
                        for sg in sgs:
                            matched = self._subtest_match_lines(sg, rel, old_lines)
                            self._add_subtest_filter(commit_plan.subtest_filters, rnum, sg["tag"], matched)
                    elif kind == "exprs":
                        subs = payload
                        for sg in sgs:
                            matched_e: set[Any] = set()
                            for sc in subs:
                                matched_e.update(sg["exprs"].get(sc, []))
                            self._add_subtest_filter(commit_plan.subtest_filters, rnum, sg["tag"], matched_e if matched_e else None)
                    elif kind == "dict_meta":
                        owners = payload
                        for sg in sgs:
                            matched_m: set[Any] = set()
                            for o in owners:
                                matched_m.update(sg["dialects"].get(o.split(".")[0], []))
                                matched_m.update(sg["dialects"].get(o, []))
                            self._add_subtest_filter(commit_plan.subtest_filters, rnum, sg["tag"], matched_m if matched_m else None)
                    elif kind == "dict_keys":
                        owners, attr, dict_keys, rel, old_lines = payload
                        for sg in sgs:
                            matched_d: set[Any] = set()
                            dk_oa = sg.get("dict_keys_by_oa")
                            if dk_oa is not None:
                                for o in owners:
                                    b = dk_oa.get(o, {}).get(attr)
                                    if b:
                                        for kr in dict_keys:
                                            matched_d.update(self._match_dict_key_in_bucket(b, kr))
                            else:
                                for dk_full, cids in sg.get("dict_keys", {}).items():
                                    do, da, dk = dk_full.split("|", 2)
                                    if da == attr and do in owners:
                                        if dk == "*":
                                            matched_d.update(cids)
                                        else:
                                            for kr in dict_keys:
                                                if dk == kr or (
                                                    not kr.startswith(("'", '"'))
                                                    and (
                                                        dk == f"TYPE:{kr.rsplit('.', 1)[-1]}"
                                                        or (dk.startswith("ENUM:") and dk.endswith(f".{kr.rsplit('.', 1)[-1]}"))
                                                        or (dk.startswith("<") and f".{kr.rsplit('.', 1)[-1]}:" in dk)
                                                    )
                                                ):
                                                    matched_d.update(cids)
                            matched_d.update(self._subtest_match_lines(sg, rel, old_lines))
                            self._add_subtest_filter(commit_plan.subtest_filters, rnum, sg["tag"], matched_d)
                    elif kind == "dialect_readers":
                        subs, base_cls, readers = payload
                        for sg in sgs:
                            d_cases: set[Any] = set()
                            for sc in subs:
                                d_cases.update(sg["dialects"].get(sc, []))
                                if "." in sc:
                                    d_cases.update(sg["dialects"].get(sc.split(".")[0], []))
                            d_cases.update(sg["dialects"].get(base_cls, []))
                            r_cases: set[Any] = set()
                            for rfile, rline in readers:
                                r_cases.update(self._subtest_match_lines(sg, rfile, {rline}))
                            inter = d_cases & r_cases if (d_cases and r_cases) else (d_cases or r_cases)
                            self._add_subtest_filter(commit_plan.subtest_filters, rnum, sg["tag"], inter if inter else None)
                    elif kind == "dialect":
                        subs, base_cls = payload
                        for sg in sgs:
                            d_cases2: set[Any] = set()
                            for sc in subs:
                                d_cases2.update(sg["dialects"].get(sc, []))
                                if "." in sc:
                                    d_cases2.update(sg["dialects"].get(sc.split(".")[0], []))
                            d_cases2.update(sg["dialects"].get(base_cls, []))
                            self._add_subtest_filter(commit_plan.subtest_filters, rnum, sg["tag"], d_cases2 if d_cases2 else None)
        return commit_plan.subtest_filters.get(rnum)

    def _handle_change_unit(
        self,
        unit: ChangeUnit,
        candidates: set[int],
        subtest_filters: dict[int, dict[str, set[Any] | None]],
        deferred_subtests: dict[int, list[tuple[str, Any]]],
    ) -> None:
        rel = unit.file
        has_sub = self._subtest_tags_by_num

        if unit.kind == "syntax_error":
            broken = self._tests_for_broken_import(rel)
            candidates.update(broken)
            for rnum in broken:
                for tag in has_sub.get(rnum, ()):
                    self._add_subtest_filter(subtest_filters, rnum, tag, None)
            return

        if rel.startswith("tests/"):
            if unit.kind == "func_body" and unit.symbol_name and "::" not in unit.symbol_name:
                parts = unit.symbol_name.split(".")
                if len(parts) == 2 and parts[1].startswith("test_"):
                    exact_id = f"{rel}::{parts[0]}::{parts[1]}"
                    rnum = self.id_to_num.get(exact_id)
                    if rnum is not None:
                        candidates.add(rnum)
                        for tag in has_sub.get(rnum, ()):
                            self._add_subtest_filter(subtest_filters, rnum, tag, None)
                        return
            affected = set(self.file_to_test_nums.get(rel, []))
            for tf in self.maps["import_file_to_test_files"].get(rel, []):
                affected.update(self.file_to_test_nums.get(tf, []))
            affected.update(self._to_rnums(self.maps["file_cov"].get(rel, [])))
            candidates.update(affected)
            for rnum in affected:
                for tag in has_sub.get(rnum, ()):
                    self._add_subtest_filter(subtest_filters, rnum, tag, None)
            return

        if rel.startswith("sqlglotc/"):
            candidates.update(self.file_to_test_nums.get("tests/test_sqlglotc.py", []))
            return

        if rel.startswith("sqlglot/typing/") and self.typing_test_num is not None:
            candidates.add(self.typing_test_num)

        if unit.kind == "func_body":
            target_lines = unit.old_lines
            hits = self._tests_covering_lines(rel, target_lines)
            # Safety net: if changed/anchor lines fell on non-executable lines (e.g. docstring/blank),
            # fall back to the full enclosing function line span so we NEVER false-noop!
            if not hits and unit.enclosing_lines:
                target_lines = unit.enclosing_lines
                hits = self._tests_covering_lines(rel, target_lines)
            candidates.update(hits)
            spec = ("lines", (rel, target_lines))
            for rnum in hits:
                if rnum in has_sub:
                    deferred_subtests.setdefault(rnum, []).append(spec)
            return

        if unit.kind == "dict_key":
            attr = unit.symbol_name or ""
            if attr == "arg_types" and unit.class_name:
                subs = self._subclasses_of(unit.class_name)
                hits: set[int] = set()
                for sc in subs:
                    hits.update(self._to_rnums(self.maps["expr_to_tests"].get(sc, [])))
                hits.update(self._tests_covering_lines(rel, unit.old_lines))
                candidates.update(hits)
                spec = ("exprs", subs)
                for rnum in hits:
                    if rnum in has_sub:
                        deferred_subtests.setdefault(rnum, []).append(spec)
                return

            owners: list[str] = []
            if unit.class_name:
                if unit.class_name == "Tokenizer":
                    owners = [o for o in self.dict_keys_by_oa if "Tokenizer" in o]
                elif unit.class_name == "Parser":
                    owners = [o for o in self.dict_keys_by_oa if "Parser" in o]
                elif unit.class_name == "Generator":
                    owners = [o for o in self.dict_keys_by_oa if "Generator" in o]
                elif unit.class_name == "Dialect":
                    owners = list(self.dict_keys_by_oa.keys())
                else:
                    owners = self._subclasses_of(unit.class_name)
                    if unit.class_name not in owners:
                        owners.append(unit.class_name)
            elif rel.startswith("sqlglot/typing/"):
                mod_owner = rel.removesuffix(".py").replace("/", ".")
                owners = [mod_owner]
                for o, o_map in self.dict_keys_by_oa.items():
                    if attr in o_map and o.startswith("sqlglot.typing.") and o not in owners:
                        owners.append(o)

            tracked = any(attr in self.dict_keys_by_oa.get(o, {}) for o in owners)
            if tracked:
                mtid_hits: set[int] = set()
                if unit.dict_keys is None:
                    for o in owners:
                        for mtids in self.dict_keys_by_oa.get(o, {}).get(attr, {}).values():
                            mtid_hits.update(mtids)
                else:
                    for o in owners:
                        for kr in unit.dict_keys:
                            mtid_hits.update(self._match_dict_key(o, attr, kr))

                hits = self._to_rnums(mtid_hits)
                hits.update(self._tests_covering_lines(rel, unit.old_lines))

                meta_rebuild_attrs = {
                    "TIME_MAPPING", "FORMAT_MAPPING", "INVERSE_TIME_MAPPING", "INVERSE_FORMAT_MAPPING",
                    "UNESCAPED_SEQUENCES", "DATE_PART_MAPPING", "CREATABLE_KIND_MAPPING",
                    "QUOTES", "IDENTIFIERS", "STRING_ESCAPES", "BYTE_STRING_ESCAPES",
                    "IDENTIFIER_ESCAPES", "SUPPORTED_JSON_PATH_PARTS",
                }
                if attr in meta_rebuild_attrs or (attr == "KEYWORDS" and (unit.dict_keys is None or "'(+)'" in unit.dict_keys)):
                    for o in owners:
                        base_d = o.split(".")[0]
                        hits.update(self._to_rnums(self.maps["dialect_to_tests"].get(base_d, [])))
                        hits.update(self._to_rnums(self.maps["dialect_to_tests"].get(o, [])))

                candidates.update(hits)
                if attr in meta_rebuild_attrs or unit.dict_keys is None:
                    spec = ("dict_meta", owners)
                else:
                    spec = ("dict_keys", (owners, attr, unit.dict_keys, rel, unit.old_lines))
                for rnum in hits:
                    if rnum in has_sub:
                        deferred_subtests.setdefault(rnum, []).append(spec)
                return

        if unit.kind in ("class_attr", "dict_key") and unit.class_name:
            attr = unit.symbol_name or "*"
            subs = self._subclasses_of(unit.class_name)
            base_cls = unit.class_name.split(".")[0]

            if base_cls in self.all_expr_classes or unit.class_name in self.all_expr_classes:
                hits = set()
                for sc in subs:
                    hits.update(self._to_rnums(self.maps["expr_to_tests"].get(sc, [])))
                hits.update(self._tests_covering_lines(rel, unit.old_lines))
                if not hits:
                    hits.update(self._to_rnums(self.maps["file_cov"].get(rel, [])))
                candidates.update(hits)
                spec = ("exprs", subs)
                for rnum in hits:
                    if rnum in has_sub:
                        deferred_subtests.setdefault(rnum, []).append(spec)
                return

            if base_cls in self.all_dialect_family or unit.class_name in self.all_dialect_family:
                dialect_hits: set[int] = set()
                for sc in subs:
                    dialect_hits.update(self._to_rnums(self.maps["dialect_to_tests"].get(sc, [])))
                    if "." in sc:
                        dialect_hits.update(self._to_rnums(self.maps["dialect_to_tests"].get(sc.split(".")[0], [])))
                dialect_hits.update(self._to_rnums(self.maps["dialect_to_tests"].get(base_cls, [])))

                readers = self.maps["attr_readers"].get(attr, []) if attr != "*" else []
                if attr != "*" and attr not in self.dialect_meta_attrs and readers:
                    reader_hits: set[int] = set()
                    for rfile, rline in readers:
                        reader_hits.update(self._tests_covering_lines(rfile, {rline}))
                    hits = dialect_hits & reader_hits
                    hits.update(self._tests_covering_lines(rel, unit.old_lines))
                    candidates.update(hits)
                    spec = ("dialect_readers", (subs, base_cls, readers))
                    for rnum in hits:
                        if rnum in has_sub:
                            deferred_subtests.setdefault(rnum, []).append(spec)
                    return

                candidates.update(dialect_hits)
                spec = ("dialect", (subs, base_cls))
                for rnum in dialect_hits:
                    if rnum in has_sub:
                        deferred_subtests.setdefault(rnum, []).append(spec)
                return

        # Fallback: module_attr / module_other / non-dialect class_attr
        sym = unit.symbol_name or "*"
        hits = set()
        if sym != "*" and sym in self.maps["name_readers"]:
            for rfile, rline in self.maps["name_readers"][sym]:
                hits.update(self._tests_covering_lines(rfile, {rline}))
        if sym != "*" and sym in self.maps["attr_readers"]:
            for rfile, rline in self.maps["attr_readers"][sym]:
                hits.update(self._tests_covering_lines(rfile, {rline}))
        if not hits or sym == "*":
            hits.update(self._to_rnums(self.maps["file_cov"].get(rel, [])))
            hits.update(self._tests_for_broken_import(rel))
        else:
            hits.update(self._tests_covering_lines(rel, unit.old_lines))
        candidates.update(hits)
        for rnum in hits:
            for tag in has_sub.get(rnum, ()):
                self._add_subtest_filter(subtest_filters, rnum, tag, None)

    def plan_commit(self, commit_id: str, patch_text: str) -> CommitPlan:
        fps = parse_unified_diff(patch_text)
        patched_files: dict[str, str] = {}
        deleted_files: set[str] = set()
        py_analyses: dict[str, PyFileAnalysis] = {}
        candidates: set[int] = set()
        subtest_filters: dict[int, dict[str, set[Any] | None]] = {}
        deferred_subtests: dict[int, list[tuple[str, Any]]] = {}

        has_non_noop = False
        has_syntax_error = False
        syntax_error_files: list[str] = []
        only_fixtures = True
        all_hot_func = True
        exec_engine_touched = False
        tpcds_fixture_touched = False
        tpch_fixture_touched = False
        has_max_nodes_edit = False

        for fp in fps:
            rel = fp.path
            if not rel:
                continue

            if fp.is_binary:
                has_non_noop = True
                only_fixtures = False
                all_hot_func = False
                if "tpc-ds/" in rel:
                    tpcds_fixture_touched = True
                    exec_engine_touched = True
                if "tpc-h/" in rel:
                    tpch_fixture_touched = True
                    exec_engine_touched = True
                candidates.update(self._to_rnums(self.maps["fixture_to_tests"].get(rel, [])))
                continue

            old_text = self._read_base_file(rel)
            new_text = apply_file_patch(old_text, fp)

            if rel.endswith(".py"):
                ana = analyze_python_change(
                    rel,
                    old_text,
                    new_text,
                    fp,
                    old_ast_cache=self._old_ast_cache,
                    pure_enums=self.pure_enums,
                    strict_enum_attrs=self.strict_enum_attrs,
                    ignore_docstrings=self.ignore_docstrings,
                )
                if ana.is_noop:
                    continue
                has_non_noop = True
                only_fixtures = False
                if fp.is_deleted:
                    deleted_files.add(rel)
                    all_hot_func = False
                else:
                    patched_files[rel] = new_text
                py_analyses[rel] = ana

                if rel.startswith(("sqlglot/executor/", "sqlglot/planner.py", "tests/test_executor.py", "tests/helpers.py")):
                    exec_engine_touched = True

                if (
                    rel in ("tests/test_parser.py", "sqlglot/parsers/athena.py")
                    or (rel == "sqlglot/parser.py" and any(fn in ("Parser.expression", "Parser.__init__") for fn in ana.changed_funcs))
                    or any(
                        hl.startswith(("+", "-"))
                        and ("max_nodes" in hl or "_node_count" in hl or "1_000_000" in hl or "1000000" in hl)
                        for h in fp.hunks
                        for hl in h.lines
                    )
                ):
                    has_max_nodes_edit = True

                if ana.syntax_error:
                    has_syntax_error = True
                    syntax_error_files.append(rel)
                    all_hot_func = False

                if rel.startswith(("tests/", "sqlglotc/", "sqlglot/typing/")):
                    all_hot_func = False

                for unit in ana.units:
                    if unit.kind != "func_body":
                        all_hot_func = False
                    self._handle_change_unit(unit, candidates, subtest_filters, deferred_subtests)
            else:
                f_hits = self._to_rnums(self.maps["fixture_to_tests"].get(rel, []))
                if not f_hits:
                    continue

                if rel.endswith(".sql"):
                    if (
                        "qualify_columns__invalid" in rel
                        or rel.endswith(("tests/fixtures/identity.sql", "tests/fixtures/partial.sql"))
                    ):
                        err_or_shift, changed_idxs = diff_sql_fixture_lines(old_text, new_text)
                    else:
                        err_or_shift, changed_idxs = diff_sql_fixture_pairs(old_text, new_text)

                    if not err_or_shift and not changed_idxs:
                        continue

                    has_non_noop = True
                    if fp.is_deleted:
                        deleted_files.add(rel)
                        all_hot_func = False
                    else:
                        patched_files[rel] = new_text
                    if "tpc-ds" in rel:
                        tpcds_fixture_touched = True
                    if "tpc-h" in rel:
                        tpch_fixture_touched = True

                    candidates.update(f_hits)
                    for rnum in f_hits:
                        for tag in self._subtest_tags_by_num.get(rnum, []):
                            if tag == rel or (
                                "tpc-ds.sql" in rel and tag == "tpc-ds"
                            ) or (
                                "tpc-h.sql" in rel and tag in ("tpc-h", "opt-tpch")
                            ):
                                self._add_subtest_filter(
                                    subtest_filters,
                                    rnum,
                                    tag,
                                    None if err_or_shift else set(changed_idxs),
                                )
                            else:
                                # Untouched companion fixture file (e.g. qualify_columns.sql when only qualify_columns_ddl.sql changed!)
                                self._add_subtest_filter(subtest_filters, rnum, tag, set())
                else:
                    if old_text == new_text:
                        continue
                    has_non_noop = True
                    all_hot_func = False
                    if fp.is_deleted:
                        deleted_files.add(rel)
                    else:
                        patched_files[rel] = new_text
                    candidates.update(f_hits)
                    for rnum in f_hits:
                        for tag in self._subtest_tags_by_num.get(rnum, []):
                            self._add_subtest_filter(subtest_filters, rnum, tag, None)

        if not has_non_noop or not candidates:
            return CommitPlan(commit=commit_id, is_noop=True, group="noop")

        if self.lazy_load_test_num in candidates:
            all_hot_func = False

        if only_fixtures:
            group = "fixture"
        elif all_hot_func:
            group = "hot_func"
        else:
            group = "full_fork"

        return CommitPlan(
            commit=commit_id,
            is_noop=False,
            has_syntax_error=has_syntax_error,
            syntax_error_files=syntax_error_files,
            patched_files=patched_files,
            deleted_files=deleted_files,
            py_analyses=py_analyses,
            candidates=candidates,
            subtest_filters=subtest_filters,
            deferred_subtests=deferred_subtests,
            group=group,
            exec_engine_touched=exec_engine_touched,
            tpcds_fixture_touched=tpcds_fixture_touched,
            tpch_fixture_touched=tpch_fixture_touched,
            has_max_nodes_edit=has_max_nodes_edit,
        )

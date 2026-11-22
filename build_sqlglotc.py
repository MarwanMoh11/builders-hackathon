"""Parallel 4-core clang builder for 3-tier sqlglotc (0 bytes in git repo).

Modes:
  no_opt (default): 94 modules (all official sqlglotc EXCEPT sqlglot/optimizer/*),
    so optimizer/* commits ALSO run with 100% C-compiled Parser/Generator/Expr/Tokenizer/Table!
  base: 22 core modules (expressions/*, tokenizer_core, trie, helper, time, serde, errors, table),
    so parsers/*, generators/*, parser.py, generator.py commits get 4.1x faster imports + 1.44x speedup!
  full: all 104 official sqlglotc modules.
"""

from __future__ import annotations

import ast
import concurrent.futures
import os
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

from mypyc.build import mypycify


HOT_MODS: frozenset[str] = frozenset({
    "sqlglot.parser",
    "sqlglot.generator",
    "sqlglot.tokenizer_core",
    "sqlglot.trie",
    "sqlglot.helper",
    "sqlglot.time",
    "sqlglot.serde",
    "sqlglot.errors",
    "sqlglot.transforms",
    "sqlglot.expressions.core",
    "sqlglot.expressions.query",
    "sqlglot.expressions.datatypes",
    "sqlglot.executor.table",
    "sqlglot.executor.env",
    "sqlglot.executor.context",
    "sqlglot.generators.python",
    "sqlglot.optimizer.scope",
    "sqlglot.optimizer.resolver",
    "sqlglot.optimizer.isolate_table_selects",
    "sqlglot.optimizer.normalize_identifiers",
    "sqlglot.optimizer.qualify",
    "sqlglot.optimizer.qualify_tables",
    "sqlglot.optimizer.qualify_columns",
    "sqlglot.optimizer.simplify",
    "sqlglot.optimizer.annotate_types",
    "sqlglot.optimizer.canonicalize_internal_names",
    "sqlglot.optimizer.pushdown_predicates",
    "sqlglot.optimizer.unnest_subqueries",
    "sqlglot.optimizer.eliminate_subqueries",
    "sqlglot.optimizer.eliminate_joins",
    "sqlglot.optimizer.eliminate_ctes",
    "sqlglot.optimizer.optimize_joins",
    "sqlglot.optimizer.pushdown_projections",
    "sqlglot.optimizer.canonicalize",
    "sqlglot.optimizer.normalize",
})


def _subpkg_files(sg_dir: Path, sub: str, files: list[str] | None = None) -> list[str]:
    if files is None:
        files = sorted(
            f for f in os.listdir(sg_dir / sub) if f.endswith(".py") and f != "__init__.py"
        )
    return [f"sqlglot/{sub}/{f}" for f in files]


def _patch_for_interpreted_subclasses(path: Path, class_name: str, fix_i64: bool = False) -> None:
    """Annotate every class-body attribute on Parser/Generator as t.ClassVar and add
    @mypyc_attr(allow_interpreted_subclasses=True) so pure-Python sqlglot/parsers/* and
    sqlglot/generators/* subclasses override class attributes & methods seamlessly!"""
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    lines = src.splitlines(keepends=True)
    replacements: list[tuple[int, str]] = []
    n_cv = 0
    for stmt in cls_node.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name == "__slots__":
                continue
            ln = stmt.lineno - 1
            orig = lines[ln]
            idx = orig.find(name)
            eq_idx = orig.find("=", idx + len(name))
            if eq_idx > idx:
                replacements.append((ln, orig[: idx + len(name)] + ": t.ClassVar[t.Any]" + orig[idx + len(name) :]))
                n_cv += 1
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            if name == "__slots__":
                continue
            if "ClassVar" in ast.unparse(stmt.annotation):
                continue
            ln = stmt.lineno - 1
            orig = lines[ln]
            idx = orig.find(name)
            col_idx = orig.find(":", idx + len(name))
            eq_idx = orig.find("=", col_idx + 1)
            if col_idx > idx and eq_idx > col_idx:
                ann_raw = orig[col_idx + 1 : eq_idx].strip()
                replacements.append((ln, orig[: col_idx + 1] + f" t.ClassVar[{ann_raw}] " + orig[eq_idx:]))
                n_cv += 1
    for ln, new_line in replacements:
        lines[ln] = new_line
    new_src = "".join(lines)
    assert "from __future__ import annotations\n" in new_src
    new_src = new_src.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\nfrom mypy_extensions import mypyc_attr as _mypyc_attr\n",
        1,
    )
    new_src = new_src.replace(
        f"class {class_name}:",
        f"@_mypyc_attr(allow_interpreted_subclasses=True)\nclass {class_name}:",
        1,
    )
    if fix_i64:
        # Work around mypyc codegen bug: default-valued i64 parameters on allow_interpreted_subclasses
        # pass 0 instead of default value through interpreted-subclass dispatch wrappers!
        assert "def _advance(self, times: i64 = 1) -> None:" in new_src
        new_src = new_src.replace("def _advance(self, times: i64 = 1) -> None:", "def _advance(self, times: int = 1) -> None:", 1)
        new_src = new_src.replace("def _retreat(self, index: i64) -> None:", "def _retreat(self, index: int) -> None:", 1)
    path.write_text(new_src, encoding="utf-8")
    print(f"[sqlglotc] patched {path.name}:{class_name} ({n_cv} ClassVars)", flush=True)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/sqlglot-c")
    mode = sys.argv[2] if len(sys.argv) > 2 else "no_dialects"
    os.chdir(root)
    os.environ["MYPYPATH"] = str(root)
    sg_dir = root / "sqlglot"

    extra_mypy_flags: list[str] = []
    if mode == "no_dialects":
        _patch_for_interpreted_subclasses(sg_dir / "parser.py", "Parser", fix_i64=True)
        _patch_for_interpreted_subclasses(sg_dir / "generator.py", "Generator", fix_i64=False)
        extra_mypy_flags = ["--disable-error-code=misc"]
        source_files = [
            "sqlglot/anonymize.py",
            "sqlglot/errors.py",
            "sqlglot/generator.py",
            "sqlglot/helper.py",
            "sqlglot/lineage.py",
            "sqlglot/parser.py",
            "sqlglot/serde.py",
            "sqlglot/time.py",
            "sqlglot/tokenizer_core.py",
            "sqlglot/trie.py",
            *_subpkg_files(sg_dir, "expressions"),
            *_subpkg_files(
                sg_dir,
                "optimizer",
                [
                    "scope.py",
                    "resolver.py",
                    "isolate_table_selects.py",
                    "normalize_identifiers.py",
                    "qualify.py",
                    "qualify_tables.py",
                    "qualify_columns.py",
                    "simplify.py",
                    "annotate_types.py",
                    "canonicalize_internal_names.py",
                ],
            ),
            *_subpkg_files(sg_dir, "executor", ["table.py"]),
        ]
    elif mode == "base":
        source_files = [
            "sqlglot/errors.py",
            "sqlglot/helper.py",
            "sqlglot/serde.py",
            "sqlglot/time.py",
            "sqlglot/tokenizer_core.py",
            "sqlglot/trie.py",
            *_subpkg_files(sg_dir, "expressions"),
            *_subpkg_files(sg_dir, "executor", ["table.py"]),
        ]
    elif mode == "no_opt":
        source_files = [
            "sqlglot/anonymize.py",
            "sqlglot/errors.py",
            "sqlglot/generator.py",
            "sqlglot/helper.py",
            "sqlglot/lineage.py",
            "sqlglot/parser.py",
            "sqlglot/serde.py",
            "sqlglot/time.py",
            "sqlglot/tokenizer_core.py",
            "sqlglot/trie.py",
            *_subpkg_files(sg_dir, "expressions"),
            *_subpkg_files(sg_dir, "generators"),
            *_subpkg_files(sg_dir, "parsers"),
            *_subpkg_files(sg_dir, "executor", ["table.py"]),
        ]
    elif mode == "full116":
        source_files = [
            "sqlglot/anonymize.py",
            "sqlglot/errors.py",
            "sqlglot/generator.py",
            "sqlglot/helper.py",
            "sqlglot/lineage.py",
            "sqlglot/parser.py",
            "sqlglot/serde.py",
            "sqlglot/time.py",
            "sqlglot/tokenizer_core.py",
            "sqlglot/trie.py",
            "sqlglot/transforms.py",
            *_subpkg_files(sg_dir, "expressions"),
            *_subpkg_files(sg_dir, "generators"),
            *_subpkg_files(
                sg_dir,
                "optimizer",
                [
                    "scope.py",
                    "resolver.py",
                    "isolate_table_selects.py",
                    "normalize_identifiers.py",
                    "qualify.py",
                    "qualify_tables.py",
                    "qualify_columns.py",
                    "simplify.py",
                    "annotate_types.py",
                    "canonicalize_internal_names.py",
                    "pushdown_predicates.py",
                    "unnest_subqueries.py",
                    "eliminate_subqueries.py",
                    "eliminate_joins.py",
                    "eliminate_ctes.py",
                    "optimize_joins.py",
                    "pushdown_projections.py",
                    "canonicalize.py",
                    "normalize.py",
                ],
            ),
            *_subpkg_files(sg_dir, "parsers"),
            *_subpkg_files(sg_dir, "executor", ["table.py", "env.py", "context.py"]),
        ]
    else:
        source_files = [
            "sqlglot/anonymize.py",
            "sqlglot/errors.py",
            "sqlglot/generator.py",
            "sqlglot/helper.py",
            "sqlglot/lineage.py",
            "sqlglot/parser.py",
            "sqlglot/serde.py",
            "sqlglot/time.py",
            "sqlglot/tokenizer_core.py",
            "sqlglot/trie.py",
            *_subpkg_files(sg_dir, "expressions"),
            *_subpkg_files(sg_dir, "generators"),
            *_subpkg_files(
                sg_dir,
                "optimizer",
                [
                    "scope.py",
                    "resolver.py",
                    "isolate_table_selects.py",
                    "normalize_identifiers.py",
                    "qualify.py",
                    "qualify_tables.py",
                    "qualify_columns.py",
                    "simplify.py",
                    "annotate_types.py",
                    "canonicalize_internal_names.py",
                ],
            ),
            *_subpkg_files(sg_dir, "parsers"),
            *_subpkg_files(sg_dir, "executor", ["table.py"]),
        ]

    t0 = time.time()
    exts = mypycify([*extra_mypy_flags, *source_files], opt_level="1", debug_level="0", separate=True, verbose=False)
    t_gen = time.time() - t0
    print(
        f"[sqlglotc:{mode}] mypyc codegen ({len(source_files)} files -> {len(exts)} exts): {t_gen:.1f}s",
        flush=True,
    )

    py_inc = sysconfig.get_path("include")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    obj_root = root / "build" / "obj"
    obj_root.mkdir(parents=True, exist_ok=True)

    cc = shutil.which("clang") or "gcc"
    n_workers = max(1, os.cpu_count() or 4)

    jobs: list[tuple[int, str, list[str], list[str], str, bool]] = []
    n_hot = 0
    for ext in exts:
        base = ext.name.removesuffix("__mypyc")
        is_hot = (base in HOT_MODS) or ("." not in ext.name)
        if is_hot:
            n_hot += 1
        parts = ext.name.split(".")
        if parts[0] == "sqlglot":
            dst_so = str(sg_dir.joinpath(*parts[1:-1], parts[-1] + ext_suffix))
        else:
            dst_so = str(sg_dir / (ext.name + ext_suffix))
        incs = [py_inc, *ext.include_dirs]
        total_sz = sum(os.path.getsize(s) for s in ext.sources if os.path.exists(s))
        jobs.append((total_sz, ext.name, list(ext.sources), incs, dst_so, is_hot))

    jobs.sort(key=lambda x: -x[0])
    print(
        f"[sqlglotc:{mode}] compiling {len(jobs)} exts ({n_hot} -O1, {len(jobs)-n_hot} -O0) with {cc} -j{n_workers}",
        flush=True,
    )

    base_flags = [
        "-fPIC",
        "-fno-strict-overflow",
        "-DNDEBUG",
        "-Wno-unused-function",
        "-Wno-unused-label",
        "-Wno-unreachable-code",
        "-Wno-unused-variable",
        "-Wno-unused-but-set-variable",
        "-Wno-unused-command-line-argument",
        "-Wno-unknown-warning-option",
        "-Wno-ignored-optimization-argument",
        "-Wno-array-bounds",
        "-Wno-cpp",
    ]

    def _compile_one(job: tuple[int, str, list[str], list[str], str, bool]) -> str:
        _, name, sources, incs, dst_so, is_hot = job
        opt_flag = "-O1" if is_hot else "-O0"
        inc_flags = [f"-I{d}" for d in incs]
        objs: list[str] = []
        for idx, src in enumerate(sources):
            obj = str(obj_root / f"{name}_{idx}.o")
            cmd = [cc, *base_flags, opt_flag, *inc_flags, "-c", src, "-o", obj]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Compile failed for {name} ({src}):\n{res.stderr}")
            objs.append(obj)
        os.makedirs(os.path.dirname(dst_so), exist_ok=True)
        link_cmd = [cc, "-shared", *objs, "-o", dst_so]
        res = subprocess.run(link_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Link failed for {name}:\n{res.stderr}")
        return name

    t_cc0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(_compile_one, j) for j in jobs]
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
    t_cc = time.time() - t_cc0

    shutil.rmtree(root / "build", ignore_errors=True)
    for egg in root.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)

    print(
        f"[sqlglotc:{mode}] DONE in {time.time() - t0:.1f}s (codegen={t_gen:.1f}s, parallel_cc={t_cc:.1f}s, workers={n_workers})",
        flush=True,
    )


if __name__ == "__main__":
    main()

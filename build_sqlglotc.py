"""Parallel 4-core clang builder for Official sqlglotc (104 modules from sqlglotc/setup.py).

Runs inside Dockerfile Stage 1 (adds 0 bytes to git repo).
Uses mypycify(separate=True) for codegen, then compiles all ~208 extensions in true
parallel across all available CPU cores (LPT-sorted largest .c first), applying
-O1 -DNDEBUG to the 29 hot core/expression/optimizer/executor/python-gen modules and
-O0 -DNDEBUG to the remaining 75 dialect/minor modules.
"""

from __future__ import annotations

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
    "sqlglot.expressions.core",
    "sqlglot.expressions.query",
    "sqlglot.expressions.datatypes",
    "sqlglot.executor.table",
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
})


def _subpkg_files(sg_dir: Path, sub: str, files: list[str] | None = None) -> list[str]:
    if files is None:
        files = sorted(
            f for f in os.listdir(sg_dir / sub) if f.endswith(".py") and f != "__init__.py"
        )
    return [f"sqlglot/{sub}/{f}" for f in files]


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/sqlglot-c")
    os.chdir(root)
    os.environ["MYPYPATH"] = str(root)
    sg_dir = root / "sqlglot"

    # Exact 104 modules from official sqlglotc/setup.py
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
    exts = mypycify(source_files, opt_level="1", debug_level="0", separate=True, verbose=False)
    t_gen = time.time() - t0
    print(f"[sqlglotc] mypyc codegen ({len(source_files)} files -> {len(exts)} exts): {t_gen:.1f}s", flush=True)

    py_inc = sysconfig.get_path("include")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    obj_root = root / "build" / "obj"
    obj_root.mkdir(parents=True, exist_ok=True)

    cc = shutil.which("clang") or "gcc"
    n_workers = max(1, os.cpu_count() or 4)

    # Build parallel compile jobs sorted largest source first (LPT) so big files never straggle!
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
            # Shared runtime lib (HASH__mypyc) lives inside sqlglot/ so sqlglot/__init__.py finds it
            dst_so = str(sg_dir / (ext.name + ext_suffix))
        incs = [py_inc, *ext.include_dirs]
        total_sz = sum(os.path.getsize(s) for s in ext.sources if os.path.exists(s))
        jobs.append((total_sz, ext.name, list(ext.sources), incs, dst_so, is_hot))

    jobs.sort(key=lambda x: -x[0])
    print(f"[sqlglotc] compiling {len(jobs)} exts ({n_hot} -O1, {len(jobs)-n_hot} -O0) with {cc} -j{n_workers}", flush=True)

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

    # Clean intermediate build artifacts to keep final layer slim
    shutil.rmtree(root / "build", ignore_errors=True)
    for egg in root.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)

    print(
        f"[sqlglotc] DONE in {time.time() - t0:.1f}s (codegen={t_gen:.1f}s, parallel_cc={t_cc:.1f}s, workers={n_workers})",
        flush=True,
    )


if __name__ == "__main__":
    main()

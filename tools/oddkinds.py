"""Hand-written patches of kinds the samples do not show, for local robustness checks.

    python tools/oddkinds.py <sqlglot checkout> <out dir>
"""

import difflib
import json
import os
import sys


def unified(path, old, new, created=False):
    a = "/dev/null" if created else f"a/{path}"
    body = "".join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), a, f"b/{path}", n=3))
    head = f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\n" if created else "")
    return head + body


def main(root, out):
    os.makedirs(os.path.join(out, "commits"), exist_ok=True)
    kinds = {}

    def read(rel):
        return open(os.path.join(root, rel), encoding="utf-8", newline="").read()

    def emit(cid, kind, *parts):
        open(os.path.join(out, "commits", f"{cid}.patch"), "w", encoding="utf-8", newline="").write("".join(parts))
        kinds[cid] = kind

    def edit(rel, old, new, count=1):
        text = read(rel)
        assert old in text, (rel, old)
        return unified(rel, text, text.replace(old, new, count))

    # 1. syntax error in a lazily loaded dialect module
    emit("xsy1", "syntax", edit("sqlglot/parsers/dremio.py", "class DremioParser(parser.Parser):", "class DremioParser(parser.Parser)"))
    # 2. a test's own assertion changed
    text = read("tests/test_parser.py")
    i = text.index("self.assertEqual(", text.index("def test_union"))
    emit("xtst", "test", unified("tests/test_parser.py", text, text[:i] + "self.assertNotEqual(" + text[i + len("self.assertEqual("):]))
    # 3. shared test helper changed
    emit("xhlp", "helper", edit("tests/helpers.py", 'if line and not line.startswith("--")', 'if line and not line.startswith("-")'))
    # 4. README only
    emit("xrdm", "docs", edit("README.md", "SQLGlot", "SQLGlot (tm)"))
    # 5. a comment line added inside a SQL fixture
    emit("xfcm", "fixture-comment", edit("tests/fixtures/optimizer/simplify.sql", "x + 3;", "-- folded\nx + 3;"))
    # 6. docstring change only
    emit("xdoc", "docstring", edit("sqlglot/optimizer/simplify.py", '"""', '"""Note: ', 1))
    # 7. new helper function added to a module and used
    text = read("sqlglot/optimizer/eliminate_joins.py")
    new = text.replace("def _should_eliminate_join(", "def _never(*args):\n    return False\n\n\ndef _should_eliminate_join(", 1)
    emit("xnew", "structural", unified("sqlglot/optimizer/eliminate_joins.py", text, new))
    # 8. pyproject edit unrelated to pytest
    emit("xpyp", "config", edit("pyproject.toml", 'description = "An easily customizable SQL parser and transpiler"', 'description = "An easily customizable SQL parser"'))
    # 9. a constant at module level in a core module
    emit("xmod", "module-const", edit("sqlglot/tokens.py", "class _TokenizerBase:", "UNUSED_CONSTANT = 1\n\n\nclass _TokenizerBase:"))
    # 10. new module plus an import of it
    created = "VALUE = 3\n"
    patch = unified("sqlglot/optimizer/_extra.py", "", created, created=True)
    patch += edit("sqlglot/optimizer/simplify.py", "from __future__ import annotations\n", "from __future__ import annotations\n\nfrom sqlglot.optimizer._extra import VALUE  # noqa: F401\n")
    emit("xnmd", "new-module", patch)
    # 11. syntax error in a core module
    emit("xsy2", "syntax-core", edit("sqlglot/expressions/core.py", "class Expr:", "class Expr(:"))
    # 12. default argument value changed
    text = read("sqlglot/helper.py")
    i = text.index("def seq_get(")
    j = text.index("\n", i)
    emit("xdef", "signature", unified("sqlglot/helper.py", text, text[:j].replace(")", ")", 1) + text[j:]) if False else edit("sqlglot/helper.py", "def csv(*args: str, sep: str = \", \")", "def csv(*args: str, sep: str = \",  \")"))
    json.dump(kinds, open(os.path.join(out, "kinds.json"), "w"), indent=1)
    print(f"wrote {len(kinds)} patches", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:3])

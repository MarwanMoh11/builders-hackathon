"""Generate sample-like patches against the pinned sqlglot, for local validation only.

    python tools/mutate.py <sqlglot checkout> <map.json.gz> <out dir> <count> [seed]

Writes <out>/commits/<id>.patch and <out>/kinds.json. Ground truth comes from running the real
test suite on each (tools/groundtruth.py); nothing here is used by the selector.
"""

from __future__ import annotations

import difflib
import gzip
import json
import os
import random
import re
import string
import sys

SWAPS = [
    (r" and ", " or "), (r" or ", " and "), (r" == ", " != "), (r" != ", " == "),
    (r" is not ", " is "), (r" is None", " == None"), (r" is ", " == "), (r" < ", " <= "),
    (r" <= ", " < "), (r" > ", " >= "), (r" >= ", " > "), (r"\bTrue\b", "False"),
    (r"\bFalse\b", "True"), (r" \+ ", " - "), (r" - ", " + "), (r"if not ", "if "),
    (r"if (?!not )", "if not "), (r"\bNone\b", "0"), (r"\breturn\b ", "return not "),
    (r"\[0\]", "[1]"), (r"\[1\]", "[0]"), (r"\[-1\]", "[0]"), (r"\bany\(", "all("),
    (r"\ball\(", "any("), (r"\bmin\(", "max("), (r"\bmax\(", "min("),
    (r"\.upper\(\)", ".lower()"), (r"\.lower\(\)", ".upper()"), (r"\bbreak\b", "continue"),
    (r"\bcontinue\b", "break"), (r", 1\)", ", 2)"), (r", 0\)", ", 1)"),
]
COMMENTS = [
    "# Cheap check first", "# Kept for backwards compatibility", "# Keep in sync with the generator",
    "# Intentionally not cached", "# Mirrors the behavior of the reference implementation",
    "# NOTE: order matters here", "# TODO: revisit",
]


def number_mutation(line: str, rng: random.Random) -> str | None:
    nums = list(re.finditer(r"(?<![\w.])(\d+)(?![\w.])", line))
    if not nums:
        return None
    m = rng.choice(nums)
    value = int(m.group(1))
    new = str(value + rng.choice([-1, 1]) if value > 0 else value + 1)
    return line[: m.start()] + new + line[m.end():]


def string_mutation(line: str, rng: random.Random) -> str | None:
    strs = list(re.finditer(r'"([A-Za-z_ ]{2,})"', line))
    if not strs:
        return None
    m = rng.choice(strs)
    s = m.group(1)
    i = rng.randrange(len(s))
    new = s[:i] + rng.choice(string.ascii_uppercase if s.isupper() else string.ascii_lowercase) + s[i + 1:]
    if new == s:
        return None
    return line[: m.start(1)] + new + line[m.end(1):]


def code_mutation(src_lines, covered, rng):
    for _ in range(400):
        lineno = rng.choice(covered)
        line = src_lines[lineno - 1]
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", '"', "'", "def ", "class ", "@", "import ", "from ")):
            continue
        choice = rng.random()
        new = None
        if choice < 0.65:
            options = [(a, b) for a, b in SWAPS if re.search(a, line)]
            if options:
                a, b = rng.choice(options)
                matches = list(re.finditer(a, line))
                m = rng.choice(matches)
                new = line[: m.start()] + re.sub(a, b, m.group(0), count=1) + line[m.end():]
        elif choice < 0.85:
            new = number_mutation(line, rng)
        else:
            new = string_mutation(line, rng)
        if new and new != line:
            try:
                compile("\n".join(src_lines[: lineno - 1] + [new] + src_lines[lineno:]), "x", "exec")
            except SyntaxError:
                continue
            return lineno, new
    return None


def comment_mutation(src_lines, rng):
    for _ in range(200):
        lineno = rng.randrange(1, len(src_lines))
        line = src_lines[lineno - 1]
        if not line.strip():
            continue
        indent = line[: len(line) - len(line.lstrip())]
        candidate = src_lines[: lineno - 1] + [indent + rng.choice(COMMENTS)] + src_lines[lineno - 1:]
        try:
            compile("\n".join(candidate), "x", "exec")
        except SyntaxError:
            continue
        return candidate
    return None


def fixture_mutation(lines, rng):
    for _ in range(200):
        i = rng.randrange(len(lines))
        line = lines[i]
        if not line.strip() or line.startswith(("--", "#")):
            continue
        tokens = line.split(" ")
        j = rng.randrange(len(tokens))
        op = rng.random()
        if op < 0.4:
            tokens.insert(j, rng.choice(["42", "x", "0", "NULL", "1"]))
        elif op < 0.7 and len(tokens) > 2:
            tokens[j] = tokens[j] + rng.choice(["42", "a", "_"])
        else:
            if len(tokens) > 3:
                del tokens[j]
            else:
                continue
        new = " ".join(tokens)
        if new != line:
            return lines[:i] + [new] + lines[i + 1:]
    return None


def unified(path, old_text, new_text):
    diff = difflib.unified_diff(
        old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
        f"a/{path}", f"b/{path}", n=3,
    )
    body = "".join(diff)
    return f"diff --git a/{path} b/{path}\n" + body


def main(root, map_path, out, count, seed=1):
    rng = random.Random(int(seed))
    cov = json.load(gzip.open(map_path, "rt"))
    py_files = [f for f in cov["lines"] if f.startswith("sqlglot/") and len(cov["lines"][f]) > 20]
    weights = [len(cov["lines"][f]) ** 0.5 for f in py_files]
    fixtures = sorted(k for k in cov["opens"] if k.endswith(".sql"))
    os.makedirs(os.path.join(out, "commits"), exist_ok=True)
    kinds = {}

    def read(rel):
        with open(os.path.join(root, rel), encoding="utf-8", newline="") as f:
            return f.read()

    def py_code_patch():
        for _ in range(50):
            rel = rng.choices(py_files, weights)[0]
            text = read(rel)
            lines = text.split("\n")
            covered = [int(l) for l in cov["lines"][rel]]
            got = code_mutation(lines, covered, rng)
            if got:
                lineno, new = got
                lines[lineno - 1] = new
                return rel, text, "\n".join(lines)
        return None

    def py_comment_patch():
        rel = rng.choice(py_files)
        text = read(rel)
        new_lines = comment_mutation(text.split("\n"), rng)
        return (rel, text, "\n".join(new_lines)) if new_lines else None

    def fixture_patch():
        rel = rng.choice(fixtures)
        text = read(rel)
        new_lines = fixture_mutation(text.split("\n"), rng)
        return (rel, text, "\n".join(new_lines)) if new_lines else None

    made = 0
    while made < int(count):
        r = rng.random()
        if r < 0.55:
            kind, parts = "code", [py_code_patch()]
        elif r < 0.70:
            kind, parts = "comment", [py_comment_patch()]
        elif r < 0.85:
            kind, parts = "fixture", [fixture_patch()]
        else:
            kind = "multi"
            parts = [py_comment_patch(), rng.choice([py_code_patch, fixture_patch])()]
        if any(p is None for p in parts) or len({p[0] for p in parts}) != len(parts):
            continue
        cid = "m" + "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(3))
        patch = "".join(unified(*p) for p in sorted(parts))
        with open(os.path.join(out, "commits", f"{cid}.patch"), "w", encoding="utf-8", newline="") as f:
            f.write(patch)
        kinds[cid] = kind
        made += 1
    json.dump(kinds, open(os.path.join(out, "kinds.json"), "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:6])

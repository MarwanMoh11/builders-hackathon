"""Unified diff parsing and applying (the runtime image has no git or patch)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)  # each begins with ' ', '-', '+' or '\\'


@dataclass
class FileDiff:
    old_path: str | None  # None for a new file
    new_path: str | None  # None for a deleted file
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False

    @property
    def path(self) -> str:
        return self.old_path or self.new_path or ""


def _strip_prefix(path: str) -> str | None:
    path = path.strip()
    if "\t" in path:
        path = path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    if path[:2] in ("a/", "b/"):
        path = path[2:]
    return path


def parse_patch(text: str) -> list[FileDiff]:
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: Hunk | None = None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            current = None
            hunk = None
            m = re.match(r"diff --git a/(.*) b/(.*)$", line)
            current = FileDiff(old_path=m.group(1) if m else None, new_path=m.group(2) if m else None)
            files.append(current)
        elif line.startswith("--- ") and (hunk is None or _hunk_done(hunk)):
            if current is None:
                current = FileDiff(None, None)
                files.append(current)
            current.old_path = _strip_prefix(line[4:])
            hunk = None
        elif line.startswith("+++ ") and (hunk is None or _hunk_done(hunk)):
            current.new_path = _strip_prefix(line[4:])
            hunk = None
        elif line.startswith("new file mode") and current is not None:
            current.old_path = None
        elif line.startswith("deleted file mode") and current is not None:
            current.new_path = None
        elif line.startswith("Binary files") and current is not None:
            current.binary = True
        elif line.startswith("@@"):
            m = HUNK_RE.match(line)
            hunk = Hunk(
                int(m.group(1)),
                int(m.group(2)) if m.group(2) is not None else 1,
                int(m.group(3)),
                int(m.group(4)) if m.group(4) is not None else 1,
            )
            current.hunks.append(hunk)
        elif hunk is not None and line[:1] in (" ", "-", "+", "\\"):
            hunk.lines.append(line)
        elif hunk is not None and line == "":
            # Some tools drop the space on blank context lines.
            hunk.lines.append(" ")
        i += 1
    return files


def _hunk_done(hunk: Hunk) -> bool:
    old = sum(1 for l in hunk.lines if l[:1] in (" ", "-"))
    new = sum(1 for l in hunk.lines if l[:1] in (" ", "+"))
    return old >= hunk.old_len and new >= hunk.new_len


class PatchError(Exception):
    pass


def apply_hunks(old_text: str, hunks: list[Hunk]) -> str:
    """Apply hunks to text. Lines keep their own endings; tolerant of small offsets."""
    old_lines = old_text.splitlines(keepends=True)
    out: list[str] = []
    cursor = 0  # index into old_lines
    for hunk in hunks:
        old_block = [l[1:] for l in hunk.lines if l[:1] in (" ", "-")]
        new_block = []
        no_newline_after: set[int] = set()
        for idx, l in enumerate(hunk.lines):
            if l[:1] in (" ", "+"):
                new_block.append(l[1:])
            elif l.startswith("\\") and idx > 0 and hunk.lines[idx - 1][:1] in (" ", "+"):
                no_newline_after.add(len(new_block) - 1)
        start = hunk.old_start - 1 if hunk.old_len else hunk.old_start
        start = _locate(old_lines, old_block, max(start, cursor), cursor)
        out.extend(old_lines[cursor:start])
        eol = "\n"
        for j, text in enumerate(new_block):
            if j in no_newline_after:
                out.append(text)
            else:
                out.append(text + eol)
        cursor = start + len(old_block)
    out.extend(old_lines[cursor:])
    return "".join(out)


def _strip_eol(s: str) -> str:
    return s.rstrip("\r\n")


def _locate(old_lines: list[str], block: list[str], guess: int, floor: int) -> int:
    def matches(at: int) -> bool:
        if at < floor or at + len(block) > len(old_lines):
            return False
        return all(_strip_eol(old_lines[at + k]) == block[k] for k in range(len(block)))

    if matches(guess):
        return guess
    for delta in range(1, 200):
        for at in (guess - delta, guess + delta):
            if matches(at):
                return at
    raise PatchError(f"hunk does not apply near line {guess + 1}")


def changed_old_lines(hunk: Hunk) -> tuple[list[int], list[int]]:
    """Old-file line numbers removed by the hunk, and old-file insertion points.

    An insertion point n means new lines were added between old lines n and n + 1.
    """
    removed: list[int] = []
    inserted_after: list[int] = []
    old_no = hunk.old_start if hunk.old_len else hunk.old_start + 1
    for l in hunk.lines:
        tag = l[:1]
        if tag == " ":
            old_no += 1
        elif tag == "-":
            removed.append(old_no)
            old_no += 1
        elif tag == "+":
            if not inserted_after or inserted_after[-1] != old_no - 1:
                inserted_after.append(old_no - 1)
    return removed, inserted_after

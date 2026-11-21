"""Turn brtrace output into the compact map the selector ships with.

    python tools/build_map.py <trace_dir> <tests.json> <out.json.gz>

The map holds only facts from the pinned commit: which tests run which lines, which tests open
which data files, and which lines run while modules are imported.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import sys


def read_attrs(attr_dir: str, number_by_id: dict) -> dict[str, list[int]]:
    attrs: dict[str, set[int]] = {}
    for path in sorted(glob.glob(os.path.join(attr_dir, "trace-*.json"))):
        data = json.load(open(path))
        keys = data.get("attr_keys", {})
        for nodeid, entry in data["tests"].items():
            number = number_by_id.get(nodeid)
            if number is None:
                continue
            for key in entry.get("attrs", []):
                attrs.setdefault(keys[str(key)], set()).add(number)
    return {k: sorted(v) for k, v in sorted(attrs.items())}


def main(trace_dir: str, tests_path: str, out_path: str, attr_dir: str | None = None) -> None:
    tests = json.load(open(tests_path))["tests"]
    number_by_id = {t["id"]: t["number"] for t in tests}

    line_tests: dict[str, dict[int, set[int]]] = {}
    open_tests: dict[str, set[int]] = {}
    collect: dict[str, set[int]] = {}
    traced_seconds: dict[int, float] = {}
    class_setup: dict[str, set[tuple[str, int]]] = {}
    class_members: dict[str, set[int]] = {}
    unknown = 0

    for path in sorted(glob.glob(os.path.join(trace_dir, "trace-*.json"))):
        data = json.load(open(path))
        files = {int(k): v for k, v in data["files"].items()}
        for fid, lines in data["collect"].items():
            collect.setdefault(files[int(fid)], set()).update(lines)
        for nodeid, entry in data["tests"].items():
            number = number_by_id.get(nodeid)
            if number is None:
                unknown += 1
                continue
            traced_seconds[number] = entry.get("seconds", 0.0)
            for fid, lines in entry.get("lines", {}).items():
                per_file = line_tests.setdefault(files[int(fid)], {})
                for line in lines:
                    per_file.setdefault(line, set()).add(number)
            for opened in entry.get("opens", []):
                open_tests.setdefault(opened, set()).add(number)
            klass = nodeid.rsplit("::", 1)[0]
            class_members.setdefault(klass, set()).add(number)
            for fid, lines in entry.get("setup", {}).items():
                class_setup.setdefault(klass, set()).update((files[int(fid)], l) for l in lines)

    # setUpClass runs once, inside the first test of its class: credit it to the whole class.
    for klass, lines in class_setup.items():
        members = class_members.get(klass, set())
        for filename, line in lines:
            line_tests.setdefault(filename, {}).setdefault(line, set()).update(members)

    # Many lines share one set of tests, so store each distinct set once.
    set_index: dict[tuple[int, ...], int] = {}
    sets: list[list[int]] = []

    def intern(numbers: set[int]) -> int:
        key = tuple(sorted(numbers))
        index = set_index.get(key)
        if index is None:
            index = set_index[key] = len(sets)
            sets.append(list(key))
        return index

    lines_out = {
        filename: {str(line): intern(numbers) for line, numbers in sorted(per_file.items())}
        for filename, per_file in sorted(line_tests.items())
    }
    out = {
        "sets": sets,
        "lines": lines_out,
        "opens": {name: sorted(numbers) for name, numbers in sorted(open_tests.items())},
        "collect": {name: sorted(lines) for name, lines in sorted(collect.items())},
        "traced": [len(traced_seconds), len(tests)],
        "attrs": read_attrs(attr_dir, number_by_id) if attr_dir else {},
    }
    with gzip.open(out_path, "wt", compresslevel=9) as f:
        json.dump(out, f, separators=(",", ":"))
    missing = [t["id"] for t in tests if t["number"] not in traced_seconds]
    print(
        f"tests traced {len(traced_seconds)}/{len(tests)}, unknown ids {unknown}, "
        f"distinct sets {len(sets)}, files {len(lines_out)}, size {os.path.getsize(out_path)} B",
        file=sys.stderr,
    )
    for test_id in missing[:20]:
        print(f"  not traced: {test_id}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:5])

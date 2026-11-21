"""Label generated patches by running the real sqlglot suite on each (inside the env image).

    python tools/groundtruth.py <patch dir> <tests.json> <out answers.json> [workers]

For each patch: copy /opt/sqlglot to a scratch dir, apply the patch, run pytest the way the
organizers did (SKIP_INTEGRATION=0, real process pools), and record failing test numbers.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "br"))
from patching import apply_hunks, parse_patch  # noqa: E402

PLUGIN = '''
import json, os
FAILED = set()
def pytest_runtest_logreport(report):
    if report.failed:
        FAILED.add(report.nodeid)
def pytest_collectreport(report):
    if report.failed:
        FAILED.add("COLLECT::" + report.nodeid)
def pytest_unconfigure(config):
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        json.dump(sorted(FAILED), open(os.environ["GT_OUT"], "w"))
'''


def main(patch_dir, tests_path, out_path, workers="4"):
    tests = json.load(open(tests_path))["tests"]
    number_by_id = {t["id"]: t["number"] for t in tests}
    by_file = {}
    for t in tests:
        by_file.setdefault(t["file"], []).append(t["number"])
    answers = json.load(open(out_path)) if os.path.exists(out_path) else {"commits": {}}
    kinds = {}
    kinds_path = os.path.join(patch_dir, "kinds.json")
    if os.path.exists(kinds_path):
        kinds = json.load(open(kinds_path))
    for patch in sorted(glob.glob(os.path.join(patch_dir, "commits", "*.patch"))):
        cid = os.path.basename(patch)[:-6]
        if cid in answers["commits"]:
            continue
        work = tempfile.mkdtemp(prefix="gt-")
        root = os.path.join(work, "sqlglot")
        shutil.copytree("/opt/sqlglot", root, ignore=shutil.ignore_patterns("__pycache__"))
        for fd in parse_patch(open(patch, encoding="utf-8", newline="").read()):
            target = os.path.join(root, fd.new_path or fd.old_path)
            old = open(target, encoding="utf-8", newline="").read() if fd.old_path else ""
            if fd.new_path is None:
                os.remove(target)
                continue
            open(target, "w", encoding="utf-8", newline="").write(apply_hunks(old, fd.hunks))
        open(os.path.join(work, "gtplugin.py"), "w").write(PLUGIN)
        result_path = os.path.join(work, "failed.json")
        env = dict(os.environ, PYTHONPATH=f"{root}:{work}", SKIP_INTEGRATION="0", GT_OUT=result_path,
                   PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "gtplugin",
             "-n", str(workers), "--timeout", "120", "tests"],
            cwd=root, env=env, capture_output=True, text=True,
        )
        try:
            failed_ids = json.load(open(result_path))
        except (OSError, ValueError):
            failed_ids = ["CRASH"]
        failing = set()
        for nodeid in failed_ids:
            if nodeid in number_by_id:
                failing.add(number_by_id[nodeid])
            elif nodeid.startswith("COLLECT::"):
                failing.update(by_file.get(nodeid[len("COLLECT::"):], []))
            elif nodeid == "CRASH":
                failing.update(number_by_id.values())
        tail = proc.stdout.strip().splitlines()[-1:] if proc.stdout else []
        print(f"{cid} {kinds.get(cid, '?')}: {len(failing)} failing {tail}", file=sys.stderr, flush=True)
        answers["commits"][cid] = {"failing": sorted(failing), "kind": kinds.get(cid, "?")}
        json.dump(answers, open(out_path, "w"), indent=1)
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main(*sys.argv[1:5])

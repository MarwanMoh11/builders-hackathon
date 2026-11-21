"""Run every test in-process with the selector's own runner on pristine sqlglot.

    python tools/measure.py <tests.json> <out costs.json>

Checks the runner reports no failures on the unpatched code, and records each test's CPU cost
when run this way (the selector pays this cost when it chooses to run a test).
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "br"))

import engine  # noqa: E402


def main(tests_path, out_path):
    tests = json.load(open(tests_path))["tests"]
    t0 = time.process_time()
    engine.setup_parent(tests)
    setup_cpu = time.process_time() - t0
    failing, cpu = engine.run_tests([t["number"] for t in tests])
    total = time.process_time() - t0
    by_id = {tests[n]["id"]: round(c, 4) for n, c in cpu.items()}
    never_run = [t["id"] for t in tests if "test_lazy_load" in t["id"]]
    class_setup = {k: round(v, 4) for k, v in engine.CLASS_SETUP_CPU.items()}
    json.dump({"cpu": by_id, "class_setup": class_setup, "never_run": never_run,
               "setup_cpu": round(setup_cpu, 3)}, open(out_path, "w"), indent=0)
    for k, v in sorted(class_setup.items(), key=lambda kv: -kv[1])[:6]:
        print(f"  class setup {v:6.3f}s  {k}", file=sys.stderr)
    print(f"setup {setup_cpu:.2f}s cpu, all tests {total - setup_cpu:.2f}s cpu, failing {len(failing)}", file=sys.stderr)
    for n in sorted(failing):
        print("  FAIL", tests[n]["id"], file=sys.stderr)
    top = sorted(cpu.items(), key=lambda kv: -kv[1])[:15]
    for n, c in top:
        print(f"  {c:7.3f}s cpu  (recorded {tests[n]['seconds']:.3f})  {tests[n]['id']}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

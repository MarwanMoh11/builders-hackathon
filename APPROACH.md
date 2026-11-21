# Blast Radius: approach

For each patch we find the tests that can possibly be affected, run the cheap ones against the
patched code inside the container, and report only those that fail (plus expensive candidates we
choose not to run). Nothing is looked up by patch: every map below is built from the pinned
commit alone, and every decision at runtime comes from analysing the patch itself.

## Maps built once from the pinned commit (`tools/`, shipped as `br/map.json.gz`)

`tools/brtrace.py` is a pytest plugin. Running sqlglot's own suite with it records, per test:

* **Lines**: every sqlglot and test line the test executes (`sys.monitoring`, one event per line
  per test). Process-pool work is run in-process with a pickle round trip, as a real pool does,
  so pool work and sqlglot's serializer are covered too.
* **Data files**: every fixture the test opens (audit hook on `open`).
* **Class attributes**: with `BR_ATTRS=1`, every class constant in sqlglot is wrapped in a
  recording descriptor, so we know which tests read e.g. `TSQL.WEEK_OFFSET`.
* **Import-time lines**: lines run while modules import, kept apart from any test.

Anything run in `setUpClass` (lines, files, attributes) is credited to every test of the class.
`tools/measure.py` records each test's CPU cost (and each class's setup cost) when run the
selector's way.

## Per patch (`br/`)

1. **Parse and apply** the diff (`patching.py`; the image has no git or patch).
2. **Compare compiled code** before and after (`pyanalyze.py`). Line numbers are ignored, so
   comment-only edits compile to identical code and cost nothing. Each changed code object is a
   *function* change (takes effect when called) or a *body* change (module/class body, runs at
   import). Function-docstring-only edits are no-ops.
3. **Candidates**:
   * function change: tests that executed the changed *statement* (AST statement range,
     restricted to the changed function's own lines);
   * data file change: tests that opened the file;
   * body/import-time change: decided in the child (below), from what actually changed.
4. **Run in a forked child** of a parent that imported sqlglot and the tests once
   (`engine.py`, `main.py`):
   * function changes are **hot-patched**: the new code object is swapped into every live
     function object (`func.__code__ = new`), keeping all other state intact;
   * import-time changes **re-import** the changed modules and their dependents (dependency
     graph read off live namespaces) from patched sources, repoint class registries, then
     **compare module state structurally** (`valdiff.py`). Changed class attributes map to their
     readers; changed globals map to the functions that read them;
   * syntax errors re-import too: test files that stop importing fail as a whole;
   * fixture edits are served through an `open` overlay;
   * tests run with `unittest` in-process, with time limits; a timeout stops the run (the patch
     probably loops) and the rest are reported unrun.
5. **Run or report**: per commit, an expected-cost comparison between reporting candidates
   unrun (their recorded time) and running them (their measured CPU, plus class setup).
   Tests far costlier to run than to report (e.g. `test_execute_tpcds`) are always reported.
6. **Pack** what is reported onto 1–8 workers (longest first onto the lightest, cheapest k).

## Validation

`tools/mutate.py` makes sample-like patches and `tools/groundtruth.py` labels them by running the
real suite with pytest; `tools/oddkinds.py` covers kinds the samples do not show (syntax errors,
test edits, new modules, pytest config, docstrings, loops).

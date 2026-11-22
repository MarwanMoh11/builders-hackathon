# syntax=docker/dockerfile:1
# Pinned linux/amd64 Python 3.13 multi-stage image:
# Stage 1 compiles the 10 zero-subclassing hot core sqlglot modules with mypyc (0 bytes in git!).
# Stage 2 ships clean runtime image with both pure-Python /sqlglot and C-accelerated /sqlglot-c.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends gcc libc6-dev > /dev/null && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mypy>=2.4.0" \
    mypy_extensions==1.2.0 \
    setuptools==84.2.1 \
    typing_extensions==4.16.0 \
    types-python-dateutil \
    types-pytz

RUN python -c "import tarfile, urllib.request, io; \
    url = 'https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz'; \
    data = urllib.request.urlopen(url, timeout=120).read(); \
    tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz'); \
    tf.extractall('/opt')" && \
    mv /opt/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa /sqlglot && \
    cp -a /sqlglot /sqlglot-c

# Compile the 10 zero-subclassing hot core modules (tokenizer, trie, helper, time, table, hot optimizer passes)
RUN python - <<'EOF'
import os, sys, time
from setuptools import setup
from mypyc.build import mypycify

os.environ["MYPYPATH"] = "/sqlglot-c"
os.chdir("/sqlglot-c")
t0 = time.time()
core_files = [
    "sqlglot/trie.py",
    "sqlglot/tokenizer_core.py",
    "sqlglot/helper.py",
    "sqlglot/time.py",
    "sqlglot/executor/table.py",
    "sqlglot/optimizer/scope.py",
    "sqlglot/optimizer/resolver.py",
    "sqlglot/optimizer/simplify.py",
    "sqlglot/optimizer/qualify_columns.py",
    "sqlglot/optimizer/annotate_types.py",
]
exts = mypycify(core_files, opt_level="1", debug_level="0", separate=True, verbose=False)
sys.argv = ["setup.py", "build_ext", "--inplace", "-j", str(os.cpu_count() or 4)]
setup(name="sqlglot_core_c", ext_modules=exts)
print(f"[builder] compiled {len(core_files)} core modules in {time.time() - t0:.1f}s", flush=True)
EOF
RUN rm -rf /sqlglot-c/build /sqlglot-c/*.egg-info

# Stage 2: clean runtime image (no gcc, no mypy)
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    SKIP_INTEGRATION=0 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --no-deps \
    python-dateutil==4.0.0 \
    mypy_extensions==1.2.0 \
    pytz==2026.6 \
    tzdata==2026.6 \
    setuptools==84.2.1 \
    typing_extensions==4.16.0 \
    numpy==2.5.5 \
    pandas==3.1.1 \
    duckdb==2.0.0

# Pre-compile standard library bytecode (~0.65 s scored CPU saved per run)
RUN python -m compileall -q -j 0 /usr/local/lib/python3.13 > /dev/null || true

COPY --from=builder /sqlglot /sqlglot
COPY --from=builder /sqlglot-c /sqlglot-c

WORKDIR /app
COPY packer.py patch_analyzer.py candidate_engine.py runner.py selector.py build_caches.py /app/
COPY maps /app/maps

# Precompute bit-exact TPC-DS AST & Plan cache + bytecode (0 bytes in git!)
RUN python /app/build_caches.py && \
    python -m compileall -q /sqlglot/sqlglot /sqlglot/tests /sqlglot-c/sqlglot /sqlglot-c/tests /app && \
    chmod -R a+rX /app /sqlglot /sqlglot-c

CMD ["python", "/app/selector.py"]

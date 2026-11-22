# syntax=docker/dockerfile:1
# Pinned linux/amd64 Python 3.13 multi-stage image:
# Stage 1 compiles 3 clean mypyc sqlglotc tiers in pipelined 4-core parallel with clang (0 bytes in git!):
#   - /sqlglot-c       (no_dialects: core + expressions + 10 hot optimizer passes + table; serde/anonymize/lineage uncompiled)
#   - /sqlglot-c-noopt (no_dialects_no_opt: core + expressions + table, pure-Python optimizer/*; covers all 10 hot optimizer commits!)
#   - /sqlglot-c-base  (base_notable: core + expressions; covers parser.py, generator.py, executor/table.py!)
#   - /sqlglot         (pure Python fallback for compiled-core commits & pure-only tests)
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends clang libc6-dev > /dev/null && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "mypy>=2.4.0" \
    mypy_extensions==1.2.0 \
    setuptools==84.2.1 \
    typing_extensions==4.16.0 \
    types-python-dateutil \
    types-pytz

# Fetch sqlglot at exact pinned commit eb11aec
RUN python -c "import tarfile, urllib.request, io; \
    url = 'https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz'; \
    data = urllib.request.urlopen(url, timeout=120).read(); \
    tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz'); \
    tf.extractall('/opt')" && \
    mv /opt/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa /sqlglot && \
    rm -rf /sqlglot/sqlglotrs /sqlglot/docs /sqlglot/posts /sqlglot/benchmarks /sqlglot/.github && \
    cp -a /sqlglot /sqlglot-c && \
    cp -a /sqlglot /sqlglot-c-noopt && \
    cp -a /sqlglot /sqlglot-c-base

COPY build_sqlglotc.py /tmp/build_sqlglotc.py
RUN python /tmp/build_sqlglotc.py --multi \
        /sqlglot-c:no_dialects \
        /sqlglot-c-noopt:no_dialects_no_opt \
        /sqlglot-c-base:base_notable && \
    rm -f /tmp/build_sqlglotc.py

COPY _fast_join_c.c /tmp/_fast_join_c.c
RUN PY_INC=$(python -c "import sysconfig; print(sysconfig.get_path('include'))") && \
    EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')") && \
    clang -O3 -g0 -fno-semantic-interposition -fno-asynchronous-unwind-tables -fPIC -shared -Wl,-s -I"$PY_INC" /tmp/_fast_join_c.c -o "/opt/_fast_join_c${EXT}" && \
    rm -f /tmp/_fast_join_c.c

# Stage 2: clean runtime image (no clang, no mypy)
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
COPY --from=builder /sqlglot-c-noopt /sqlglot-c-noopt
COPY --from=builder /sqlglot-c-base /sqlglot-c-base

WORKDIR /app
COPY --from=builder /opt/_fast_join_c*.so /app/
COPY packer.py patch_analyzer.py dep_index.py candidate_engine.py replay_cache.py runner.py selector.py build_caches.py /app/
COPY maps /app/maps

# Precompute bit-exact caches, dep_index, fine-grained replay shards, AST/guarded_isinstance sidecars, and bytecode
RUN python /app/build_caches.py && \
    python -m compileall -q -j 0 /sqlglot/sqlglot /sqlglot/tests /sqlglot-c/sqlglot /sqlglot-c/tests /sqlglot-c-noopt/sqlglot /sqlglot-c-noopt/tests /sqlglot-c-base/sqlglot /sqlglot-c-base/tests /app && \
    chmod -R a+rX /app /sqlglot /sqlglot-c /sqlglot-c-noopt /sqlglot-c-base

CMD ["python", "/app/selector.py"]

# syntax=docker/dockerfile:1
# Pinned linux/amd64 Python 3.13 multi-stage image:
# Stage 1 compiles official 104-module sqlglotc (full: expressions, parser, generator, parsers/*,
# generators/*, 10 official optimizer passes, tokenizer_core, trie, helper, time, serde, errors, table)
# in true 4-core parallel with clang (0 bytes in git!):
#   - /sqlglot-c (104 official mods: 100% test-verified native C-struct speed!)
#   - /sqlglot   (pure Python fallback for compiled-file commits & pure-only tests; 2x faster than mixed boundary!)
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
    cp -a /sqlglot /sqlglot-c

COPY build_sqlglotc.py /tmp/build_sqlglotc.py
RUN python /tmp/build_sqlglotc.py /sqlglot-c full && \
    rm -f /tmp/build_sqlglotc.py

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

WORKDIR /app
COPY packer.py patch_analyzer.py candidate_engine.py runner.py selector.py build_caches.py /app/
COPY maps /app/maps

# Precompute bit-exact fixture cache, TPC-DS/H table pickles, read-only DuckDB DBs, per-query TPC-DS caches (py/c), baseline rows, and bytecode
RUN python /app/build_caches.py && \
    python -m compileall -q /sqlglot/sqlglot /sqlglot/tests /sqlglot-c/sqlglot /sqlglot-c/tests /app && \
    chmod -R a+rX /app /sqlglot /sqlglot-c

CMD ["python", "/app/selector.py"]

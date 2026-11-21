# Pinned linux/amd64 Python 3.13 image matching the scorer's exact Python & SQLite
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    SKIP_INTEGRATION=0 \
    PYTHONUNBUFFERED=1

# Install only the exact pinned runtime wheels used by sqlglot & its test suite (from requirements.lock)
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

# The slim image ships the standard library without bytecode (~0.65 s scored CPU saved per run)
RUN python -m compileall -q -j 0 /usr/local/lib/python3.13 > /dev/null || true

# Fetch sqlglot at the exact pinned commit (no git apt install needed!)
RUN python -c "import tarfile, urllib.request, io; \
    url = 'https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz'; \
    data = urllib.request.urlopen(url, timeout=120).read(); \
    tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz'); \
    tf.extractall('/opt')" && \
    mv /opt/sqlglot-eb11aec4be8cb2122afe713a4816662c3f0fb2fa /sqlglot

# Copy selector code and precomputed maps derived from sqlglot@eb11aec
WORKDIR /app
COPY packer.py patch_analyzer.py candidate_engine.py runner.py selector.py /app/
COPY maps /app/maps

# Pre-compile all bytecode so runtime imports are fast on read-only disk, and make readable by nobody
RUN python -m compileall -q /sqlglot/sqlglot /sqlglot/tests /app && \
    chmod -R a+rX /app /sqlglot

CMD ["python", "/app/selector.py"]

# The scorer builds this on linux/amd64 and runs it with no network. The
# digest is the image every commit was labeled in: its Python and SQLite
# decide which sqlglot tests run, so we keep it to run tests the same way.
FROM python:3.13-slim@sha256:e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SKIP_INTEGRATION=0 \
    SQLGLOT_ROOT=/opt/sqlglot

# Exactly the packages the labeling environment had.
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --disable-pip-version-check -r /app/requirements.lock
# The slim image ships the standard library without bytecode: compile it once here instead of
# on every run (CPU time is scored).
RUN python -m compileall -q -j 0 /usr/local/lib/python3.13 > /dev/null || true

# sqlglot at the pinned commit, from sqlglot's own repository.
RUN python -c "import urllib.request as u; u.urlretrieve('https://github.com/tobymao/sqlglot/archive/eb11aec4be8cb2122afe713a4816662c3f0fb2fa.tar.gz', '/tmp/sqlglot.tgz')" \
 && mkdir -p /opt/sqlglot \
 && tar -xzf /tmp/sqlglot.tgz -C /opt/sqlglot --strip-components=1 \
 && rm /tmp/sqlglot.tgz \
 && python -m compileall -q /opt/sqlglot/sqlglot /opt/sqlglot/tests \
 && chmod -R a+rX /opt/sqlglot

COPY br /app/br
COPY tools /app/tools
RUN python /app/br/candidates.py /app/br/map.json.gz /app/br/map.pickle  && chmod -R a+rX /app && python -m compileall -q /app/br

CMD ["python", "/app/br/main.py"]

# syntax=docker/dockerfile:1

# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Install uv, resolve the lockfile, and materialise site-packages for copying.
# No dev extras — the runtime image only needs production deps.
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv (the package manager used by this project).
RUN pip install --no-cache-dir uv

# Copy the lockfile + packaging manifests first so layer caching survives
# source-only edits that don't change the dependency graph.
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# Materialise prod-only site-packages into /install.
# --frozen: fail loudly if the lockfile needs updating (reproducibility).
# --no-dev:  skip dev-only extras (pytest, mypy, ruff, …).
# --prefix:  isolate to /install so we can copy a clean tree to runtime.
RUN uv sync --frozen --no-dev --prefix /install

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="vdbbench" \
      org.opencontainers.image.description="Reproducible vector-DB benchmark harness" \
      org.opencontainers.image.source="https://github.com/BishBish123/vector-db-bench"

# System deps required by psutil (libffi) and matplotlib (libgomp, libGL).
# Kept minimal — only what the installed wheels need at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libffi-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the isolated site-packages tree from the builder stage.
COPY --from=builder /install /usr/local

# Copy package source (needed by the editable/hatchling layout).
COPY --from=builder /build/src ./src

# Volume mount points documented here so `docker inspect` surfaces them.
# /app/results — parquet output written by every bench run.
# /app/data    — encoded input bundles (produced by `vdbbench prep`).
VOLUME ["/app/results", "/app/data"]

# Default behaviour: run the smoke benchmark (synthetic, no Docker services
# needed inside the container — uses the in-process `exact` adapter).
# Override CMD on the `docker run` / `docker compose` command line to run
# the full pgvector+qdrant sweep:
#   docker run vdbbench bench --encoded data/... --all ...
ENTRYPOINT ["vdbbench"]
CMD ["bench-demo"]

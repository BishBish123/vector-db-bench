"""Shared fixtures for adapter integration tests.

Both `test_pgvector_integration.py` and `test_qdrant_integration.py`
exercise live containers. The contract suite (`test_contract.py`) already
probes reachability and `pytest.skip()`s when the service is dead — the
dedicated integration suites used to skip the import / env-var check, but
hard-fail when the env var was set and the container had died (common on
CI when a job loses its postgres / qdrant sidecar mid-run).

The fixtures below centralise the probe so a dead container produces a
clean skip from every integration test, not just the contract ones.
"""

from __future__ import annotations

import os

import pytest


def _probe_pgvector(dsn: str, *, timeout: float = 2.0) -> None:
    """Open a one-shot psycopg connection; raise to signal unreachable.

    Kept narrow on purpose — the only thing we care about here is whether
    the DSN points at a live Postgres. All real adapter wiring (extension
    install, table creation) lives in PgVectorAdapter.setup() and gets
    exercised by the actual tests.
    """
    pytest.importorskip("psycopg")
    import psycopg  # noqa: PLC0415

    with psycopg.connect(dsn, connect_timeout=int(timeout)) as _:
        pass


def _probe_qdrant(url: str, *, timeout: float = 2.0) -> None:
    """Hit the qdrant `/healthz` endpoint; non-2xx or transport errors raise."""
    pytest.importorskip("httpx")
    import httpx  # noqa: PLC0415

    response = httpx.get(f"{url}/healthz", timeout=timeout)
    response.raise_for_status()


@pytest.fixture
def pgvector_dsn() -> str:
    """Yield a reachable pgvector DSN, or `pytest.skip()` if unavailable.

    Skips when:
      * `psycopg` / `pgvector` aren't installed, or
      * `PGVECTOR_DSN` is unset and no default container is reachable, or
      * the resolved DSN points at a service that won't accept a
        connection within the probe timeout.
    """
    pytest.importorskip("pgvector")
    dsn = os.environ.get("PGVECTOR_DSN", "postgresql://bench:bench@localhost:5433/bench")
    try:
        _probe_pgvector(dsn)
    except Exception as exc:  # pragma: no cover - reachability skip path
        pytest.skip(f"pgvector unreachable at {dsn}: {exc}")
    return dsn


@pytest.fixture
def qdrant_url() -> str:
    """Yield a reachable qdrant URL, or `pytest.skip()` if unavailable."""
    pytest.importorskip("qdrant_client")
    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    try:
        _probe_qdrant(url)
    except Exception as exc:  # pragma: no cover - reachability skip path
        pytest.skip(f"qdrant unreachable at {url}: {exc}")
    return url

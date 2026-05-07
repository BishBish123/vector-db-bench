"""Notebook execution smoke test.

Executes analysis.ipynb against results/demo/summary.parquet using nbclient
and asserts:
  - no cells errored
  - the expected output PNGs were written under assets/

Runs in CI without GPU or Docker — the notebook only reads a committed parquet
and writes static PNG files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = REPO_ROOT / "analysis.ipynb"
PARQUET_PATH = REPO_ROOT / "results" / "demo" / "summary.parquet"
ASSETS_DIR = REPO_ROOT / "assets"

# PNGs that must exist after the notebook runs.
EXPECTED_PNGS = [
    "analysis-cost-pareto.png",
    "analysis-recall-pareto.png",
    "analysis-ingest.png",
]

try:
    import nbformat  # type: ignore[import-untyped]
    from nbclient import NotebookClient  # type: ignore[import-untyped]

    _NBCLIENT_AVAILABLE = True
except ImportError:
    _NBCLIENT_AVAILABLE = False


def _run_notebook() -> list[dict]:  # type: ignore[return]
    """Execute the notebook and return its cells (with outputs)."""
    if not _NBCLIENT_AVAILABLE:
        pytest.skip("nbclient / nbformat not installed")

    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    client = NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(REPO_ROOT)}},
    )
    client.execute()
    return list(nb.cells)


def _cells_with_errors(cells: list[dict]) -> list[str]:
    """Return cell source strings for any cell whose outputs contain errors."""
    bad: list[str] = []
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                bad.append(cell.get("source", "")[:200])
    return bad


@pytest.mark.slow
def test_notebook_executes_without_errors() -> None:
    """analysis.ipynb runs cleanly against the committed demo parquet."""
    if not _NBCLIENT_AVAILABLE:
        pytest.skip("nbclient / nbformat not installed")

    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    assert PARQUET_PATH.exists(), (
        f"Demo parquet not found: {PARQUET_PATH}. "
        "Run 'make bench-demo' to generate it."
    )

    cells = _run_notebook()
    errors = _cells_with_errors(cells)
    assert not errors, "Notebook cells errored:\n" + "\n---\n".join(errors)


@pytest.mark.slow
def test_notebook_writes_expected_pngs() -> None:
    """After notebook execution the expected analysis-*.png files exist."""
    if not _NBCLIENT_AVAILABLE:
        pytest.skip("nbclient / nbformat not installed")

    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    assert PARQUET_PATH.exists(), f"Demo parquet not found: {PARQUET_PATH}"

    _run_notebook()

    missing = [p for p in EXPECTED_PNGS if not (ASSETS_DIR / p).exists()]
    assert not missing, f"Expected PNG(s) not written: {missing}"


@pytest.mark.slow
def test_notebook_is_valid_nbformat() -> None:
    """analysis.ipynb parses as valid nbformat 4 without executing it."""
    if not _NBCLIENT_AVAILABLE:
        pytest.skip("nbclient / nbformat not installed")

    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    nbformat.validate(nb)
    assert nb.nbformat == 4


def test_notebook_json_is_parseable() -> None:
    """analysis.ipynb is valid JSON and has the expected top-level keys."""
    assert NOTEBOOK_PATH.exists(), f"Notebook not found: {NOTEBOOK_PATH}"
    with open(NOTEBOOK_PATH) as fh:
        data = json.load(fh)
    assert "cells" in data
    assert "nbformat" in data
    assert data["nbformat"] == 4
    assert len(data["cells"]) >= 7, "Expected at least 7 cells (setup+load+5 charts)"

"""Lightweight reproducibility-infrastructure tests.

These tests verify that the Docker/compose/script artifacts are
syntactically valid and self-consistent.  They do NOT build Docker images
or spin up containers — all assertions are file-system and YAML checks.

Marked unit (no marker needed — not integration, not slow) so they run in
the default `make test` / CI unit-test path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _repo(path: str) -> Path:
    return REPO_ROOT / path


# ---------------------------------------------------------------------------
# 1. Dockerfile is present and has the expected stages
# ---------------------------------------------------------------------------


def test_dockerfile_exists_and_is_nonempty() -> None:
    """Dockerfile must exist and contain content."""
    df = _repo("Dockerfile")
    assert df.is_file(), f"Dockerfile not found at {df}"
    assert df.stat().st_size > 0, "Dockerfile is empty"


def test_dockerfile_has_builder_stage() -> None:
    """Multi-stage build must declare the builder stage."""
    content = _repo("Dockerfile").read_text()
    assert "AS builder" in content, "Dockerfile missing 'AS builder' stage"


def test_dockerfile_has_runtime_stage() -> None:
    """Multi-stage build must declare the runtime stage."""
    content = _repo("Dockerfile").read_text()
    assert "AS runtime" in content, "Dockerfile missing 'AS runtime' stage"


def test_dockerfile_sets_entrypoint_to_vdbbench() -> None:
    """Entrypoint must invoke the vdbbench CLI."""
    content = _repo("Dockerfile").read_text()
    assert "vdbbench" in content and "ENTRYPOINT" in content


def test_dockerfile_installs_uv() -> None:
    """Builder stage must install uv (the project's package manager)."""
    content = _repo("Dockerfile").read_text()
    assert "uv" in content


def test_dockerfile_uses_uv_sync_frozen() -> None:
    """Builder must use --frozen to guarantee lockfile parity."""
    content = _repo("Dockerfile").read_text()
    assert "--frozen" in content, "Dockerfile must pass --frozen to uv sync"


# ---------------------------------------------------------------------------
# 2. docker-compose.full.yml is valid YAML with required services
# ---------------------------------------------------------------------------


def test_docker_compose_full_exists() -> None:
    """docker-compose.full.yml must be present."""
    cf = _repo("docker-compose.full.yml")
    assert cf.is_file(), f"docker-compose.full.yml not found at {cf}"


def test_docker_compose_full_parses_as_valid_yaml() -> None:
    """docker-compose.full.yml must be parseable YAML."""
    content = _repo("docker-compose.full.yml").read_text()
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "Top-level must be a mapping"


def test_docker_compose_full_has_all_three_services() -> None:
    """Stack must declare pgvector, qdrant, and vdbbench services."""
    data = yaml.safe_load(_repo("docker-compose.full.yml").read_text())
    services = set(data.get("services", {}).keys())
    for required in ("pgvector", "qdrant", "vdbbench"):
        assert required in services, f"Missing service '{required}' in docker-compose.full.yml"


def test_docker_compose_full_vdbbench_depends_on_both_backends() -> None:
    """vdbbench service must wait for pgvector and qdrant to be healthy."""
    data = yaml.safe_load(_repo("docker-compose.full.yml").read_text())
    vdbbench_svc = data["services"]["vdbbench"]
    depends = vdbbench_svc.get("depends_on", {})
    dep_keys = set(depends.keys()) if isinstance(depends, dict) else set(depends)
    assert "pgvector" in dep_keys, "vdbbench must depend_on pgvector"
    assert "qdrant" in dep_keys, "vdbbench must depend_on qdrant"


def test_docker_compose_full_has_healthchecks() -> None:
    """pgvector and qdrant services must have healthcheck definitions."""
    data = yaml.safe_load(_repo("docker-compose.full.yml").read_text())
    for svc_name in ("pgvector", "qdrant"):
        svc = data["services"][svc_name]
        assert "healthcheck" in svc, f"Service '{svc_name}' is missing a healthcheck"


# ---------------------------------------------------------------------------
# 3. scripts/reproduce.sh passes bash syntax check
# ---------------------------------------------------------------------------


def test_reproduce_sh_exists_and_is_executable() -> None:
    """scripts/reproduce.sh must exist and have the executable bit set."""
    sh = _repo("scripts/reproduce.sh")
    assert sh.is_file(), f"scripts/reproduce.sh not found at {sh}"
    assert sh.stat().st_mode & 0o111, "scripts/reproduce.sh is not executable"


def test_reproduce_sh_is_bash_syntax_clean() -> None:
    """bash -n must report no syntax errors in scripts/reproduce.sh."""
    sh = _repo("scripts/reproduce.sh")
    result = subprocess.run(
        ["bash", "-n", str(sh)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"bash -n reported syntax errors in scripts/reproduce.sh:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# 4. .dockerignore is present and excludes the right directories
# ---------------------------------------------------------------------------


def test_dockerignore_exists() -> None:
    """.dockerignore must be present so the build context is lean."""
    assert _repo(".dockerignore").is_file()


@pytest.mark.parametrize(
    "excluded",
    [".venv/", "tests/", ".git/", "__pycache__/", ".pytest_cache/"],
)
def test_dockerignore_excludes_expected_paths(excluded: str) -> None:
    """.dockerignore must exclude heavyweight dev-only directories."""
    content = _repo(".dockerignore").read_text()
    assert excluded in content, f".dockerignore is missing entry for '{excluded}'"


def test_dockerignore_keeps_results_demo() -> None:
    """results/demo/ (the baseline reference) must NOT be excluded by .dockerignore.

    A blanket `results/` line would strip the committed baseline from the build
    context, breaking the reproduce.sh diff step.  We check that no non-comment
    line reads exactly `results/` or `results/demo` (without a re-inclusion
    `!results/demo` negation).
    """
    lines = _repo(".dockerignore").read_text().splitlines()
    # Strip comment lines and blank lines before checking.
    active_lines = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    negations = {ln[1:] for ln in active_lines if ln.startswith("!")}
    for line in active_lines:
        if line.startswith("!"):
            continue
        # A blanket `results/` without a matching negation would be wrong.
        if line in ("results/", "results"):
            assert "results/demo" in negations or "results/demo/" in negations, (
                f".dockerignore has '{line}' without a '!results/demo' re-inclusion — "
                "the reference baseline would be excluded from the build context"
            )


# ---------------------------------------------------------------------------
# 5. CI workflow for reproducibility exists
# ---------------------------------------------------------------------------


def test_reproducibility_workflow_exists() -> None:
    """The reproducibility CI workflow file must be present."""
    wf = _repo(".github/workflows/reproducibility.yml")
    assert wf.is_file(), f"reproducibility.yml not found at {wf}"


def test_reproducibility_workflow_is_valid_yaml() -> None:
    """Reproducibility workflow must be parseable YAML."""
    content = _repo(".github/workflows/reproducibility.yml").read_text()
    data = yaml.safe_load(content)
    assert isinstance(data, dict)
    assert "jobs" in data


def test_reproducibility_workflow_has_weekly_cron() -> None:
    """Workflow must run on a weekly schedule (cron trigger).

    PyYAML parses bare `on:` as boolean True (YAML 1.1 spec); access the
    triggers via the boolean key True rather than the string "on".
    """
    data = yaml.safe_load(_repo(".github/workflows/reproducibility.yml").read_text())
    # PyYAML 1.1 maps bare `on` → True; GitHub Actions files always use `on:`.
    triggers = data.get(True) or data.get("on") or {}
    assert "schedule" in triggers, "reproducibility workflow must have a schedule trigger"
    cron_entries = triggers["schedule"]
    assert any("cron" in entry for entry in cron_entries), (
        "schedule trigger must include a cron expression"
    )


# ---------------------------------------------------------------------------
# 6. Reference baseline exists (needed by reproduce.sh diff)
# ---------------------------------------------------------------------------


def test_results_demo_summary_parquet_exists() -> None:
    """results/demo/summary.parquet must be committed as the reference baseline."""
    parquet = _repo("results/demo/summary.parquet")
    assert parquet.is_file(), (
        f"Reference baseline not found at {parquet}. "
        "Run `make bench-demo` and commit the result."
    )

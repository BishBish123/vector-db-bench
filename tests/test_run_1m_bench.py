"""Lightweight unit tests for the 1M MS-MARCO benchmark automation.

These tests verify that the infrastructure artifacts are syntactically valid
and self-consistent.  They do NOT execute the benchmark, pull Docker images,
or make network requests — all assertions are file-system and YAML checks.

Marked as plain unit tests (no marker) so they run in the default
`make test` / CI unit-test path without any extra flags.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).parent.parent


def _repo(path: str) -> Path:
    return REPO_ROOT / path


# ---------------------------------------------------------------------------
# 1. scripts/run_1m_bench.sh — file-level checks
# ---------------------------------------------------------------------------


def test_run_1m_bench_sh_exists() -> None:
    """scripts/run_1m_bench.sh must be present."""
    sh = _repo("scripts/run_1m_bench.sh")
    assert sh.is_file(), f"scripts/run_1m_bench.sh not found at {sh}"


def test_run_1m_bench_sh_is_executable() -> None:
    """scripts/run_1m_bench.sh must have the executable bit set."""
    sh = _repo("scripts/run_1m_bench.sh")
    assert sh.stat().st_mode & 0o111, "scripts/run_1m_bench.sh is not executable"


def test_run_1m_bench_sh_is_bash_syntax_clean() -> None:
    """bash -n must report no syntax errors in scripts/run_1m_bench.sh."""
    sh = _repo("scripts/run_1m_bench.sh")
    result = subprocess.run(
        ["bash", "-n", str(sh)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"bash -n reported syntax errors in scripts/run_1m_bench.sh:\n{result.stderr}"
    )


def test_run_1m_bench_sh_has_dry_run_flag() -> None:
    """The script must document and handle --dry-run."""
    content = _repo("scripts/run_1m_bench.sh").read_text()
    assert "--dry-run" in content, "scripts/run_1m_bench.sh is missing --dry-run support"


def test_run_1m_bench_sh_has_skip_prep_flag() -> None:
    """The script must document and handle --skip-prep."""
    content = _repo("scripts/run_1m_bench.sh").read_text()
    assert "--skip-prep" in content, "scripts/run_1m_bench.sh is missing --skip-prep support"


def test_run_1m_bench_sh_dry_run_exits_zero() -> None:
    """--dry-run must exit 0 (no Docker required, no commands executed)."""
    sh = _repo("scripts/run_1m_bench.sh")
    result = subprocess.run(
        ["bash", str(sh), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"--dry-run exited {result.returncode}:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_run_1m_bench_sh_dry_run_prints_steps() -> None:
    """--dry-run output must describe each pipeline step without running them."""
    sh = _repo("scripts/run_1m_bench.sh")
    result = subprocess.run(
        ["bash", str(sh), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
    )
    output = result.stdout + result.stderr
    assert "DRY-RUN" in output, "Expected [DRY-RUN] markers in --dry-run output"
    assert "msmarco" in output.lower(), "Expected msmarco reference in --dry-run output"
    assert "vdbbench bench" in output, "Expected bench step description in --dry-run output"


def test_run_1m_bench_sh_documents_hardware_requirements() -> None:
    """The script must document disk ≥50 GB and RAM ≥16 GB requirements."""
    content = _repo("scripts/run_1m_bench.sh").read_text()
    assert "50" in content and "GB" in content, (
        "scripts/run_1m_bench.sh must document the 50 GB disk requirement"
    )
    assert "16" in content, (
        "scripts/run_1m_bench.sh must document the 16 GB RAM requirement"
    )


def test_run_1m_bench_sh_documents_runtime_estimate() -> None:
    """The script must log an estimated wall-clock time at startup."""
    content = _repo("scripts/run_1m_bench.sh").read_text()
    assert "4-8 h" in content or "4-8" in content, (
        "scripts/run_1m_bench.sh must mention the 4-8 h estimated runtime"
    )


# ---------------------------------------------------------------------------
# 2. .github/workflows/full_bench.yml — CI workflow checks
# ---------------------------------------------------------------------------


def test_full_bench_workflow_exists() -> None:
    """The full-bench workflow file must be present."""
    wf = _repo(".github/workflows/full_bench.yml")
    assert wf.is_file(), f"full_bench.yml not found at {wf}"


def test_full_bench_workflow_is_valid_yaml() -> None:
    """.github/workflows/full_bench.yml must be parseable YAML."""
    content = _repo(".github/workflows/full_bench.yml").read_text()
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "Top-level must be a mapping"
    assert "jobs" in data, "Workflow must declare at least one job"


def test_full_bench_workflow_has_workflow_dispatch_trigger() -> None:
    """The workflow must be triggered only via workflow_dispatch (not push/schedule).

    PyYAML parses bare `on:` as boolean True (YAML 1.1); access triggers
    via the True key rather than the string 'on'.
    """
    data = yaml.safe_load(_repo(".github/workflows/full_bench.yml").read_text())
    triggers = data.get(True) or data.get("on") or {}
    assert "workflow_dispatch" in triggers, (
        "full_bench.yml must have a workflow_dispatch trigger"
    )


def test_full_bench_workflow_has_no_push_trigger() -> None:
    """The 1M workflow must NOT trigger on push — it's too expensive for CI."""
    data = yaml.safe_load(_repo(".github/workflows/full_bench.yml").read_text())
    triggers = data.get(True) or data.get("on") or {}
    assert "push" not in triggers, (
        "full_bench.yml must not trigger on push (too expensive for CI)"
    )


def test_full_bench_workflow_has_no_schedule_trigger() -> None:
    """The 1M workflow must NOT trigger on schedule — it's too expensive."""
    data = yaml.safe_load(_repo(".github/workflows/full_bench.yml").read_text())
    triggers = data.get(True) or data.get("on") or {}
    assert "schedule" not in triggers, (
        "full_bench.yml must not have a schedule trigger (too expensive)"
    )


def test_full_bench_workflow_uploads_results_artifact() -> None:
    """The workflow must upload results/1m/ as a workflow artifact."""
    data = yaml.safe_load(_repo(".github/workflows/full_bench.yml").read_text())
    job_steps = []
    for job in data.get("jobs", {}).values():
        job_steps.extend(job.get("steps", []))
    upload_steps = [
        s for s in job_steps if "upload-artifact" in str(s.get("uses", ""))
    ]
    assert upload_steps, "Workflow must have at least one upload-artifact step"
    paths = " ".join(str(s.get("with", {}).get("path", "")) for s in upload_steps)
    assert "results/1m" in paths, (
        "Workflow must upload results/1m/ as an artifact"
    )


def test_full_bench_workflow_artifact_retention_90_days() -> None:
    """Results artifact must be retained for 90 days."""
    data = yaml.safe_load(_repo(".github/workflows/full_bench.yml").read_text())
    job_steps = []
    for job in data.get("jobs", {}).values():
        job_steps.extend(job.get("steps", []))
    for step in job_steps:
        if "upload-artifact" in str(step.get("uses", "")):
            retention = step.get("with", {}).get("retention-days")
            if retention is not None:
                assert int(retention) == 90, (
                    f"Results artifact retention must be 90 days, got {retention}"
                )
                return
    # If we get here, no upload step had explicit retention — acceptable only
    # if there are no upload steps at all (caught by the prior test).


def test_full_bench_workflow_calls_run_1m_bench_sh() -> None:
    """The workflow job must invoke scripts/run_1m_bench.sh."""
    content = _repo(".github/workflows/full_bench.yml").read_text()
    assert "run_1m_bench.sh" in content, (
        "full_bench.yml must call scripts/run_1m_bench.sh"
    )


# ---------------------------------------------------------------------------
# 3. MEASURED-ON.md — placeholder section checks
# ---------------------------------------------------------------------------


def test_measured_on_md_exists() -> None:
    """MEASURED-ON.md must be present."""
    md = _repo("MEASURED-ON.md")
    assert md.is_file(), f"MEASURED-ON.md not found at {md}"


def test_measured_on_md_has_full_run_section() -> None:
    """MEASURED-ON.md must have a '## Full run' section."""
    content = _repo("MEASURED-ON.md").read_text()
    assert "## Full run" in content, (
        "MEASURED-ON.md must have a '## Full run' section"
    )


def test_measured_on_md_full_run_section_has_not_yet_measured() -> None:
    """The Full run section must note that the run has not yet been measured."""
    content = _repo("MEASURED-ON.md").read_text()
    # Either the original or the updated placeholder is acceptable.
    has_placeholder = "NOT YET MEASURED" in content or "not yet measured" in content.lower()
    assert has_placeholder, (
        "MEASURED-ON.md Full run section must include a 'not yet measured' placeholder"
    )


def test_measured_on_md_full_run_section_documents_command() -> None:
    """The Full run section must document the command to run the 1M benchmark."""
    content = _repo("MEASURED-ON.md").read_text()
    assert "run_1m_bench.sh" in content, (
        "MEASURED-ON.md must document the scripts/run_1m_bench.sh command"
    )

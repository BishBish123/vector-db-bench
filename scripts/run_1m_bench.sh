#!/usr/bin/env bash
# scripts/run_1m_bench.sh — Full 1M MS-MARCO benchmark automation.
#
# What this does (in order):
#   1. Checks prerequisites (Docker running, disk ≥50 GB, RAM ≥16 GB).
#   2. Brings up the full stack via docker-compose.full.yml.
#   3. Downloads + encodes 1M MS-MARCO passages (skipped with --skip-prep).
#   4. Runs vdbbench bench --all over the encoded bundle.
#   5. Generates the HTML report.
#   6. Generates plots from the summary parquet.
#   7. Tears down the docker-compose stack.
#   8. Prints a summary of what ran and where results landed.
#
# Usage:
#   ./scripts/run_1m_bench.sh [--dry-run] [--skip-prep]
#
#   --dry-run     Print every step without executing it.
#   --skip-prep   Skip the `vdbbench prep` step; assumes data/encoded-1m already exists.
#
# Environment overrides:
#   PGVECTOR_DSN   DSN for pgvector (default: postgresql://bench:bench@localhost:5433/bench)
#   QDRANT_URL     URL for qdrant   (default: http://localhost:6333)
#   COMPOSE_FILE   Path to the full-stack compose file (default: docker-compose.full.yml)
#
# Hardware requirements:
#   - Docker (Engine or Desktop) running and reachable via `docker info`
#   - Disk: ≥50 GB free on the partition holding the repo (HF cache + encoded bundle + results)
#   - RAM:  ≥16 GB (the encode step and HNSW index build peak around 12–14 GB)
#   - CPU:  ≥4 cores recommended; 8 cores cuts wall-clock roughly in half
#
# Estimated wall-clock (shared-cpu-1x, 4 cores):
#   - MS-MARCO download + encode: 2–4 h (depends on HF bandwidth and CPU)
#   - pgvector ingest + index:    1–2 h
#   - qdrant ingest + index:      30–60 min
#   - lancedb ingest + index:     30–60 min (Linux / arm64 only)
#   - chroma ingest + index:      30–60 min (Linux / arm64 only)
#   - Total:                      4–8 h
#
# Exit codes:
#   0  All steps completed successfully.
#   1  Prerequisite check failed or a pipeline step returned non-zero.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${REPO_ROOT}/docker-compose.full.yml}"
ENCODED_DIR="${REPO_ROOT}/data/encoded-1m"
RESULTS_BASE="${REPO_ROOT}/results/1m"
ASSETS_DIR="${REPO_ROOT}/assets/1m"
PGVECTOR_DSN="${PGVECTOR_DSN:-postgresql://bench:bench@localhost:5433/bench}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

DRY_RUN=false
SKIP_PREP=false

# ── Argument parsing ────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dry-run)   DRY_RUN=true ;;
        --skip-prep) SKIP_PREP=true ;;
        *)
            echo "[run_1m_bench] Unknown argument: $arg" >&2
            echo "Usage: $0 [--dry-run] [--skip-prep]" >&2
            exit 1
            ;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────────

log()  { echo "[run_1m_bench] $*"; }
step() { echo ""; echo "[run_1m_bench] ── $* ──"; }

run() {
    # run <description> <cmd> [args...]
    local desc="$1"; shift
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $desc"
        echo "  cmd: $*"
    else
        log "$desc"
        "$@"
    fi
}

# ── Banner ──────────────────────────────────────────────────────────────────

step "1M MS-MARCO benchmark"
log "repo root    : $REPO_ROOT"
log "compose file : $COMPOSE_FILE"
log "encoded dir  : $ENCODED_DIR"
log "results base : $RESULTS_BASE"
log "assets dir   : $ASSETS_DIR"
log "pgvector DSN : $PGVECTOR_DSN"
log "qdrant URL   : $QDRANT_URL"
log "dry-run      : $DRY_RUN"
log "skip-prep    : $SKIP_PREP"
log ""
log "Estimated wall-clock: 4-8 h on shared-cpu-1x (4 cores)"
log "  breakdown: encode 2-4 h | pgvector ingest 1-2 h | other adapters 1-2 h"

# ── Step 1: Prerequisite checks ─────────────────────────────────────────────

step "Step 1/7: Prerequisite checks"

# 1a. Docker must be running.
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] check: docker info (Docker must be running)"
else
    if ! docker info >/dev/null 2>&1; then
        log "ERROR: Docker is not running or not reachable." >&2
        log "       Start Docker and re-run this script." >&2
        exit 1
    fi
    log "OK  Docker is running"
fi

# 1b. Disk space — warn if <50 GB free.
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] check: df -k (≥50 GB free on repo partition)"
else
    _free_kb=$(df -k "$REPO_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')
    _free_gb=$(( ${_free_kb:-0} / 1048576 ))
    if (( _free_gb < 50 )); then
        log "WARNING: only ${_free_gb} GB free on $(df -k "$REPO_ROOT" | awk 'NR==2{print $1}')." >&2
        log "         The 1M prep + results need ≥50 GB.  Continuing, but you may hit ENOSPC." >&2
    else
        log "OK  ${_free_gb} GB free (≥50 GB required)"
    fi
fi

# 1c. RAM — warn if <16 GB total.
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] check: /proc/meminfo or sysctl (≥16 GB RAM recommended)"
else
    if [[ -r /proc/meminfo ]]; then
        _mem_kb=$(awk '/MemTotal/{print $2}' /proc/meminfo)
        _mem_gb=$(( ${_mem_kb:-0} / 1048576 ))
    elif command -v sysctl >/dev/null 2>&1; then
        _mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo 0)
        _mem_gb=$(( _mem_bytes / 1073741824 ))
    else
        _mem_gb=0
    fi
    if (( _mem_gb < 16 )); then
        log "WARNING: ${_mem_gb} GB RAM detected; ≥16 GB is recommended for the 1M run." >&2
        log "         Continuing — the bench will use swap if needed, which will be slower." >&2
    else
        log "OK  ${_mem_gb} GB RAM (≥16 GB required)"
    fi
fi

# ── Step 2: Bring up the full stack ─────────────────────────────────────────

step "Step 2/7: Bring up docker-compose full stack"

run "docker-compose down (clean slate)" \
    docker compose -f "$COMPOSE_FILE" down --remove-orphans

run "docker-compose up -d --wait" \
    docker compose -f "$COMPOSE_FILE" up -d --wait

# ── Step 3: Download + encode MS-MARCO (optional) ───────────────────────────

step "Step 3/7: Prep — download + encode 1M MS-MARCO passages"

if [[ "$SKIP_PREP" == "true" ]]; then
    if [[ "$DRY_RUN" == "false" ]] && [[ ! -d "$ENCODED_DIR" ]]; then
        log "ERROR: --skip-prep requested but $ENCODED_DIR does not exist." >&2
        log "       Remove --skip-prep or run without it to build the encoded bundle." >&2
        exit 1
    fi
    log "Skipping prep (--skip-prep flag set); using existing $ENCODED_DIR"
else
    run "vdbbench prep --dataset msmarco --limit 1000000 --out $ENCODED_DIR" \
        uv run vdbbench prep \
            --dataset msmarco \
            --limit 1000000 \
            --out "$ENCODED_DIR"
fi

# ── Step 4: Run the benchmark ────────────────────────────────────────────────

step "Step 4/7: Bench — run all adapters"

RUN_DIR="${RESULTS_BASE}/run-$(date +%Y%m%d)"

run "vdbbench bench --all --encoded $ENCODED_DIR --out $RUN_DIR" \
    uv run vdbbench bench \
        --all \
        --encoded "$ENCODED_DIR" \
        --out "$RUN_DIR" \
        --pgvector-dsn "$PGVECTOR_DSN" \
        --qdrant-url "$QDRANT_URL"

# ── Step 5: Generate HTML report ─────────────────────────────────────────────

step "Step 5/7: Report — generate HTML"

run "vdbbench report html --from $RUN_DIR --out $RUN_DIR/report.html" \
    uv run vdbbench report html \
        --from "$RUN_DIR" \
        --out "$RUN_DIR/report.html"

# ── Step 6: Generate plots ────────────────────────────────────────────────────

step "Step 6/7: Plot — generate charts"

run "vdbbench plot --summary $RUN_DIR/summary.parquet --out $ASSETS_DIR" \
    uv run vdbbench plot \
        --summary "$RUN_DIR/summary.parquet" \
        --out "$ASSETS_DIR"

# ── Step 7: Tear down docker-compose ─────────────────────────────────────────

step "Step 7/7: Tear down docker-compose"

run "docker-compose down" \
    docker compose -f "$COMPOSE_FILE" down --remove-orphans

# ── Summary ──────────────────────────────────────────────────────────────────

step "Complete"
if [[ "$DRY_RUN" == "true" ]]; then
    log "DRY-RUN complete — no commands were executed."
    log ""
    log "To run for real:"
    log "  ./scripts/run_1m_bench.sh"
    log "  ./scripts/run_1m_bench.sh --skip-prep   # if $ENCODED_DIR already exists"
else
    log "1M MS-MARCO benchmark finished."
    log ""
    log "Results:"
    log "  Parquet:     $RUN_DIR/summary.parquet"
    log "  HTML report: $RUN_DIR/report.html"
    log "  Charts:      $ASSETS_DIR/"
    log ""
    log "To update MEASURED-ON.md, fill in the 'Full run' section with"
    log "the metadata from $RUN_DIR/bench_manifest.json."
fi

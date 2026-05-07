#!/usr/bin/env bash
# scripts/reproduce.sh — containerised reproducibility check.
#
# What this does:
#   1. Builds the vdbbench Docker image.
#   2. Brings up the full stack (pgvector + qdrant + vdbbench) via
#      docker-compose.full.yml.
#   3. Runs the smoke benchmark inside the container (synthetic, 5k vectors).
#   4. Copies results out to ./results/reproduced/.
#   5. Diffs the produced summary.parquet against the committed reference at
#      ./results/demo/summary.parquet.
#   6. Exits non-zero if the result differs beyond the documented tolerance.
#
# Usage:
#   ./scripts/reproduce.sh [--smoke-only]
#
#   --smoke-only   Skip Docker / compose — run `vdbbench bench --adapter exact`
#                  offline. Useful for CI where Docker-in-Docker is unavailable.
#
# Environment overrides:
#   TOLERANCE_LATENCY_PCT   Max % difference in p95 latency before flagging drift
#                           (default: 50, reflecting single-machine noise).
#   TOLERANCE_RECALL_ABS    Max absolute recall@10 difference (default: 0.05).
#   RESULTS_DIR             Host path to write reproduced results into
#                           (default: ./results/reproduced).
#
# Exit codes:
#   0   Results within tolerance (or --smoke-only produced a valid parquet).
#   1   Drift detected, Docker failure, or missing reference baseline.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.full.yml"
REFERENCE="${REPO_ROOT}/results/demo/summary.parquet"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results/reproduced}"
TOLERANCE_LATENCY_PCT="${TOLERANCE_LATENCY_PCT:-50}"
TOLERANCE_RECALL_ABS="${TOLERANCE_RECALL_ABS:-0.05}"
SMOKE_ONLY=false
ENCODED_DIR="${REPO_ROOT}/data/encoded-demo"

# ── Argument parsing ────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --smoke-only) SMOKE_ONLY=true ;;
        *) echo "[reproduce] Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "[reproduce] repo root : $REPO_ROOT"
echo "[reproduce] reference  : $REFERENCE"
echo "[reproduce] results dir: $RESULTS_DIR"
echo "[reproduce] tolerance  : latency ±${TOLERANCE_LATENCY_PCT}%, recall ±${TOLERANCE_RECALL_ABS}"

# ── Verify reference baseline exists ───────────────────────────────────────
if [[ ! -f "$REFERENCE" ]]; then
    echo "[reproduce] ERROR: reference baseline not found at $REFERENCE" >&2
    echo "[reproduce]        Run \`make bench-demo\` first to produce it." >&2
    exit 1
fi

mkdir -p "$RESULTS_DIR"

# ── Smoke-only path (no Docker) ────────────────────────────────────────────
if [[ "$SMOKE_ONLY" == "true" ]]; then
    echo "[reproduce] smoke-only mode: running offline exact adapter"
    # Prep a synthetic encoded bundle if one doesn't exist yet.
    if [[ ! -d "${ENCODED_DIR}" ]]; then
        echo "[reproduce] building synthetic encoded bundle at ${ENCODED_DIR}"
        uv run vdbbench prep \
            --dataset synthetic \
            --sample-size 5000 \
            --dim 64 \
            --out "${ENCODED_DIR}"
    fi
    uv run vdbbench bench \
        --adapter exact \
        --encoded "${ENCODED_DIR}" \
        --out "${RESULTS_DIR}"
    PRODUCED="${RESULTS_DIR}/summary.parquet"
else
    # ── Docker path ───────────────────────────────────────────────────────────
    echo "[reproduce] step 1/4: building Docker image"
    docker build --tag vdbbench:latest "${REPO_ROOT}"

    # Prep a synthetic encoded bundle on the host so the container can read it.
    if [[ ! -d "${ENCODED_DIR}" ]]; then
        echo "[reproduce] step 1b: building encoded bundle (not cached)"
        uv run vdbbench prep \
            --dataset synthetic \
            --sample-size 5000 \
            --dim 64 \
            --out "${ENCODED_DIR}"
    fi

    echo "[reproduce] step 2/4: bringing up full stack"
    export RESULTS_DIR DATA_DIR="${REPO_ROOT}/data"
    docker compose -f "${COMPOSE_FILE}" down --remove-orphans 2>/dev/null || true
    docker compose -f "${COMPOSE_FILE}" up --build -d pgvector qdrant
    # Wait for services to be healthy before starting the bench runner.
    docker compose -f "${COMPOSE_FILE}" up --no-build --exit-code-from vdbbench vdbbench

    echo "[reproduce] step 3/4: collecting results"
    PRODUCED="${RESULTS_DIR}/summary.parquet"

    echo "[reproduce] step 4/4: tearing down"
    docker compose -f "${COMPOSE_FILE}" down --remove-orphans 2>/dev/null || true
fi

# ── Parquet drift check ─────────────────────────────────────────────────────
echo "[reproduce] diffing ${PRODUCED} against ${REFERENCE}"

if [[ ! -f "$PRODUCED" ]]; then
    echo "[reproduce] ERROR: bench did not produce $PRODUCED" >&2
    exit 1
fi

# Use the Python environment to inspect the parquets.
uv run python - <<PYEOF
import sys
import pandas as pd

ref_path  = "${REFERENCE}"
new_path  = "${PRODUCED}"
tol_lat   = float("${TOLERANCE_LATENCY_PCT}")
tol_rec   = float("${TOLERANCE_RECALL_ABS}")

ref = pd.read_parquet(ref_path)
new = pd.read_parquet(new_path)

errors = []

# Every DB in the reference must appear in the reproduction.
ref_dbs = set(ref["db"].unique())
new_dbs = set(new["db"].unique())
missing = ref_dbs - new_dbs
if missing:
    errors.append(f"Missing DBs in reproduction: {missing}")

# For each DB present in both, check metrics within tolerance.
common = ref_dbs & new_dbs
for db in sorted(common):
    r = ref[ref["db"] == db].iloc[0]
    n = new[new["db"] == db].iloc[0]

    if "latency_ms_p95" in ref.columns and "latency_ms_p95" in new.columns:
        ref_lat = float(r["latency_ms_p95"])
        new_lat = float(n["latency_ms_p95"])
        if ref_lat > 0:
            pct_diff = abs(new_lat - ref_lat) / ref_lat * 100
            if pct_diff > tol_lat:
                errors.append(
                    f"{db}: p95 latency {new_lat:.1f}ms vs reference {ref_lat:.1f}ms "
                    f"(±{pct_diff:.1f}% > tolerance {tol_lat}%)"
                )

    if "recall_at_k_mean" in ref.columns and "recall_at_k_mean" in new.columns:
        ref_rec = float(r["recall_at_k_mean"])
        new_rec = float(n["recall_at_k_mean"])
        abs_diff = abs(new_rec - ref_rec)
        if abs_diff > tol_rec:
            errors.append(
                f"{db}: recall@k {new_rec:.4f} vs reference {ref_rec:.4f} "
                f"(Δ={abs_diff:.4f} > tolerance {tol_rec})"
            )

if errors:
    print("[reproduce] DRIFT DETECTED:", file=sys.stderr)
    for e in errors:
        print(f"  {e}", file=sys.stderr)
    sys.exit(1)

print(f"[reproduce] OK — {len(common)} DB(s) within tolerance")
PYEOF

echo "[reproduce] reproducibility check passed"

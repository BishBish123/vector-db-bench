.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON ?= python3
UV ?= uv
# `make prep` / `make bench-100k` use SAMPLE_SIZE — 100k is the right
# default for the 100k-scale sweep target. The CLI's own --sample-size
# default is 5000 (demo scale) so `uv run vdbbench prep` without a flag
# matches results/demo/ — see `make bench-demo` for the demo wrapper.
SAMPLE_SIZE ?= 100000
EMBED_MODEL ?= BAAI/bge-small-en-v1.5

# ENCODER selects which embedding model to use for `make bench-demo`.
# Choices: bge (default, 384-dim) | nomic (768-dim, requires trust_remote_code)
# Example: make bench-demo ENCODER=nomic
ENCODER ?= bge
# Derive EMBED_MODEL from ENCODER when not set explicitly.
ifeq ($(ENCODER),nomic)
  EMBED_MODEL := nomic-ai/nomic-embed-text-v1.5
endif

# Container ports — overridable so collisions with an existing local
# Postgres / Qdrant don't force the user to edit docker-compose.yml.
# `make up PGVECTOR_PORT=5444 QDRANT_PORT=6343` is the documented escape
# hatch. QDRANT_PORT and QDRANT_GRPC_PORT are independent overrides
# (the gRPC port does NOT shift automatically when you change the HTTP
# port); set `QDRANT_GRPC_PORT=6344` explicitly if 6334 is also taken.
# The gRPC port is currently exposed for operator-side conflict
# resolution and reserved for future gRPC support — the bench harness
# itself only talks HTTP today.
PGVECTOR_PORT ?= 5433
QDRANT_PORT ?= 6333
QDRANT_GRPC_PORT ?= 6334
export PGVECTOR_PORT
export QDRANT_PORT
export QDRANT_GRPC_PORT

# Derive the bench-side DSN / URL from the same vars so `make bench-demo
# PGVECTOR_PORT=5444` actually reaches the rebound container instead of
# silently hitting whatever happens to be on 5433. Override these
# directly if you're benching against a non-Docker service.
PGVECTOR_DSN ?= postgresql://bench:bench@localhost:$(PGVECTOR_PORT)/bench
QDRANT_URL ?= http://localhost:$(QDRANT_PORT)

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Install all extras + dev tooling (platform-gated extras are skipped where wheels are unavailable)
	$(UV) sync --extra dev --extra embed --extra chroma --extra lance
	@# Surface platform gating: on Intel macOS the embed/chroma/lance
	@# extras silently no-op (no wheels) and `make install` still exits
	@# 0, leaving the operator with a 2-adapter harness and no signal.
	@# This probe prints OK / SKIP per optional dep so the gap is obvious
	@# at install time rather than at bench-demo time.
	@$(UV) run python scripts/check_adapter_availability.py

.PHONY: install-min
install-min: ## Install only core + dev (no embed/chroma/lance — Intel macOS path)
	$(UV) sync --extra dev

.PHONY: lock
lock: ## Refresh uv lockfile
	$(UV) lock

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
.PHONY: fmt
fmt: ## Format with ruff
	$(UV) run ruff format src tests

.PHONY: lint
lint: ## Lint with ruff (no fixes)
	$(UV) run ruff check src tests

.PHONY: lint-fix
lint-fix: ## Lint and apply safe fixes
	$(UV) run ruff check --fix src tests

.PHONY: typecheck
typecheck: ## Type-check with mypy
	$(UV) run mypy src

.PHONY: check
check: lint typecheck ## Lint + typecheck

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
.PHONY: test
test: ## Run unit tests
	$(UV) run pytest -m "not integration and not slow"

.PHONY: test-integration
test-integration: ## Run integration tests (requires Docker)
	$(UV) run pytest -m integration

.PHONY: test-all
test-all: ## Run the full test suite
	$(UV) run pytest

# ---------------------------------------------------------------------------
# Benchmark pipeline (filled in by later phases)
# ---------------------------------------------------------------------------
.PHONY: prep
prep: ## Build corpus + ground-truth (deterministic)
	$(UV) run vdbbench prep --sample-size $(SAMPLE_SIZE) --embed-model $(EMBED_MODEL)

.PHONY: bench
bench: ## Run benchmark across all DBs at SAMPLE_SIZE (default 100k) — writes results/100k/
	$(UV) run vdbbench bench --all --out results/100k \
		--pgvector-dsn $(PGVECTOR_DSN) --qdrant-url $(QDRANT_URL)

.PHONY: smoke
smoke: ## End-to-end offline smoke (no Docker, no model download) — exercises corpus → encode → bench → plot
	$(UV) run python scripts/smoke_pipeline.py

.PHONY: bench-demo
bench-demo: ## Demo pipeline: synthetic 5k vectors @ dim=64, writes results/demo/. ENCODER=bge|nomic
	@# The committed `results/demo/summary.parquet` was captured on an
	@# Intel macOS host where lancedb and chromadb have no wheels, so it
	@# contains 2 rows (pgvector + qdrant). A run on Linux / Apple
	@# Silicon / WSL2 will produce 4 rows; that's a wider sweep, not a
	@# regression. See the README's "Platform support" table.
	@# Use ENCODER=nomic to swap in the 768-dim nomic model (requires
	@# trust_remote_code and a ~500 MB model download on first run).
	$(UV) run vdbbench prep  --out data/encoded-demo --dataset synthetic --sample-size 5000 --dim 64
	$(UV) run vdbbench bench --encoded data/encoded-demo --out results/demo --all \
		--pgvector-dsn $(PGVECTOR_DSN) --qdrant-url $(QDRANT_URL)
	$(UV) run vdbbench plot  --summary results/demo/summary.parquet --out assets

.PHONY: bench-demo-nomic
bench-demo-nomic: ## Demo with nomic-embed-text-v1.5 (768-dim). Alias for: make bench-demo ENCODER=nomic
	@# nomic-embed-text-v1.5 is a 768-dim SBERT-compatible model.
	@# First run downloads ~500 MB of model weights from HuggingFace.
	@# The model requires trust_remote_code=True (custom modeling file).
	@# Read https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 before use.
	$(MAKE) bench-demo ENCODER=nomic EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5

.PHONY: bench-100k
bench-100k: prep bench plots ## 100k-scale sweep (uses SAMPLE_SIZE/EMBED_MODEL); writes results/100k/

.PHONY: bench-1m
bench-1m: ## Full 1M MS-MARCO sweep (results/full/summary.parquet; ~hours, no commit)
	$(UV) run vdbbench prep  --dataset msmarco --sample-size 1000000 --out data/encoded-1m
	$(UV) run vdbbench bench --encoded data/encoded-1m --out results/full --all --profile p99 \
		--pgvector-dsn $(PGVECTOR_DSN) --qdrant-url $(QDRANT_URL)
	$(UV) run vdbbench plot  --summary results/full/summary.parquet --out assets/full

.PHONY: bench-all
bench-all: bench-1m  ## Run the full 1M benchmark (alias for bench-1m)

.PHONY: bench-1m-real
bench-1m-real: ## Full 1M automated pipeline via scripts/run_1m_bench.sh (press-one-button reproduction)
	@# Requires Docker running, ≥50 GB free disk, ≥16 GB RAM.
	@# Estimated wall-clock: 4-8 h on a 4-core shared-cpu-1x host.
	@# Use --skip-prep if data/encoded-1m already exists from a previous run.
	@# Use --dry-run to print the steps without executing.
	./scripts/run_1m_bench.sh

.PHONY: plots
plots: ## Regenerate analysis plots from results/100k/summary.parquet
	$(UV) run vdbbench plot --summary results/100k/summary.parquet --out assets/100k

.PHONY: report
report: ## Render the demo HTML report → results/demo/report.html (self-contained, no external deps)
	$(UV) run vdbbench report html --from results/demo/ --out results/demo/report.html --charts-dir assets

# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
.PHONY: up-pull
up-pull: ## Pre-pull pgvector + qdrant images so `up --wait` doesn't time out on first run
	docker compose pull

.PHONY: up-wait
up-wait: ## Start containers and wait for healthy (assumes images already pulled)
	@# Pre-flight: warn early if a host port the bench will publish is
	@# already allocated. Without this the operator only sees the
	@# collision after `docker compose` has created the network and
	@# started pulling layers, which is several seconds of confusing
	@# "Bind for 0.0.0.0:5433 failed" output. `lsof` is in the macOS /
	@# Linux base image; `command -v` keeps the recipe portable on the
	@# rare host without it.
	@bash -c 'command -v lsof >/dev/null 2>&1 || exit 0; \
		busy=$$(lsof -nP -iTCP:$(PGVECTOR_PORT) -iTCP:$(QDRANT_PORT) -sTCP:LISTEN 2>/dev/null | awk "NR>1 {print \$$9}" | sed "s/.*://" | sort -u | tr "\n" " "); \
		if [ -n "$$busy" ]; then \
			echo "[make up] ports already allocated: $$busy"; \
			echo "  override the publish port and re-run, e.g.:"; \
			echo "    make up PGVECTOR_PORT=5444 QDRANT_PORT=6343"; \
			echo "  current targets: pgvector=$(PGVECTOR_PORT), qdrant=$(QDRANT_PORT)"; \
			exit 1; \
		fi'
	@# Sweep stale "Created" containers from prior failed runs first —
	@# otherwise `up -d --wait` errors out with "container is already in
	@# use" and the operator has to manually `docker compose down` before
	@# they can iterate. `|| true` because a clean host has nothing to
	@# remove and `down` reports a non-zero on missing project state.
	@docker compose down --remove-orphans 2>/dev/null || true
	@docker compose up -d --wait || { \
		echo ""; \
		echo "[make up] docker compose failed."; \
		echo "  If a host port is already allocated, override the published port:"; \
		echo "    make up PGVECTOR_PORT=5444 QDRANT_PORT=6343"; \
		echo "  Current published ports: pgvector=$(PGVECTOR_PORT), qdrant=$(QDRANT_PORT)."; \
		exit 1; \
	}

.PHONY: clean-containers
clean-containers: ## Force-remove pgvector + qdrant containers and their volumes (use when 'make up' is wedged)
	docker compose down -v --remove-orphans

.PHONY: up
up: up-pull up-wait ## Bring up pgvector + qdrant containers (pulls images first, then waits for healthy)

.PHONY: down
down: ## Tear down containers and remove their volumes
	docker compose down -v

.PHONY: ps
ps: ## Show container status
	docker compose ps

# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------
.PHONY: sweep
sweep: ## Knob-grid Pareto sweep — exact adapter, tiny grid (6 trials), writes results/sweep/
	@# Runs entirely offline (no Docker). Produces results/sweep/sweep.parquet
	@# and results/sweep/sweep_pareto.{png,svg} so the README reference resolves.
	@# Uses smoke-out/encoded — the bundle produced by scripts/smoke_pipeline.py.
	$(UV) run vdbbench sweep \
		--adapter exact \
		--grid metric=cosine,l2 \
		--grid k_neighbors=4,8,16 \
		--encoded smoke-out/encoded \
		--out results/sweep

# ---------------------------------------------------------------------------
# Analysis notebook
# ---------------------------------------------------------------------------
.PHONY: analysis
analysis: ## Execute analysis.ipynb and regenerate assets/analysis-*.png
	$(UV) run jupyter nbconvert --to notebook --execute analysis.ipynb \
		--output analysis.executed.ipynb
	@echo "Executed notebook written to analysis.executed.ipynb"
	@echo "Charts written to assets/analysis-*.png"

.PHONY: profile-demo
profile-demo: ## Flame-graph run: exact adapter, smoke encoded data → results/profile-demo/ (requires py-spy)
	@# Runs entirely offline (no Docker). Produces results/profile-demo/profile.svg
	@# alongside summary.parquet and timings.parquet. Requires py-spy:
	@#   pip install "vdbbench[profile]"  or  uv sync --extra profile
	@# Uses the smoke-out/encoded bundle produced by scripts/smoke_pipeline.py.
	@# If smoke-out/encoded does not exist, run `make smoke` first.
	$(UV) run vdbbench bench \
		--adapter exact \
		--profiler py-spy \
		--encoded smoke-out/encoded \
		--out results/profile-demo

.PHONY: profile-demo-mem
profile-demo-mem: ## ps_mem memory-breakdown run: exact adapter, smoke encoded data → results/profile-demo-mem/ (requires ps_mem)
	@# Runs entirely offline (no Docker). Produces results/profile-demo-mem/ps_mem.log
	@# with timestamped per-process memory snapshots (private/shared/swap breakdown)
	@# alongside summary.parquet and timings.parquet. Requires the ps_mem system binary:
	@#   macOS:  brew install ps_mem
	@#   Linux:  sudo apt install ps_mem
	@# Uses the smoke-out/encoded bundle produced by scripts/smoke_pipeline.py.
	@# If smoke-out/encoded does not exist, run `make smoke` first.
	$(UV) run vdbbench bench \
		--adapter exact \
		--profiler ps_mem \
		--encoded smoke-out/encoded \
		--out results/profile-demo-mem

# ---------------------------------------------------------------------------
# Encoder comparison
# ---------------------------------------------------------------------------
# ENCODERS selects which embedding models to compare.
# Default: bge only (nomic requires ~500 MB model download + trust_remote_code).
# To add nomic: make compare-encoders ENCODERS=bge,nomic
ENCODERS ?= bge

.PHONY: compare-encoders
compare-encoders: ## Side-by-side encoder comparison (bge vs nomic). Default ENCODERS=bge. No Docker needed.
	@# Runs entirely offline (no Docker) with ENCODERS=bge (the default).
	@# The smoke artifact at results/encoder-compare/ uses bge only because
	@# nomic requires a ~500 MB model download and trust_remote_code=True.
	@# To run both encoders: make compare-encoders ENCODERS=bge,nomic
	@# (reads the model card at https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
	@#  before enabling nomic in a security-sensitive environment).
	$(UV) run vdbbench compare-encoders \
		--adapter exact \
		--encoders $(ENCODERS) \
		--dataset synthetic \
		--out results/encoder-compare

.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Docker / Reproducibility
# ---------------------------------------------------------------------------
.PHONY: docker-build
docker-build: ## Build the vdbbench Docker image (multi-stage, production deps only)
	docker build --tag vdbbench:latest .

.PHONY: docker-run
docker-run: ## Bring up the full stack (pgvector + qdrant + vdbbench) and run the bench
	@# Requires the encoded bundle at data/encoded-demo. Run `make bench-demo` or
	@# `uv run vdbbench prep ...` first to produce it, then re-run this target.
	docker compose -f docker-compose.full.yml up --build

.PHONY: reproduce
reproduce: ## Build image + run full stack + diff results against results/demo/ baseline
	./scripts/reproduce.sh

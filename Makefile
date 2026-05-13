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

# Container ports — overridable so collisions with an existing local
# Postgres / Qdrant don't force the user to edit docker-compose.yml.
# `make up PGVECTOR_PORT=5444 QDRANT_PORT=6343` is the documented escape hatch.
PGVECTOR_PORT ?= 5433
QDRANT_PORT ?= 6333
export PGVECTOR_PORT
export QDRANT_PORT

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
bench-demo: ## Demo pipeline reproducing the README numbers: synthetic 5k vectors @ dim=64, writes results/demo/
	$(UV) run vdbbench prep  --out data/encoded-demo --dataset synthetic --sample-size 5000 --dim 64
	$(UV) run vdbbench bench --encoded data/encoded-demo --out results/demo --all \
		--pgvector-dsn $(PGVECTOR_DSN) --qdrant-url $(QDRANT_URL)
	$(UV) run vdbbench plot  --summary results/demo/summary.parquet --out assets

.PHONY: bench-100k
bench-100k: prep bench plots ## 100k-scale sweep (uses SAMPLE_SIZE/EMBED_MODEL); writes results/100k/

.PHONY: bench-1m
bench-1m: ## Full 1M MS-MARCO sweep (results/full/summary.parquet; ~hours, no commit)
	$(UV) run vdbbench prep  --dataset msmarco --sample-size 1000000 --out data/encoded-1m
	$(UV) run vdbbench bench --encoded data/encoded-1m --out results/full --all --profile p99 \
		--pgvector-dsn $(PGVECTOR_DSN) --qdrant-url $(QDRANT_URL)
	$(UV) run vdbbench plot  --summary results/full/summary.parquet --out assets/full

.PHONY: plots
plots: ## Regenerate analysis plots from results/100k/summary.parquet
	$(UV) run vdbbench plot --summary results/100k/summary.parquet --out assets/100k

# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------
.PHONY: up-pull
up-pull: ## Pre-pull pgvector + qdrant images so `up --wait` doesn't time out on first run
	docker compose pull

.PHONY: up-wait
up-wait: ## Start containers and wait for healthy (assumes images already pulled)
	@docker compose up -d --wait || { \
		echo ""; \
		echo "[make up] docker compose failed."; \
		echo "  If a host port is already allocated, override the published port:"; \
		echo "    make up PGVECTOR_PORT=5444 QDRANT_PORT=6343"; \
		echo "  Current published ports: pgvector=$(PGVECTOR_PORT), qdrant=$(QDRANT_PORT)."; \
		exit 1; \
	}

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
.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON ?= python3
UV ?= uv
SAMPLE_SIZE ?= 100000
EMBED_MODEL ?= BAAI/bge-small-en-v1.5

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
bench: ## Run benchmark across all DBs
	$(UV) run vdbbench bench --all

.PHONY: bench-all
bench-all: prep bench plots ## Full pipeline: prep + bench + plots

.PHONY: plots
plots: ## Regenerate analysis plots from results/raw.parquet
	$(UV) run vdbbench plot

# ---------------------------------------------------------------------------
# Containers — wired in Phase 2 alongside the docker-compose.yml.
# ---------------------------------------------------------------------------
# .PHONY: up down
# up:    docker compose up -d        # pgvector, qdrant
# down:  docker compose down -v

# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------
.PHONY: clean
clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist
	find . -name __pycache__ -type d -exec rm -rf {} +

PYTHON ?= python
DIST_DIR ?= dist
PY_TARGETS := dlhub tracks ml_algorithms/python optimization/python tests scripts .github/scripts

.PHONY: help test coverage lint format contract lesson-entrypoints stats zoo-integrity fidelity narrative evidence verify check
.PHONY: doctor smoke docs package package-smoke release-check

help:
	@echo "DL-Hub developer commands"
	@echo "  make verify        Fast repository checks (includes offline Zoo imports)"
	@echo "  make lesson-entrypoints Run all 339 lesson --help entrypoints (~4-7 min)"
	@echo "  make stats         Check generated repository statistics"
	@echo "  make evidence      Check lesson data/benchmark evidence profiles"
	@echo "  make zoo-integrity Deterministic offline Model Zoo registry checks"
	@echo "  make check         Repository checks plus the full pytest suite"
	@echo "  make smoke         Curated offline lesson smoke suite"
	@echo "  make docs          Strict MkDocs build"
	@echo "  make package       Build and validate sdist/wheel metadata"
	@echo "  make package-smoke Validate wheel and sdist in isolated temporary venvs"
	@echo "  make release-check Full tests, docs, package, and install validation"

test:
	$(PYTHON) -m pytest

coverage:
	$(PYTHON) -m pytest --cov=dlhub --cov=ml_algorithms --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check $(PY_TARGETS)

format:
	$(PYTHON) -m isort $(PY_TARGETS)
	$(PYTHON) -m black $(PY_TARGETS)
	$(PYTHON) -m ruff check --fix $(PY_TARGETS)

contract:
	$(PYTHON) scripts/lesson_contracts.py --check

lesson-entrypoints:
	$(PYTHON) scripts/lesson_entrypoint_check.py --check

stats:
	$(PYTHON) scripts/project_stats.py --check

zoo-integrity:
	$(PYTHON) scripts/zoo_integrity.py --check

fidelity:
	$(PYTHON) scripts/model_fidelity.py --check

narrative:
	$(PYTHON) scripts/narrative_check.py

evidence:
	$(PYTHON) scripts/benchmark_profiles.py --check

verify: lint contract stats zoo-integrity fidelity narrative evidence

check: verify test

smoke:
	$(PYTHON) scripts/smoke_check.py

doctor:
	$(PYTHON) scripts/doctor.py

docs:
	$(PYTHON) -m mkdocs build --strict

package:
	$(PYTHON) .github/scripts/package_gate.py clean --dist-dir "$(DIST_DIR)"
	$(PYTHON) -m build --outdir "$(DIST_DIR)"
	$(PYTHON) .github/scripts/package_gate.py check --dist-dir "$(DIST_DIR)"

package-smoke: package
	$(PYTHON) .github/scripts/package_gate.py smoke --dist-dir "$(DIST_DIR)"

release-check: check docs package-smoke

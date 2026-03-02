PY_TARGETS = dlhub tracks ml_algorithms/python optimization/python tests scripts

.PHONY: test lint format check
.PHONY: doctor smoke

test:
	pytest

lint:
	ruff check $(PY_TARGETS)

format:
	isort $(PY_TARGETS)
	black $(PY_TARGETS)
	ruff check --fix $(PY_TARGETS)

check: lint test

smoke:
	python scripts/smoke_check.py

doctor:
	python scripts/doctor.py

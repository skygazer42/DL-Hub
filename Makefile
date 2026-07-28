PY_TARGETS = dlhub tracks ml_algorithms/python optimization/python tests scripts

.PHONY: test lint format contract fidelity narrative verify check
.PHONY: doctor smoke

test:
	pytest

lint:
	ruff check $(PY_TARGETS)

format:
	isort $(PY_TARGETS)
	black $(PY_TARGETS)
	ruff check --fix $(PY_TARGETS)

contract:
	python scripts/lesson_contracts.py --check

fidelity:
	python scripts/model_fidelity.py --check

narrative:
	python scripts/narrative_check.py

verify: lint contract fidelity narrative

check: verify test

smoke:
	python scripts/smoke_check.py

doctor:
	python scripts/doctor.py

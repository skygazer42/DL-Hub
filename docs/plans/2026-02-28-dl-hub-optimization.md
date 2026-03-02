# DL-Hub Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make this repo easier to navigate, run, and maintain by adding a lightweight engineering baseline (docs + tooling + tests) without breaking existing learning materials.

**Architecture:** Treat `DL-Hub` as a "multi-project learning hub" instead of a single app. Keep existing folders intact (no renames), add repo-level documentation and developer tooling, and focus code-quality investment on the most reusable Python modules (`ml_algorithms/python` and `optimization/python`).

**Tech Stack:** Python 3.10+, NumPy, PyTorch (for some demos), `pytest`, `ruff`, `black`, `isort`, `pre-commit`, GitHub Actions (optional but recommended).

---

## Guiding principles

- **Non-destructive:** avoid renaming large directories or removing content.
- **Focused correctness:** tests target the reusable NumPy modules first.
- **Fast feedback loops:** `make test` / `make lint` should work from repo root.
- **Documentation-first:** a new contributor should know where to start in <5 minutes.

## Scope (what we will optimize)

- **Docs & navigation:** root `README.md` and new `docs/` entrypoints.
- **Python dev baseline:** lint/format/test configs scoped to our reusable modules.
- **Import ergonomics:** enable `ml_algorithms.python.*` and `optimization.python.*` imports.
- **CI (optional):** run lint + tests automatically on PRs.

## Out of scope (explicitly not doing)

- Renaming top-level folders with spaces (risky for existing links).
- Refactoring or "standardizing" every deep-learning subproject (too large).
- Packaging to PyPI (this repo is a learning collection; keep it local-first).

---

## Tasks (20)

### Task 1: Add a root `.gitignore`

**Files:**
- Create: `.gitignore`

**Steps:**
1. Add common ignores: Python caches, virtualenvs, Jupyter, `.data/`, model checkpoints, plots.
2. Verify: `git status --porcelain` stays clean after running demos/tests.

---

### Task 2: Add repo structure documentation

**Files:**
- Create: `docs/STRUCTURE.md`

**Steps:**
1. Describe top-level directories and what each is for.
2. Add "Where to start" section and links into key subfolders.

---

### Task 3: Add "how to run" documentation

**Files:**
- Create: `docs/RUNNING.md`

**Steps:**
1. Provide a short Python env recommendation (venv / conda).
2. Add a "quick smoke test" section (NumPy modules + LeNet example).

---

### Task 4: Fix/refresh the root `README.md` navigation

**Files:**
- Modify: `README.md`

**Steps:**
1. Ensure every listed top-level item exists (remove or mark TODO when missing).
2. Add links to `docs/STRUCTURE.md` and `docs/RUNNING.md`.
3. Add a small "Quick Start" snippet for the most reliable runnable example.

---

### Task 5: Fix Markdown code fences in `graph/gin/README.md`

**Files:**
- Modify: `graph/gin/README.md`

**Steps:**
1. Fix broken fenced-code blocks (backticks).
2. Keep content unchanged aside from formatting.

---

### Task 6: Add `pyproject.toml` with ruff/black/isort config

**Files:**
- Create: `pyproject.toml`

**Steps:**
1. Configure `ruff` for Python 3.10 and a small, practical rule set.
2. Configure `black`/`isort` for consistent formatting.
3. Exclude large non-target directories from tooling by default.

---

### Task 7: Add pytest configuration

**Files:**
- Create: `pytest.ini`

**Steps:**
1. Restrict test discovery to `tests/`.
2. Add a couple of useful defaults (`-q`, strict markers optional).

---

### Task 8: Make `ml_algorithms` importable

**Files:**
- Create: `ml_algorithms/__init__.py`
- Create: `ml_algorithms/python/__init__.py`

**Steps:**
1. Add minimal package markers, no behavior change.
2. Verify imports work: `python -c "from ml_algorithms.python.linear_models import LogisticRegression"`

---

### Task 9: Make `optimization` importable

**Files:**
- Create: `optimization/__init__.py`
- Create: `optimization/python/__init__.py`

**Steps:**
1. Add minimal package markers.
2. Verify imports work: `python -c "from optimization.python.optimizers import Adam"`

---

### Task 10: Update ML algorithms README imports

**Files:**
- Modify: `ml_algorithms/python/README.md`

**Steps:**
1. Update examples to use package imports (`ml_algorithms.python.*`).
2. Add a note about running from repo root.

---

### Task 11: Update optimization README imports

**Files:**
- Modify: `optimization/python/README.md`

**Steps:**
1. Update examples to use package imports (`optimization.python.*`).
2. Add a note about running from repo root.

---

### Task 12: Add unit tests for linear models

**Files:**
- Create: `tests/test_linear_models.py`

**Steps:**
1. Test `LinearRegression` fits a simple linear relationship.
2. Test `LogisticRegression` separates a simple linearly separable dataset.
3. Run: `pytest -q` (expect PASS).

---

### Task 13: Add unit tests for clustering + PCA

**Files:**
- Create: `tests/test_unsupervised.py`

**Steps:**
1. Test `KMeans` recovers obvious clusters.
2. Test `PCA` output shape and variance ordering basic sanity.

---

### Task 14: Add unit tests for Naive Bayes + kNN

**Files:**
- Create: `tests/test_classical_ml.py`

**Steps:**
1. Test Gaussian Naive Bayes on synthetic data.
2. Test kNN predicts nearest labels on a small dataset.

---

### Task 15: Add unit tests for optimization utilities

**Files:**
- Create: `tests/test_optimization.py`

**Steps:**
1. Test optimizers update params in expected direction.
2. Test LR schedulers are deterministic and bounded.
3. Test losses/metrics return expected values for simple cases.

---

### Task 16: Add a `Makefile` for common commands

**Files:**
- Create: `Makefile`

**Steps:**
1. `make test`: run pytest.
2. `make lint`: ruff check.
3. `make format`: black + isort + ruff format (if enabled).

---

### Task 17: Add `requirements-dev.txt`

**Files:**
- Create: `requirements-dev.txt`

**Steps:**
1. Pin minimal developer tools (`pytest`, `ruff`, `black`, `isort`, `pre-commit`).
2. Keep runtime deps out of it.

---

### Task 18: Add pre-commit hooks

**Files:**
- Create: `.pre-commit-config.yaml`

**Steps:**
1. Add `ruff`, `black`, `isort` hooks.
2. Document install: `pre-commit install`.

---

### Task 19: Add GitHub Actions CI (lint + tests)

**Files:**
- Create: `.github/workflows/python-ci.yml`

**Steps:**
1. Run on PR/push to `main`.
2. Execute `ruff check` and `pytest`.

---

### Task 20: Add a lightweight "smoke check" script

**Files:**
- Create: `scripts/smoke_check.py`

**Steps:**
1. Import the core modules and run a tiny fit/predict for sanity.
2. Document usage in `docs/RUNNING.md`.


# Code Style And Algorithm Expansion Implementation Plan

**Goal:** Normalize the repository to its declared formatting/linting standards, then add a small, well-tested batch of new NumPy ML algorithms.

**Architecture:** First align the existing codebase with the repo's declared `black`/`isort`/`ruff` standards using automated formatting plus targeted lint fixes. Then expand `ml_algorithms/python` with deterministic, lightweight algorithms that fit the current educational scope and are easy to regression test.

**Tech Stack:** Python 3.10, NumPy, PyTorch, pytest, black, isort, ruff

---

### Task 1: Audit and baseline

**Files:**
- Modify: `docs/plans/2026-03-10-code-style-and-algorithm-expansion.md`
- Check: `pyproject.toml`
- Check: `Makefile`

**Step 1: Capture baseline formatter and linter failures**

Run:

```bash
black --check dlhub tracks ml_algorithms/python optimization/python tests scripts
isort --check-only dlhub tracks ml_algorithms/python optimization/python tests scripts
ruff check dlhub tracks ml_algorithms/python optimization/python tests scripts
```

**Step 2: Record the main drift categories**

Expect:
- repo-wide `black` drift
- repo-wide `isort` drift
- remaining `ruff` findings after auto-fixable issues

### Task 2: Normalize repository style

**Files:**
- Modify: `dlhub/**/*.py`
- Modify: `tracks/**/*.py`
- Modify: `ml_algorithms/python/*.py`
- Modify: `optimization/python/*.py`
- Modify: `tests/*.py`
- Modify: `scripts/*.py`

**Step 1: Apply automated formatting**

Run:

```bash
isort dlhub tracks ml_algorithms/python optimization/python tests scripts
black dlhub tracks ml_algorithms/python optimization/python tests scripts
ruff check --fix dlhub tracks ml_algorithms/python optimization/python tests scripts
```

**Step 2: Fix non-auto-fixable lint issues**

Examples to handle:
- undefined type-hint names caused by eager annotation evaluation
- unused imports/variables left by generator output
- ambiguous variable names

**Step 3: Re-run formatter and linter commands**

Run until clean:

```bash
black --check dlhub tracks ml_algorithms/python optimization/python tests scripts
isort --check-only dlhub tracks ml_algorithms/python optimization/python tests scripts
ruff check dlhub tracks ml_algorithms/python optimization/python tests scripts
```

### Task 3: Add failing tests for new algorithms

**Files:**
- Modify: `tests/test_linear_models.py`
- Modify: `tests/test_unsupervised.py`
- Modify: `tests/test_classical_ml.py`

**Step 1: Add one failing regression/classification test**

Candidate behaviors:
- `RidgeRegression` fits a noisy linear system with low MSE
- `SoftmaxRegression` separates a simple 3-class dataset
- `AgglomerativeClustering` recovers obvious clusters

**Step 2: Run targeted tests and verify RED**

Run:

```bash
pytest -q tests/test_linear_models.py tests/test_unsupervised.py tests/test_classical_ml.py
```

Expected:
- failures because new classes do not yet exist

### Task 4: Implement algorithms minimally

**Files:**
- Modify: `ml_algorithms/python/linear_models.py`
- Modify: `ml_algorithms/python/clustering.py`
- Modify: `ml_algorithms/python/README.md`
- Modify: `README.md`

**Step 1: Implement the smallest working versions**

Add:
- `RidgeRegression`
- `SoftmaxRegression`
- `AgglomerativeClustering`

**Step 2: Keep APIs consistent with existing classes**

Conventions:
- dataclass configuration
- `fit(...) -> self`
- learned attributes like `weights_`, `labels_`, `cluster_centers_` when applicable
- deterministic behavior where practical

**Step 3: Re-run the targeted tests and verify GREEN**

Run:

```bash
pytest -q tests/test_linear_models.py tests/test_unsupervised.py tests/test_classical_ml.py
```

### Task 5: Final verification

**Files:**
- Verify: `README.md`
- Verify: `ml_algorithms/python/README.md`
- Verify: touched Python files

**Step 1: Re-run style gates**

```bash
black --check dlhub tracks ml_algorithms/python optimization/python tests scripts
isort --check-only dlhub tracks ml_algorithms/python optimization/python tests scripts
ruff check dlhub tracks ml_algorithms/python optimization/python tests scripts
```

**Step 2: Re-run focused tests**

```bash
pytest -q tests/test_linear_models.py tests/test_unsupervised.py tests/test_classical_ml.py tests/test_import_regressions.py
```

**Step 3: Re-run repo smoke**

```bash
python scripts/smoke_check.py
```

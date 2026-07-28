# More ML Algorithms Pack Implementation Plan

**Goal:** Add 8 new classic ML algorithms implemented in NumPy under `ml_algorithms/python/`, with fast, deterministic pytest coverage and a low-conflict “8 branches -> 1 integration branch” workflow.

**Architecture:** Each algorithm lives in its own module file (one file per branch) and follows existing repo conventions (`dataclass`, `fit(...) -> self`, `np.float64`). A single integration-branch test file validates module discoverability + basic correctness on synthetic data.

**Tech Stack:** Python 3.10+ style, NumPy, pytest.

---

## Branch Strategy (8 feature branches)

- Integration branch: `feat/more-ml-algorithms-pack`
- Feature branches (one module file each):
  - `feat/alg-lasso`
  - `feat/alg-elastic-net`
  - `feat/alg-kernel-ridge`
  - `feat/alg-gaussian-process`
  - `feat/alg-kernel-pca`
  - `feat/alg-mds`
  - `feat/alg-lle`
  - `feat/alg-kde`

Rules:

- Feature branches: only create the algorithm module file + its minimal self-check (if desired). No README/test edits (avoid conflicts).
- Integration branch: add tests + README update + full verification.

---

### Task 1: Add a failing integration-branch test (TDD RED)

**Files:**
- Create: `tests/test_more_ml_algorithms_pack.py`

**Step 1: Write tests that fail without raising ImportError**

Use `importlib.util.find_spec(...)` to assert module presence first, so missing modules fail as test failures (not errors).

**Step 2: Verify RED**

Run: `pytest -q tests/test_more_ml_algorithms_pack.py`  
Expected: FAIL (assertion that module spec is not None).

**Step 3: Commit**

```bash
git add tests/test_more_ml_algorithms_pack.py
git commit -m "test: add more-ml-algorithms pack smokes (failing)"
```

---

### Task 2: Implement Lasso Regression (branch `feat/alg-lasso`)

**Files:**
- Create: `ml_algorithms/python/lasso.py`

**Steps:**
1. `git switch -c feat/alg-lasso`
2. Implement `LassoRegression` (coordinate descent + soft threshold) with `fit/predict`.
3. Run: `pytest -q tests/test_more_ml_algorithms_pack.py -k lasso`
4. Commit: `git commit -am "feat: add Lasso regression (NumPy)"`
5. Merge into integration branch:
   - `git switch feat/more-ml-algorithms-pack`
   - `git merge --no-ff feat/alg-lasso`

---

### Task 3: Implement Elastic Net Regression (branch `feat/alg-elastic-net`)

**Files:**
- Create: `ml_algorithms/python/elastic_net.py`

Same steps as Task 2; implement `ElasticNetRegression`.

---

### Task 4: Implement Kernel Ridge Regression (branch `feat/alg-kernel-ridge`)

**Files:**
- Create: `ml_algorithms/python/kernel_ridge.py`

Implement `KernelRidgeRegression` with `kernel="linear"|"rbf"`, `gamma` for RBF, and `alpha` regularization.

---

### Task 5: Implement Gaussian Process Regressor (branch `feat/alg-gaussian-process`)

**Files:**
- Create: `ml_algorithms/python/gaussian_process.py`

Implement `GaussianProcessRegressor` with RBF kernel, Cholesky solve, and `predict(return_std=True)`.

---

### Task 6: Implement Kernel PCA (branch `feat/alg-kernel-pca`)

**Files:**
- Create: `ml_algorithms/python/kernel_pca.py`

Implement `KernelPCA` with `fit/transform/fit_transform`. Support linear and RBF kernels, and proper kernel centering.

---

### Task 7: Implement Metric MDS (branch `feat/alg-mds`)

**Files:**
- Create: `ml_algorithms/python/mds.py`

Implement classic MDS: pairwise distances -> double-centering -> eigendecomposition -> embedding.

---

### Task 8: Implement Locally Linear Embedding (branch `feat/alg-lle`)

**Files:**
- Create: `ml_algorithms/python/lle.py`

Implement LLE with kNN neighbors, local weight solving (with regularization), and embedding via smallest non-trivial eigenvectors.

---

### Task 9: Implement Gaussian KDE (branch `feat/alg-kde`)

**Files:**
- Create: `ml_algorithms/python/kde.py`

Implement `GaussianKDE` with `fit` and `score_samples` (log density) + `pdf`.

---

### Task 10: Update algorithm list documentation (integration branch only)

**Files:**
- Modify: `ml_algorithms/python/README.md`

Add the 8 new algorithms to the list.

Commit: `git commit -am "docs: list new NumPy ML algorithms"`

---

### Task 11: Final verification (integration branch)

Run:

- `pytest -q tests/test_more_ml_algorithms_pack.py`
- `pytest -q`

Expected: PASS.

---

### Task 12: Finish branch

Merge integration branch to `main` (or open a PR), then clean up feature branches.

# Unsupervised Algorithms Expansion (Design)

**Date:** 2026-03-10  
**Repo:** DL-Hub (`ml_algorithms/python`)  

## Goal

Add a small batch of **NumPy-only** unsupervised algorithms that:
- match the existing educational scope (`@dataclass`, `fit(...) -> self`, deterministic tests),
- run fast in CI (small datasets in tests),
- integrate cleanly with existing algorithms (e.g. reuse `KMeans` where appropriate).

## Candidate Batches (2–3 algorithms)

### Option 1 (Recommended): “Clustering + Factorization + ICA”

- **Spectral Clustering** (`SpectralClustering`): graph Laplacian embedding + `KMeans`.
- **NMF** (`NMF`): non-negative matrix factorization (multiplicative updates).
- **FastICA** (`FastICA`): independent component analysis via fixed-point iterations.

**Why:** broad coverage across unsupervised families, NumPy-friendly math, deterministic tests feasible.

### Option 2: “Manifold / embedding heavy”

- `TSNE` + `Isomap` (+ maybe `MDS`).

**Trade-off:** more brittle/slow optimization in tests; harder to keep CI stable.

### Option 3: “Anomaly detection”

- `IsolationForest` / `LOF` style algorithms.

**Trade-off:** requires more design for scoring APIs and robust evaluation metrics.

## Proposed APIs (Option 1)

### SpectralClustering

- `@dataclass` fields: `n_clusters: int = 2`, `gamma: float = 1.0`, `random_state: int | None = None`
- `fit(x) -> self` sets `labels_`
- Steps: RBF affinity → normalized Laplacian → eigenvectors → row-normalize embedding → `KMeans`

### NMF

- `@dataclass` fields: `n_components`, `max_iter`, `tol`, `random_state`
- `fit(x) -> self`, `fit_transform(x)`, `transform(x)`, `inverse_transform(w)`
- Learned attrs: `components_` (H), `basis_` (W), `reconstruction_err_`

### FastICA

- `@dataclass` fields: `n_components`, `max_iter`, `tol`, `random_state`
- `fit(x) -> self`, `transform(x)`, `fit_transform(x)`
- Learned attrs: `components_` (unmixing), `mixing_`, `mean_`, `whitening_`
- Note: handle sign/permutation ambiguity in tests via best-correlation matching.

## Testing Strategy

- Each algorithm gets its own dedicated test module under `tests/`.
- Tests must be deterministic:
  - fixed `np.random.default_rng(seed)` datasets,
  - convergence-based assertions with comfortable margins.
- Keep datasets small (≤ ~200 samples) to keep runtime low.


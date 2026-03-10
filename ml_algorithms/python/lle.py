"""Locally Linear Embedding (LLE) manifold learning in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pairwise_squared_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq_norms = np.sum(x**2, axis=1, keepdims=True)
    sq = sq_norms + sq_norms.T - 2.0 * (x @ x.T)
    return np.maximum(sq, 0.0)


@dataclass
class LocallyLinearEmbedding:
    n_neighbors: int = 10
    n_components: int = 2
    reg: float = 1e-3

    def fit(self, x: np.ndarray) -> LocallyLinearEmbedding:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")

        n_samples = int(x.shape[0])
        if n_samples < 2:
            raise ValueError("x must contain at least 2 samples")

        k = int(self.n_neighbors)
        if k < 1:
            raise ValueError("n_neighbors must be >= 1")
        if k >= n_samples:
            raise ValueError("n_neighbors must be < n_samples")

        n_components = int(self.n_components)
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if n_components >= n_samples:
            raise ValueError("n_components must be < n_samples")

        reg = float(self.reg)
        if reg <= 0.0:
            raise ValueError("reg must be > 0")

        sq = _pairwise_squared_distances(x)
        np.fill_diagonal(sq, np.inf)
        neighbor_idx = np.argpartition(sq, kth=k - 1, axis=1)[:, :k]

        w = np.zeros((n_samples, n_samples), dtype=np.float64)
        ones = np.ones((k,), dtype=np.float64)

        for i in range(n_samples):
            nbrs = neighbor_idx[i]
            z = x[nbrs] - x[i]
            c = z @ z.T  # (k,k)
            trace = float(np.trace(c))
            if trace <= 0.0:
                trace = 1.0
            c = c + (reg * trace) * np.eye(k, dtype=np.float64)
            weights = np.linalg.solve(c, ones)
            weights = weights / float(weights.sum())
            w[i, nbrs] = weights

        i = np.eye(n_samples, dtype=np.float64)
        m = (i - w).T @ (i - w)

        eigenvalues, eigenvectors = np.linalg.eigh(m)
        order = np.argsort(eigenvalues)  # ascending

        self.embedding_ = eigenvectors[:, order[1 : n_components + 1]]
        self.eigenvalues_ = eigenvalues[order[: n_components + 1]]
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.embedding_


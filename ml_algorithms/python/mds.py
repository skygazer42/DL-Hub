"""Metric multidimensional scaling (classic MDS) in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pairwise_euclidean_distances(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq_norms = np.sum(x**2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * (x @ x.T)
    sq_dists = np.maximum(sq_dists, 0.0)
    return np.sqrt(sq_dists)


@dataclass
class MDS:
    n_components: int = 2

    def fit(self, x: np.ndarray) -> MDS:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")

        n_samples = int(x.shape[0])
        if n_samples < 2:
            raise ValueError("x must contain at least 2 samples")

        n_components = int(self.n_components)
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if n_components > n_samples:
            raise ValueError("n_components must be <= n_samples")

        d = _pairwise_euclidean_distances(x)
        d2 = d**2

        row_mean = d2.mean(axis=1, keepdims=True)
        col_mean = d2.mean(axis=0, keepdims=True)
        total_mean = float(d2.mean())
        b = -0.5 * (d2 - row_mean - col_mean + total_mean)

        eigenvalues, eigenvectors = np.linalg.eigh(b)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        top = eigenvalues[:n_components]
        top = np.maximum(top, 0.0)
        self.eigenvalues_ = top
        self.embedding_ = eigenvectors[:, :n_components] * np.sqrt(top[None, :])
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.embedding_


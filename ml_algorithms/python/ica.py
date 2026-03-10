"""FastICA (symmetric) in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sym_decorrelation(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    s = w @ w.T
    eigenvalues, eigenvectors = np.linalg.eigh(s)
    eigenvalues = np.clip(eigenvalues, 1e-12, None)
    inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return inv_sqrt @ w


@dataclass
class FastICA:
    n_components: int = 2
    max_iter: int = 200
    tol: float = 1e-4
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> FastICA:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        n_samples, n_features = x.shape

        if n_samples < 2:
            raise ValueError("x must contain at least 2 samples")
        if int(self.n_components) < 1:
            raise ValueError("n_components must be >= 1")
        if int(self.n_components) > n_features:
            raise ValueError("n_components must be <= n_features")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be >= 1")
        if float(self.tol) < 0.0:
            raise ValueError("tol must be >= 0")

        n_components = int(self.n_components)

        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_

        cov = (centered.T @ centered) / float(n_samples)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order][:n_components]
        eigenvectors = eigenvectors[:, order][:, :n_components]

        if np.any(eigenvalues <= 1e-12):
            raise ValueError("Data appears rank-deficient for the requested n_components.")

        # Fix eigenvector sign for deterministic whitening.
        for j in range(n_components):
            col = eigenvectors[:, j]
            idx = int(np.argmax(np.abs(col)))
            if col[idx] < 0.0:
                eigenvectors[:, j] = -col

        whitening = (eigenvectors / np.sqrt(eigenvalues)[None, :]).T
        z = centered @ whitening.T

        rng = np.random.default_rng(self.random_state)
        w = rng.normal(size=(n_components, n_components))
        w = _sym_decorrelation(w)

        for _ in range(int(self.max_iter)):
            y = z @ w.T
            g = np.tanh(y)
            g_prime = 1.0 - g**2

            w_new = (g.T @ z) / float(n_samples) - np.diag(g_prime.mean(axis=0)) @ w
            w_new = _sym_decorrelation(w_new)

            lim = np.max(np.abs(np.abs(np.diag(w_new @ w.T)) - 1.0))
            w = w_new
            if lim < float(self.tol):
                break

        self.components_ = w @ whitening
        self.mixing_ = np.linalg.pinv(self.components_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.shape[1] != self.mean_.shape[0]:
            raise ValueError("x must have the same number of features as the data used in fit()")
        centered = x - self.mean_
        return centered @ self.components_.T

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

"""Kernel PCA in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pairwise_squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a2 = np.sum(a**2, axis=1, keepdims=True)
    b2 = np.sum(b**2, axis=1, keepdims=True).T
    sq = a2 + b2 - 2.0 * (a @ b.T)
    return np.maximum(sq, 0.0)


def _kernel(x1: np.ndarray, x2: np.ndarray, *, kind: str, gamma: float | None) -> np.ndarray:
    kind = str(kind).lower().strip()
    if kind == "linear":
        return np.asarray(x1, dtype=np.float64) @ np.asarray(x2, dtype=np.float64).T
    if kind == "rbf":
        g = 1.0 if gamma is None else float(gamma)
        sq = _pairwise_squared_distances(x1, x2)
        return np.exp(-g * sq)
    raise ValueError("kernel must be one of: 'linear', 'rbf'")


@dataclass
class KernelPCA:
    n_components: int = 2
    kernel: str = "rbf"
    gamma: float | None = None

    def fit(self, x: np.ndarray) -> KernelPCA:
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

        self.x_fit_ = x

        k = _kernel(x, x, kind=self.kernel, gamma=self.gamma)
        self.k_fit_mean_ = float(k.mean())
        self.k_fit_col_mean_ = k.mean(axis=0)

        # Center training kernel: Kc = K - 1K - K1 + 1K1
        k_row_mean = k.mean(axis=1, keepdims=True)
        k_col_mean = self.k_fit_col_mean_[None, :]
        k_centered = k - k_row_mean - k_col_mean + self.k_fit_mean_

        eigenvalues, eigenvectors = np.linalg.eigh(k_centered)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        top_vals = eigenvalues[:n_components]
        top_vals = np.maximum(top_vals, 0.0)
        self.lambdas_ = top_vals
        self.alphas_ = eigenvectors[:, :n_components]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        k = _kernel(x, self.x_fit_, kind=self.kernel, gamma=self.gamma)
        k_row_mean = k.mean(axis=1, keepdims=True)
        k_centered = k - k_row_mean - self.k_fit_col_mean_[None, :] + self.k_fit_mean_

        denom = np.sqrt(self.lambdas_)
        denom = np.where(denom > 0.0, denom, 1.0)
        return k_centered @ (self.alphas_ / denom[None, :])

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


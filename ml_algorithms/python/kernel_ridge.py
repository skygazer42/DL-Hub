"""Kernel ridge regression in NumPy (dual form)."""

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
class KernelRidgeRegression:
    alpha: float = 1.0
    kernel: str = "rbf"
    gamma: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> KernelRidgeRegression:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")

        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        alpha = float(self.alpha)
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")

        self.x_train_ = x
        k = _kernel(x, x, kind=self.kernel, gamma=self.gamma)
        k = k + alpha * np.eye(k.shape[0], dtype=np.float64)
        self.dual_coef_ = np.linalg.solve(k, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        k = _kernel(x, self.x_train_, kind=self.kernel, gamma=self.gamma)
        return k @ self.dual_coef_


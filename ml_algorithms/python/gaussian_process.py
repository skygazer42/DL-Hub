"""Gaussian Process regression (RBF kernel) in NumPy."""

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


def _rbf_kernel(x1: np.ndarray, x2: np.ndarray, *, length_scale: float, sigma_f: float) -> np.ndarray:
    ls = float(length_scale)
    if ls <= 0.0:
        raise ValueError("length_scale must be > 0")
    sf = float(sigma_f)
    if sf <= 0.0:
        raise ValueError("sigma_f must be > 0")
    sq = _pairwise_squared_distances(x1 / ls, x2 / ls)
    return (sf**2) * np.exp(-0.5 * sq)


@dataclass
class GaussianProcessRegressor:
    """Gaussian Process regression with an RBF kernel.

    Notes:
    - Uses Cholesky factorization for stability.
    - Provides predictive mean and (optionally) standard deviation.
    """

    length_scale: float = 1.0
    sigma_f: float = 1.0
    noise: float = 1e-6
    jitter: float = 1e-9

    def fit(self, x: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")

        y = np.asarray(y, dtype=np.float64).ravel()
        if y.shape[0] != x.shape[0]:
            raise ValueError("x and y must have the same number of samples")

        noise = float(self.noise)
        if noise < 0.0:
            raise ValueError("noise must be >= 0")

        jitter = float(self.jitter)
        if jitter <= 0.0:
            raise ValueError("jitter must be > 0")

        self.x_train_ = x
        self.y_train_ = y

        k = _rbf_kernel(
            x,
            x,
            length_scale=float(self.length_scale),
            sigma_f=float(self.sigma_f),
        )
        k = k + (noise + jitter) * np.eye(k.shape[0], dtype=np.float64)

        self.l_ = np.linalg.cholesky(k)

        # Solve K^-1 y using Cholesky: alpha = L^-T (L^-1 y)
        v = np.linalg.solve(self.l_, y)
        self.alpha_ = np.linalg.solve(self.l_.T, v)
        return self

    def predict(self, x: np.ndarray, *, return_std: bool = False):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        k_star = _rbf_kernel(
            x,
            self.x_train_,
            length_scale=float(self.length_scale),
            sigma_f=float(self.sigma_f),
        )
        mean = k_star @ self.alpha_

        if not return_std:
            return mean

        # Predictive variance:
        # var = k(x*,x*) - v^T v, where v = L^-1 k(x*,X)^T
        v = np.linalg.solve(self.l_, k_star.T)  # (n_train, n_test)
        k_xx = float(self.sigma_f) ** 2
        var = np.maximum(k_xx - np.sum(v**2, axis=0), 0.0)
        std = np.sqrt(var)
        return mean, std


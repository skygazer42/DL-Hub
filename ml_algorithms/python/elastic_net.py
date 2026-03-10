"""Elastic Net regression (L1 + L2 regularized linear regression) in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _soft_threshold(z: float, gamma: float) -> float:
    if z > gamma:
        return z - gamma
    if z < -gamma:
        return z + gamma
    return 0.0


@dataclass
class ElasticNetRegression:
    """Coordinate descent Elastic Net regression.

    Objective (fit_intercept=True default):
        (1 / (2n)) * ||y - (X w + b)||^2
        + alpha * l1_ratio * ||w||_1
        + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
    """

    alpha: float = 1.0
    l1_ratio: float = 0.5
    max_iter: int = 1000
    tol: float = 1e-6
    fit_intercept: bool = True

    def fit(self, x: np.ndarray, y: np.ndarray) -> ElasticNetRegression:
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

        l1_ratio = float(self.l1_ratio)
        if not (0.0 <= l1_ratio <= 1.0):
            raise ValueError("l1_ratio must be in [0, 1]")

        max_iter = int(self.max_iter)
        if max_iter <= 0:
            raise ValueError("max_iter must be > 0")

        tol = float(self.tol)
        if tol <= 0.0:
            raise ValueError("tol must be > 0")

        n_samples, n_features = x.shape

        if self.fit_intercept:
            self.x_mean_ = x.mean(axis=0)
            self.y_mean_ = float(y.mean())
        else:
            self.x_mean_ = np.zeros((n_features,), dtype=np.float64)
            self.y_mean_ = 0.0

        x_centered = x - self.x_mean_
        scale = x_centered.std(axis=0)
        scale = np.where(scale > 0.0, scale, 1.0).astype(np.float64)
        self.x_scale_ = scale

        xs = x_centered / self.x_scale_
        y_centered = y - self.y_mean_

        w = np.zeros((n_features,), dtype=np.float64)
        col_norm = (xs**2).mean(axis=0)

        y_pred = xs @ w

        l1 = alpha * l1_ratio
        l2 = alpha * (1.0 - l1_ratio)

        for _ in range(max_iter):
            w_prev = w.copy()

            for j in range(n_features):
                r_j = y_centered - y_pred + w[j] * xs[:, j]
                rho = float((xs[:, j] * r_j).mean())
                w_j = _soft_threshold(rho, l1) / float(col_norm[j] + l2 + 1e-12)

                if w_j != w[j]:
                    y_pred += (w_j - w[j]) * xs[:, j]
                    w[j] = w_j

            if float(np.max(np.abs(w - w_prev))) < tol:
                break

        self.weights_ = w / self.x_scale_
        self.bias_ = self.y_mean_ - float(self.x_mean_ @ self.weights_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x @ self.weights_ + float(self.bias_)


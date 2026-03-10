"""Gaussian kernel density estimation (KDE) in NumPy."""

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


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis=axis)


@dataclass
class GaussianKDE:
    bandwidth: float = 1.0

    def fit(self, x: np.ndarray) -> GaussianKDE:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if int(x.shape[0]) < 1:
            raise ValueError("x must be non-empty")

        bw = float(self.bandwidth)
        if bw <= 0.0:
            raise ValueError("bandwidth must be > 0")

        self.x_train_ = x
        self.n_train_, self.n_features_ = (int(v) for v in x.shape)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if int(x.shape[1]) != int(self.n_features_):
            raise ValueError("x must have the same number of features as the training data")

        bw = float(self.bandwidth)
        sq = _pairwise_squared_distances(x / bw, self.x_train_ / bw)  # (m, n)
        log_terms = -0.5 * sq

        log_norm = -np.log(float(self.n_train_))
        log_norm -= float(self.n_features_) * np.log(bw)
        log_norm -= 0.5 * float(self.n_features_) * np.log(2.0 * np.pi)

        return _logsumexp(log_terms, axis=1) + log_norm

    def pdf(self, x: np.ndarray) -> np.ndarray:
        return np.exp(self.score_samples(x))


"""Non-negative matrix factorization (NMF) in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NMF:
    n_components: int = 2
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> NMF:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.size == 0:
            raise ValueError("x must be non-empty")
        if np.any(x < 0.0):
            raise ValueError("NMF requires x to be non-negative.")

        n_samples, n_features = x.shape
        n_components = int(self.n_components)
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if n_components > min(n_samples, n_features):
            raise ValueError("n_components must be <= min(n_samples, n_features)")

        max_iter = int(self.max_iter)
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1")

        tol = float(self.tol)
        if tol < 0.0:
            raise ValueError("tol must be >= 0")

        rng = np.random.default_rng(self.random_state)
        eps = 1e-12

        mean_x = float(x.mean())
        scale = float(np.sqrt(mean_x / float(n_components))) if mean_x > 0.0 else 1.0

        w = rng.random((n_samples, n_components), dtype=np.float64) * scale + eps
        h = rng.random((n_components, n_features), dtype=np.float64) * scale + eps

        prev_err: float | None = None
        for _ in range(max_iter):
            wh = w @ h

            numerator = w.T @ x
            denominator = w.T @ wh + eps
            h *= numerator / denominator

            wh = w @ h
            numerator = x @ h.T
            denominator = wh @ h.T + eps
            w *= numerator / denominator

            err = float(np.linalg.norm(x - (w @ h), ord="fro"))
            if prev_err is not None:
                if abs(prev_err - err) <= tol * (prev_err + eps):
                    prev_err = err
                    break
            prev_err = err

        self.basis_ = w
        self.components_ = h
        self.reconstruction_err_ = float(prev_err if prev_err is not None else 0.0)
        return self

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.basis_

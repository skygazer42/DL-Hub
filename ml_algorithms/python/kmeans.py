"""K-means clustering in NumPy."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be >= 1")
    return result


@dataclass
class KMeans:
    n_clusters: int = 8
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> KMeans:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("x must contain at least one sample and one feature")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")

        n_clusters = _positive_integer("n_clusters", self.n_clusters)
        if n_clusters > x.shape[0]:
            raise ValueError("n_clusters must be <= n_samples")
        max_iter = _positive_integer("max_iter", self.max_iter)
        tol = float(self.tol)
        if not np.isfinite(tol) or tol < 0.0:
            raise ValueError("tol must be finite and >= 0")

        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(x.shape[0], n_clusters, replace=False)
        centroids = x[indices]
        for _ in range(max_iter):
            distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centroids = np.vstack(
                [
                    x[labels == idx].mean(axis=0) if np.any(labels == idx) else centroids[idx]
                    for idx in range(n_clusters)
                ]
            )
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < tol:
                break
        self.cluster_centers_ = centroids
        self.n_features_in_ = int(x.shape[1])
        self.labels_ = self.predict(x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("KMeans is not fitted; call fit() first")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"x must have {self.n_features_in_} features, got {x.shape[1]}")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values")
        distances = ((x[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

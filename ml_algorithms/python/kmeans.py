"""K-means clustering in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KMeans:
    n_clusters: int = 8
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> KMeans:
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=np.float64)
        indices = rng.choice(x.shape[0], self.n_clusters, replace=False)
        centroids = x[indices]
        for _ in range(self.max_iter):
            distances = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centroids = np.vstack(
                [
                    x[labels == idx].mean(axis=0) if np.any(labels == idx) else centroids[idx]
                    for idx in range(self.n_clusters)
                ]
            )
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break
        self.cluster_centers_ = centroids
        self.labels_ = labels
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        distances = ((x[:, None, :] - self.cluster_centers_[None, :, :]) ** 2).sum(axis=2)
        return distances.argmin(axis=1)

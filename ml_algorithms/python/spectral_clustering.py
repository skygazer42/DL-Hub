"""Spectral clustering in NumPy (RBF affinity + normalized Laplacian + k-means)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_algorithms.python.kmeans import KMeans


def _pairwise_squared_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq_norms = np.sum(x**2, axis=1, keepdims=True)
    sq_dists = sq_norms + sq_norms.T - 2.0 * (x @ x.T)
    return np.maximum(sq_dists, 0.0)


@dataclass
class SpectralClustering:
    n_clusters: int = 2
    gamma: float = 1.0
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> SpectralClustering:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_samples = int(x.shape[0])
        if n_samples < 1:
            raise ValueError("x must be non-empty")
        if int(self.n_clusters) < 2:
            raise ValueError("n_clusters must be >= 2")
        if int(self.n_clusters) > n_samples:
            raise ValueError("n_clusters must be <= n_samples")
        if float(self.gamma) <= 0.0:
            raise ValueError("gamma must be > 0")

        sq_dists = _pairwise_squared_euclidean(x)
        affinity = np.exp(-float(self.gamma) * sq_dists)
        np.fill_diagonal(affinity, 0.0)

        degree = affinity.sum(axis=1)
        inv_sqrt_degree = np.where(degree > 0.0, 1.0 / np.sqrt(degree), 0.0)

        normalized_affinity = affinity * inv_sqrt_degree[:, None] * inv_sqrt_degree[None, :]
        laplacian = np.eye(n_samples, dtype=np.float64) - normalized_affinity

        _, eigenvectors = np.linalg.eigh(laplacian)
        embedding = eigenvectors[:, : int(self.n_clusters)]

        norms = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = np.divide(
            embedding,
            norms,
            out=np.zeros_like(embedding),
            where=norms > 0.0,
        )

        kmeans = KMeans(n_clusters=int(self.n_clusters), random_state=self.random_state).fit(
            embedding
        )
        self.labels_ = np.asarray(kmeans.labels_, dtype=int)
        return self

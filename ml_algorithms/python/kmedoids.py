"""K-medoids clustering in NumPy.

This is a simple Partitioning Around Medoids (PAM)-style implementation using
L2 distances. It is intentionally minimal and educational.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _pairwise_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances between two 2D arrays."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # (n, 1, d) - (1, m, d) => (n, m, d)
    diff = x[:, None, :] - y[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


@dataclass
class KMedoids:
    n_clusters: int = 8
    max_iter: int = 300
    random_state: int | None = None

    def fit(self, x: np.ndarray) -> KMedoids:
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {x.shape}")
        n_samples = x.shape[0]
        k = int(self.n_clusters)
        if k <= 0:
            raise ValueError("n_clusters must be > 0")
        if n_samples < k:
            raise ValueError("n_clusters must be <= number of samples")

        # Initialize medoids as random unique points.
        medoid_indices = rng.choice(n_samples, size=k, replace=False)

        for _ in range(int(self.max_iter)):
            medoids = x[medoid_indices]
            distances = _pairwise_distances(x, medoids)
            labels = distances.argmin(axis=1)

            new_medoid_indices = medoid_indices.copy()
            for cluster_id in range(k):
                members = np.where(labels == cluster_id)[0]
                if members.size == 0:
                    continue

                member_points = x[members]
                # Distances within cluster: (m, m)
                intra = _pairwise_distances(member_points, member_points)
                # Choose point with minimal total distance to others.
                best_member = members[np.argmin(intra.sum(axis=1))]
                new_medoid_indices[cluster_id] = int(best_member)

            # Stop if medoids no longer change.
            if np.array_equal(new_medoid_indices, medoid_indices):
                medoid_indices = new_medoid_indices
                break
            medoid_indices = new_medoid_indices

        # Final assignment
        self.medoid_indices_ = medoid_indices.astype(int)
        self.cluster_centers_ = x[self.medoid_indices_]
        self.labels_ = self.predict(x)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        distances = _pairwise_distances(x, self.cluster_centers_)
        return distances.argmin(axis=1)

"""Principal component analysis in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCA:
    n_components: int

    def fit(self, x: np.ndarray) -> PCA:
        x = np.asarray(x, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        centered = x - self.mean_
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        self.components_ = eigenvectors[:, order[: self.n_components]]
        self.explained_variance_ = eigenvalues[order[: self.n_components]]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        centered = x - self.mean_
        return centered @ self.components_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

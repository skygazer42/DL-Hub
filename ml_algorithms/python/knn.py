"""K-nearest neighbors classifier and regressor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KNNClassifier:
    k: int = 5

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.x_ = np.asarray(x, dtype=np.float64)
        self.y_ = np.asarray(y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        distances = ((x[:, None, :] - self.x_[None, :, :]) ** 2).sum(axis=2)
        neighbors = np.argpartition(distances, self.k, axis=1)[:, : self.k]
        labels = self.y_[neighbors]
        return np.apply_along_axis(self._majority_vote, 1, labels)

    @staticmethod
    def _majority_vote(labels: np.ndarray) -> np.ndarray:
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]


@dataclass
class KNNRegressor:
    k: int = 5

    def fit(self, x: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.x_ = np.asarray(x, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.float64)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        distances = ((x[:, None, :] - self.x_[None, :, :]) ** 2).sum(axis=2)
        neighbors = np.argpartition(distances, self.k, axis=1)[:, : self.k]
        return self.y_[neighbors].mean(axis=1)

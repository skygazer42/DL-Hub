"""K-nearest neighbors classifier and regressor."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import numpy as np


def _validate_training_data(
    x: np.ndarray, y: np.ndarray, *, numeric_targets: bool
) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(x, dtype=np.float64)
    targets = np.asarray(y, dtype=np.float64 if numeric_targets else None)
    if features.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("x must contain at least one sample and one feature")
    if targets.ndim != 1:
        raise ValueError("y must be a 1D array")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    if not np.all(np.isfinite(features)):
        raise ValueError("x must contain only finite values")
    if numeric_targets and not np.all(np.isfinite(targets)):
        raise ValueError("y must contain only finite values")
    return features, targets


def _validate_k(k: int, n_samples: int) -> int:
    if isinstance(k, bool):
        raise ValueError("k must be an integer in [1, n_samples]")
    try:
        value = operator.index(k)
    except TypeError as exc:
        raise ValueError("k must be an integer in [1, n_samples]") from exc
    if not 1 <= value <= n_samples:
        raise ValueError(f"k must be in [1, {n_samples}], got {value}")
    return value


def _validate_query(x: np.ndarray, *, n_features: int) -> np.ndarray:
    query = np.asarray(x, dtype=np.float64)
    if query.ndim != 2:
        raise ValueError("x must be a 2D array of shape (n_samples, n_features)")
    if query.shape[1] != n_features:
        raise ValueError(f"x must have {n_features} features, got {query.shape[1]}")
    if not np.all(np.isfinite(query)):
        raise ValueError("x must contain only finite values")
    return query


@dataclass
class KNNClassifier:
    k: int = 5

    def fit(self, x: np.ndarray, y: np.ndarray) -> KNNClassifier:
        self.x_, self.y_ = _validate_training_data(x, y, numeric_targets=False)
        _validate_k(self.k, int(self.x_.shape[0]))
        self.n_features_in_ = int(self.x_.shape[1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x_"):
            raise RuntimeError("KNNClassifier is not fitted; call fit() first")
        x = _validate_query(x, n_features=self.n_features_in_)
        k = _validate_k(self.k, int(self.x_.shape[0]))
        distances = ((x[:, None, :] - self.x_[None, :, :]) ** 2).sum(axis=2)
        neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
        labels = self.y_[neighbors]
        return np.apply_along_axis(self._majority_vote, 1, labels)

    @staticmethod
    def _majority_vote(labels: np.ndarray) -> np.ndarray:
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]


@dataclass
class KNNRegressor:
    k: int = 5

    def fit(self, x: np.ndarray, y: np.ndarray) -> KNNRegressor:
        self.x_, self.y_ = _validate_training_data(x, y, numeric_targets=True)
        _validate_k(self.k, int(self.x_.shape[0]))
        self.n_features_in_ = int(self.x_.shape[1])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self, "x_"):
            raise RuntimeError("KNNRegressor is not fitted; call fit() first")
        x = _validate_query(x, n_features=self.n_features_in_)
        k = _validate_k(self.k, int(self.x_.shape[0]))
        distances = ((x[:, None, :] - self.x_[None, :, :]) ** 2).sum(axis=2)
        neighbors = np.argpartition(distances, k - 1, axis=1)[:, :k]
        return self.y_[neighbors].mean(axis=1)

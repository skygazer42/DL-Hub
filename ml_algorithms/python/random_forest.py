"""Random forest classifier and regressor in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor


@dataclass
class RandomForestClassifier:
    n_estimators: int = 10
    max_depth: int = 5
    min_samples_split: int = 2
    max_features: int | None = None
    random_state: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.trees_: list[DecisionTreeClassifier] = []
        n_samples, n_features = x.shape
        max_features = self.max_features or int(np.sqrt(n_features))
        for _ in range(self.n_estimators):
            indices = rng.integers(0, n_samples, n_samples)
            feature_indices = rng.choice(n_features, max_features, replace=False)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(x[indices][:, feature_indices], y[indices])
            tree.feature_indices_ = feature_indices
            self.trees_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        predictions = []
        for tree in self.trees_:
            preds = tree.predict(x[:, tree.feature_indices_])
            predictions.append(preds)
        predictions = np.vstack(predictions).T
        return np.apply_along_axis(self._majority_vote, 1, predictions)

    @staticmethod
    def _majority_vote(labels: np.ndarray) -> np.ndarray:
        values, counts = np.unique(labels, return_counts=True)
        return values[np.argmax(counts)]


@dataclass
class RandomForestRegressor:
    n_estimators: int = 10
    max_depth: int = 5
    min_samples_split: int = 2
    max_features: int | None = None
    random_state: int | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
        rng = np.random.default_rng(self.random_state)
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.trees_: list[DecisionTreeRegressor] = []
        n_samples, n_features = x.shape
        max_features = self.max_features or int(np.sqrt(n_features))
        for _ in range(self.n_estimators):
            indices = rng.integers(0, n_samples, n_samples)
            feature_indices = rng.choice(n_features, max_features, replace=False)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(x[indices][:, feature_indices], y[indices])
            tree.feature_indices_ = feature_indices
            self.trees_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        predictions = []
        for tree in self.trees_:
            preds = tree.predict(x[:, tree.feature_indices_])
            predictions.append(preds)
        predictions = np.vstack(predictions).T
        return predictions.mean(axis=1)

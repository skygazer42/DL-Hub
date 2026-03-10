"""Gradient boosting algorithms in NumPy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decision_tree import DecisionTreeRegressor


@dataclass
class GradientBoostingRegressor:
    n_estimators: int = 50
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_split: int = 2
    estimators_: list[DecisionTreeRegressor] = field(init=False, default_factory=list)
    init_: float = field(init=False, default=0.0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.init_ = float(np.mean(y))
        preds = np.full_like(y, fill_value=self.init_, dtype=np.float64)
        self.estimators_ = []

        for _ in range(int(self.n_estimators)):
            residual = y - preds
            tree = DecisionTreeRegressor(
                max_depth=int(self.max_depth),
                min_samples_split=int(self.min_samples_split),
            ).fit(x, residual)
            update = np.asarray(tree.predict(x), dtype=np.float64)
            preds += float(self.learning_rate) * update
            self.estimators_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        preds = np.full(x.shape[0], fill_value=float(self.init_), dtype=np.float64)
        for tree in self.estimators_:
            preds += float(self.learning_rate) * np.asarray(tree.predict(x), dtype=np.float64)
        return preds

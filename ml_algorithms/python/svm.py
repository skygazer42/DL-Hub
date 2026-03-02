"""Linear SVM classifier (hinge loss) in NumPy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearSVM:
    learning_rate: float = 1e-3
    epochs: int = 1000
    regularization: float = 1e-2

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearSVM:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y = np.where(y <= 0, -1.0, 1.0)
        self.weights_ = np.zeros(x.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            margins = y * (x @ self.weights_ + self.bias_)
            misclassified = margins < 1
            grad_w = self.weights_ - self.regularization * (x[misclassified].T @ y[misclassified])
            grad_b = -self.regularization * y[misclassified].sum()
            self.weights_ -= self.learning_rate * grad_w
            self.bias_ -= self.learning_rate * grad_b
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return x @ self.weights_ + self.bias_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.decision_function(x) >= 0).astype(int)

"""Simple NumPy implementations of classic linear models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearRegression:
    learning_rate: float = 1e-2
    epochs: int = 1000

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        self.weights_ = np.zeros((x_bias.shape[1], 1), dtype=np.float64)
        for _ in range(self.epochs):
            preds = x_bias @ self.weights_
            grad = (x_bias.T @ (preds - y)) / len(x_bias)
            self.weights_ -= self.learning_rate * grad
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        return (x_bias @ self.weights_).ravel()


@dataclass
class LogisticRegression:
    learning_rate: float = 1e-2
    epochs: int = 1000

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticRegression:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        self.weights_ = np.zeros((x_bias.shape[1], 1), dtype=np.float64)
        for _ in range(self.epochs):
            preds = self._sigmoid(x_bias @ self.weights_)
            grad = (x_bias.T @ (preds - y)) / len(x_bias)
            self.weights_ -= self.learning_rate * grad
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_bias = np.c_[np.ones((x.shape[0], 1)), x]
        return self._sigmoid(x_bias @ self.weights_).ravel()

    def predict(self, x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(x) >= threshold).astype(int)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

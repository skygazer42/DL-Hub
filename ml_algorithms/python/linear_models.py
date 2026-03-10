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


@dataclass
class RidgeRegression:
    alpha: float = 1.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> RidgeRegression:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        x_bias = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]
        xtx = x_bias.T @ x_bias
        xty = x_bias.T @ y

        reg = float(self.alpha) * np.eye(xtx.shape[0], dtype=np.float64)
        reg[0, 0] = 0.0  # don't regularize bias

        self.weights_ = np.linalg.solve(xtx + reg, xty)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_bias = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]
        return (x_bias @ self.weights_).ravel()


@dataclass
class SoftmaxRegression:
    learning_rate: float = 1e-2
    epochs: int = 1000

    def fit(self, x: np.ndarray, y: np.ndarray) -> SoftmaxRegression:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_idx = np.array([class_to_index[v] for v in y], dtype=int)

        x_bias = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]
        n_samples, n_features = x_bias.shape
        n_classes = int(self.classes_.shape[0])

        y_onehot = np.zeros((n_samples, n_classes), dtype=np.float64)
        y_onehot[np.arange(n_samples), y_idx] = 1.0

        self.weights_ = np.zeros((n_features, n_classes), dtype=np.float64)

        for _ in range(int(self.epochs)):
            scores = x_bias @ self.weights_
            probs = self._softmax(scores)
            grad = (x_bias.T @ (probs - y_onehot)) / float(n_samples)
            self.weights_ -= float(self.learning_rate) * grad
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        x_bias = np.c_[np.ones((x.shape[0], 1), dtype=np.float64), x]
        return self._softmax(x_bias @ self.weights_)

    def predict(self, x: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(x)
        return self.classes_[np.argmax(probs, axis=1)]

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

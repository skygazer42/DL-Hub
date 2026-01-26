"""A minimal two-layer MLP classifier in NumPy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLPClassifier:
    hidden_units: int = 32
    learning_rate: float = 1e-2
    epochs: int = 500

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MLPClassifier":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        y_onehot = self._one_hot(y)
        rng = np.random.default_rng(42)
        self.w1_ = rng.normal(0, 0.1, size=(x.shape[1], self.hidden_units))
        self.b1_ = np.zeros(self.hidden_units)
        self.w2_ = rng.normal(0, 0.1, size=(self.hidden_units, len(self.classes_)))
        self.b2_ = np.zeros(len(self.classes_))
        for _ in range(self.epochs):
            hidden = self._relu(x @ self.w1_ + self.b1_)
            logits = hidden @ self.w2_ + self.b2_
            probs = self._softmax(logits)
            grad_logits = (probs - y_onehot) / x.shape[0]
            grad_w2 = hidden.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_hidden = grad_logits @ self.w2_.T
            grad_hidden[hidden <= 0] = 0
            grad_w1 = x.T @ grad_hidden
            grad_b1 = grad_hidden.sum(axis=0)
            self.w2_ -= self.learning_rate * grad_w2
            self.b2_ -= self.learning_rate * grad_b2
            self.w1_ -= self.learning_rate * grad_w1
            self.b1_ -= self.learning_rate * grad_b1
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        hidden = self._relu(x @ self.w1_ + self.b1_)
        logits = hidden @ self.w2_ + self.b2_
        probs = self._softmax(logits)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        y_indices = np.searchsorted(self.classes_, y)
        one_hot = np.zeros((y.shape[0], len(self.classes_)), dtype=np.float64)
        one_hot[np.arange(y.shape[0]), y_indices] = 1.0
        return one_hot

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)

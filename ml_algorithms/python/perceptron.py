"""Perceptron classifier in NumPy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Perceptron:
    learning_rate: float = 1e-2
    epochs: int = 1000

    def fit(self, x: np.ndarray, y: np.ndarray) -> "Perceptron":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y)
        y = np.where(y <= 0, -1.0, 1.0)
        self.weights_ = np.zeros(x.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        for _ in range(self.epochs):
            for sample, label in zip(x, y):
                if label * (sample @ self.weights_ + self.bias_) <= 0:
                    self.weights_ += self.learning_rate * label * sample
                    self.bias_ += self.learning_rate * label
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return ((x @ self.weights_ + self.bias_) >= 0).astype(int)

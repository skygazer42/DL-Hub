"""Loss functions commonly used in deep learning."""
from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(((y_true - y_pred) ** 2).mean())


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.abs(y_true - y_pred).mean())


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)
    loss = -y_true * np.log(y_pred) - (1.0 - y_true) * np.log(1.0 - y_pred)
    return float(loss.mean())


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return float(loss.mean())

"""Loss functions commonly used in deep learning."""

import numpy as np


def _validate_numeric_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}"
        )
    if true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must contain only finite values")
    return true, pred


def _validate_eps(eps: float) -> float:
    value = float(eps)
    if not np.isfinite(value) or not 0.0 < value < 0.5:
        raise ValueError("eps must be finite and in the interval (0, 0.5)")
    return value


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_numeric_pair(y_true, y_pred)
    return float(((y_true - y_pred) ** 2).mean())


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_numeric_pair(y_true, y_pred)
    return float(np.abs(y_true - y_pred).mean())


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = _validate_numeric_pair(y_true, y_pred)
    if np.any((y_true < 0.0) | (y_true > 1.0)):
        raise ValueError("y_true values must be in [0, 1]")
    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("y_pred probabilities must be in [0, 1]")
    eps = _validate_eps(eps)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -y_true * np.log(y_pred) - (1.0 - y_true) * np.log(1.0 - y_pred)
    return float(loss.mean())


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true, y_pred = _validate_numeric_pair(y_true, y_pred)
    if y_true.ndim != 2:
        raise ValueError("y_true and y_pred must be 2D arrays of shape (n_samples, n_classes)")
    if np.any((y_true < 0.0) | (y_true > 1.0)):
        raise ValueError("y_true values must be in [0, 1]")
    if np.any((y_pred < 0.0) | (y_pred > 1.0)):
        raise ValueError("y_pred probabilities must be in [0, 1]")
    if not np.allclose(y_true.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("each y_true row must sum to 1")
    if not np.allclose(y_pred.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("each y_pred probability row must sum to 1")
    eps = _validate_eps(eps)
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    return float(loss.mean())

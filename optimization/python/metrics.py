"""Metrics for ML/DL experimentation."""

import numpy as np


def _validate_matching_arrays(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {true.shape} and {pred.shape}"
        )
    if true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    for name, value in (("y_true", true), ("y_pred", pred)):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
    return true, pred


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_matching_arrays(y_true, y_pred)
    return float((y_true == y_pred).mean())


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true, y_pred = _validate_matching_arrays(y_true, y_pred)
    if not np.all(np.isin(y_true, (0, 1))) or not np.all(np.isin(y_pred, (0, 1))):
        raise ValueError("precision_recall_f1 supports binary labels {0, 1} only")
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = _validate_matching_arrays(
        np.asarray(y_true, dtype=np.float64),
        np.asarray(y_pred, dtype=np.float64),
    )
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        raise ValueError("y_true and y_pred must contain only finite values")
    total = ((y_true - y_true.mean()) ** 2).sum()
    residual = ((y_true - y_pred) ** 2).sum()
    if total == 0.0:
        return 1.0 if residual == 0.0 else 0.0
    return float(1.0 - residual / total)

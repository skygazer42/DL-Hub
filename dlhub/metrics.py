from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def accuracy_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty")
    for name, value in (("y_true", y_true), ("y_pred", y_pred)):
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain only finite values")
    return float((y_true == y_pred).mean())


def accuracy_torch(logits: torch.Tensor, targets: torch.Tensor) -> float:
    import torch

    if not torch.is_tensor(logits) or not torch.is_tensor(targets):
        raise TypeError("logits and targets must be torch tensors")
    if logits.ndim < 2:
        raise ValueError(f"logits must have at least 2 dimensions, got shape {tuple(logits.shape)}")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one class")
    expected_target_shape = tuple(logits.shape[:1]) + tuple(logits.shape[2:])
    if tuple(targets.shape) != expected_target_shape:
        raise ValueError(
            f"target shape must be {expected_target_shape} for logits shape {tuple(logits.shape)}, "
            f"got {tuple(targets.shape)}"
        )
    if targets.numel() == 0:
        raise ValueError("logits and targets must be non-empty")
    if logits.device != targets.device:
        raise ValueError("logits and targets must be on the same device")
    if logits.is_complex() or targets.is_complex():
        raise ValueError("logits and targets must be real-valued")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("logits must contain only finite values")
    if not bool(torch.isfinite(targets).all().item()):
        raise ValueError("targets must contain only finite values")

    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())

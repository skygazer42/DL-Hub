from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def accuracy_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def accuracy_torch(logits: torch.Tensor, targets: torch.Tensor) -> float:
    import torch

    preds = torch.argmax(logits, dim=1)
    return float((preds == targets).float().mean().item())

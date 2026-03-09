
from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """Masked MSE loss for blind-spot denoising (Noise2Self/Noise2Void-style).

    Expected target format:
    - `target["target"]`: Tensor (B, C, H, W)
    - `target["mask"]`: Tensor (B, C, H, W) with 0/1 or bool values
    """

    def __init__(self, *, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor | Mapping[str, torch.Tensor]) -> torch.Tensor:
        if torch.is_tensor(target):
            return F.mse_loss(pred, target)

        if not isinstance(target, Mapping):
            raise TypeError(f"MaskedMSELoss target must be a Tensor or Mapping, got: {type(target).__name__}")

        y = target.get("target", None)
        m = target.get("mask", None)
        if y is None or m is None:
            raise KeyError("MaskedMSELoss expects target mapping with keys {'target', 'mask'}")
        if not torch.is_tensor(y) or not torch.is_tensor(m):
            raise TypeError("MaskedMSELoss expects 'target' and 'mask' to be torch Tensors")
        if pred.shape != y.shape or y.shape != m.shape:
            raise ValueError(f"Shape mismatch: pred={tuple(pred.shape)} target={tuple(y.shape)} mask={tuple(m.shape)}")

        pred = pred.to(torch.float32)
        y = y.to(torch.float32)
        m = m.to(dtype=pred.dtype)

        se = (pred - y).pow(2) * m
        denom = m.sum().clamp_min(self.eps)
        return se.sum() / denom


__all__ = ["MaskedMSELoss"]



from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.instance_segmentation.yolact import build_yolact_instance_segmenter


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    num_classes: int = 1
    variant: str = "yolact_tiny"
    width_mult: float = 1.0


class TinyYOLACT(nn.Module):
    """Toy-first YOLACT-style model specialized for 1-instance-per-image lessons.

    We set `num_anchors=1` to avoid extra anchor dimensions in the output tensors.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = build_yolact_instance_segmenter(
            in_channels=int(cfg.in_channels),
            num_classes=int(cfg.num_classes),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
            num_anchors=1,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.model(x)
        # For this lesson we treat bbox deltas as ltrb distances -> enforce non-negativity.
        out = dict(out)
        out["bbox_deltas"] = torch.relu(out["bbox_deltas"])
        return out


def mask_logits_from_proto(
    *,
    proto: torch.Tensor,
    mask_coeffs: torch.Tensor,
    pos_mask: torch.Tensor,
    out_hw: tuple[int, int],
) -> torch.Tensor:
    """Compute mask logits from YOLACT prototypes and per-cell coefficients.

    Args:
        proto: (B, P, H4, W4)
        mask_coeffs: (B, P, Gh, Gw) for A=1 anchors
        pos_mask: (B, 1, Gh, Gw) one-hot positive location per sample
        out_hw: (H, W) output resolution
    Returns:
        mask_logits: (B, 1, H, W)
    """

    if proto.ndim != 4 or mask_coeffs.ndim != 4 or pos_mask.ndim != 4:
        raise ValueError("Expected proto/mask_coeffs/pos_mask to be 4D tensors.")
    if pos_mask.shape[1] != 1:
        raise ValueError("pos_mask must have shape (B, 1, Gh, Gw)")
    if proto.shape[0] != mask_coeffs.shape[0] or proto.shape[0] != pos_mask.shape[0]:
        raise ValueError("Batch size mismatch between proto/mask_coeffs/pos_mask")
    if mask_coeffs.shape[1] != proto.shape[1]:
        raise ValueError("Prototype channels P must match in proto and mask_coeffs.")
    if mask_coeffs.shape[-2:] != pos_mask.shape[-2:]:
        raise ValueError("mask_coeffs and pos_mask spatial sizes must match.")

    # Select the coefficient vector at the positive cell (one-hot, so sum works).
    coeff = (mask_coeffs * pos_mask).sum(dim=(2, 3))  # (B, P)
    mask_small = (proto * coeff[:, :, None, None]).sum(dim=1, keepdim=True)  # (B, 1, H4, W4)
    return F.interpolate(mask_small, size=tuple(map(int, out_hw)), mode="bilinear", align_corners=False)


__all__ = ["ModelConfig", "TinyYOLACT", "mask_logits_from_proto"]

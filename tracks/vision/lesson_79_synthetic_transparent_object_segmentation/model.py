from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.transparent_object_segmentation import (
    build_boundary_glass_seg_transparent_segmenter,
    build_camotransparent_seg_transparent_segmenter,
    build_diffusion_transparent_transparent_segmenter,
    build_glassseg_toy_transparent_segmenter,
    build_mamba_transparent_transparent_segmenter,
    build_prompt_transparent_transparent_segmenter,
    build_refractmask_seg_transparent_segmenter,
    build_transformer_transparent_transparent_segmenter,
    build_translab_seg_transparent_segmenter,
    build_trimap_transparent_transparent_segmenter,
)


_ARCH_BUILDERS = {
    "glassseg_toy": build_glassseg_toy_transparent_segmenter,
    "boundary_glass_seg": build_boundary_glass_seg_transparent_segmenter,
    "camotransparent_seg": build_camotransparent_seg_transparent_segmenter,
    "diffusion_transparent": build_diffusion_transparent_transparent_segmenter,
    "prompt_transparent": build_prompt_transparent_transparent_segmenter,
    "refractmask_seg": build_refractmask_seg_transparent_segmenter,
    "transformer_transparent": build_transformer_transparent_transparent_segmenter,
    "translab_seg": build_translab_seg_transparent_segmenter,
    "trimap_transparent": build_trimap_transparent_transparent_segmenter,
    "mamba_transparent": build_mamba_transparent_transparent_segmenter,
}


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "glassseg_toy"
    variant: str = "glassseg_toy_small"
    width_mult: float = 1.0


class TransparentObjectSegmentationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.arch not in _ARCH_BUILDERS:
            raise ValueError(f"Unknown arch: {cfg.arch!r}. Supported: {sorted(_ARCH_BUILDERS)}")
        builder = _ARCH_BUILDERS[cfg.arch]
        self.net = builder(
            in_channels=int(cfg.in_channels),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(image.to(torch.float32))


def transparent_segmentation_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    mask_target = targets["mask"].to(torch.float32)
    alpha_target = targets["alpha"].to(torch.float32)
    boundary_target = targets["boundary"].to(torch.float32)

    mask_bce = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], mask_target)
    alpha_l1 = torch.nn.functional.l1_loss(outputs["alpha"], alpha_target)
    boundary_l1 = torch.nn.functional.l1_loss(outputs["boundary"], boundary_target)
    total = mask_bce + 0.3 * alpha_l1 + 0.2 * boundary_l1
    parts = {
        "mask_bce": float(mask_bce.item()),
        "alpha_l1": float(alpha_l1.item()),
        "boundary_l1": float(boundary_l1.item()),
    }
    return total, parts


def build_model(cfg: ModelConfig) -> TransparentObjectSegmentationModel:
    return TransparentObjectSegmentationModel(cfg)


__all__ = [
    "ModelConfig",
    "TransparentObjectSegmentationModel",
    "build_model",
    "transparent_segmentation_loss",
]

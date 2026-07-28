from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.generative.video_to_video import (
    build_diffusion_v2v_video_to_video,
    build_mamba_v2v_video_to_video,
    build_transformer_v2v_video_to_video,
)


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    family: str = "diffusion_v2v"
    variant: str = "diffusion_v2v_tiny"
    width_mult: float = 1.0


def _build_model(cfg: ModelConfig) -> nn.Module:
    family = str(cfg.family)
    builders = {
        "diffusion_v2v": build_diffusion_v2v_video_to_video,
        "mamba_v2v": build_mamba_v2v_video_to_video,
        "transformer_v2v": build_transformer_v2v_video_to_video,
    }
    if family not in builders:
        known = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unsupported family '{family}'. Available: {known}")
    return builders[family](
        in_channels=int(cfg.in_channels),
        variant=str(cfg.variant),
        width_mult=float(cfg.width_mult),
    )


class CompactVideoToVideoModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _build_model(cfg)

    def forward(self, source_video: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(source_video)


def video_to_video_loss(
    outputs: dict[str, torch.Tensor],
    target_video: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    video_loss = torch.nn.functional.mse_loss(outputs["video"], target_video)
    residual_reg = outputs["residual"].abs().mean()
    total = video_loss + 0.05 * residual_reg
    parts = {
        "video_loss": float(video_loss.detach().item()),
        "residual_reg": float(residual_reg.detach().item()),
    }
    return total, parts


__all__ = ["ModelConfig", "CompactVideoToVideoModel", "video_to_video_loss"]

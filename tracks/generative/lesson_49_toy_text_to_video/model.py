from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    family: str = "diffusion_t2v"
    variant: str = "diffusion_t2v_tiny"
    width_mult: float = 1.0


class ToyTextToVideoModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        builders = import_module("dlhub.generative.text_to_video")
        builder_name = f"build_{cfg.family}_text_to_video"
        builder = getattr(builders, builder_name)
        self.generator = builder(
            in_channels=int(cfg.in_channels),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
        )

    def forward(self, prompts: list[str] | tuple[str, ...]) -> dict[str, torch.Tensor]:
        batch_size = len(prompts)
        if batch_size == 0:
            raise ValueError("prompts must be non-empty")
        device = next(self.parameters()).device
        outputs = self.generator(prompt=list(prompts), batch_size=batch_size, device=device)
        video = outputs["video"].permute(0, 1, 2, 3, 4).contiguous()
        return {
            "video": video,
            "prompt_features": outputs["prompt_features"],
            "motion": outputs["motion"],
        }


def text_to_video_loss(
    outputs: dict[str, torch.Tensor],
    target_video: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_video = outputs["video"]
    video_loss = torch.nn.functional.mse_loss(pred_video, target_video)
    motion_reg = outputs["motion"].pow(2).mean()
    loss = video_loss + 0.01 * motion_reg
    parts = {"video_loss": video_loss.detach(), "motion_reg": motion_reg.detach()}
    return loss, parts


__all__ = ["ModelConfig", "ToyTextToVideoModel", "text_to_video_loss"]

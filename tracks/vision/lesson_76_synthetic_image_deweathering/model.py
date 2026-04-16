from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.image_deweathering_zoo import build_local_model, list_local_arches


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "deweather:deweather_cnn_small"
    width_mult: float = 1.0


class DeweatheringModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.model = build_local_model(
            str(cfg.arch),
            in_channels=int(cfg.in_channels),
            width_mult=float(cfg.width_mult),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.model(image)


def build_model(cfg: ModelConfig) -> DeweatheringModel:
    return DeweatheringModel(cfg)


def list_supported_arches() -> list[str]:
    return list_local_arches()


def deweathering_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = torch.nn.functional.l1_loss(
        outputs["restored"].to(torch.float32),
        targets["clean"].to(torch.float32),
    )
    weather_loss = torch.nn.functional.l1_loss(
        outputs["weather_residual"].to(torch.float32),
        targets["weather_residual"].to(torch.float32),
    )
    total_loss = reconstruction_loss + 0.5 * weather_loss
    return total_loss, {
        "reconstruction_loss": float(reconstruction_loss.detach().item()),
        "weather_loss": float(weather_loss.detach().item()),
    }


__all__ = [
    "DeweatheringModel",
    "ModelConfig",
    "build_model",
    "deweathering_loss",
    "list_supported_arches",
]

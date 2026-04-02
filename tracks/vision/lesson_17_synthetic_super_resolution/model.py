from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from dlhub.vision.super_resolution_zoo import build_local_model, list_local_arches


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "sr:srcnn_tiny"
    in_channels: int = 3
    upscale_factor: int = 2
    image_size: int = 32
    width_mult: float = 1.0
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    return list_local_arches()


def build_model(cfg: ModelConfig) -> nn.Module:
    return build_local_model(
        cfg.arch,
        in_channels=int(cfg.in_channels),
        upscale_factor=int(cfg.upscale_factor),
        image_size=int(cfg.image_size),
        width_mult=float(cfg.width_mult),
        dropout=float(cfg.dropout),
    )


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]

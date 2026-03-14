from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "mot2d:sort_tiny"
    in_channels: int = 3
    num_classes: int = 3
    seq_len: int = 4
    image_size: int = 64
    width_mult: float = 1.0
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.vision.mot_zoo import list_local_arches

    return list_local_arches()


def build_model(cfg: ModelConfig) -> nn.Module:
    from dlhub.vision.mot_zoo import build_local_model

    return build_local_model(
        str(cfg.arch),
        in_channels=int(cfg.in_channels),
        num_classes=int(cfg.num_classes),
        seq_len=int(cfg.seq_len),
        image_size=int(cfg.image_size),
        width_mult=float(cfg.width_mult),
        dropout=float(cfg.dropout),
    )


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "dldet:pedestrian_fcos"
    in_channels: int = 3
    num_classes: int = 1
    width_mult: float = 0.5


def list_supported_arches() -> list[str]:
    from dlhub.vision.detection_zoo import list_local_arches

    local = list_local_arches()
    return [a.removeprefix("dldet:") for a in local] + local


def build_model(cfg: ModelConfig) -> nn.Module:
    from dlhub.vision.detection_zoo import build_local_model

    return build_local_model(
        str(cfg.arch),
        in_channels=int(cfg.in_channels),
        num_classes=int(cfg.num_classes),
        width_mult=float(cfg.width_mult),
    )


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]

from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "pointnet"  # local: any id from dlhub.pointcloud.local_zoo (try --list-arch)
    in_channels: int = 3
    num_classes: int = 2
    num_points: int = 128
    width_mult: float = 1.0
    dropout: float = 0.1


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.local_zoo import list_local_arches

    local = list_local_arches()
    return [a.removeprefix("pc:") for a in local] + local


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw if ":" in arch_raw else f"pc:{arch_raw}"

    from dlhub.pointcloud.local_zoo import build_local_model

    return build_local_model(
        arch,
        in_channels=int(cfg.in_channels),
        num_classes=int(cfg.num_classes),
        num_points=int(cfg.num_points),
        width_mult=float(cfg.width_mult),
        dropout=float(cfg.dropout),
    )


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]

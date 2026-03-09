
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "byol_pointnet"
    variant: str = "byol_pointnet_small"
    in_channels: int = 3
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.byol import _VARIANTS as variants

    return [f"byol_pointnet:{k}" for k in sorted(variants)] + ["byol_pointnet"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"byol_pointnet", "byol"}:
        from dlhub.pointcloud.selfsupervised.byol import build_byol_pointnet

        return build_byol_pointnet(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "byol_pointnet_small",
            dropout=float(cfg.dropout),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: byol_pointnet:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


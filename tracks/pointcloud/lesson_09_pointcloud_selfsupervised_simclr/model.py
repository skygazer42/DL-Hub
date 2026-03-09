
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "simclr_pointnet"
    variant: str = "simclr_pointnet_small"
    in_channels: int = 3
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.simclr import _VARIANTS as variants

    return [f"simclr_pointnet:{k}" for k in sorted(variants)] + ["simclr_pointnet"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"simclr_pointnet", "simclr"}:
        from dlhub.pointcloud.selfsupervised.simclr import build_simclr_pointnet

        return build_simclr_pointnet(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "simclr_pointnet_small",
            dropout=float(cfg.dropout),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: simclr_pointnet:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


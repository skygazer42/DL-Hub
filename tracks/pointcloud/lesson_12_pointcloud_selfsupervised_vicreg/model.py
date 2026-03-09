
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "vicreg_pointnet"
    variant: str = "vicreg_pointnet_small"
    in_channels: int = 3
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.vicreg import _VARIANTS as variants

    return [f"vicreg_pointnet:{k}" for k in sorted(variants)] + ["vicreg_pointnet"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"vicreg_pointnet", "vicreg"}:
        from dlhub.pointcloud.selfsupervised.vicreg import build_vicreg_pointnet

        return build_vicreg_pointnet(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "vicreg_pointnet_small",
            dropout=float(cfg.dropout),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: vicreg_pointnet:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


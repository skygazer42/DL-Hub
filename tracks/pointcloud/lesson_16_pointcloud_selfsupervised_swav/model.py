from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "swav_pointnet"
    variant: str = "swav_pointnet_small"
    in_channels: int = 3
    dropout: float = 0.0
    num_prototypes: int | None = None


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.swav import _VARIANTS as variants

    return [f"swav_pointnet:{k}" for k in sorted(variants)] + ["swav_pointnet"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"swav_pointnet", "swav"}:
        from dlhub.pointcloud.selfsupervised.swav import build_swav_pointnet

        return build_swav_pointnet(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "swav_pointnet_small",
            dropout=float(cfg.dropout),
            num_prototypes=None if cfg.num_prototypes is None else int(cfg.num_prototypes),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: swav_pointnet:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


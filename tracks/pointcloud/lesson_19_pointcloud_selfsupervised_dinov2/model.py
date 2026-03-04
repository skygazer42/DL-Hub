from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "dinov2_pointmae"
    variant: str = "dinov2_pointmae_small"
    in_channels: int = 3
    dropout: float = 0.0
    out_dim: int | None = None


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.dinov2 import _VARIANTS as variants

    return [f"dinov2_pointmae:{k}" for k in sorted(variants)] + ["dinov2_pointmae"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"dinov2_pointmae", "dinov2"}:
        from dlhub.pointcloud.selfsupervised.dinov2 import build_dinov2_pointmae

        return build_dinov2_pointmae(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "dinov2_pointmae_small",
            dropout=float(cfg.dropout),
            out_dim=None if cfg.out_dim is None else int(cfg.out_dim),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: dinov2_pointmae:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


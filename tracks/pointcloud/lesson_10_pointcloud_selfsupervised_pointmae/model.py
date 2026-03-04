from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "pointmae"
    variant: str = "pointmae_small"
    in_channels: int = 3
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.pointmae import _VARIANTS as variants

    return [f"pointmae:{k}" for k in sorted(variants)] + ["pointmae"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"pointmae", "mae"}:
        from dlhub.pointcloud.selfsupervised.pointmae import build_pointmae_pretrainer

        return build_pointmae_pretrainer(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "pointmae_small",
            dropout=float(cfg.dropout),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: pointmae:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


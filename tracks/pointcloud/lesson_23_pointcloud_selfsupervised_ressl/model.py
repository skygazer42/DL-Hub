from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "ressl_pointnet"
    variant: str = "ressl_pointnet_small"
    in_channels: int = 3
    dropout: float = 0.0
    queue_size: int | None = None


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.ressl import _VARIANTS as variants

    return [f"ressl_pointnet:{k}" for k in sorted(variants)] + ["ressl_pointnet"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"ressl_pointnet", "ressl"}:
        from dlhub.pointcloud.selfsupervised.ressl import build_ressl_pointnet

        return build_ressl_pointnet(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "ressl_pointnet_small",
            dropout=float(cfg.dropout),
            queue_size=None if cfg.queue_size is None else int(cfg.queue_size),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: ressl_pointnet:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


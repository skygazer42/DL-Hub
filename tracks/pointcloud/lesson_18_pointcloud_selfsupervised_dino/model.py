from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "dino_pointnet"
    variant: str = "dino_pointnet_small"
    in_channels: int = 3
    dropout: float = 0.0
    out_dim: int | None = None


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.dino import _VARIANTS as variants

    return [f"dino_pointnet:{k}" for k in sorted(variants)] + ["dino_pointnet"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"dino_pointnet", "dino"}:
        from dlhub.pointcloud.selfsupervised.dino import build_dino_pointnet

        return build_dino_pointnet(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "dino_pointnet_small",
            dropout=float(cfg.dropout),
            out_dim=None if cfg.out_dim is None else int(cfg.out_dim),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: dino_pointnet:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]

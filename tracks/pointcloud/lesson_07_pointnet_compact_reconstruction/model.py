from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    num_points: int = 128
    arch: str = "pointnet_ae"
    variant: str = "pointnet_ae_small"
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.reconstruction.pointnet_ae import _VARIANTS as pn_variants

    return [f"pointnet_ae:{k}" for k in sorted(pn_variants)] + ["pointnet_ae"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"pointnet_ae", "pointnetae"}:
        from dlhub.pointcloud.reconstruction.pointnet_ae import build_pointnet_autoencoder

        return build_pointnet_autoencoder(
            in_channels=int(cfg.in_channels),
            num_points=int(cfg.num_points),
            variant=str(variant) if variant else "pointnet_ae_small",
            dropout=float(cfg.dropout),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: pointnet_ae:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]

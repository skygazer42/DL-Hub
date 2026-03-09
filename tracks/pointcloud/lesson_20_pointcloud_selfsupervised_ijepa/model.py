
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "ijepa_pointmae"
    variant: str = "ijepa_pointmae_small"
    in_channels: int = 3
    dropout: float = 0.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.ijepa import _VARIANTS as variants

    return [f"ijepa_pointmae:{k}" for k in sorted(variants)] + ["ijepa_pointmae"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"ijepa_pointmae", "ijepa"}:
        from dlhub.pointcloud.selfsupervised.ijepa import build_ijepa_pointmae

        return build_ijepa_pointmae(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "ijepa_pointmae_small",
            dropout=float(cfg.dropout),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: ijepa_pointmae:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


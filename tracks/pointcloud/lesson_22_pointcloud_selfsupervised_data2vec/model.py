from __future__ import annotations

from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "data2vec_pointmae"
    variant: str = "data2vec_pointmae_small"
    in_channels: int = 3
    dropout: float = 0.0
    predictor_hidden: int | None = None


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.selfsupervised.data2vec import _VARIANTS as variants

    return [f"data2vec_pointmae:{k}" for k in sorted(variants)] + ["data2vec_pointmae"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    if arch in {"data2vec_pointmae", "data2vec"}:
        from dlhub.pointcloud.selfsupervised.data2vec import build_data2vec_pointmae

        return build_data2vec_pointmae(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "data2vec_pointmae_small",
            dropout=float(cfg.dropout),
            predictor_hidden=None if cfg.predictor_hidden is None else int(cfg.predictor_hidden),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: data2vec_pointmae:<variant>")


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]


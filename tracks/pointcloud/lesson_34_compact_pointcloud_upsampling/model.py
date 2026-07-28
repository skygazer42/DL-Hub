from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "punet_upsample"
    variant: str = "punet_upsample_tiny"
    width_mult: float = 1.0


_ARCH_TO_MODULE = {
    "punet_upsample": "punet_upsample",
    "pugan_upsample": "pugan_upsample",
    "diffusion_upsample": "diffusion_upsample",
    "transformer_upsample": "transformer_upsample",
}


def list_supported_arches() -> list[str]:
    arches: list[str] = []
    for arch, module_name in sorted(_ARCH_TO_MODULE.items()):
        module = import_module(f"dlhub.pointcloud.pointcloud_upsampling.{module_name}")
        variants = getattr(module, "_VARIANTS", {})
        for key in sorted(variants):
            arches.append(f"{arch}:{key}")
    return arches


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        arch_name, variant_name = arch_raw.split(":", 1)
        arch = arch_name.strip()
        variant = variant_name.strip()

    if arch not in _ARCH_TO_MODULE:
        raise ValueError(
            f"Unknown arch: {arch_raw!r}. Supported prefixes: {', '.join(sorted(_ARCH_TO_MODULE))}"
        )

    module = import_module(f"dlhub.pointcloud.pointcloud_upsampling.{_ARCH_TO_MODULE[arch]}")
    builder = getattr(module, f"build_{arch}_upsampler")
    chosen_variant = variant or str(cfg.variant)
    return builder(
        in_channels=int(cfg.in_channels),
        variant=chosen_variant,
        width_mult=float(cfg.width_mult),
    )


__all__ = ["ModelConfig", "build_model", "list_supported_arches"]

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "dncnn"  # dncnn | restormer | noise2noise_unet | bm3d | ffdnet | nafnet | drunet | swinir | ridnet
    variant: str = "dncnn_9"
    in_channels: int = 1
    sigma: float = 0.1  # for BM3D baseline


def list_supported_arches() -> list[str]:
    from dlhub.vision.denoising.bm3d import _VARIANTS as bm3d_variants
    from dlhub.vision.denoising.dncnn import _VARIANTS as dncnn_variants
    from dlhub.vision.denoising.drunet import _VARIANTS as drunet_variants
    from dlhub.vision.denoising.ffdnet import _VARIANTS as ffdnet_variants
    from dlhub.vision.denoising.nafnet import _VARIANTS as nafnet_variants
    from dlhub.vision.denoising.noise2noise import _VARIANTS as n2n_variants
    from dlhub.vision.denoising.restormer import _VARIANTS as restormer_variants
    from dlhub.vision.denoising.ridnet import _VARIANTS as ridnet_variants
    from dlhub.vision.denoising.swinir import _VARIANTS as swinir_variants

    out: list[str] = []
    out.extend([f"dncnn:{k}" for k in sorted(dncnn_variants)])
    out.extend([f"restormer:{k}" for k in sorted(restormer_variants)])
    out.extend([f"nafnet:{k}" for k in sorted(nafnet_variants)])
    out.extend([f"swinir:{k}" for k in sorted(swinir_variants)])
    out.extend([f"ridnet:{k}" for k in sorted(ridnet_variants)])
    out.extend([f"ffdnet:{k}" for k in sorted(ffdnet_variants)])
    out.extend([f"drunet:{k}" for k in sorted(drunet_variants)])
    out.extend([f"noise2noise_unet:{k}" for k in sorted(n2n_variants)])
    out.extend([f"bm3d:{k}" for k in sorted(bm3d_variants)])
    return out


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()

    # Allow either "dncnn" + variant, or single string like "dncnn:dncnn_17".
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        pref, name = arch_raw.split(":", 1)
        arch = pref.strip().lower()
        variant = name.strip()

    in_channels = int(cfg.in_channels)

    if arch in {"dncnn"}:
        from dlhub.vision.denoising.dncnn import build_dncnn_denoiser

        return build_dncnn_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"restormer"}:
        from dlhub.vision.denoising.restormer import build_restormer_denoiser

        return build_restormer_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"nafnet"}:
        from dlhub.vision.denoising.nafnet import build_nafnet_denoiser

        return build_nafnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"swinir"}:
        from dlhub.vision.denoising.swinir import build_swinir_denoiser

        return build_swinir_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"ridnet"}:
        from dlhub.vision.denoising.ridnet import build_ridnet_denoiser

        return build_ridnet_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"ffdnet"}:
        from dlhub.vision.denoising.ffdnet import build_ffdnet_denoiser

        return build_ffdnet_denoiser(in_channels=in_channels, sigma=float(cfg.sigma), variant=variant)

    if arch in {"drunet"}:
        from dlhub.vision.denoising.drunet import build_drunet_denoiser

        return build_drunet_denoiser(in_channels=in_channels, sigma=float(cfg.sigma), variant=variant)

    if arch in {"noise2noise_unet", "n2n_unet", "noise2noise"}:
        from dlhub.vision.denoising.noise2noise import build_noise2noise_denoiser

        return build_noise2noise_denoiser(in_channels=in_channels, variant=variant)

    if arch in {"bm3d"}:
        from dlhub.vision.denoising.bm3d import build_bm3d_denoiser

        return build_bm3d_denoiser(in_channels=in_channels, sigma=float(cfg.sigma), variant=variant)

    raise ValueError(
        f"Unknown arch: {arch_raw!r}. Examples: dncnn:dncnn_17 | restormer:restormer_tiny | bm3d:bm3d_fast"
    )


class DenoiserAdapter(nn.Module):
    """Adapter to ensure a consistent forward signature (noisy -> denoised)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        if not isinstance(y, torch.Tensor):
            raise TypeError(f"Denoiser must return a Tensor, got: {type(y).__name__}")
        if y.shape != x.shape:
            raise ValueError(f"Denoiser output shape {tuple(y.shape)} must match input {tuple(x.shape)}")
        return y


__all__ = ["DenoiserAdapter", "ModelConfig", "build_model", "list_supported_arches"]

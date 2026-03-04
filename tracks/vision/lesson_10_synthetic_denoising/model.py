from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "dncnn"  # dncnn | restormer | noise2noise_unet | bm3d
    variant: str = "dncnn_9"
    in_channels: int = 1
    sigma: float = 0.1  # for BM3D baseline


def list_supported_arches() -> list[str]:
    from dlhub.vision.denoising.bm3d import _VARIANTS as bm3d_variants
    from dlhub.vision.denoising.dncnn import _VARIANTS as dncnn_variants
    from dlhub.vision.denoising.noise2noise import _VARIANTS as n2n_variants
    from dlhub.vision.denoising.restormer import _VARIANTS as restormer_variants

    out: list[str] = []
    out.extend([f"dncnn:{k}" for k in sorted(dncnn_variants)])
    out.extend([f"restormer:{k}" for k in sorted(restormer_variants)])
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


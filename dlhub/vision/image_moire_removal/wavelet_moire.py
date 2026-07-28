from __future__ import annotations

from torch import nn

from ._common import build_baseline_moire_remover, smoke_test_moire_remover


_VARIANTS: dict[str, dict[str, int]] = {
    "wavelet_moire_tiny": {"width": 24, "depth": 1, "passes": 1},
    "wavelet_moire_small": {"width": 36, "depth": 2, "passes": 2},
    "wavelet_moire_base": {"width": 48, "depth": 3, "passes": 2},
}


def build_wavelet_moire_moire_remover(
    *,
    in_channels: int,
    variant: str = "wavelet_moire_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_moire_remover(
        family="wavelet_moire",
        mode="wavelet",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_moire_remover(build_wavelet_moire_moire_remover, "wavelet_moire_tiny")

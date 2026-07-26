from __future__ import annotations

from torch import nn

from ._common import build_toy_exposure_corrector, smoke_test_exposure_corrector


_VARIANTS: dict[str, dict[str, int]] = {
    "retinex_exposure_tiny": {"width": 24, "depth": 1, "steps": 1},
    "retinex_exposure_small": {"width": 36, "depth": 2, "steps": 1},
    "retinex_exposure_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_retinex_exposure_exposure_corrector(
    *, in_channels: int, variant: str = "retinex_exposure_small", width_mult: float = 1.0
) -> nn.Module:
    return build_toy_exposure_corrector(
        family="retinex_exposure",
        mode="retinex",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_exposure_corrector(
        build_retinex_exposure_exposure_corrector, "retinex_exposure_tiny"
    )

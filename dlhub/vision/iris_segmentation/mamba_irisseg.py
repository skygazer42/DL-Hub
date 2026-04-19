from __future__ import annotations

from torch import nn

from ._common import build_toy_iris_segmentor, smoke_test_iris_segmentor


_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_irisseg_tiny": {"width": 24, "depth": 1},
    "mamba_irisseg_small": {"width": 36, "depth": 2},
    "mamba_irisseg_base": {"width": 48, "depth": 3},
}


def build_mamba_irisseg_iris_segmentor(
    *,
    in_channels: int,
    variant: str = "mamba_irisseg_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_iris_segmentor(
        family="mamba_irisseg",
        mode="mamba",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_iris_segmentor(build_mamba_irisseg_iris_segmentor, "mamba_irisseg_tiny")

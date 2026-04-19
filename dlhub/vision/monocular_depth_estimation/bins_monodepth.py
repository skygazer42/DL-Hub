from __future__ import annotations

from torch import nn

from ._common import build_toy_depth_model, smoke_test_depth_model


_VARIANTS: dict[str, dict[str, int]] = {'bins_monodepth_tiny': {'width': 24, 'depth': 1}, 'bins_monodepth_small': {'width': 36, 'depth': 2}, 'bins_monodepth_base': {'width': 48, 'depth': 3}}


def build_bins_monodepth_depth_model(
    *,
    in_channels: int,
    variant: str = 'bins_monodepth_small',
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_depth_model(
        family='bins_monodepth',
        mode='bins',
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_depth_model(build_bins_monodepth_depth_model, 'bins_monodepth_tiny')

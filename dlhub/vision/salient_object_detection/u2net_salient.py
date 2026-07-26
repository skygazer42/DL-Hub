from __future__ import annotations

from torch import nn

from ._common import build_toy_salient_detector, smoke_test_salient_detector


_VARIANTS: dict[str, dict[str, int]] = {
    "u2net_salient_tiny": {"width": 24, "depth": 1},
    "u2net_salient_small": {"width": 36, "depth": 2},
    "u2net_salient_base": {"width": 48, "depth": 3},
}


def build_u2net_salient_salient_detector(
    *,
    in_channels: int,
    variant: str = "u2net_salient_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_salient_detector(
        family="u2net_salient",
        mode="u2net",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_salient_detector(build_u2net_salient_salient_detector, "u2net_salient_tiny")

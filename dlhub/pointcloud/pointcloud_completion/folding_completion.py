from __future__ import annotations

from torch import nn

from ._common import build_toy_completer, smoke_test_completer

_VARIANTS: dict[str, dict[str, int]] = {
    "folding_completion_tiny": {"width": 48, "depth": 1, "points": 144},
    "folding_completion_small": {"width": 64, "depth": 2, "points": 192},
    "folding_completion_base": {"width": 96, "depth": 3, "points": 256},
}


def build_folding_completion_completer(
    *,
    in_channels: int,
    variant: str = "folding_completion_small",
    width_mult: float = 1.0,
    num_output_points: int | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_completer(
        family="folding_completion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_output_points=num_output_points,
        encoder_kind="pointnet",
        decoder_kind="fold",
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_completer(build_folding_completion_completer, "folding_completion_tiny")

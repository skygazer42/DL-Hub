from __future__ import annotations

from torch import nn

from ._common import build_toy_completer, smoke_test_completer

_VARIANTS: dict[str, dict[str, int]] = {
    "mamba_completion_tiny": {"width": 48, "depth": 1, "points": 128},
    "mamba_completion_small": {"width": 64, "depth": 2, "points": 192},
    "mamba_completion_base": {"width": 96, "depth": 3, "points": 256},
}


def build_mamba_completion_completer(
    *,
    in_channels: int,
    variant: str = "mamba_completion_small",
    width_mult: float = 1.0,
    num_output_points: int | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    return build_toy_completer(
        family="mamba_completion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_output_points=num_output_points,
        encoder_kind="state_space",
        decoder_kind="state_space",
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_completer(build_mamba_completion_completer, "mamba_completion_tiny")

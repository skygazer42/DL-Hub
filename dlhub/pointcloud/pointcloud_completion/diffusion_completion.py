from __future__ import annotations

from torch import nn

from ._common import build_baseline_completer, smoke_test_completer

_VARIANTS: dict[str, dict[str, int]] = {
    "diffusion_completion_tiny": {"width": 48, "depth": 1, "points": 128},
    "diffusion_completion_small": {"width": 64, "depth": 2, "points": 192},
    "diffusion_completion_base": {"width": 96, "depth": 3, "points": 256},
}


def build_diffusion_completion_completer(
    *,
    in_channels: int,
    variant: str = "diffusion_completion_small",
    width_mult: float = 1.0,
    num_output_points: int | None = None,
    dropout: float = 0.0,
) -> nn.Module:
    return build_baseline_completer(
        family="diffusion_completion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
        num_output_points=num_output_points,
        encoder_kind="transformer",
        decoder_kind="diffusion",
        dropout=float(dropout),
    )


if __name__ == "__main__":
    smoke_test_completer(build_diffusion_completion_completer, "diffusion_completion_tiny")

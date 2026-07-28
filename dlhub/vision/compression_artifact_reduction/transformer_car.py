from __future__ import annotations

from torch import nn

from ._common import build_baseline_artifact_reducer, smoke_test_artifact_reducer


_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_car_tiny": {"width": 24, "depth": 1, "steps": 1},
    "transformer_car_small": {"width": 36, "depth": 2, "steps": 1},
    "transformer_car_base": {"width": 48, "depth": 3, "steps": 2},
}


def build_transformer_car_artifact_reducer(
    *,
    in_channels: int,
    variant: str = "transformer_car_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_artifact_reducer(
        family="transformer_car",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_artifact_reducer(build_transformer_car_artifact_reducer, "transformer_car_tiny")

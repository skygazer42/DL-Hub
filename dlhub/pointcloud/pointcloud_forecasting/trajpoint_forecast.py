from __future__ import annotations

from torch import nn

from ._common import build_toy_forecasting_model, smoke_test_forecasting_model


_VARIANTS: dict[str, dict[str, int]] = {
    "trajpoint_forecast_tiny": {"width": 24, "depth": 1, "horizon": 2},
    "trajpoint_forecast_small": {"width": 32, "depth": 2, "horizon": 3},
    "trajpoint_forecast_base": {"width": 48, "depth": 3, "horizon": 4},
}


def build_trajpoint_forecast_forecasting_model(
    *,
    in_channels: int,
    variant: str = "trajpoint_forecast_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_forecasting_model(
        family="trajpoint_forecast",
        mode="traj",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_forecasting_model(build_trajpoint_forecast_forecasting_model, "trajpoint_forecast_tiny")

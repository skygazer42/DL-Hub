from __future__ import annotations

from torch import nn

from ._common import build_toy_forecasting_model, smoke_test_forecasting_model


_VARIANTS: dict[str, dict[str, int]] = {
    "motion_field_forecast_tiny": {"width": 24, "depth": 1, "horizon": 2},
    "motion_field_forecast_small": {"width": 32, "depth": 2, "horizon": 3},
    "motion_field_forecast_base": {"width": 48, "depth": 3, "horizon": 5},
}


def build_motion_field_forecast_forecasting_model(
    *,
    in_channels: int,
    variant: str = "motion_field_forecast_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_forecasting_model(
        family="motion_field_forecast",
        mode="motion",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_forecasting_model(
        build_motion_field_forecast_forecasting_model, "motion_field_forecast_tiny"
    )

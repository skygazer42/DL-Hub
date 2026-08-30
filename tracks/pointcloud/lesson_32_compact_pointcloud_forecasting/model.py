from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    forecast_horizon: int = 2
    arch: str = "trajpoint_forecast:trajpoint_forecast_small"
    variant: str = ""
    width_mult: float = 1.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.pointcloud_forecasting.trajpoint_forecast import (
        _VARIANTS as traj_variants,
    )

    return [f"trajpoint_forecast:{name}" for name in sorted(traj_variants)] + ["trajpoint_forecast"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        prefix, name = arch_raw.split(":", 1)
        arch = prefix.strip().lower()
        variant = name.strip()

    if arch in {"trajpoint_forecast", "forecasting"}:
        from dlhub.pointcloud.pointcloud_forecasting.trajpoint_forecast import (
            build_trajpoint_forecast_forecasting_model,
        )

        forecast_horizon = int(cfg.forecast_horizon)
        if forecast_horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")
        model = build_trajpoint_forecast_forecasting_model(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "trajpoint_forecast_small",
            width_mult=float(cfg.width_mult),
        )
        model.horizon = forecast_horizon
        return model

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: trajpoint_forecast:<variant>")


def forecasting_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    future = targets["future"].to(torch.float32)
    pred = outputs["forecast"].to(torch.float32)
    forecast_loss = torch.nn.functional.l1_loss(pred, future)
    step_mae = torch.abs(pred - future).mean()
    return forecast_loss, {
        "forecast_loss": float(forecast_loss.detach().item()),
        "step_mae": float(step_mae.detach().item()),
    }


__all__ = ["ModelConfig", "build_model", "forecasting_loss", "list_supported_arches"]

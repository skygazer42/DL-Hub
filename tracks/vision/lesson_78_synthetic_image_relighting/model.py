from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import dlhub.vision.image_relighting as image_relighting


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "deep_relight:deep_relight_small"
    width_mult: float = 1.0


def _parse_arch(arch: str) -> tuple[str, str]:
    family, sep, variant = str(arch).partition(":")
    if not family or not sep or not variant:
        raise ValueError("arch must look like '<family>:<variant>'")
    return family, variant


def _build_relighter(cfg: ModelConfig) -> nn.Module:
    family, variant = _parse_arch(cfg.arch)
    builder_name = f"build_{family}_relighter"
    builder = getattr(image_relighting, builder_name)
    return builder(
        in_channels=int(cfg.in_channels),
        variant=str(variant),
        width_mult=float(cfg.width_mult),
    )


class RelightingModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.net = _build_relighter(cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self.net(x.to(torch.float32))
        relit = outputs["relit"]
        light_map = outputs["light_map"]
        residual = outputs["residual"]
        return {"relit": relit, "light_map": light_map, "residual": residual}


def relighting_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    relit_loss = torch.nn.functional.l1_loss(
        outputs["relit"],
        targets["relit"].to(torch.float32),
    )
    light_map_loss = torch.nn.functional.l1_loss(
        outputs["light_map"],
        targets["light_map"].to(torch.float32),
    )
    total_loss = relit_loss + 0.2 * light_map_loss
    return total_loss, {
        "relit_loss": float(relit_loss.item()),
        "light_map_loss": float(light_map_loss.item()),
    }


def build_model(cfg: ModelConfig) -> RelightingModel:
    return RelightingModel(cfg)


__all__ = ["ModelConfig", "RelightingModel", "build_model", "relighting_loss"]

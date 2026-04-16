from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    latent_dim: int = 64
    family: str = "dreamfusion_toy"
    variant: str = "dreamfusion_toy_tiny"
    width_mult: float = 1.0


_FAMILY_TO_BUILDER = {
    "dreamfusion_toy": "build_dreamfusion_toy_text3d_generator",
    "gaussian_text3d": "build_gaussian_text3d_text3d_generator",
    "magic3d_toy": "build_magic3d_toy_text3d_generator",
}


def _build_generator(cfg: ModelConfig) -> nn.Module:
    builders = import_module("dlhub.generative.text_to_3d")
    if cfg.family not in _FAMILY_TO_BUILDER:
        raise ValueError(f"Unsupported family: {cfg.family}")
    builder = getattr(builders, _FAMILY_TO_BUILDER[cfg.family])
    return builder(
        in_channels=int(cfg.in_channels),
        latent_dim=int(cfg.latent_dim),
        variant=str(cfg.variant),
        width_mult=float(cfg.width_mult),
    )


class ToyTextTo3DModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = _build_generator(cfg)

    def forward(self, text_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if text_features.ndim != 2 or text_features.shape[1] != 32:
            raise ValueError(f"Expected text features with shape (B, 32), got {tuple(text_features.shape)}")
        outputs = self.generator.forward(
            batch_size=int(text_features.shape[0]),
            device=text_features.device,
            text=text_features.to(dtype=torch.float32),
        )
        return {
            "triplanes": outputs["triplanes"],
            "density": outputs["density"],
            "mesh_tokens": outputs["mesh_tokens"],
        }


def text_to_3d_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    density_loss = torch.nn.functional.mse_loss(outputs["density"], targets["density"])
    mesh_loss = torch.nn.functional.l1_loss(outputs["mesh_tokens"], targets["mesh_tokens"])
    loss = density_loss + 0.5 * mesh_loss
    parts = {"density_loss": float(density_loss.item()), "mesh_loss": float(mesh_loss.item())}
    return loss, parts


__all__ = ["ModelConfig", "ToyTextTo3DModel", "text_to_3d_loss"]

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.generative.image_to_3d import (
    build_diffusion_i23d_image_to_3d_generator,
    build_gaussian_i23d_image_to_3d_generator,
    build_lift3d_i23d_image_to_3d_generator,
    build_mamba_i23d_image_to_3d_generator,
    build_mesh_i23d_image_to_3d_generator,
    build_prompt_i23d_image_to_3d_generator,
    build_sdf_i23d_image_to_3d_generator,
    build_transformer_i23d_image_to_3d_generator,
    build_triplane_i23d_image_to_3d_generator,
    build_zero123_baseline_image_to_3d_generator,
)


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    family: str = "zero123_baseline"
    variant: str = "zero123_baseline_tiny"
    width_mult: float = 1.0


_BUILDERS: dict[str, callable] = {
    "zero123_baseline": build_zero123_baseline_image_to_3d_generator,
    "gaussian_i23d": build_gaussian_i23d_image_to_3d_generator,
    "diffusion_i23d": build_diffusion_i23d_image_to_3d_generator,
    "triplane_i23d": build_triplane_i23d_image_to_3d_generator,
    "sdf_i23d": build_sdf_i23d_image_to_3d_generator,
    "mesh_i23d": build_mesh_i23d_image_to_3d_generator,
    "mamba_i23d": build_mamba_i23d_image_to_3d_generator,
    "lift3d_i23d": build_lift3d_i23d_image_to_3d_generator,
    "transformer_i23d": build_transformer_i23d_image_to_3d_generator,
    "prompt_i23d": build_prompt_i23d_image_to_3d_generator,
}


class CompactImageTo3DModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        family = str(cfg.family)
        if family not in _BUILDERS:
            raise ValueError(f"Unsupported family: {family}")
        self.backbone = _BUILDERS[family](
            in_channels=int(cfg.in_channels),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(image)


def _match_density_shape(pred_density: torch.Tensor, target_density: torch.Tensor) -> torch.Tensor:
    if pred_density.shape == target_density.shape:
        return target_density
    return F.interpolate(
        target_density,
        size=pred_density.shape[2:],
        mode="trilinear",
        align_corners=False,
    )


def _match_mesh_shape(pred_mesh: torch.Tensor, target_mesh: torch.Tensor) -> torch.Tensor:
    if pred_mesh.shape == target_mesh.shape:
        return target_mesh
    resized = F.interpolate(
        target_mesh.transpose(1, 2),
        size=pred_mesh.shape[1],
        mode="linear",
        align_corners=False,
    )
    return resized.transpose(1, 2)


def image_to_3d_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_density = outputs["density"]
    pred_mesh = outputs["mesh_tokens"]
    target_density = _match_density_shape(pred_density, targets["density"])
    target_mesh = _match_mesh_shape(pred_mesh, targets["mesh_tokens"])

    density_loss = F.l1_loss(pred_density, target_density)
    mesh_loss = F.mse_loss(pred_mesh, target_mesh)
    loss = density_loss + 0.3 * mesh_loss
    return loss, {
        "density_loss": float(density_loss.detach().item()),
        "mesh_loss": float(mesh_loss.detach().item()),
    }


__all__ = ["ModelConfig", "CompactImageTo3DModel", "image_to_3d_loss"]

from __future__ import annotations

from torch import nn

from ._common import build_toy_world_model, smoke_test_world_model

_VARIANTS: dict[str, dict[str, int]] = {
    "diffusion_world_tiny": {"width": 52, "depth": 2, "latent": 68, "action": 4, "context": 14},
    "diffusion_world_small": {"width": 76, "depth": 3, "latent": 92, "action": 6, "context": 18},
    "diffusion_world_base": {"width": 100, "depth": 4, "latent": 116, "action": 8, "context": 26},
}


def build_diffusion_world_world_model(*, in_channels: int = 3, action_dim: int = 4, context_dim: int = 14, variant: str = "diffusion_world_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_world_model(family="diffusion_world", mode="diffusion", variants=_VARIANTS, in_channels=int(in_channels), action_dim=int(action_dim), context_dim=int(context_dim), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_world_model(build_diffusion_world_world_model, "diffusion_world_tiny")

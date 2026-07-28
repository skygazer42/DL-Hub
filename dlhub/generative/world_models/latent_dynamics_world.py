from __future__ import annotations

from torch import nn

from ._common import build_baseline_world_model, smoke_test_world_model

_VARIANTS: dict[str, dict[str, int]] = {
    "latent_dynamics_world_tiny": {
        "width": 48,
        "depth": 2,
        "latent": 56,
        "action": 5,
        "context": 12,
    },
    "latent_dynamics_world_small": {
        "width": 72,
        "depth": 3,
        "latent": 84,
        "action": 7,
        "context": 16,
    },
    "latent_dynamics_world_base": {
        "width": 96,
        "depth": 4,
        "latent": 112,
        "action": 9,
        "context": 24,
    },
}


def build_latent_dynamics_world_world_model(
    *,
    in_channels: int = 3,
    action_dim: int = 5,
    context_dim: int = 12,
    variant: str = "latent_dynamics_world_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_world_model(
        family="latent_dynamics_world",
        mode="latent_dynamics",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        action_dim=int(action_dim),
        context_dim=int(context_dim),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_world_model(build_latent_dynamics_world_world_model, "latent_dynamics_world_tiny")

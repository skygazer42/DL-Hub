from __future__ import annotations

from torch import nn

from ._common import build_toy_world_model, smoke_test_world_model

_VARIANTS: dict[str, dict[str, int]] = {
    "dreamer_world_tiny": {"width": 56, "depth": 2, "latent": 56, "action": 4, "context": 16},
    "dreamer_world_small": {"width": 80, "depth": 3, "latent": 80, "action": 6, "context": 20},
    "dreamer_world_base": {"width": 104, "depth": 4, "latent": 104, "action": 8, "context": 28},
}


def build_dreamer_world_world_model(
    *,
    in_channels: int = 3,
    action_dim: int = 4,
    context_dim: int = 16,
    variant: str = "dreamer_world_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_world_model(
        family="dreamer_world",
        mode="dreamer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        action_dim=int(action_dim),
        context_dim=int(context_dim),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_world_model(build_dreamer_world_world_model, "dreamer_world_tiny")

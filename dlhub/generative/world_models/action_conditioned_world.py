from __future__ import annotations

from torch import nn

from ._common import build_toy_world_model, smoke_test_world_model

_VARIANTS: dict[str, dict[str, int]] = {
    "action_conditioned_world_tiny": {"width": 52, "depth": 2, "latent": 52, "action": 6, "context": 10},
    "action_conditioned_world_small": {"width": 76, "depth": 3, "latent": 76, "action": 8, "context": 14},
    "action_conditioned_world_base": {"width": 100, "depth": 4, "latent": 100, "action": 10, "context": 18},
}


def build_action_conditioned_world_world_model(*, in_channels: int = 3, action_dim: int = 6, context_dim: int = 10, variant: str = "action_conditioned_world_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_world_model(family="action_conditioned_world", mode="action_conditioned", variants=_VARIANTS, in_channels=int(in_channels), action_dim=int(action_dim), context_dim=int(context_dim), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_world_model(build_action_conditioned_world_world_model, "action_conditioned_world_tiny")

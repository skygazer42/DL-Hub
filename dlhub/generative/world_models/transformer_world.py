from __future__ import annotations

from torch import nn

from ._common import build_baseline_world_model, smoke_test_world_model

_VARIANTS: dict[str, dict[str, int]] = {
    "transformer_world_tiny": {"width": 56, "depth": 2, "latent": 64, "action": 4, "context": 16},
    "transformer_world_small": {"width": 80, "depth": 3, "latent": 96, "action": 6, "context": 20},
    "transformer_world_base": {"width": 104, "depth": 4, "latent": 128, "action": 8, "context": 28},
}


def build_transformer_world_world_model(
    *,
    in_channels: int = 3,
    action_dim: int = 4,
    context_dim: int = 16,
    variant: str = "transformer_world_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_world_model(
        family="transformer_world",
        mode="transformer",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        action_dim=int(action_dim),
        context_dim=int(context_dim),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_world_model(build_transformer_world_world_model, "transformer_world_tiny")

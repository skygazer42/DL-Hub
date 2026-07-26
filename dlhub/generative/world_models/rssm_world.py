from __future__ import annotations

from torch import nn

from ._common import build_toy_world_model, smoke_test_world_model

_VARIANTS: dict[str, dict[str, int]] = {
    "rssm_world_tiny": {"width": 48, "depth": 2, "latent": 48, "action": 4, "context": 12},
    "rssm_world_small": {"width": 72, "depth": 3, "latent": 72, "action": 6, "context": 16},
    "rssm_world_base": {"width": 96, "depth": 4, "latent": 96, "action": 8, "context": 24},
}


def build_rssm_world_world_model(
    *,
    in_channels: int = 3,
    action_dim: int = 4,
    context_dim: int = 12,
    variant: str = "rssm_world_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_toy_world_model(
        family="rssm_world",
        mode="rssm",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        action_dim=int(action_dim),
        context_dim=int(context_dim),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_world_model(build_rssm_world_world_model, "rssm_world_tiny")

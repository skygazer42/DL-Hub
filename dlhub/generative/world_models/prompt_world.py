from __future__ import annotations

from torch import nn

from ._common import build_toy_world_model, smoke_test_world_model

_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_world_tiny": {"width": 48, "depth": 2, "latent": 56, "action": 4, "context": 24},
    "prompt_world_small": {"width": 72, "depth": 3, "latent": 80, "action": 6, "context": 32},
    "prompt_world_base": {"width": 96, "depth": 4, "latent": 104, "action": 8, "context": 40},
}


def build_prompt_world_world_model(*, in_channels: int = 3, action_dim: int = 4, context_dim: int = 24, variant: str = "prompt_world_small", width_mult: float = 1.0) -> nn.Module:
    return build_toy_world_model(family="prompt_world", mode="prompt", variants=_VARIANTS, in_channels=int(in_channels), action_dim=int(action_dim), context_dim=int(context_dim), variant=str(variant), width_mult=float(width_mult))


if __name__ == "__main__":
    smoke_test_world_model(build_prompt_world_world_model, "prompt_world_tiny")

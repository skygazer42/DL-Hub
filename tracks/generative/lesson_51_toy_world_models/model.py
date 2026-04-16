from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    action_dim: int = 4
    context_dim: int = 12
    family: str = "rssm_world"
    variant: str = "rssm_world_tiny"
    width_mult: float = 1.0


def _build_model(cfg: ModelConfig) -> nn.Module:
    family = str(cfg.family)
    builders = {
        "rssm_world": "build_rssm_world_world_model",
        "dreamer_world": "build_dreamer_world_world_model",
        "transformer_world": "build_transformer_world_world_model",
        "action_conditioned_world": "build_action_conditioned_world_world_model",
        "diffusion_world": "build_diffusion_world_world_model",
        "memory_world": "build_memory_world_world_model",
        "video_world": "build_video_world_world_model",
        "latent_dynamics_world": "build_latent_dynamics_world_world_model",
        "mamba_world": "build_mamba_world_world_model",
        "prompt_world": "build_prompt_world_world_model",
    }
    if family not in builders:
        known = ", ".join(sorted(builders.keys()))
        raise ValueError(f"Unsupported family '{family}'. Available: {known}")

    from dlhub.generative import world_models as world_models_pkg

    builder = getattr(world_models_pkg, builders[family])
    return builder(
        in_channels=int(cfg.in_channels),
        action_dim=int(cfg.action_dim),
        context_dim=int(cfg.context_dim),
        variant=str(cfg.variant),
        width_mult=float(cfg.width_mult),
    )


class ToyWorldModelsModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = _build_model(cfg)

    def forward(
        self,
        *,
        obs: torch.Tensor,
        action: torch.Tensor,
        prompt: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.model(obs, action, prompt)


def world_models_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_recon = torch.nn.functional.interpolate(
        targets["next_obs"].to(torch.float32),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )
    reconstruction_loss = torch.nn.functional.mse_loss(outputs["reconstruction"], target_recon)
    reward_loss = torch.nn.functional.mse_loss(outputs["reward"], targets["reward"].to(torch.float32))
    done_loss = torch.nn.functional.binary_cross_entropy(outputs["done"], targets["done"].to(torch.float32))
    total = reconstruction_loss + reward_loss + 0.5 * done_loss
    return total, {
        "reconstruction_loss": float(reconstruction_loss.detach().item()),
        "reward_loss": float(reward_loss.detach().item()),
        "done_loss": float(done_loss.detach().item()),
    }


def reward_mae(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> float:
    pred = outputs["reward"].to(torch.float32)
    truth = targets["reward"].to(torch.float32)
    return float((pred - truth).abs().mean().item())


__all__ = ["ModelConfig", "ToyWorldModelsModel", "reward_mae", "world_models_loss"]


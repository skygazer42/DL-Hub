from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _check_obs(obs: torch.Tensor) -> torch.Tensor:
    obs = obs.to(torch.float32)
    if obs.ndim != 4:
        raise ValueError(f"Expected obs shape (B, C, H, W), got {tuple(obs.shape)}")
    return obs


class ToyWorldModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        action_dim: int,
        context_dim: int,
        width: int,
        depth: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.in_channels = int(in_channels)
        self.action_dim = int(action_dim)
        self.context_dim = int(context_dim)
        self.latent_dim = int(latent_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, int(width), kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(int(width), int(width), kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        fused_dim = int(width) * 16
        self.obs_proj = nn.Linear(fused_dim, self.latent_dim)
        self.action_proj = nn.Linear(self.action_dim, self.latent_dim)
        self.prompt_proj = nn.Linear(self.context_dim, self.latent_dim)
        blocks: list[nn.Module] = []
        for _ in range(max(1, int(depth))):
            blocks.extend(
                [
                    nn.Linear(self.latent_dim, self.latent_dim),
                    nn.GELU(),
                ]
            )
        self.transition = nn.Sequential(*blocks)
        self.state_head = nn.Linear(self.latent_dim, self.latent_dim)
        self.reward_head = nn.Linear(self.latent_dim, 1)
        self.done_head = nn.Linear(self.latent_dim, 1)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, fused_dim),
            nn.GELU(),
        )
        self.recon_head = nn.Conv2d(int(width), self.in_channels, kernel_size=1)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        prompt: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        obs = _check_obs(obs)
        batch = int(obs.shape[0])
        dev = obs.device
        feat = self.pool(self.stem(obs)).flatten(1)
        latent = self.obs_proj(feat)
        if action is None:
            action = torch.zeros(batch, self.action_dim, device=dev, dtype=obs.dtype)
        elif tuple(action.shape) != (batch, self.action_dim):
            raise ValueError(f"Expected action shape {(batch, self.action_dim)}, got {tuple(action.shape)}")
        if prompt is None:
            prompt = torch.zeros(batch, self.context_dim, device=dev, dtype=obs.dtype)
        elif tuple(prompt.shape) != (batch, self.context_dim):
            raise ValueError(f"Expected prompt shape {(batch, self.context_dim)}, got {tuple(prompt.shape)}")
        fused = latent + self.action_proj(action) + self.prompt_proj(prompt)
        hidden = self.transition(fused)
        next_state = torch.tanh(self.state_head(hidden))
        reward = self.reward_head(hidden)
        done = torch.sigmoid(self.done_head(hidden))
        recon_feat = self.decoder(hidden).view(batch, -1, 4, 4)
        reconstruction = torch.tanh(self.recon_head(recon_feat))
        return {
            "latent": latent,
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "reconstruction": reconstruction,
        }


def build_toy_world_model(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    action_dim: int,
    context_dim: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    return ToyWorldModel(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        action_dim=max(int(action_dim), int(cfg.get("action", action_dim))),
        context_dim=max(int(context_dim), int(cfg.get("context", context_dim))),
        width=width,
        depth=int(cfg["depth"]),
        latent_dim=int(cfg["latent"]),
    )


def smoke_test_world_model(builder: Callable[..., nn.Module], variant: str) -> None:
    model = builder(variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 16, 16))
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    print(variant, shapes)


__all__ = ["ToyWorldModel", "build_toy_world_model", "smoke_test_world_model"]

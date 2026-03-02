from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    z_dim: int = 64
    hidden_dim: int = 256


class Generator(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        out_dim = 28 * 28
        self.net = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim * 2, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.net(z)
        images = torch.sigmoid(logits).view(-1, 1, 28, 28)
        return images


class Discriminator(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_dim = 28 * 28
        self.net = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images.view(images.size(0), -1)
        return self.net(x).view(-1)


class GAN(nn.Module):
    """A simple wrapper so checkpoints store both G and D weights."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.generator = Generator(cfg)
        self.discriminator = Discriminator(cfg)


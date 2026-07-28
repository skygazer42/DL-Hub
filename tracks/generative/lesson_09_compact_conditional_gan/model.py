from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    z_dim: int = 64
    hidden_dim: int = 128
    num_classes: int = 4
    image_size: int = 28


class ConditionalGenerator(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        out_dim = int(cfg.image_size) * int(cfg.image_size)
        self.label_embed = nn.Embedding(int(cfg.num_classes), int(cfg.z_dim))
        self.net = nn.Sequential(
            nn.Linear(int(cfg.z_dim) * 2, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim) * 2),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim) * 2, out_dim),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.label_embed(labels)
        logits = self.net(torch.cat([z, cond], dim=1))
        size = int(self.cfg.image_size)
        return torch.sigmoid(logits).view(-1, 1, size, size)


class ConditionalDiscriminator(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_dim = int(cfg.image_size) * int(cfg.image_size)
        self.label_embed = nn.Embedding(int(cfg.num_classes), in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, int(cfg.hidden_dim) * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(int(cfg.hidden_dim) * 2, int(cfg.hidden_dim)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        x = images.view(images.size(0), -1)
        cond = self.label_embed(labels)
        return self.net(torch.cat([x, cond], dim=1)).view(-1)


class ConditionalGAN(nn.Module):
    """Wrapper to checkpoint generator and discriminator together."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.generator = ConditionalGenerator(cfg)
        self.discriminator = ConditionalDiscriminator(cfg)

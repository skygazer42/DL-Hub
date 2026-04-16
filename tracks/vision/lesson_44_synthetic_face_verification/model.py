from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    hidden_channels: int = 24
    num_blocks: int = 3
    embedding_dim: int = 64
    dropout: float = 0.1


class FaceVerificationModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        in_ch = int(cfg.in_channels)
        hidden = int(cfg.hidden_channels)
        blocks: list[nn.Module] = []
        for idx in range(int(cfg.num_blocks)):
            out_ch = hidden * (2**idx)
            blocks.extend(
                [
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*blocks)
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_ch, int(cfg.embedding_dim)),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(int(cfg.embedding_dim) * 2, 2),
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(images.to(torch.float32)))

    def forward(self, images_a: torch.Tensor, images_b: torch.Tensor) -> torch.Tensor:
        emb_a = self.encode(images_a)
        emb_b = self.encode(images_b)
        features = torch.cat([torch.abs(emb_a - emb_b), emb_a * emb_b], dim=1)
        return self.classifier(features)


def verification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, labels)


def verification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        return float((logits.argmax(dim=1) == labels).to(torch.float32).mean().item())


__all__ = ["FaceVerificationModel", "ModelConfig", "verification_loss", "verification_accuracy"]

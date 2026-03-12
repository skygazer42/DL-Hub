from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def dynamic_threshold(x: torch.Tensor, *, percentile: float = 0.95) -> torch.Tensor:
    flat = x.abs().flatten(start_dim=1)
    scale = torch.quantile(flat, q=float(percentile), dim=1, keepdim=True).clamp_min(1.0)
    scale = scale.view(x.shape[0], *([1] * (x.ndim - 1)))
    return (x / scale).clamp(-1.0, 1.0)


class FrozenTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(hidden_size))
        self.proj = nn.Linear(int(hidden_size), int(hidden_size))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids.to(torch.long))
        return self.proj(embedded)


class CrossAttentionBlock(nn.Module):
    def __init__(self, channels: int, text_hidden_size: int) -> None:
        super().__init__()
        self.query = nn.Linear(int(channels), int(channels))
        self.key = nn.Linear(int(text_hidden_size), int(channels))
        self.value = nn.Linear(int(text_hidden_size), int(channels))

    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        q = self.query(tokens)
        k = self.key(text_embeddings)
        v = self.value(text_embeddings)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (channels ** -0.5)
        probs = torch.softmax(scores, dim=-1)
        attended = torch.matmul(probs, v)
        attended = attended.view(batch, height, width, channels).permute(0, 3, 1, 2)
        return x + attended


class ImagenUNet(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, text_hidden_size: int) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1)
        self.cross_attention = CrossAttentionBlock(int(hidden_channels), int(text_hidden_size))
        self.out_conv = nn.Conv2d(int(hidden_channels), 3, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.in_conv(x))
        h = self.cross_attention(h, text_embeddings)
        return self.out_conv(self.activation(h))


class ImagenDiffusionStage(nn.Module):
    uses_classifier_free_guidance = True

    def __init__(
        self,
        image_size: int,
        base_channels: int,
        text_hidden_size: int,
        *,
        cond_drop_prob: float,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.cond_drop_prob = float(cond_drop_prob)
        self.unet = ImagenUNet(3, int(base_channels), int(text_hidden_size))

    def _drop_condition(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cond_drop_prob <= 0.0:
            return text_embeddings
        if self.cond_drop_prob >= 1.0:
            return torch.zeros_like(text_embeddings)
        keep_mask = (
            torch.rand((text_embeddings.shape[0], 1, 1), device=text_embeddings.device)
            >= self.cond_drop_prob
        )
        return text_embeddings * keep_mask.to(dtype=text_embeddings.dtype)

    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        return self.unet(x, self._drop_condition(text_embeddings))


class ImagenSuperResolutionStage(nn.Module):
    def __init__(self, target_size: int, text_hidden_size: int, *, cond_drop_prob: float) -> None:
        super().__init__()
        self.target_size = int(target_size)
        self.cond_drop_prob = float(cond_drop_prob)
        self.condition_proj = nn.Linear(int(text_hidden_size), 3)
        self.refine = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def _drop_condition(self, text_embeddings: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cond_drop_prob <= 0.0:
            return text_embeddings
        if self.cond_drop_prob >= 1.0:
            return torch.zeros_like(text_embeddings)
        keep_mask = (
            torch.rand((text_embeddings.shape[0], 1, 1), device=text_embeddings.device)
            >= self.cond_drop_prob
        )
        return text_embeddings * keep_mask.to(dtype=text_embeddings.dtype)

    def forward(self, x: torch.Tensor, text_embeddings: torch.Tensor) -> torch.Tensor:
        upsampled = F.interpolate(
            x,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )
        text_embeddings = self._drop_condition(text_embeddings)
        conditioning = self.condition_proj(text_embeddings.mean(dim=1)).view(x.shape[0], 3, 1, 1)
        return self.refine(upsampled + conditioning)


@dataclass(frozen=True)
class ImagenConfig:
    text_vocab_size: int
    text_hidden_size: int = 512
    base_channels: int = 64
    image_size: int = 64
    superres_sizes: tuple[int, ...] = (128, 256)
    cond_drop_prob: float = 0.1


class ImagenModel(nn.Module):
    uses_dynamic_thresholding = True

    def __init__(self, config: ImagenConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = FrozenTextEncoder(
            int(config.text_vocab_size),
            int(config.text_hidden_size),
        )
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.base_diffusion = ImagenDiffusionStage(
            int(config.image_size),
            int(config.base_channels),
            int(config.text_hidden_size),
            cond_drop_prob=float(config.cond_drop_prob),
        )
        self.super_resolution_models = nn.ModuleList(
            [
                ImagenSuperResolutionStage(
                    int(size),
                    int(config.text_hidden_size),
                    cond_drop_prob=float(config.cond_drop_prob),
                )
                for size in config.superres_sizes
            ]
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        text_embeddings = self.text_encoder(token_ids)
        batch_size = token_ids.shape[0]
        images = token_ids.new_zeros(
            (batch_size, 3, int(self.config.image_size), int(self.config.image_size)),
            dtype=torch.float32,
        )
        images = self.base_diffusion(images, text_embeddings)
        images = dynamic_threshold(images)
        for stage in self.super_resolution_models:
            images = stage(images, text_embeddings)
            images = dynamic_threshold(images)
        return images


__all__ = [
    "ImagenConfig",
    "ImagenDiffusionStage",
    "ImagenModel",
    "ImagenSuperResolutionStage",
    "ImagenUNet",
    "dynamic_threshold",
]

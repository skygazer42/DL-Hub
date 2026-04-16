from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyVisionEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(16, int(width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected image shape (B, C, H, W), got {tuple(image.shape)}")
        return self.net(image.to(torch.float32))


class TextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(width), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(width), int(width))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")
        token_embed = self.embedding(input_ids.to(torch.long))
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (token_embed * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_text_length: int
    image_size: int
    embed_dim: int = 32
    vision_width: int = 32
    text_width: int = 32
    init_temperature: float = 0.07


class ToyPersonSearchModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = TinyVisionEncoder(width=int(cfg.vision_width))
        self.text_encoder = TextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            width=int(cfg.text_width),
        )
        self.image_projection = nn.Linear(int(cfg.vision_width), int(cfg.embed_dim), bias=False)
        self.text_projection = nn.Linear(int(cfg.text_width), int(cfg.embed_dim), bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(cfg.init_temperature))))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image = batch["image"].to(torch.float32)
        input_ids = batch["input_ids"].to(torch.long)
        attention_mask = batch["attention_mask"].to(torch.float32)

        if int(input_ids.shape[1]) != int(self.cfg.max_text_length):
            raise ValueError(
                "input_ids length does not match model config: "
                f"{int(input_ids.shape[1])} != {int(self.cfg.max_text_length)}"
            )

        image_features = self.vision_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)
        image_embed = F.normalize(self.image_projection(image_features), dim=-1)
        text_embed = F.normalize(self.text_projection(text_features), dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_embed @ text_embed.transpose(0, 1)
        logits_per_text = logits_per_image.transpose(0, 1)
        return {
            "image_embed": image_embed,
            "text_embed": text_embed,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
        }


def person_search_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    batch_size = int(logits_per_image.shape[0])
    targets = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)


@torch.no_grad()
def retrieval_accuracy(
    logits_per_image: torch.Tensor, logits_per_text: torch.Tensor
) -> tuple[float, float]:
    targets = torch.arange(int(logits_per_image.shape[0]), device=logits_per_image.device)
    image_to_text = (logits_per_image.argmax(dim=-1) == targets).to(torch.float32).mean()
    text_to_image = (logits_per_text.argmax(dim=-1) == targets).to(torch.float32).mean()
    return float(image_to_text.item()), float(text_to_image.item())


@torch.no_grad()
def recall_at_k(
    logits_per_image: torch.Tensor, logits_per_text: torch.Tensor, *, k: int
) -> tuple[float, float]:
    topk_image = logits_per_image.topk(k=int(k), dim=-1).indices
    topk_text = logits_per_text.topk(k=int(k), dim=-1).indices
    targets = torch.arange(int(logits_per_image.shape[0]), device=logits_per_image.device).unsqueeze(1)
    image_to_text = topk_image.eq(targets).any(dim=-1).to(torch.float32).mean()
    text_to_image = topk_text.eq(targets).any(dim=-1).to(torch.float32).mean()
    return float(image_to_text.item()), float(text_to_image.item())


__all__ = [
    "ModelConfig",
    "TextEncoder",
    "TinyVisionEncoder",
    "ToyPersonSearchModel",
    "person_search_loss",
    "recall_at_k",
    "retrieval_accuracy",
]

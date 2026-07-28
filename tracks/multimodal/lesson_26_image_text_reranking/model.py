from __future__ import annotations

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


class CandidateTextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, pad_id: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(width), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(width), int(width))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 3:
            raise ValueError(
                f"Expected input_ids shape (B, K, T), got {tuple(input_ids.shape)}"
            )
        token_embed = self.embedding(input_ids.to(torch.long))
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (token_embed * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)
        return self.proj(pooled)


@dataclass(frozen=True)
class ImageTextRerankerConfig:
    vocab_size: int
    pad_id: int
    num_candidates: int
    max_text_length: int
    embed_dim: int = 32
    vision_width: int = 32
    text_width: int = 32
    hidden_dim: int = 48


class CompactImageTextReranker(nn.Module):
    def __init__(self, cfg: ImageTextRerankerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = TinyVisionEncoder(int(cfg.vision_width))
        self.text_encoder = CandidateTextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            width=int(cfg.text_width),
        )
        self.image_projection = nn.Linear(int(cfg.vision_width), int(cfg.embed_dim))
        self.text_projection = nn.Linear(int(cfg.text_width), int(cfg.embed_dim))
        self.scorer = nn.Sequential(
            nn.Linear(int(cfg.embed_dim) * 3, int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        image = batch["image"].to(torch.float32)
        candidate_input_ids = batch["candidate_input_ids"].to(torch.long)
        candidate_attention_mask = batch["candidate_attention_mask"].to(torch.float32)

        if int(candidate_input_ids.shape[1]) != int(self.cfg.num_candidates):
            raise ValueError(
                "num_candidates does not match model config: "
                f"{int(candidate_input_ids.shape[1])} != {int(self.cfg.num_candidates)}"
            )
        if int(candidate_input_ids.shape[2]) != int(self.cfg.max_text_length):
            raise ValueError(
                "candidate text length does not match model config: "
                f"{int(candidate_input_ids.shape[2])} != {int(self.cfg.max_text_length)}"
            )

        image_features = self.vision_encoder(image)
        candidate_features = self.text_encoder(candidate_input_ids, candidate_attention_mask)
        image_embed = F.normalize(self.image_projection(image_features), dim=-1)
        candidate_embed = F.normalize(self.text_projection(candidate_features), dim=-1)

        expanded_image = image_embed.unsqueeze(1).expand(-1, candidate_embed.shape[1], -1)
        fused = torch.cat(
            [expanded_image, candidate_embed, (expanded_image - candidate_embed).abs()], dim=-1
        )
        scores = self.scorer(fused).squeeze(-1)
        probabilities = torch.softmax(scores, dim=-1)
        return {
            "scores": scores,
            "image_embed": image_embed,
            "candidate_embed": candidate_embed,
            "probabilities": probabilities,
        }


def reranking_loss(scores: torch.Tensor, label_index: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(scores, label_index.to(torch.long))


@torch.no_grad()
def reranking_accuracy(scores: torch.Tensor, label_index: torch.Tensor) -> float:
    preds = scores.argmax(dim=-1)
    return float((preds == label_index.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "CandidateTextEncoder",
    "ImageTextRerankerConfig",
    "TinyVisionEncoder",
    "CompactImageTextReranker",
    "reranking_accuracy",
    "reranking_loss",
]

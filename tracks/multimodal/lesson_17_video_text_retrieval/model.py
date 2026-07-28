from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class TinyFrameEncoder(nn.Module):
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

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"Expected video shape (B, T, C, H, W), got {tuple(frames.shape)}")
        batch_size, num_frames = int(frames.shape[0]), int(frames.shape[1])
        encoded = self.net(frames.view(batch_size * num_frames, *frames.shape[2:]))
        return encoded.view(batch_size, num_frames, -1)


class TemporalPoolingEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(width), int(width))

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError(
                f"Expected frame_features shape (B, T, D), got {tuple(frame_features.shape)}"
            )
        return self.proj(frame_features.mean(dim=1))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, width: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(width), padding_idx=int(pad_id))
        self.proj = nn.Linear(int(width), int(width))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")
        x = self.embedding(input_ids.to(torch.long))
        mask = attention_mask.to(torch.float32).unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.proj(pooled)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(out_dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_text_length: int
    num_frames: int
    image_size: int
    embed_dim: int = 32
    vision_width: int = 32
    temporal_width: int = 32
    text_width: int = 32
    init_temperature: float = 0.07


class CompactVideoTextRetrievalModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = TinyFrameEncoder(width=int(cfg.vision_width))
        self.temporal_encoder = TemporalPoolingEncoder(width=int(cfg.vision_width))
        self.text_encoder = TextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            width=int(cfg.text_width),
        )
        self.video_projection = ProjectionHead(int(cfg.vision_width), int(cfg.embed_dim))
        self.text_projection = ProjectionHead(int(cfg.text_width), int(cfg.embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(cfg.init_temperature))))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video = batch["video"].to(torch.float32)
        input_ids = batch["input_ids"].to(torch.long)
        attention_mask = batch["attention_mask"].to(torch.float32)

        if int(video.shape[1]) != int(self.cfg.num_frames):
            raise ValueError(
                "video num_frames does not match model config: "
                f"{int(video.shape[1])} != {int(self.cfg.num_frames)}"
            )
        if int(input_ids.shape[1]) != int(self.cfg.max_text_length):
            raise ValueError(
                "input_ids length does not match model config: "
                f"{int(input_ids.shape[1])} != {int(self.cfg.max_text_length)}"
            )

        frame_features = self.frame_encoder(video)
        pooled_video_features = self.temporal_encoder(frame_features)
        text_features = self.text_encoder(input_ids, attention_mask)

        video_embed = F.normalize(self.video_projection(pooled_video_features), dim=-1)
        text_embed = F.normalize(self.text_projection(text_features), dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_video = logit_scale * video_embed @ text_embed.transpose(0, 1)
        logits_per_text = logits_per_video.transpose(0, 1)
        return {
            "video_embed": video_embed,
            "text_embed": text_embed,
            "frame_features": frame_features,
            "pooled_video_features": pooled_video_features,
            "logits_per_video": logits_per_video,
            "logits_per_text": logits_per_text,
        }


def clip_contrastive_loss(
    logits_per_video: torch.Tensor, logits_per_text: torch.Tensor
) -> torch.Tensor:
    batch_size = int(logits_per_video.shape[0])
    targets = torch.arange(batch_size, device=logits_per_video.device)
    loss_v = F.cross_entropy(logits_per_video, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_v + loss_t)


@torch.no_grad()
def retrieval_accuracy(
    logits_per_video: torch.Tensor, logits_per_text: torch.Tensor
) -> tuple[float, float]:
    targets = torch.arange(int(logits_per_video.shape[0]), device=logits_per_video.device)
    video_to_text = (logits_per_video.argmax(dim=-1) == targets).to(torch.float32).mean()
    text_to_video = (logits_per_text.argmax(dim=-1) == targets).to(torch.float32).mean()
    return float(video_to_text.item()), float(text_to_video.item())


@torch.no_grad()
def recall_at_k(
    logits_per_video: torch.Tensor,
    logits_per_text: torch.Tensor,
    *,
    k: int,
) -> tuple[float, float]:
    top_k = max(1, min(int(k), int(logits_per_video.shape[-1])))
    targets = torch.arange(int(logits_per_video.shape[0]), device=logits_per_video.device).view(-1, 1)
    v2t = logits_per_video.topk(top_k, dim=-1).indices
    t2v = logits_per_text.topk(top_k, dim=-1).indices
    video_to_text = (v2t == targets).any(dim=-1).to(torch.float32).mean()
    text_to_video = (t2v == targets).any(dim=-1).to(torch.float32).mean()
    return float(video_to_text.item()), float(text_to_video.item())


__all__ = [
    "ModelConfig",
    "TemporalPoolingEncoder",
    "TextEncoder",
    "TinyFrameEncoder",
    "CompactVideoTextRetrievalModel",
    "clip_contrastive_loss",
    "recall_at_k",
    "retrieval_accuracy",
]

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

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected video shape (B, T, C, H, W), got {tuple(video.shape)}")
        batch_size, num_frames = int(video.shape[0]), int(video.shape[1])
        encoded = self.net(video.view(batch_size * num_frames, *video.shape[2:]))
        return encoded.view(batch_size, num_frames, -1).mean(dim=1)


class TinyAudioEncoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        hidden = max(16, int(width) // 2)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, int(width), kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, audio_spectrogram: torch.Tensor) -> torch.Tensor:
        if audio_spectrogram.ndim != 4:
            raise ValueError(
                "Expected audio_spectrogram shape (B, 1, M, T), "
                f"got {tuple(audio_spectrogram.shape)}"
            )
        return self.net(audio_spectrogram)


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
class AudioGroundedRetrievalConfig:
    vocab_size: int
    pad_id: int
    max_text_length: int
    num_frames: int
    image_size: int
    num_mel_bins: int
    num_audio_steps: int
    embed_dim: int = 32
    vision_width: int = 32
    audio_width: int = 32
    text_width: int = 32
    fusion_width: int = 48
    init_temperature: float = 0.07


class ToyAudioGroundedRetrievalModel(nn.Module):
    def __init__(self, cfg: AudioGroundedRetrievalConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.video_encoder = TinyFrameEncoder(width=int(cfg.vision_width))
        self.audio_encoder = TinyAudioEncoder(width=int(cfg.audio_width))
        self.text_encoder = TextEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            width=int(cfg.text_width),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(cfg.vision_width + cfg.audio_width), int(cfg.fusion_width)),
            nn.ReLU(),
        )
        self.clip_projection = ProjectionHead(int(cfg.fusion_width), int(cfg.embed_dim))
        self.query_projection = ProjectionHead(int(cfg.text_width), int(cfg.embed_dim))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(cfg.init_temperature))))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video = batch["video"].to(torch.float32)
        audio_spectrogram = batch["audio_spectrogram"].to(torch.float32)
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
        if int(audio_spectrogram.shape[2]) != int(self.cfg.num_mel_bins):
            raise ValueError(
                "audio_spectrogram num_mel_bins does not match model config: "
                f"{int(audio_spectrogram.shape[2])} != {int(self.cfg.num_mel_bins)}"
            )
        if int(audio_spectrogram.shape[3]) != int(self.cfg.num_audio_steps):
            raise ValueError(
                "audio_spectrogram num_audio_steps does not match model config: "
                f"{int(audio_spectrogram.shape[3])} != {int(self.cfg.num_audio_steps)}"
            )

        video_features = self.video_encoder(video)
        audio_features = self.audio_encoder(audio_spectrogram)
        fused_features = self.fusion(torch.cat([video_features, audio_features], dim=-1))
        query_features = self.text_encoder(input_ids, attention_mask)

        clip_embed = F.normalize(self.clip_projection(fused_features), dim=-1)
        query_embed = F.normalize(self.query_projection(query_features), dim=-1)

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_clip = logit_scale * clip_embed @ query_embed.transpose(0, 1)
        logits_per_query = logits_per_clip.transpose(0, 1)
        return {
            "clip_embed": clip_embed,
            "query_embed": query_embed,
            "video_features": video_features,
            "audio_features": audio_features,
            "fused_features": fused_features,
            "logits_per_clip": logits_per_clip,
            "logits_per_query": logits_per_query,
        }


def clip_contrastive_loss(
    logits_per_clip: torch.Tensor, logits_per_query: torch.Tensor
) -> torch.Tensor:
    batch_size = int(logits_per_clip.shape[0])
    targets = torch.arange(batch_size, device=logits_per_clip.device)
    loss_c = F.cross_entropy(logits_per_clip, targets)
    loss_q = F.cross_entropy(logits_per_query, targets)
    return 0.5 * (loss_c + loss_q)


@torch.no_grad()
def retrieval_accuracy(
    logits_per_clip: torch.Tensor, logits_per_query: torch.Tensor
) -> tuple[float, float]:
    targets = torch.arange(int(logits_per_clip.shape[0]), device=logits_per_clip.device)
    clip_to_query = (logits_per_clip.argmax(dim=-1) == targets).to(torch.float32).mean()
    query_to_clip = (logits_per_query.argmax(dim=-1) == targets).to(torch.float32).mean()
    return float(clip_to_query.item()), float(query_to_clip.item())


__all__ = [
    "AudioGroundedRetrievalConfig",
    "TextEncoder",
    "TinyAudioEncoder",
    "TinyFrameEncoder",
    "ToyAudioGroundedRetrievalModel",
    "clip_contrastive_loss",
    "retrieval_accuracy",
]

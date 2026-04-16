from __future__ import annotations

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
        encoded = self.net(video.view(batch_size * num_frames, *video.shape[2:]).to(torch.float32))
        return encoded.view(batch_size, num_frames, -1)


class TinyAudioEncoder(nn.Module):
    def __init__(self, audio_window: int, width: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(int(audio_window), int(width)),
            nn.ReLU(),
            nn.Linear(int(width), int(width)),
            nn.ReLU(),
        )

    def forward(self, audio_clip: torch.Tensor) -> torch.Tensor:
        if audio_clip.ndim != 3:
            raise ValueError(f"Expected audio_clip shape (B, T, W), got {tuple(audio_clip.shape)}")
        return self.proj(audio_clip.to(torch.float32))


class QueryEncoder(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(vocab_size), int(text_dim), padding_idx=int(pad_id))
        self.rnn = nn.GRU(int(text_dim), int(hidden_dim), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, query_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        del attention_mask
        emb = self.embedding(query_ids.to(torch.long))
        _seq, hidden = self.rnn(emb)
        return self.norm(hidden[-1])


@dataclass(frozen=True)
class AudioVisualEventLocalizationConfig:
    vocab_size: int
    pad_id: int
    num_frames: int
    audio_window: int
    hidden_dim: int = 64
    vision_width: int = 32
    audio_width: int = 24
    text_dim: int = 32


class ToyAudioVisualEventLocalizationModel(nn.Module):
    def __init__(self, cfg: AudioVisualEventLocalizationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.frame_encoder = TinyFrameEncoder(int(cfg.vision_width))
        self.audio_encoder = TinyAudioEncoder(int(cfg.audio_window), int(cfg.audio_width))
        self.query_encoder = QueryEncoder(
            vocab_size=int(cfg.vocab_size),
            pad_id=int(cfg.pad_id),
            text_dim=int(cfg.text_dim),
            hidden_dim=int(cfg.hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(int(cfg.vision_width + cfg.audio_width + cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.hidden_dim), int(cfg.hidden_dim)),
            nn.ReLU(),
        )
        self.head = nn.Linear(int(cfg.hidden_dim), 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video = batch["video"]
        audio_clip = batch["audio_clip"]
        if int(video.shape[1]) != int(self.cfg.num_frames):
            raise ValueError(
                "video num_frames does not match model config: "
                f"{int(video.shape[1])} != {int(self.cfg.num_frames)}"
            )
        if int(audio_clip.shape[1]) != int(self.cfg.num_frames):
            raise ValueError(
                "audio_clip num_frames does not match model config: "
                f"{int(audio_clip.shape[1])} != {int(self.cfg.num_frames)}"
            )

        video_feat = self.frame_encoder(video)
        audio_feat = self.audio_encoder(audio_clip)
        query_feat = self.query_encoder(batch["query_ids"], batch["attention_mask"])
        query_map = query_feat.unsqueeze(1).expand(-1, int(video_feat.shape[1]), -1)
        fused = self.fusion(torch.cat([video_feat, audio_feat, query_map], dim=-1))
        frame_logits = self.head(fused).squeeze(-1)
        frame_probs = torch.sigmoid(frame_logits)
        pred_frame = frame_logits.argmax(dim=-1).to(torch.long)
        return {
            "frame_logits": frame_logits,
            "frame_probs": frame_probs,
            "pred_frame": pred_frame,
        }


def localization_loss(frame_logits: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(frame_logits, event_mask.to(torch.float32))


@torch.no_grad()
def frame_accuracy(pred_frame: torch.Tensor, event_frame: torch.Tensor) -> float:
    return float((pred_frame == event_frame.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "AudioVisualEventLocalizationConfig",
    "QueryEncoder",
    "TinyAudioEncoder",
    "TinyFrameEncoder",
    "ToyAudioVisualEventLocalizationModel",
    "frame_accuracy",
    "localization_loss",
]


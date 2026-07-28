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
        return encoded.view(batch_size, num_frames, -1)


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


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(int(in_dim), int(out_dim), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@dataclass(frozen=True)
class AudioVisualLearningConfig:
    num_frames: int
    image_size: int
    num_mel_bins: int
    num_audio_steps: int
    num_events: int
    num_motions: int = 4
    embed_dim: int = 32
    vision_width: int = 32
    audio_width: int = 32
    fusion_width: int = 48
    init_temperature: float = 0.07


class CompactAudioVisualLearningModel(nn.Module):
    def __init__(self, cfg: AudioVisualLearningConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.video_encoder = TinyFrameEncoder(width=int(cfg.vision_width))
        self.audio_encoder = TinyAudioEncoder(width=int(cfg.audio_width))
        self.video_projection = ProjectionHead(int(cfg.vision_width), int(cfg.embed_dim))
        self.audio_projection = ProjectionHead(int(cfg.audio_width), int(cfg.embed_dim))
        self.fusion = nn.Sequential(
            nn.Linear(int(cfg.vision_width + cfg.audio_width), int(cfg.fusion_width)),
            nn.ReLU(),
        )
        self.event_classifier = nn.Linear(int(cfg.fusion_width), int(cfg.num_events))
        self.motion_classifier = nn.Linear(int(cfg.fusion_width), int(cfg.num_motions))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / float(cfg.init_temperature))))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        video = batch["video"].to(torch.float32)
        audio_spectrogram = batch["audio_spectrogram"].to(torch.float32)

        if int(video.shape[1]) != int(self.cfg.num_frames):
            raise ValueError(
                "video num_frames does not match model config: "
                f"{int(video.shape[1])} != {int(self.cfg.num_frames)}"
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

        frame_features = self.video_encoder(video)
        pooled_video = frame_features.mean(dim=1)
        pooled_audio = self.audio_encoder(audio_spectrogram)

        video_embed = F.normalize(self.video_projection(pooled_video), dim=-1)
        audio_embed = F.normalize(self.audio_projection(pooled_audio), dim=-1)
        fused_embed = self.fusion(torch.cat([pooled_video, pooled_audio], dim=-1))

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_video = logit_scale * video_embed @ audio_embed.transpose(0, 1)
        logits_per_audio = logits_per_video.transpose(0, 1)
        event_logits = self.event_classifier(fused_embed)
        motion_logits = self.motion_classifier(fused_embed)
        return {
            "video_embed": video_embed,
            "audio_embed": audio_embed,
            "fused_embed": fused_embed,
            "logits_per_video": logits_per_video,
            "logits_per_audio": logits_per_audio,
            "event_logits": event_logits,
            "motion_logits": motion_logits,
        }


def clip_contrastive_loss(
    logits_per_video: torch.Tensor, logits_per_audio: torch.Tensor
) -> torch.Tensor:
    batch_size = int(logits_per_video.shape[0])
    targets = torch.arange(batch_size, device=logits_per_video.device)
    loss_v = F.cross_entropy(logits_per_video, targets)
    loss_a = F.cross_entropy(logits_per_audio, targets)
    return 0.5 * (loss_v + loss_a)


def classification_loss(
    event_logits: torch.Tensor,
    motion_logits: torch.Tensor,
    event_targets: torch.Tensor,
    motion_targets: torch.Tensor,
) -> torch.Tensor:
    event_loss = F.cross_entropy(event_logits, event_targets.to(torch.long))
    motion_loss = F.cross_entropy(motion_logits, motion_targets.to(torch.long))
    return event_loss + 0.5 * motion_loss


@torch.no_grad()
def retrieval_accuracy(
    logits_per_video: torch.Tensor, logits_per_audio: torch.Tensor
) -> tuple[float, float]:
    targets = torch.arange(int(logits_per_video.shape[0]), device=logits_per_video.device)
    video_to_audio = (logits_per_video.argmax(dim=-1) == targets).to(torch.float32).mean()
    audio_to_video = (logits_per_audio.argmax(dim=-1) == targets).to(torch.float32).mean()
    return float(video_to_audio.item()), float(audio_to_video.item())


@torch.no_grad()
def classification_accuracy(
    event_logits: torch.Tensor,
    motion_logits: torch.Tensor,
    event_targets: torch.Tensor,
    motion_targets: torch.Tensor,
) -> tuple[float, float]:
    event_acc = (event_logits.argmax(dim=-1) == event_targets).to(torch.float32).mean()
    motion_acc = (motion_logits.argmax(dim=-1) == motion_targets).to(torch.float32).mean()
    return float(event_acc.item()), float(motion_acc.item())


def multitask_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    contrastive = clip_contrastive_loss(outputs["logits_per_video"], outputs["logits_per_audio"])
    cls = classification_loss(
        outputs["event_logits"],
        outputs["motion_logits"],
        batch["event_id"],
        batch["motion_id"],
    )
    return contrastive + 0.5 * cls


__all__ = [
    "AudioVisualLearningConfig",
    "TinyAudioEncoder",
    "TinyFrameEncoder",
    "CompactAudioVisualLearningModel",
    "classification_accuracy",
    "classification_loss",
    "clip_contrastive_loss",
    "multitask_loss",
    "retrieval_accuracy",
]

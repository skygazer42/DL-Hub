from __future__ import annotations

import torch
from torch import nn


class ToyAudioVisualModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        width: int,
        depth: int,
        audio_bins: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.audio_bins = int(audio_bins)
        c = int(width)
        layers: list[nn.Module] = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)])
        self.video_encoder = nn.Sequential(*layers)
        self.video_pool = nn.AdaptiveAvgPool2d(1)
        self.audio_encoder = nn.Sequential(
            nn.Linear(self.audio_bins, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, c),
        )
        self.fusion = nn.Sequential(nn.Linear(c * 2, c), nn.ReLU(inplace=True))

    def forward(
        self, video: torch.Tensor, audio: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        x = video.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
        pooled = self.video_pool(self.video_encoder(x)).flatten(1)
        if audio is None:
            audio = torch.zeros(x.shape[0], self.audio_bins, dtype=x.dtype, device=x.device)
        else:
            audio = audio.to(torch.float32)
        if audio.ndim != 2:
            raise ValueError(f"Expected audio shape (B,F), got {tuple(audio.shape)}")
        if int(audio.shape[1]) != self.audio_bins:
            raise ValueError(
                f"Expected audio feature size {self.audio_bins} for family {self.family!r}, got {int(audio.shape[1])}"
            )
        audio_tokens = self.audio_encoder(audio)
        joint = self.fusion(torch.cat([pooled, audio_tokens], dim=1))
        return {
            "video_embedding": pooled,
            "audio_embedding": audio_tokens,
            "joint_embedding": joint,
        }


def build_toy_audio_visual_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    audio_bins: int = 32,
):
    variant = str(variant)
    if variant not in variants:
        available = ", ".join(sorted(variants))
        raise KeyError(f"Unknown {family} variant {variant!r}. Available variants: {available}")
    spec = variants[variant]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyAudioVisualModel(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        audio_bins=int(audio_bins),
    )


def smoke_test_audio_visual_model(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 32, 32), torch.randn(2, 32))
    print(variant, tuple(out["video_embedding"].shape), tuple(out["joint_embedding"].shape))

from __future__ import annotations

import torch
from torch import nn


def check_ncthw(video: torch.Tensor) -> torch.Tensor:
    video = video.to(torch.float32)
    if video.ndim != 5:
        raise ValueError(f"Expected input shape (B, C, T, H, W), got {tuple(video.shape)}")
    return video


class CompactVideoToVideo(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        layers = [
            nn.Conv3d(int(in_channels), int(width), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend(
                [nn.Conv3d(int(width), int(width), kernel_size=3, padding=1), nn.ReLU(inplace=True)]
            )
        self.encoder = nn.Sequential(*layers)
        self.residual_head = nn.Conv3d(int(width), int(in_channels), kernel_size=3, padding=1)
        self.mix_head = nn.Conv3d(int(width), int(in_channels), kernel_size=1)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_ncthw(video)
        feat = self.encoder(x)
        residual = torch.tanh(self.residual_head(feat))
        mix = torch.sigmoid(self.mix_head(feat))
        stylized = torch.clamp(x + 0.5 * residual, -1.0, 1.0)
        out = torch.lerp(x, stylized, mix)
        return {"video": out, "residual": residual, "mix": mix}


def build_baseline_video_to_video(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int = 3,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    return CompactVideoToVideo(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
    )


def smoke_test_video_to_video(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 4, 32, 32))
    print(variant, tuple(out["video"].shape))

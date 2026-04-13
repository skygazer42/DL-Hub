from __future__ import annotations
import torch
from torch import nn

def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4: raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image

class ToyImageToVideo(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int, frames: int) -> None:
        super().__init__(); self.family = str(family); self.frames = max(2, int(frames)); layers = [nn.Conv2d(int(in_channels), int(width), 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 1)): layers.extend([nn.Conv2d(int(width), int(width), 3, 1, 1), nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*layers); self.frame_head = nn.Conv2d(int(width), int(in_channels), 3, padding=1); self.motion_head = nn.Conv2d(int(width), int(in_channels), 1)
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(image); feat = self.encoder(x); base = self.frame_head(feat); motion = torch.tanh(self.motion_head(feat)); frames=[]
        for step in range(self.frames):
            alpha = float(step) / max(1, self.frames - 1); frames.append(torch.clamp(base + alpha * motion, -1.0, 1.0))
        return {'video': torch.stack(frames, dim=1), 'motion': motion}

def build_toy_image_to_video(*, family: str, mode: str, variants: dict[str, dict[str, int]], in_channels: int, variant: str, width_mult: float = 1.0) -> nn.Module:
    cfg = variants[str(variant)]; width = max(16, int(int(cfg['width']) * float(width_mult))); return ToyImageToVideo(family=str(family), mode=str(mode), in_channels=int(in_channels), width=width, depth=int(cfg['depth']), frames=int(cfg.get('frames', 4)))

def smoke_test_image_to_video(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5); out = model(torch.randn(2, 3, 32, 32)); print(variant, tuple(out['video'].shape))

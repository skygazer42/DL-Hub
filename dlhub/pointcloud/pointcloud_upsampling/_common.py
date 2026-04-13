from __future__ import annotations
import torch
from torch import nn

def check_points(points: torch.Tensor, in_channels: int) -> torch.Tensor:
    points = points.to(torch.float32)
    if points.ndim != 3: raise ValueError(f"Expected input shape (B, N, C), got {tuple(points.shape)}")
    if points.shape[-1] != int(in_channels): raise ValueError(f"Expected {int(in_channels)} channels, got {int(points.shape[-1])}")
    return points

class TinyUpsampler(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int, up_factor: int) -> None:
        super().__init__(); self.family = str(family); self.mode = str(mode); self.in_channels = int(in_channels); self.up_factor = max(2, int(up_factor)); self.input_proj = nn.Linear(int(in_channels), int(width)); self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(int(width)), nn.Linear(int(width), int(width)), nn.GELU()) for _ in range(max(1, int(depth)))]); self.offset_head = nn.Linear(int(width), int(in_channels)); self.mix_head = nn.Linear(int(width), int(in_channels))
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        pts = check_points(points, self.in_channels); feat = self.input_proj(pts)
        for block in self.blocks: feat = feat + block(feat)
        repeated = pts.repeat_interleave(self.up_factor, dim=1); offsets = self.offset_head(feat).repeat_interleave(self.up_factor, dim=1); mix = self.mix_head(feat).repeat_interleave(self.up_factor, dim=1)
        if self.mode in {'pugan', 'diffusion'}: offsets = 0.5 * torch.tanh(offsets) + 0.1 * mix
        elif self.mode in {'punet', 'mpu'}: offsets = 0.25 * offsets
        return repeated + offsets

def build_toy_upsampler(*, family: str, mode: str, variants: dict[str, dict[str, int]], in_channels: int, variant: str, width_mult: float = 1.0) -> nn.Module:
    cfg = variants[str(variant)]; width = max(16, int(int(cfg['width']) * float(width_mult))); return TinyUpsampler(family=str(family), mode=str(mode), in_channels=int(in_channels), width=width, depth=int(cfg['depth']), up_factor=int(cfg.get('up_factor', 2)))

def smoke_test_upsampler(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5); out = model(torch.randn(2, 32, 3)); print(variant, tuple(out.shape))

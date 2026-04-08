from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x

class TinyRestorationNet(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, out_key: str) -> None:
        super().__init__()
        self.family = str(family)
        self.out_key = str(out_key)
        c = int(width)
        self.stem = nn.Conv2d(int(in_channels), c, 3, 1, 1)
        self.blocks = nn.Sequential(*sum([[nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, 1, 1)] for _ in range(max(1, int(depth)))], []))
        self.head = nn.Conv2d(c, int(in_channels), 3, 1, 1)
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(image)
        y = self.head(self.blocks(self.stem(x)))
        out = torch.clamp(x + y, -1.0, 1.0)
        return {self.out_key: out}

def build_toy_restoration(*, family: str, variants: dict[str, dict[str, int]], in_channels: int, variant: str, width_mult: float = 1.0, out_key: str = 'image') -> nn.Module:
    spec = variants[str(variant)]
    width = max(16, int(int(spec['width']) * float(width_mult)))
    return TinyRestorationNet(family=str(family), in_channels=int(in_channels), width=width, depth=int(spec['depth']), out_key=str(out_key))

def smoke_test_restoration(builder, variant: str, out_key: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    print(variant, tuple(out[out_key].shape))
    print('ok')

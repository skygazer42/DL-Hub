from __future__ import annotations
import torch
from torch import nn

def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4: raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image

class ToyRetriever(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__(); self.family = str(family); layers = [nn.Conv2d(int(in_channels), int(width), 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 1)): layers.extend([nn.Conv2d(int(width), int(width), 3, 1, 1), nn.ReLU(inplace=True)])
        self.image_encoder = nn.Sequential(*layers); self.pool = nn.AdaptiveAvgPool2d(1); self.text_encoder = nn.Sequential(nn.Linear(32, int(width)), nn.ReLU(inplace=True), nn.Linear(int(width), int(width)))
    def forward(self, image: torch.Tensor, text: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = check_nchw(image); image_embedding = self.pool(self.image_encoder(x)).flatten(1); text = torch.zeros(x.shape[0], 32, dtype=x.dtype, device=x.device) if text is None else text.to(torch.float32); text_embedding = self.text_encoder(text); similarity = (image_embedding * text_embedding).sum(dim=1, keepdim=True); return {'image_embedding': image_embedding, 'text_embedding': text_embedding, 'similarity': similarity}

def build_toy_retriever(*, family: str, mode: str, variants: dict[str, dict[str, int]], in_channels: int, variant: str, width_mult: float = 1.0) -> nn.Module:
    cfg = variants[str(variant)]; width = max(16, int(int(cfg['width']) * float(width_mult))); return ToyRetriever(family=str(family), mode=str(mode), in_channels=int(in_channels), width=width, depth=int(cfg['depth']))

def smoke_test_retriever(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5); out = model(torch.randn(2, 3, 32, 32), torch.randn(2, 32)); print(variant, tuple(out['image_embedding'].shape), tuple(out['similarity'].shape))

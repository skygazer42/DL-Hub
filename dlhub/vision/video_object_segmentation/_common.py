from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def check_btchw(x):
    x = x.to(torch.float32)
    if x.ndim != 5:
        raise ValueError(f"Expected input shape (B,T,C,H,W), got {tuple(x.shape)}")
    return x


class ToyVOSModel(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, num_masks: int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        self.num_masks = int(num_masks)
        self.frame = nn.Sequential(
            nn.Conv2d(int(in_channels), c, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.GRU(c, c, batch_first=True)
        self.head = nn.Conv2d(c, int(num_masks), 1)

    def forward(self, video):
        x = check_btchw(video)
        b, t, c, h, w = x.shape
        f = self.frame(x.view(b * t, c, h, w))
        pooled = F.adaptive_avg_pool2d(f, (1, 1)).flatten(1).view(b, t, -1)
        seq, _ = self.temporal(pooled)
        context = seq.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w).reshape(b * t, -1, h, w)
        logits = self.head(f + context)
        logits = logits.view(b, t, self.num_masks, h, w)
        masks = logits.argmax(dim=2)
        return {"logits": logits, "masks": masks}


def build_toy_vos(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    num_masks: int = 2,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyVOSModel(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        num_masks=int(num_masks),
    )


def smoke_test_vos(builder, variant: str):
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 4, 3, 64, 64))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})

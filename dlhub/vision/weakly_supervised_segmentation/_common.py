from __future__ import annotations
import torch
from torch import nn


def check_nchw(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class CompactWSSegmenter(nn.Module):
    def __init__(self, *, family: str, in_channels: int, num_classes: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        layers = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Conv2d(c, int(num_classes), 1)

    def forward(self, image):
        feat = self.backbone(check_nchw(image))
        logits = self.head(feat)
        pseudo = torch.softmax(logits, dim=1).argmax(dim=1)
        return {"logits": logits, "pseudo_mask": pseudo}


def build_baseline_ws_segmenter(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactWSSegmenter(
        family=str(family),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_wss(builder, variant: str):
    model = builder(in_channels=3, num_classes=4, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})

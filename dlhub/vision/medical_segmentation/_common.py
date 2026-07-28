from __future__ import annotations
import torch
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CompactMedicalSegmenter(nn.Module):
    def __init__(self, *, family: str, in_channels: int, num_classes: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        w = int(width)
        self.enc1 = DoubleConv(int(in_channels), w)
        self.down = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(w, w * 2)
        self.bott = DoubleConv(w * 2, w * 4)
        self.up2 = nn.ConvTranspose2d(w * 4, w * 2, 2, 2)
        self.dec2 = DoubleConv(w * 4, w * 2)
        self.up1 = nn.ConvTranspose2d(w * 2, w, 2, 2)
        self.dec1 = DoubleConv(w * 2, w)
        self.head = nn.Conv2d(w, int(num_classes), 1)

    def forward(self, image):
        x = check_nchw(image)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down(e1))
        b = self.bott(self.down(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        logits = self.head(d1)
        mask = logits.argmax(dim=1)
        return {"logits": logits, "mask": mask}


def build_baseline_medical_segmenter(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(8, int(int(spec["width"]) * float(width_mult)))
    return CompactMedicalSegmenter(
        family=str(family),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(spec["depth"]),
    )


def smoke_test_med(builder, variant: str):
    model = builder(in_channels=1, num_classes=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 1, 64, 64))
    print(variant, {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})

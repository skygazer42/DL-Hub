from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class TinyHOIBackbone(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int):
        super().__init__()
        c = int(width)
        layers = [nn.Conv2d(int(in_channels), c, 3, 2, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.net(check_nchw(x))


class CompactHOIDetector(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        num_verbs: int,
        num_objects: int,
        width: int,
        depth: int,
        num_queries: int,
    ):
        super().__init__()
        self.family = str(family)
        self.backbone = TinyHOIBackbone(in_channels, width, depth)
        c = self.backbone.out_channels
        self.query = nn.Parameter(torch.randn(int(num_queries), c) * 0.02)
        self.verb = nn.Linear(c, int(num_verbs))
        self.obj = nn.Linear(c, int(num_objects))
        self.box = nn.Linear(c, 4)

    def forward(self, image):
        feat = self.backbone(image)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        q = self.query.unsqueeze(0).expand(image.shape[0], -1, -1) + pooled.unsqueeze(1)
        return {
            "verb_logits": self.verb(q),
            "object_logits": self.obj(q),
            "boxes": torch.sigmoid(self.box(q)),
        }


def build_baseline_hoi_detector(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_verbs: int,
    num_objects: int,
    variant: str,
    width_mult: float = 1.0,
    num_queries: int = 16,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactHOIDetector(
        family=str(family),
        in_channels=int(in_channels),
        num_verbs=int(num_verbs),
        num_objects=int(num_objects),
        width=width,
        depth=int(spec["depth"]),
        num_queries=int(num_queries),
    )


def smoke_test_hoi(builder, variant: str):
    model = builder(in_channels=3, num_verbs=6, num_objects=8, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 128, 128))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})

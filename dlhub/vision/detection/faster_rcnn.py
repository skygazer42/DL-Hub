from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _BackboneStride4(nn.Module):
    """Tiny backbone producing a stride-4 feature map."""

    def __init__(self, *, in_channels: int, stem_channels: int, feat_channels: int, depth: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        feat = int(feat_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),  # /4
        ]
        for _ in range(d):
            layers.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RPNHead(nn.Module):
    def __init__(self, channels: int, *, num_anchors: int = 3) -> None:
        super().__init__()
        c = int(channels)
        na = int(num_anchors)
        if na <= 0:
            raise ValueError("num_anchors must be > 0")
        self.conv = ConvBNAct(c, c, kernel_size=3, stride=1, act="relu")
        self.obj = nn.Conv2d(c, na, kernel_size=1)
        self.box = nn.Conv2d(c, na * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.conv(x)
        return {"rpn_obj_logits": self.obj(x), "rpn_bbox_deltas": self.box(x)}


class RoIHead(nn.Module):
    def __init__(self, channels: int, num_classes: int, *, num_rois: int = 32, hidden_dim: int | None = None) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        r = int(num_rois)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        if r <= 0:
            raise ValueError("num_rois must be > 0")
        h = int(hidden_dim) if hidden_dim is not None else c
        self.num_rois = r
        self.roi_queries = nn.Parameter(torch.randn(r, c) * 0.02)
        self.fc1 = nn.Linear(c, h)
        self.fc2 = nn.Linear(h, h)
        self.cls = nn.Linear(h, nc)
        self.box = nn.Linear(h, 4)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c, _, _ = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)  # (B,C)
        q = self.roi_queries.unsqueeze(0).expand(b, -1, -1)  # (B,R,C)
        x = pooled.unsqueeze(1) + q
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return {"roi_cls_logits": self.cls(x), "roi_boxes": torch.sigmoid(self.box(x))}


class FasterRCNNDetector(nn.Module):
    """Faster R-CNN-style two-stage detector (toy-first).

    Forward returns raw RPN and ROI head outputs.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 96,
        backbone_depth: int = 2,
        num_anchors: int = 3,
        num_rois: int = 32,
    ) -> None:
        super().__init__()
        self.backbone = _BackboneStride4(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        self.rpn = RPNHead(int(feat_channels), num_anchors=int(num_anchors))
        self.roi = RoIHead(int(feat_channels), int(num_classes), num_rois=int(num_rois))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        feat = self.backbone(x)
        out = {}
        out.update(self.rpn(feat))
        out.update(self.roi(feat))
        return out


_VARIANTS: dict[str, dict] = {
    "faster_rcnn_tiny": {"stem": 24, "feat": 64, "depth": 1, "rois": 16},
    "faster_rcnn_small": {"stem": 32, "feat": 96, "depth": 2, "rois": 32},
    "faster_rcnn_base": {"stem": 48, "feat": 128, "depth": 3, "rois": 64},
}


def build_faster_rcnn_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "faster_rcnn_tiny",
    width_mult: float = 1.0,
    num_anchors: int = 3,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Faster R-CNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    return FasterRCNNDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        backbone_depth=int(spec["depth"]),
        num_anchors=int(num_anchors),
        num_rois=int(spec["rois"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_faster_rcnn_detector(in_channels=3, num_classes=2, variant="faster_rcnn_tiny", width_mult=0.5)
    out = m(x)
    print("faster_rcnn_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")


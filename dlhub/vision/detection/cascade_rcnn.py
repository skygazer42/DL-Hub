
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels


class _BackboneStride4(nn.Module):
    def __init__(self, *, in_channels: int, stem_channels: int, feat_channels: int, depth: int) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        feat = int(feat_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),
        ]
        for _ in range(d):
            layers.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CascadeStage(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        self.fc1 = nn.Linear(c, c)
        self.fc2 = nn.Linear(c, c)
        self.cls = nn.Linear(c, nc)
        self.box = nn.Linear(c, 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.cls(x), self.box(x)


class CascadeRCNNDetector(nn.Module):
    """Cascade R-CNN-style multi-stage refinement (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 96,
        backbone_depth: int = 2,
        num_rois: int = 32,
        stages: int = 3,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        r = int(num_rois)
        s = int(stages)
        if r <= 0:
            raise ValueError("num_rois must be > 0")
        if s <= 0:
            raise ValueError("stages must be > 0")

        self.backbone = _BackboneStride4(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        c = int(feat_channels)
        self.roi_queries = nn.Parameter(torch.randn(r, c) * 0.02)
        self.stages = nn.ModuleList([_CascadeStage(c, nc) for _ in range(s)])

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        feat = self.backbone(x)
        b, c, _, _ = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)
        h = pooled.unsqueeze(1) + self.roi_queries.unsqueeze(0).expand(b, -1, -1)

        cls_out: list[torch.Tensor] = []
        box_out: list[torch.Tensor] = []
        boxes = torch.zeros(b, h.shape[1], 4, device=h.device, dtype=h.dtype)
        for stage in self.stages:
            cls_logits, delta = stage(h)
            boxes = torch.sigmoid(boxes + delta)
            h = h + torch.tanh(h) * 0.1  # tiny refinement signal
            cls_out.append(cls_logits)
            box_out.append(boxes)
        return {"roi_cls_logits": cls_out, "roi_boxes": box_out}


_VARIANTS: dict[str, dict] = {
    "cascade_rcnn_tiny": {"stem": 24, "feat": 64, "depth": 1, "rois": 16, "stages": 2},
    "cascade_rcnn_small": {"stem": 32, "feat": 96, "depth": 2, "rois": 32, "stages": 3},
    "cascade_rcnn_base": {"stem": 48, "feat": 128, "depth": 3, "rois": 64, "stages": 3},
}


def build_cascade_rcnn_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "cascade_rcnn_tiny",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Cascade R-CNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    return CascadeRCNNDetector(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        backbone_depth=int(spec["depth"]),
        num_rois=int(spec["rois"]),
        stages=int(spec["stages"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    m = build_cascade_rcnn_detector(in_channels=3, num_classes=2, variant="cascade_rcnn_tiny", width_mult=0.5)
    out = m(x)
    print("cascade_rcnn_tiny", len(out["roi_cls_logits"]), len(out["roi_boxes"]))
    loss = sum(t.mean() for t in out["roi_cls_logits"]) + sum(t.mean() for t in out["roi_boxes"])
    loss.backward()
    print("ok")


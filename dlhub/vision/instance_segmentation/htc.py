from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import check_nchw


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


class _HTCStage(nn.Module):
    def __init__(self, channels: int, num_classes: int, mask_size: int) -> None:
        super().__init__()
        c = int(channels)
        nc = int(num_classes)
        ms = int(mask_size)
        self.fc1 = nn.Linear(c, c)
        self.fc2 = nn.Linear(c, c)
        self.cls = nn.Linear(c, nc)
        self.box = nn.Linear(c, 4)
        self.mask = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, nc * ms * ms),
        )
        self.mask_size = ms
        self.num_classes = nc

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        cls = self.cls(h)
        box = torch.sigmoid(self.box(h))
        mask = self.mask(h)
        return h, cls, box, mask


class HTC(nn.Module):
    """HTC (Hybrid Task Cascade) style instance segmenter (toy-first)."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 96,
        backbone_depth: int = 2,
        num_rois: int = 32,
        num_stages: int = 3,
        mask_size: int = 14,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        r = int(num_rois)
        if r <= 0:
            raise ValueError("num_rois must be > 0")
        s = int(num_stages)
        if s <= 0:
            raise ValueError("num_stages must be > 0")

        self.backbone = _BackboneStride4(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        c = int(feat_channels)
        self.roi_queries = nn.Parameter(torch.randn(r, c) * 0.02)

        self.stages = nn.ModuleList([_HTCStage(c, nc, int(mask_size)) for _ in range(s)])
        self.num_rois = r
        self.num_classes = nc
        self.mask_size = int(mask_size)

    def forward(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        x = check_nchw(x)
        feat = self.backbone(x)
        b, c, _, _ = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)
        h = pooled.unsqueeze(1) + self.roi_queries.unsqueeze(0).expand(b, -1, -1)

        stage_cls: list[torch.Tensor] = []
        stage_boxes: list[torch.Tensor] = []
        stage_masks: list[torch.Tensor] = []
        for stage in self.stages:
            h, cls, box, mask = stage(h)
            stage_cls.append(cls)
            stage_boxes.append(box)
            stage_masks.append(mask.view(b, self.num_rois, self.num_classes, self.mask_size, self.mask_size))

        return {"stage_cls_logits": stage_cls, "stage_boxes": stage_boxes, "stage_mask_logits": stage_masks}


_VARIANTS: dict[str, dict] = {
    "htc_tiny": {"stem": 24, "feat": 64, "depth": 1, "rois": 16, "stages": 2, "mask": 14},
    "htc_small": {"stem": 32, "feat": 96, "depth": 2, "rois": 32, "stages": 3, "mask": 14},
    "htc_base": {"stem": 48, "feat": 128, "depth": 3, "rois": 64, "stages": 3, "mask": 28},
}


def build_htc_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "htc_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown HTC variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    rois = max(1, int(round(int(spec["rois"]) * float(width_mult))))
    return HTC(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        backbone_depth=int(spec["depth"]),
        num_rois=int(rois),
        num_stages=int(spec["stages"]),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_htc_instance_segmenter(in_channels=3, num_classes=3, variant="htc_tiny", width_mult=0.5)
    out = m(x)
    shapes = {k: [tuple(t.shape) for t in v] for k, v in out.items()}
    print("htc_tiny", shapes)
    loss = sum(t.mean() for v in out.values() for t in v)
    loss.backward()
    print("ok")


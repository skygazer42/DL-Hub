import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import check_nchw


class _BackboneStride4(nn.Module):
    def __init__(
        self, *, in_channels: int, stem_channels: int, feat_channels: int, depth: int
    ) -> None:
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


class MaskScoringRCNN(nn.Module):
    """Mask Scoring R-CNN-style instance segmenter (toy-first).

    Adds a mask-quality prediction head (mask_iou) on top of a Mask R-CNN-like skeleton.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 96,
        backbone_depth: int = 2,
        num_rois: int = 32,
        mask_size: int = 14,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        r = int(num_rois)
        if r <= 0:
            raise ValueError("num_rois must be > 0")
        ms = int(mask_size)
        if ms <= 0:
            raise ValueError("mask_size must be > 0")

        self.backbone = _BackboneStride4(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        c = int(feat_channels)

        self.roi_queries = nn.Parameter(torch.randn(r, c) * 0.02)
        self.fc1 = nn.Linear(c, c)
        self.fc2 = nn.Linear(c, c)
        self.cls = nn.Linear(c, nc)
        self.box = nn.Linear(c, 4)

        self.mask = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, nc * ms * ms),
        )
        # Mask scoring head: predicts IoU/quality per class for each ROI.
        self.mask_iou = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, nc),
        )

        self.num_rois = r
        self.num_classes = nc
        self.mask_size = ms

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        feat = self.backbone(x)
        b, c, _, _ = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)
        h = pooled.unsqueeze(1) + self.roi_queries.unsqueeze(0).expand(b, -1, -1)
        h = torch.relu(self.fc1(h))
        h = torch.relu(self.fc2(h))
        cls_logits = self.cls(h)
        boxes = torch.sigmoid(self.box(h))

        mask_logits = self.mask(h).view(
            b, self.num_rois, self.num_classes, self.mask_size, self.mask_size
        )
        mask_iou = torch.sigmoid(self.mask_iou(h))  # (B,R,C)
        return {
            "roi_cls_logits": cls_logits,
            "roi_boxes": boxes,
            "mask_logits": mask_logits,
            "mask_iou": mask_iou,
        }


_VARIANTS: dict[str, dict] = {
    "mask_scoring_rcnn_tiny": {"stem": 24, "feat": 64, "depth": 1, "rois": 16, "mask": 14},
    "mask_scoring_rcnn_small": {"stem": 32, "feat": 96, "depth": 2, "rois": 32, "mask": 14},
    "mask_scoring_rcnn_base": {"stem": 48, "feat": 128, "depth": 3, "rois": 64, "mask": 28},
}


def build_mask_scoring_rcnn_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mask_scoring_rcnn_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Mask Scoring R-CNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    rois = max(1, int(round(int(spec["rois"]) * float(width_mult))))

    return MaskScoringRCNN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        backbone_depth=int(spec["depth"]),
        num_rois=int(rois),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mask_scoring_rcnn_instance_segmenter(
        in_channels=3, num_classes=3, variant="mask_scoring_rcnn_tiny", width_mult=0.5
    )
    out = m(x)
    print("mask_scoring_rcnn_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

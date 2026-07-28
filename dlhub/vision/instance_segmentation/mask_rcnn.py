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


class MaskRCNNInstanceSegmenter(nn.Module):
    """Mask R-CNN-style instance segmenter (compact-first).

    Returns RPN outputs + ROI cls/box + ROI masks.
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

        self.backbone = _BackboneStride4(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        c = int(feat_channels)

        self.rpn_conv = ConvBNAct(c, c, kernel_size=3, stride=1, act="relu")
        self.rpn_obj = nn.Conv2d(c, 3, kernel_size=1)
        self.rpn_box = nn.Conv2d(c, 3 * 4, kernel_size=1)

        self.roi_queries = nn.Parameter(torch.randn(r, c) * 0.02)
        self.roi_fc1 = nn.Linear(c, c)
        self.roi_fc2 = nn.Linear(c, c)
        self.roi_cls = nn.Linear(c, nc)
        self.roi_box = nn.Linear(c, 4)

        ms = int(mask_size)
        self.mask_mlp = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, nc * ms * ms),
        )
        self.mask_size = ms
        self.num_rois = r
        self.num_classes = nc

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        feat = self.backbone(x)
        rpn_feat = self.rpn_conv(feat)
        rpn_obj_logits = self.rpn_obj(rpn_feat)
        rpn_bbox_deltas = self.rpn_box(rpn_feat)

        b, c, _, _ = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)
        q = self.roi_queries.unsqueeze(0).expand(b, -1, -1)
        h = pooled.unsqueeze(1) + q
        h = torch.relu(self.roi_fc1(h))
        h = torch.relu(self.roi_fc2(h))
        roi_cls_logits = self.roi_cls(h)
        roi_boxes = torch.sigmoid(self.roi_box(h))

        masks = self.mask_mlp(h).view(
            b, self.num_rois, self.num_classes, self.mask_size, self.mask_size
        )
        return {
            "rpn_obj_logits": rpn_obj_logits,
            "rpn_bbox_deltas": rpn_bbox_deltas,
            "roi_cls_logits": roi_cls_logits,
            "roi_boxes": roi_boxes,
            "mask_logits": masks,
        }


_VARIANTS: dict[str, dict] = {
    "mask_rcnn_tiny": {"stem": 24, "feat": 64, "depth": 1, "rois": 16, "mask": 14},
    "mask_rcnn_small": {"stem": 32, "feat": 96, "depth": 2, "rois": 32, "mask": 14},
    "mask_rcnn_base": {"stem": 48, "feat": 128, "depth": 3, "rois": 64, "mask": 28},
}


def build_mask_rcnn_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mask_rcnn_small",
    width_mult: float = 1.0,
    num_rois: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Mask R-CNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    rois = int(spec["rois"]) if num_rois is None else int(num_rois)

    return MaskRCNNInstanceSegmenter(
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
    m = build_mask_rcnn_instance_segmenter(
        in_channels=3, num_classes=3, variant="mask_rcnn_tiny", width_mult=0.5
    )
    out = m(x)
    print("mask_rcnn_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

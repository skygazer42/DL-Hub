
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import check_nchw, fuse_panoptic


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


class SCNetPanoptic(nn.Module):
    """SCNet-style panoptic segmentation (toy-first).

    ROI queries gated by global context + lightweight mask head, plus a semantic head for stuff.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 96,
        backbone_depth: int = 2,
        num_rois: int = 32,
        mask_size: int = 14,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
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

        self.semantic = nn.Sequential(
            ConvBNAct(c, c, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(c, nt + ns, kernel_size=1, bias=True),
        )

        self.roi_queries = nn.Parameter(torch.randn(r, c) * 0.02)
        self.gate = nn.Sequential(nn.Linear(c, c), nn.Sigmoid())

        self.fc1 = nn.Linear(c, c)
        self.fc2 = nn.Linear(c, c)
        self.cls = nn.Linear(c, nt)
        self.box = nn.Linear(c, 4)
        # Class-agnostic mask per ROI for simplicity.
        self.mask = nn.Sequential(
            nn.Linear(c, c),
            nn.ReLU(inplace=True),
            nn.Linear(c, ms * ms),
        )

        self.num_rois = r
        self.mask_size = ms
        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        feat = self.backbone(x)
        b, c, _, _ = feat.shape

        semantic_logits = self.semantic(feat)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, c)
        gate = self.gate(pooled).unsqueeze(1)  # (B,1,C)

        hq = pooled.unsqueeze(1) + self.roi_queries.unsqueeze(0).expand(b, -1, -1)
        hq = hq * gate
        hq = torch.relu(self.fc1(hq))
        hq = torch.relu(self.fc2(hq))
        query_cls_logits = self.cls(hq)  # (B,R,nt)
        query_boxes = torch.sigmoid(self.box(hq))

        mask_small = self.mask(hq).view(b, self.num_rois, self.mask_size, self.mask_size)
        mask_logits = F.interpolate(mask_small, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "query_boxes": query_boxes,
            "mask_logits": mask_logits,
            "gate": gate,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "scnet_panoptic_tiny": {"stem": 24, "feat": 64, "depth": 1, "rois": 16, "mask": 14},
    "scnet_panoptic_small": {"stem": 32, "feat": 96, "depth": 2, "rois": 32, "mask": 14},
    "scnet_panoptic_base": {"stem": 48, "feat": 128, "depth": 3, "rois": 64, "mask": 28},
}


def build_scnet_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "scnet_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SCNet-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    rois = max(1, int(round(int(spec["rois"]) * float(width_mult))))
    return SCNetPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        stem_channels=int(stem),
        feat_channels=int(feat),
        backbone_depth=int(spec["depth"]),
        num_rois=int(rois),
        mask_size=int(spec["mask"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_scnet_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="scnet_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("scnet_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


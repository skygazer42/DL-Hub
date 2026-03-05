from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    BackboneLowDet,
    DensePredHead,
    ProtoNet,
    check_nchw,
    fuse_panoptic,
    masks_from_prototypes,
)


class BlendMaskPanoptic(nn.Module):
    """BlendMask-style panoptic segmentation (toy-first, pure torch).

    Prototype masks are modulated by a learned blending map; dense coefficients produce instance masks.
    A lightweight semantic head runs in parallel on low-level features.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        num_protos: int = 32,
        num_anchors: int = 3,
        stem_channels: int = 24,
        low_channels: int = 48,
        det_channels: int = 96,
        backbone_depth: int = 2,
        head_channels: int = 96,
        head_convs: int = 2,
        proto_depth: int = 3,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        np = int(num_protos)
        if np <= 0:
            raise ValueError("num_protos must be > 0")
        na = int(num_anchors)
        if na <= 0:
            raise ValueError("num_anchors must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )

        self.semantic = nn.Sequential(
            ConvBNAct(int(low_channels), int(low_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(low_channels), nt + ns, kernel_size=1, bias=True),
        )

        self.proto = ProtoNet(int(low_channels), np, depth=int(proto_depth), act="relu")
        self.blend = nn.Sequential(
            ConvBNAct(int(low_channels), int(low_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(low_channels), 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.head = DensePredHead(
            in_ch=int(det_channels),
            num_classes=nt,
            num_anchors=na,
            num_protos=np,
            head_channels=int(head_channels),
            num_convs=int(head_convs),
            act="relu",
        )

        self.num_thing_classes = nt
        self.num_stuff_classes = ns
        self.num_anchors = na
        self.num_protos = np

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        low, det = self.backbone(x)

        semantic_logits = self.semantic(low)  # (B, nt+ns, H/4, W/4)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(low)  # (B,P,H/4,W/4)
        blend = self.blend(low)  # (B,1,H/4,W/4)
        proto = proto * blend

        cls_logits, bbox_deltas, mask_coeffs = self.head(det)  # cls (B,A*nt,H/8,W/8); coeff (B,A*P,H/8,W/8)

        a = self.num_anchors
        p = self.num_protos
        _, _, h8, w8 = mask_coeffs.shape
        coeff = mask_coeffs.view(b, a, p, h8, w8).permute(0, 3, 4, 1, 2).reshape(b, -1, p)  # (B,N,P)
        mask_logits = masks_from_prototypes(proto, coeff)  # (B,N,H/4,W/4)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        # Instance scores from dense classification.
        nt = self.num_thing_classes
        cls = cls_logits.view(b, a, nt, h8, w8).permute(0, 3, 4, 1, 2).reshape(b, -1, nt)
        instance_scores = cls.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, instance_scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "mask_coeffs": mask_coeffs,
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "blendmask_panoptic_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "protos": 16},
    "blendmask_panoptic_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "protos": 32},
    "blendmask_panoptic_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "protos": 48},
}


def build_blendmask_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "blendmask_panoptic_small",
    width_mult: float = 1.0,
    num_anchors: int = 3,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BlendMask-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))

    return BlendMaskPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        num_protos=int(protos),
        num_anchors=int(num_anchors),
        stem_channels=int(stem),
        low_channels=int(low),
        det_channels=int(det),
        backbone_depth=int(spec["depth"]),
        head_channels=int(head),
        head_convs=2,
        proto_depth=3,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_blendmask_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="blendmask_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("blendmask_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


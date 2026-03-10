import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.instance_segmentation._common import (
    BackboneLowDet,
    DensePredHead,
    ProtoNet,
    check_nchw,
)


class OrienMask(nn.Module):
    """OrienMask-style polar instance segmentation."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int,
        low_channels: int,
        det_channels: int,
        head_channels: int,
        backbone_depth: int,
        num_anchors: int,
        num_protos: int,
        num_rays: int,
    ) -> None:
        super().__init__()
        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
        )
        self.proto_net = ProtoNet(int(low_channels), int(num_protos), depth=3)
        self.pred_head = DensePredHead(
            in_ch=int(det_channels),
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            num_protos=int(num_protos),
            head_channels=int(head_channels),
            num_convs=2,
        )
        self.ray_head = nn.Conv2d(
            int(det_channels), int(num_anchors) * int(num_rays), kernel_size=3, padding=1
        )
        self.num_anchors = int(num_anchors)
        self.num_protos = int(num_protos)
        self.num_rays = int(num_rays)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        low, det = self.backbone(x)
        proto = self.proto_net(low)
        cls_logits, bbox_deltas, coeff_map = self.pred_head(det)
        ray_offsets = self.ray_head(det)

        b = x.shape[0]
        coeff = F.adaptive_avg_pool2d(coeff_map, (1, 1)).view(b, self.num_anchors, self.num_protos)
        mask_logits = torch.einsum("bkp,bphw->bkhw", coeff, proto)
        ray_offsets = F.adaptive_avg_pool2d(ray_offsets, (1, 1)).view(
            b, self.num_anchors, self.num_rays
        )
        return {
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "ray_offsets": ray_offsets,
            "proto": proto,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict[str, int]] = {
    "orienmask_tiny": {
        "stem": 24,
        "low": 40,
        "det": 72,
        "head": 72,
        "depth": 1,
        "anchors": 8,
        "protos": 16,
        "rays": 18,
    },
    "orienmask_small": {
        "stem": 24,
        "low": 48,
        "det": 96,
        "head": 96,
        "depth": 2,
        "anchors": 12,
        "protos": 24,
        "rays": 24,
    },
    "orienmask_base": {
        "stem": 32,
        "low": 64,
        "det": 128,
        "head": 128,
        "depth": 3,
        "anchors": 16,
        "protos": 32,
        "rays": 36,
    },
}


def build_orienmask_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "orienmask_small",
    width_mult: float = 1.0,
    num_anchors: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown OrienMask variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    return OrienMask(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8),
        low_channels=scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8),
        det_channels=scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8),
        head_channels=scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8),
        backbone_depth=int(spec["depth"]),
        num_anchors=int(spec["anchors"]) if num_anchors is None else int(num_anchors),
        num_protos=int(spec["protos"]),
        num_rays=int(spec["rays"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_orienmask_instance_segmenter(
        in_channels=3, num_classes=3, variant="orienmask_tiny", width_mult=0.5
    )
    out = m(x)
    print("orienmask_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

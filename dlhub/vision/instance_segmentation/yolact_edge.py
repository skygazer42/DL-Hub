import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.instance_segmentation._common import (
    BackboneLowDet,
    DensePredHead,
    ProtoNet,
    check_nchw,
)


class YolactEdge(nn.Module):
    """YolactEdge-style one-stage instance segmentation (compact-first).

    Prototype masks are modulated by a learned blending map, then combined using per-location coefficients.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
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
        np = int(num_protos)
        if np <= 0:
            raise ValueError("num_protos must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )
        self.proto = ProtoNet(int(low_channels), np, depth=int(proto_depth))
        self.blend = nn.Sequential(
            ConvBNAct(int(low_channels), int(low_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(low_channels), 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.head = DensePredHead(
            in_ch=int(det_channels),
            num_classes=int(num_classes),
            num_anchors=int(num_anchors),
            num_protos=np,
            head_channels=int(head_channels),
            num_convs=int(head_convs),
            act="relu",
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        low, det = self.backbone(x)
        proto = self.proto(low)  # (B,P,H/4,W/4)
        blend = self.blend(low)  # (B,1,H/4,W/4)
        proto = proto * blend

        cls_logits, bbox_deltas, mask_coeffs = self.head(det)  # coeffs (B,A*P,H/8,W/8)

        b, p, h4, w4 = proto.shape
        a = mask_coeffs.shape[1] // p
        slots = mask_coeffs.shape[-2] * mask_coeffs.shape[-1] * a
        coeff = mask_coeffs.view(b, a, p, mask_coeffs.shape[-2], mask_coeffs.shape[-1]).permute(
            0, 3, 4, 1, 2
        )
        coeff = coeff.reshape(b, -1, p)  # (B,S,P)
        proto_flat = proto.reshape(b, p, h4 * w4)
        mask_flat = torch.bmm(coeff, proto_flat)
        mask_logits = mask_flat.view(b, slots, h4, w4)

        return {
            "proto": proto,
            "blend": blend,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "mask_coeffs": mask_coeffs,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict] = {
    "yolact_edge_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "protos": 16},
    "yolact_edge_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "protos": 32},
    "yolact_edge_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "protos": 48},
}


def build_yolact_edge_instance_segmenter(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "yolact_edge_small",
    width_mult: float = 1.0,
    num_anchors: int = 3,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YolactEdge variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))

    return YolactEdge(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
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
    m = build_yolact_edge_instance_segmenter(
        in_channels=3, num_classes=3, variant="yolact_edge_tiny", width_mult=0.5
    )
    out = m(x)
    print("yolact_edge_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

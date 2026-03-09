
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneLowDet, ProtoNet, check_nchw, masks_from_prototypes


class YOLACTPanoptic(nn.Module):
    """YOLACT-style panoptic segmentation (toy-first).

    Instance branch: prototypes + dense mask coefficients at stride /8.
    Semantic branch: stride /4 semantic logits.
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
        if nt <= 0 or ns <= 0:
            raise ValueError("num_thing_classes/num_stuff_classes must be > 0")
        np = int(num_protos)
        na = int(num_anchors)
        if np <= 0 or na <= 0:
            raise ValueError("num_protos/num_anchors must be > 0")

        self.backbone = BackboneLowDet(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            low_channels=int(low_channels),
            det_channels=int(det_channels),
            depth=int(backbone_depth),
            act="relu",
        )
        self.proto = ProtoNet(int(low_channels), np, depth=int(proto_depth), act="relu")

        # Semantic branch at low-res (/4).
        self.semantic = nn.Sequential(
            ConvBNAct(int(low_channels), int(low_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(low_channels), nt + ns, kernel_size=1, bias=True),
        )

        # Dense instance prediction at /8.
        tower: list[nn.Module] = [ConvBNAct(int(det_channels), int(head_channels), kernel_size=3, stride=1, act="relu")]
        for _ in range(int(head_convs) - 1):
            tower.append(ConvBNAct(int(head_channels), int(head_channels), kernel_size=3, stride=1, act="relu"))
        self.tower = nn.Sequential(*tower)

        self.cls_logits = nn.Conv2d(int(head_channels), na * nt, kernel_size=3, padding=1)
        self.box_pred = nn.Conv2d(int(head_channels), na * 4, kernel_size=3, padding=1)
        self.mask_coeff = nn.Conv2d(int(head_channels), na * np, kernel_size=3, padding=1)

        self.num_anchors = na
        self.num_protos = np
        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        low, det = self.backbone(x)

        semantic_logits = self.semantic(low)  # (B, nt+ns, H/4, W/4)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(low)  # (B,P,H/4,W/4)
        t = self.tower(det)
        cls_logits = self.cls_logits(t)  # (B,A*nt,H/8,W/8)
        bbox_deltas = self.box_pred(t)
        coeffs = self.mask_coeff(t)  # (B,A*P,H/8,W/8)

        # Flatten coefficients: (B, N, P) with N=A*H8*W8.
        a = self.num_anchors
        p = self.num_protos
        coeff = coeffs.view(b, a, p, coeffs.shape[-2], coeffs.shape[-1]).permute(0, 3, 4, 1, 2)
        coeff = coeff.reshape(b, -1, p)
        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        return {
            "semantic_logits": semantic_logits,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "mask_coeffs": coeffs,
            "proto": proto,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict] = {
    "yolact_panoptic_tiny": {"stem": 24, "low": 40, "det": 80, "depth": 1, "head": 80, "protos": 16},
    "yolact_panoptic_small": {"stem": 24, "low": 48, "det": 96, "depth": 2, "head": 96, "protos": 32},
    "yolact_panoptic_base": {"stem": 32, "low": 64, "det": 128, "depth": 3, "head": 128, "protos": 48},
}


def build_yolact_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "yolact_panoptic_small",
    width_mult: float = 1.0,
    num_anchors: int = 3,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown YOLACT-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    low = scale_channels(int(spec["low"]), float(width_mult), min_ch=16, divisor=8)
    det = scale_channels(int(spec["det"]), float(width_mult), min_ch=16, divisor=8)
    head = scale_channels(int(spec["head"]), float(width_mult), min_ch=16, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))

    return YOLACTPanoptic(
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
    m = build_yolact_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="yolact_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("yolact_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["cls_logits"].mean() + out["bbox_deltas"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


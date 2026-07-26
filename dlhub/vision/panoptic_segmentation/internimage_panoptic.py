import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    FPN4,
    BackboneC2C3C4C5,
    ProtoNet,
    check_nchw,
    masks_from_prototypes,
)


class InternImagePanoptic(nn.Module):
    """InternImagePanoptic (Unified Panoptic Segmentation Network) (toy-first).

    Two-branch network:
    - Semantic segmentation head (stuff + thing)
    - Instance mask head (prototype masks + per-instance coefficients)
    - A small "panoptic head" that can learn to resolve overlaps (toy: 1x1 conv on concatenated maps)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        stem_channels: int = 24,
        c2_channels: int = 32,
        c3_channels: int = 64,
        c4_channels: int = 96,
        c5_channels: int = 128,
        depth: int = 2,
        fpn_channels: int = 96,
        num_instances: int = 32,
        num_protos: int = 32,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0 or ns <= 0:
            raise ValueError("num_thing_classes/num_stuff_classes must be > 0")
        fpn = int(fpn_channels)
        ni = int(num_instances)
        np = int(num_protos)
        if fpn <= 0 or ni <= 0 or np <= 0:
            raise ValueError("fpn/instances/protos must be > 0")

        self.backbone = BackboneC2C3C4C5(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            c2_channels=int(c2_channels),
            c3_channels=int(c3_channels),
            c4_channels=int(c4_channels),
            c5_channels=int(c5_channels),
            depth=int(depth),
            act="relu",
        )
        self.fpn = FPN4(
            (int(c2_channels), int(c3_channels), int(c4_channels), int(c5_channels)),
            fpn,
            act="relu",
        )

        self.semantic = nn.Sequential(
            nn.Conv2d(fpn, fpn, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn, nt + ns, kernel_size=1, bias=True),
        )

        self.proto = ProtoNet(fpn, np, depth=3, act="relu")
        self.query = nn.Parameter(torch.randn(ni, fpn) * 0.02)
        self.proj = nn.Linear(fpn, fpn)
        self.cls = nn.Linear(fpn, nt)
        self.coeff = nn.Linear(fpn, np)

        # Panoptic head (toy): learns to mix semantic logits with an aggregated instance map.
        self.panoptic_head = nn.Conv2d((nt + ns) + 1, nt + ns, kernel_size=1, bias=True)

        self.num_thing_classes = nt
        self.num_stuff_classes = ns
        self.num_instances = ni

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2, _, _, _ = self.fpn(c2, c3, c4, c5)

        semantic_logits = self.semantic(p2)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(p2)
        pooled = F.adaptive_avg_pool2d(p2, (1, 1)).view(b, -1)
        base = torch.relu(self.proj(pooled)).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        inst = base + q
        query_cls_logits = self.cls(inst)
        coeff = self.coeff(inst)
        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        # Aggregate instance masks into a single "thingness" map.
        thing_map = mask_logits.sigmoid().max(dim=1, keepdim=True).values  # (B,1,H,W)
        fused_logits = self.panoptic_head(torch.cat([semantic_logits, thing_map], dim=1))
        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "panoptic_logits": fused_logits,
        }


_VARIANTS: dict[str, dict] = {
    "internimage_panoptic_tiny": {
        "stem": 24,
        "c2": 24,
        "c3": 48,
        "c4": 64,
        "c5": 96,
        "depth": 1,
        "fpn": 64,
        "instances": 16,
        "protos": 16,
    },
    "internimage_panoptic_small": {
        "stem": 24,
        "c2": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "fpn": 96,
        "instances": 32,
        "protos": 32,
    },
    "internimage_panoptic_base": {
        "stem": 32,
        "c2": 40,
        "c3": 80,
        "c4": 128,
        "c5": 160,
        "depth": 2,
        "fpn": 128,
        "instances": 64,
        "protos": 48,
    },
}


def build_internimage_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "internimage_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown InternImagePanoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    def sc(v: int, *, min_ch: int = 16) -> int:
        return scale_channels(int(v), float(width_mult), min_ch=min_ch, divisor=8)

    stem = sc(spec["stem"])
    c2 = sc(spec["c2"])
    c3 = sc(spec["c3"])
    c4 = sc(spec["c4"])
    c5 = sc(spec["c5"])
    fpn = sc(spec["fpn"], min_ch=32)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return InternImagePanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        fpn_channels=int(fpn),
        num_instances=int(instances),
        num_protos=int(protos),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_internimage_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="internimage_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("internimage_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean()
        + out["query_cls_logits"].mean()
        + out["mask_logits"].mean()
        + out["panoptic_logits"].mean()
    )
    loss.backward()
    print("ok")


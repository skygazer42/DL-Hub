import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    FPN4,
    BackboneC2C3C4C5,
    ProtoNet,
    check_nchw,
    fuse_panoptic,
    masks_from_prototypes,
)


class UberPanopticNet(nn.Module):
    """A simple "uber" panoptic model (toy-first).

    FPN semantic head + prototype masks + query refinement with cross-attention over deep tokens.
    This is not a paper-faithful implementation; it's a compact educational architecture.
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
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        ni = int(num_instances)
        if ni <= 0:
            raise ValueError("num_instances must be > 0")
        np = int(num_protos)
        if np <= 0:
            raise ValueError("num_protos must be > 0")

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
            int(fpn_channels),
            act="relu",
        )

        fpn = int(fpn_channels)
        self.semantic = nn.Sequential(
            ConvBNAct(fpn, fpn, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(fpn, nt + ns, kernel_size=1, bias=True),
        )

        self.proto = ProtoNet(fpn, np, depth=3, act="relu")

        self.query = nn.Parameter(torch.randn(ni, fpn) * 0.02)
        self.deep_proj = nn.Conv2d(fpn, fpn, kernel_size=1, bias=True)
        self.cross = nn.MultiheadAttention(fpn, int(num_heads), batch_first=True)
        self.norm = nn.LayerNorm(fpn)

        self.cls = nn.Linear(fpn, nt)
        self.coeff = nn.Linear(fpn, np)

        self.num_instances = ni
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2, _, _, p5 = self.fpn(c2, c3, c4, c5)

        semantic_logits = self.semantic(p2)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(p2)
        deep = self.deep_proj(p5).flatten(2).transpose(1, 2)  # (B,N,fpn)
        q = self.query.unsqueeze(0).expand(b, -1, -1)  # (B,I,fpn)
        hq, _ = self.cross(q, deep, deep, need_weights=False)
        q = self.norm(q + hq)

        query_cls_logits = self.cls(q)
        coeff = self.coeff(q)
        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(
            semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes)
        )

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "uberpanoptic_tiny": {
        "stem": 24,
        "c2": 24,
        "c3": 48,
        "c4": 64,
        "c5": 96,
        "depth": 1,
        "fpn": 64,
        "instances": 16,
        "protos": 16,
        "heads": 4,
    },
    "uberpanoptic_small": {
        "stem": 24,
        "c2": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "fpn": 96,
        "instances": 32,
        "protos": 32,
        "heads": 4,
    },
    "uberpanoptic_base": {
        "stem": 32,
        "c2": 40,
        "c3": 80,
        "c4": 128,
        "c5": 160,
        "depth": 2,
        "fpn": 128,
        "instances": 64,
        "protos": 48,
        "heads": 8,
    },
}


def build_uberpanoptic_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "uberpanoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown UberPanoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
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
    heads = int(spec["heads"])
    while heads > 1 and int(fpn) % heads != 0:
        heads -= 1

    return UberPanopticNet(
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
        num_heads=int(heads),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_uberpanoptic_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="uberpanoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("uberpanoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")

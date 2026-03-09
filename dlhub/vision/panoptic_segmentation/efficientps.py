
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneC2C3C4C5, FPN4, ProtoNet, check_nchw, masks_from_prototypes


class _PPM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, bins: tuple[int, ...] = (1, 2, 3)) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((b, b)),
                    nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
                for b in (int(x) for x in bins)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_in + c_out * len(self.blocks), c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [x]
        for blk in self.blocks:
            y = blk(x)
            outs.append(F.interpolate(y, size=(h, w), mode="nearest"))
        return self.fuse(torch.cat(outs, dim=1))


class EfficientPS(nn.Module):
    """EfficientPS panoptic segmentation (toy-first).

    Uses an FPN and lightweight context module for semantic head, plus prototype-based instance masks.
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
        context_channels: int = 96,
        num_instances: int = 32,
        num_protos: int = 32,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0 or ns <= 0:
            raise ValueError("num_thing_classes/num_stuff_classes must be > 0")
        fpn = int(fpn_channels)
        ctx = int(context_channels)
        ni = int(num_instances)
        np = int(num_protos)
        if fpn <= 0 or ctx <= 0 or ni <= 0 or np <= 0:
            raise ValueError("channels/instances/protos must be > 0")

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
        self.fpn = FPN4((int(c2_channels), int(c3_channels), int(c4_channels), int(c5_channels)), fpn, act="relu")

        self.context = _PPM(fpn, ctx, bins=(1, 2, 3))
        self.semantic = nn.Sequential(
            ConvBNAct(ctx, ctx, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(ctx, nt + ns, kernel_size=1, bias=True),
        )

        self.proto = ProtoNet(fpn, np, depth=3, act="relu")
        self.query = nn.Parameter(torch.randn(ni, fpn) * 0.02)
        self.proj = nn.Linear(fpn, fpn)
        self.cls = nn.Linear(fpn, nt)
        self.coeff = nn.Linear(fpn, np)

        self.num_instances = ni

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)

        # Semantic head (use P2 with context).
        ctx = self.context(p2)
        semantic_logits = self.semantic(ctx)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        # Instance head (use P3 pooled for queries, prototypes from P2).
        proto = self.proto(p2)
        pooled = F.adaptive_avg_pool2d(p3, (1, 1)).view(b, -1)
        base = torch.relu(self.proj(pooled)).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        inst = base + q
        query_cls_logits = self.cls(inst)
        coeff = self.coeff(inst)
        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        return {"semantic_logits": semantic_logits, "query_cls_logits": query_cls_logits, "mask_logits": mask_logits, "p5": p5}


_VARIANTS: dict[str, dict] = {
    "efficientps_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "fpn": 64, "ctx": 64, "instances": 16, "protos": 16},
    "efficientps_small": {"stem": 24, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96, "ctx": 96, "instances": 32, "protos": 32},
    "efficientps_base": {"stem": 32, "c2": 40, "c3": 80, "c4": 128, "c5": 160, "depth": 2, "fpn": 128, "ctx": 128, "instances": 64, "protos": 48},
}


def build_efficientps_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "efficientps_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EfficientPS variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    def sc(v: int, *, min_ch: int = 16) -> int:
        return scale_channels(int(v), float(width_mult), min_ch=min_ch, divisor=8)

    stem = sc(spec["stem"])
    c2 = sc(spec["c2"])
    c3 = sc(spec["c3"])
    c4 = sc(spec["c4"])
    c5 = sc(spec["c5"])
    fpn = sc(spec["fpn"], min_ch=32)
    ctx = sc(spec["ctx"], min_ch=32)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return EfficientPS(
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
        context_channels=int(ctx),
        num_instances=int(instances),
        num_protos=int(protos),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_efficientps_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="efficientps_tiny", width_mult=0.5
    )
    out = m(x)
    print("efficientps_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


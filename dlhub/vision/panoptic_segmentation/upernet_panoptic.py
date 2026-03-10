import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    BackboneC2C3C4C5,
    ProtoNet,
    check_nchw,
    fuse_panoptic,
    masks_from_prototypes,
)


class PPM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, bins: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.bins = tuple(int(b) for b in bins)
        self.proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((b, b)),
                    nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                )
                for b in self.bins
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_in + c_out * len(self.bins), c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs: list[torch.Tensor] = [x]
        for p in self.proj:
            y = p(x)
            outs.append(F.interpolate(y, size=(h, w), mode="nearest"))
        return self.fuse(torch.cat(outs, dim=1))


class UPerNetPanoptic(nn.Module):
    """UPerNet-style panoptic segmentation (toy-first).

    PPM on top feature + FPN fusion for semantic logits, plus a lightweight instance-query+prototype branch.
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
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        fpn = int(fpn_channels)
        if fpn <= 0:
            raise ValueError("fpn_channels must be > 0")
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

        self.ppm = PPM(int(c5_channels), fpn, bins=(1, 2, 3, 6))
        self.l2 = nn.Conv2d(int(c2_channels), fpn, kernel_size=1)
        self.l3 = nn.Conv2d(int(c3_channels), fpn, kernel_size=1)
        self.l4 = nn.Conv2d(int(c4_channels), fpn, kernel_size=1)
        self.l5 = nn.Identity()

        self.p2 = ConvBNAct(fpn, fpn, kernel_size=3, stride=1, act="relu")
        self.p3 = ConvBNAct(fpn, fpn, kernel_size=3, stride=1, act="relu")
        self.p4 = ConvBNAct(fpn, fpn, kernel_size=3, stride=1, act="relu")
        self.p5 = ConvBNAct(fpn, fpn, kernel_size=3, stride=1, act="relu")

        self.semantic_head = nn.Sequential(
            ConvBNAct(fpn, fpn, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(fpn, nt + ns, kernel_size=1, bias=True),
        )

        self.proto = ProtoNet(fpn, np, depth=3, act="relu")
        self.query = nn.Parameter(torch.randn(ni, fpn) * 0.02)
        self.proj = nn.Sequential(nn.Linear(fpn, fpn), nn.ReLU(inplace=True))
        self.cls = nn.Linear(fpn, nt)
        self.coeff = nn.Linear(fpn, np)

        self.num_instances = ni
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)

        p5 = self.l5(self.ppm(c5))
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="nearest")
        p2, p3, p4, p5 = self.p2(p2), self.p3(p3), self.p4(p4), self.p5(p5)

        semantic_logits = self.semantic_head(p2)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(p2)
        pooled = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1)
        base = self.proj(pooled).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        inst = base + q
        query_cls_logits = self.cls(inst)
        coeff = self.coeff(inst)
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
    "upernet_panoptic_tiny": {
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
    "upernet_panoptic_small": {
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
    "upernet_panoptic_base": {
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


def build_upernet_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "upernet_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown UPerNet-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    fpn = scale_channels(int(spec["fpn"]), float(width_mult), min_ch=32, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return UPerNetPanoptic(
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
    m = build_upernet_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="upernet_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("upernet_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")

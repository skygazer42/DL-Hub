import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    ProtoNet,
    check_nchw,
    fuse_panoptic,
    masks_from_prototypes,
)


class HRNetPanoptic(nn.Module):
    """HRNet-style panoptic segmentation (compact-first).

    Maintains a high-resolution branch and a lower-resolution branch with simple fusion,
    plus a prototype+query instance head.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        base_channels: int = 32,
        depth: int = 2,
        num_protos: int = 32,
        num_instances: int = 32,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        base = int(base_channels)
        if base <= 0:
            raise ValueError("base_channels must be > 0")
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        np = int(num_protos)
        if np <= 0:
            raise ValueError("num_protos must be > 0")
        ni = int(num_instances)
        if ni <= 0:
            raise ValueError("num_instances must be > 0")

        # Stem to stride 4.
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), base, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(base, base, kernel_size=3, stride=2, act="relu"),
        )

        # High-res branch (stride 4).
        high: list[nn.Module] = []
        for _ in range(d):
            high.append(ConvBNAct(base, base, kernel_size=3, stride=1, act="relu"))
        self.high = nn.Sequential(*high)

        # Low-res branch (stride 8).
        self.to_low = ConvBNAct(base, base * 2, kernel_size=3, stride=2, act="relu")
        low: list[nn.Module] = []
        for _ in range(d):
            low.append(ConvBNAct(base * 2, base * 2, kernel_size=3, stride=1, act="relu"))
        self.low = nn.Sequential(*low)

        self.low_to_high = nn.Conv2d(base * 2, base, kernel_size=1, bias=True)

        self.semantic_head = nn.Sequential(
            ConvBNAct(base, base, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(base, nt + ns, kernel_size=1, bias=True),
        )

        self.proto = ProtoNet(base, np, depth=3, act="relu")
        self.query = nn.Parameter(torch.randn(ni, base) * 0.02)
        self.proj = nn.Sequential(nn.Linear(base, base), nn.ReLU(inplace=True))
        self.cls = nn.Linear(base, nt)
        self.coeff = nn.Linear(base, np)

        self.num_instances = ni
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        feat = self.stem(x)  # (B,base,H/4,W/4)
        high = self.high(feat)
        low = self.low(self.to_low(high))  # (B,2base,H/8,W/8)
        low_up = F.interpolate(self.low_to_high(low), size=high.shape[-2:], mode="nearest")
        fused = high + low_up

        semantic_logits = self.semantic_head(fused)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        proto = self.proto(fused)  # (B,P,H/4,W/4)
        pooled = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)
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
    "hrnet_panoptic_tiny": {"base": 24, "depth": 1, "protos": 16, "instances": 16},
    "hrnet_panoptic_small": {"base": 32, "depth": 2, "protos": 32, "instances": 32},
    "hrnet_panoptic_base": {"base": 48, "depth": 3, "protos": 48, "instances": 64},
}


def build_hrnet_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "hrnet_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown HRNet-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    base = scale_channels(int(spec["base"]), float(width_mult), min_ch=16, divisor=8)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return HRNetPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        base_channels=int(base),
        depth=int(spec["depth"]),
        num_protos=int(protos),
        num_instances=int(instances),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_hrnet_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="hrnet_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("hrnet_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")

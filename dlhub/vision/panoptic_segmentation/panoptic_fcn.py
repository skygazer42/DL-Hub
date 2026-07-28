import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import (
    BackboneC2C3C4C5,
    ProtoNet,
    check_nchw,
    masks_from_prototypes,
)


class PanopticFCN(nn.Module):
    """Panoptic FCN (compact-first).

    Bottom-up style:
    - semantic logits (stuff + thing)
    - instance masks from per-pixel embeddings + prototype masks (compact simplification)
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
        head_channels: int = 96,
        num_instances: int = 32,
        num_protos: int = 32,
        embed_dim: int = 16,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0 or ns <= 0:
            raise ValueError("num_thing_classes/num_stuff_classes must be > 0")
        hc = int(head_channels)
        if hc <= 0:
            raise ValueError("head_channels must be > 0")
        ni = int(num_instances)
        np = int(num_protos)
        ed = int(embed_dim)
        if ni <= 0 or np <= 0 or ed <= 0:
            raise ValueError("instances/protos/embed_dim must be > 0")

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

        # Use /8 feature (C3).
        self.proj = ConvBNAct(int(c3_channels), hc, kernel_size=1, stride=1, padding=0, act="relu")
        self.semantic = nn.Conv2d(hc, nt + ns, kernel_size=1, bias=True)

        # Pixel embeddings for grouping (compact output).
        self.embed = nn.Sequential(
            ConvBNAct(hc, hc, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(hc, ed, kernel_size=1, bias=True),
        )

        # Prototype masks + instance coefficients (compact).
        self.proto = ProtoNet(hc, np, depth=3, act="relu")
        self.query = nn.Parameter(torch.randn(ni, hc) * 0.02)
        self.q_proj = nn.Linear(hc, hc)
        self.q_cls = nn.Linear(hc, nt)
        self.q_coeff = nn.Linear(hc, np)

        self.num_instances = ni
        self.embed_dim = ed

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        _, c3, _, _ = self.backbone(x)
        feat = self.proj(c3)

        semantic_logits = self.semantic(feat)
        pixel_embed = self.embed(feat)

        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")
        pixel_embed = F.interpolate(pixel_embed, size=(h, w), mode="nearest")

        proto = self.proto(feat)
        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).view(b, -1)
        base = torch.relu(self.q_proj(pooled)).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        hq = base + q
        query_cls_logits = self.q_cls(hq)
        coeff = self.q_coeff(hq)
        mask_logits = masks_from_prototypes(proto, coeff)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        return {
            "semantic_logits": semantic_logits,
            "pixel_embed": pixel_embed,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
        }


_VARIANTS: dict[str, dict] = {
    "panoptic_fcn_tiny": {
        "stem": 24,
        "c2": 24,
        "c3": 48,
        "c4": 64,
        "c5": 96,
        "depth": 1,
        "head": 64,
        "instances": 16,
        "protos": 16,
        "embed": 12,
    },
    "panoptic_fcn_small": {
        "stem": 24,
        "c2": 32,
        "c3": 64,
        "c4": 96,
        "c5": 128,
        "depth": 2,
        "head": 96,
        "instances": 32,
        "protos": 32,
        "embed": 16,
    },
    "panoptic_fcn_base": {
        "stem": 32,
        "c2": 40,
        "c3": 80,
        "c4": 128,
        "c5": 160,
        "depth": 2,
        "head": 128,
        "instances": 64,
        "protos": 48,
        "embed": 24,
    },
}


def build_panoptic_fcn_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "panoptic_fcn_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Panoptic FCN variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]

    def sc(v: int, *, min_ch: int = 16) -> int:
        return scale_channels(int(v), float(width_mult), min_ch=min_ch, divisor=8)

    stem = sc(spec["stem"])
    c2 = sc(spec["c2"])
    c3 = sc(spec["c3"])
    c4 = sc(spec["c4"])
    c5 = sc(spec["c5"])
    head = sc(spec["head"], min_ch=32)
    protos = max(8, int(round(int(spec["protos"]) * float(width_mult))))
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))
    embed = max(8, int(spec["embed"]))

    return PanopticFCN(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        head_channels=int(head),
        num_instances=int(instances),
        num_protos=int(protos),
        embed_dim=int(embed),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_panoptic_fcn_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="panoptic_fcn_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("panoptic_fcn_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean()
        + out["pixel_embed"].mean()
        + out["query_cls_logits"].mean()
        + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")

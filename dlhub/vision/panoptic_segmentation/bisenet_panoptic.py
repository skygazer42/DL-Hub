from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import check_nchw, fuse_panoptic


class BiSeNetPanoptic(nn.Module):
    """BiSeNet-style panoptic segmentation (toy-first).

    Spatial path (detail) + context path (semantics) fused for semantic logits and query masks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        spatial_channels: int = 64,
        context_channels: int = 128,
        fused_channels: int = 96,
        num_instances: int = 32,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        sc = int(spatial_channels)
        cc = int(context_channels)
        fc = int(fused_channels)
        if sc <= 0 or cc <= 0 or fc <= 0:
            raise ValueError("channels must be > 0")
        ni = int(num_instances)
        if ni <= 0:
            raise ValueError("num_instances must be > 0")

        # Spatial path: keep higher resolution (/4).
        self.spatial = nn.Sequential(
            ConvBNAct(int(in_channels), sc, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(sc, sc, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(sc, sc, kernel_size=3, stride=1, act="relu"),
        )

        # Context path: more downsampling (/16).
        self.context = nn.Sequential(
            ConvBNAct(int(in_channels), cc // 2, kernel_size=3, stride=2, act="relu"),  # /2
            ConvBNAct(cc // 2, cc // 2, kernel_size=3, stride=2, act="relu"),  # /4
            ConvBNAct(cc // 2, cc, kernel_size=3, stride=2, act="relu"),  # /8
            ConvBNAct(cc, cc, kernel_size=3, stride=2, act="relu"),  # /16
        )

        self.global_gate = nn.Sequential(
            nn.Linear(cc, cc),
            nn.Sigmoid(),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(sc + cc, fc, kernel_size=1, bias=False),
            nn.BatchNorm2d(fc),
            nn.ReLU(inplace=True),
            ConvBNAct(fc, fc, kernel_size=3, stride=1, act="relu"),
        )

        self.semantic_head = nn.Conv2d(fc, nt + ns, kernel_size=1, bias=True)

        self.query = nn.Parameter(torch.randn(ni, fc) * 0.02)
        self.proj = nn.Sequential(nn.Linear(fc, fc), nn.ReLU(inplace=True))
        self.cls = nn.Linear(fc, nt)
        self.mask_embed = nn.Linear(fc, fc)

        self.num_instances = ni
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        s = self.spatial(x)  # (B,sc,H/4,W/4)
        c = self.context(x)  # (B,cc,H/16,W/16)

        # Global context gating.
        pooled = F.adaptive_avg_pool2d(c, (1, 1)).flatten(1)
        gate = self.global_gate(pooled).view(b, -1, 1, 1)
        c = c * gate

        c_up = F.interpolate(c, size=s.shape[-2:], mode="nearest")
        fused = self.fuse(torch.cat([s, c_up], dim=1))

        semantic_logits4 = self.semantic_head(fused)
        semantic_logits = F.interpolate(semantic_logits4, size=(h, w), mode="nearest")

        pooled_f = F.adaptive_avg_pool2d(fused, (1, 1)).flatten(1)
        base = self.proj(pooled_f).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        inst = base + q
        query_cls_logits = self.cls(inst)
        me = self.mask_embed(inst)
        mask_flat = torch.bmm(me, fused.flatten(2))
        mask_logits4 = mask_flat.view(b, int(self.num_instances), fused.shape[-2], fused.shape[-1])
        mask_logits = F.interpolate(mask_logits4, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "bisenet_panoptic_tiny": {"spatial": 48, "context": 96, "fused": 64, "instances": 16},
    "bisenet_panoptic_small": {"spatial": 64, "context": 128, "fused": 96, "instances": 32},
    "bisenet_panoptic_base": {"spatial": 96, "context": 192, "fused": 128, "instances": 64},
}


def build_bisenet_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "bisenet_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BiSeNet-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    spatial = scale_channels(int(spec["spatial"]), float(width_mult), min_ch=16, divisor=8)
    context = scale_channels(int(spec["context"]), float(width_mult), min_ch=16, divisor=8)
    fused = scale_channels(int(spec["fused"]), float(width_mult), min_ch=16, divisor=8)
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return BiSeNetPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        spatial_channels=int(spatial),
        context_channels=int(context),
        fused_channels=int(fused),
        num_instances=int(instances),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_bisenet_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="bisenet_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("bisenet_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


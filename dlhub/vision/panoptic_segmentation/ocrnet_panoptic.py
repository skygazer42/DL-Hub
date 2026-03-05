from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneC2C3C4C5, check_nchw, fuse_panoptic


class OCRNetPanoptic(nn.Module):
    """OCRNet-style panoptic segmentation (toy-first).

    Uses coarse logits to compute object-context features, refines semantic logits,
    and adds a lightweight query-based instance mask head.
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
        feat_channels: int = 96,
        num_instances: int = 32,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        fc = int(feat_channels)
        if fc <= 0:
            raise ValueError("feat_channels must be > 0")
        ni = int(num_instances)
        if ni <= 0:
            raise ValueError("num_instances must be > 0")

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

        self.proj = ConvBNAct(int(c4_channels), fc, kernel_size=1, stride=1, padding=0, act="relu")
        self.coarse = nn.Conv2d(fc, nt + ns, kernel_size=1, bias=True)
        self.refine_feat = ConvBNAct(fc * 2, fc, kernel_size=3, stride=1, act="relu")
        self.semantic_head = nn.Conv2d(fc, nt + ns, kernel_size=1, bias=True)

        self.query = nn.Parameter(torch.randn(ni, fc) * 0.02)
        self.proj_q = nn.Sequential(nn.Linear(fc, fc), nn.ReLU(inplace=True))
        self.cls = nn.Linear(fc, nt)
        self.mask_embed = nn.Linear(fc, fc)

        self.num_instances = ni
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        _, _, c4, _ = self.backbone(x)  # /16 feature
        feat = self.proj(c4)
        coarse = self.coarse(feat)  # (B,nt+ns,H16,W16)

        b, c, h16, w16 = feat.shape
        k = coarse.shape[1]
        n = h16 * w16

        probs = torch.softmax(coarse, dim=1).view(b, k, n)  # (B,K,N)
        f = feat.view(b, c, n)  # (B,C,N)

        region = torch.bmm(probs, f.transpose(1, 2))  # (B,K,C)
        context = torch.bmm(region.transpose(1, 2), probs).view(b, c, h16, w16)  # (B,C,H16,W16)

        refined = self.refine_feat(torch.cat([feat, context], dim=1))  # (B,fc,H16,W16)
        semantic_logits16 = self.semantic_head(refined)
        semantic_logits = F.interpolate(semantic_logits16, size=(h, w), mode="nearest")

        pooled = F.adaptive_avg_pool2d(refined, (1, 1)).flatten(1)
        base = self.proj_q(pooled).unsqueeze(1)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        inst = base + q
        query_cls_logits = self.cls(inst)
        me = self.mask_embed(inst)
        mask_flat = torch.bmm(me, refined.flatten(2))
        mask_logits16 = mask_flat.view(b, int(self.num_instances), h16, w16)
        mask_logits = F.interpolate(mask_logits16, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "ocrnet_panoptic_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "feat": 64, "instances": 16},
    "ocrnet_panoptic_small": {"stem": 24, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "feat": 96, "instances": 32},
    "ocrnet_panoptic_base": {"stem": 32, "c2": 40, "c3": 80, "c4": 128, "c5": 160, "depth": 2, "feat": 128, "instances": 64},
}


def build_ocrnet_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "ocrnet_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown OCRNet-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
    c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
    c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
    c5 = scale_channels(int(spec["c5"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return OCRNetPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        stem_channels=int(stem),
        c2_channels=int(c2),
        c3_channels=int(c3),
        c4_channels=int(c4),
        c5_channels=int(c5),
        depth=int(spec["depth"]),
        feat_channels=int(feat),
        num_instances=int(instances),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_ocrnet_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="ocrnet_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("ocrnet_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


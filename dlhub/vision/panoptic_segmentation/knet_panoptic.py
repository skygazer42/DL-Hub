from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneC2C3C4C5, FPN4, check_nchw, fuse_panoptic


class KNetPanoptic(nn.Module):
    """K-Net-style panoptic segmentation (toy-first).

    Maintains a set of mask kernels (queries) that are iteratively refined and applied to a mask feature map.
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
        mask_dim: int = 96,
        num_kernels: int = 32,
        num_iters: int = 3,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        nk = int(num_kernels)
        if nk <= 0:
            raise ValueError("num_kernels must be > 0")
        iters = int(num_iters)
        if iters <= 0:
            raise ValueError("num_iters must be > 0")
        md = int(mask_dim)
        if md <= 0:
            raise ValueError("mask_dim must be > 0")

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
        self.fpn = FPN4((int(c2_channels), int(c3_channels), int(c4_channels), int(c5_channels)), int(fpn_channels), act="relu")

        self.semantic = nn.Sequential(
            ConvBNAct(int(fpn_channels), int(fpn_channels), kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(int(fpn_channels), nt + ns, kernel_size=1, bias=True),
        )

        self.mask_feat = nn.Sequential(
            nn.Conv2d(int(fpn_channels), md, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            ConvBNAct(md, md, kernel_size=3, stride=1, act="relu"),
        )

        self.kernels = nn.Parameter(torch.randn(nk, md) * 0.02)
        hidden = max(8, int(round(md * float(mlp_ratio))))
        self.update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(md),
                    nn.Linear(md, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, md),
                )
                for _ in range(iters)
            ]
        )

        self.cls = nn.Linear(md, nt)
        self.mask_embed = nn.Linear(md, md)

        self.num_kernels = nk
        self.num_iters = iters
        self.num_thing_classes = nt
        self.num_stuff_classes = ns
        self.mask_dim = md

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2, _, _, _ = self.fpn(c2, c3, c4, c5)

        semantic_logits = self.semantic(p2)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        mf = self.mask_feat(p2)  # (B,MD,H/4,W/4)
        ctx = F.adaptive_avg_pool2d(mf, (1, 1)).view(b, 1, -1)  # (B,1,MD)

        k = self.kernels.unsqueeze(0).expand(b, -1, -1)  # (B,NK,MD)
        for upd in self.update:
            k = k + upd(k + ctx)

        query_cls_logits = self.cls(k)  # (B,NK,nt)
        me = self.mask_embed(k)  # (B,NK,MD)

        mf_flat = mf.flatten(2)  # (B,MD,HW)
        mask_flat = torch.bmm(me, mf_flat)  # (B,NK,HW)
        mask_logits = mask_flat.view(b, int(self.num_kernels), mf.shape[-2], mf.shape[-1])
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "kernels": k,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "knet_panoptic_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "fpn": 64, "mask": 64, "kernels": 16, "iters": 2},
    "knet_panoptic_small": {"stem": 24, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96, "mask": 96, "kernels": 32, "iters": 3},
    "knet_panoptic_base": {"stem": 32, "c2": 40, "c3": 80, "c4": 128, "c5": 160, "depth": 2, "fpn": 128, "mask": 128, "kernels": 64, "iters": 4},
}


def build_knet_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "knet_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown KNet-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    def sc(v: int, *, min_ch: int = 16) -> int:
        return scale_channels(int(v), float(width_mult), min_ch=min_ch, divisor=8)

    stem = sc(spec["stem"])
    c2 = sc(spec["c2"])
    c3 = sc(spec["c3"])
    c4 = sc(spec["c4"])
    c5 = sc(spec["c5"])
    fpn = sc(spec["fpn"], min_ch=32)
    mask = sc(spec["mask"], min_ch=32)
    kernels = max(4, int(round(int(spec["kernels"]) * float(width_mult))))

    return KNetPanoptic(
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
        mask_dim=int(mask),
        num_kernels=int(kernels),
        num_iters=int(spec["iters"]),
        mlp_ratio=2.0,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_knet_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="knet_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("knet_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


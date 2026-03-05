from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.panoptic_segmentation._common import BackboneC2C3C4C5, FPN4, check_nchw, fuse_panoptic


class MaXDeepLabPanoptic(nn.Module):
    """MaX-DeepLab-style panoptic segmentation (toy-first).

    Uses a global memory token set concatenated with pixel tokens for a Transformer encoder.
    Memory tokens serve as instance queries; pixel tokens serve as the segmentation embedding map.
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
        embed_dim: int = 96,
        encoder_layers: int = 3,
        num_heads: int = 4,
        num_mem_tokens: int = 32,
        num_instances: int = 16,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        ed = int(embed_dim)
        if ed <= 0:
            raise ValueError("embed_dim must be > 0")
        h = int(num_heads)
        if h <= 0 or ed % h != 0:
            raise ValueError("num_heads must be > 0 and divide embed_dim")
        el = int(encoder_layers)
        if el <= 0:
            raise ValueError("encoder_layers must be > 0")
        m = int(num_mem_tokens)
        if m <= 0:
            raise ValueError("num_mem_tokens must be > 0")
        ni = int(num_instances)
        if ni <= 0 or ni > m:
            raise ValueError("num_instances must be in [1, num_mem_tokens]")

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

        self.pix_proj = nn.Conv2d(int(fpn_channels), ed, kernel_size=1, bias=True)
        self.mem = nn.Parameter(torch.randn(m, ed) * 0.02)

        ff = int(round(ed * float(mlp_ratio)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ed,
            nhead=h,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=el)

        self.semantic_head = nn.Sequential(
            ConvBNAct(ed, ed, kernel_size=3, stride=1, act="relu"),
            nn.Conv2d(ed, nt + ns, kernel_size=1, bias=True),
        )

        self.cls = nn.Linear(ed, nt)
        self.mask_embed = nn.Linear(ed, ed)

        self.num_mem_tokens = m
        self.num_instances = ni
        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape
        c2, c3, c4, c5 = self.backbone(x)
        p2, _, _, _ = self.fpn(c2, c3, c4, c5)

        pix = self.pix_proj(p2)  # (B,ED,H/4,W/4)
        b, ed, h4, w4 = pix.shape
        pix_tok = pix.permute(0, 2, 3, 1).reshape(b, h4 * w4, ed)  # (B,N,ED)

        mem = self.mem.unsqueeze(0).expand(b, -1, -1)  # (B,M,ED)
        tok = torch.cat([mem, pix_tok], dim=1)  # (B,M+N,ED)
        tok = self.encoder(tok)

        mem_out = tok[:, : self.num_mem_tokens]
        pix_out = tok[:, self.num_mem_tokens :]

        pix_feat = pix_out.view(b, h4, w4, ed).permute(0, 3, 1, 2).contiguous()
        semantic_logits = self.semantic_head(pix_feat)
        semantic_logits = F.interpolate(semantic_logits, size=(h, w), mode="nearest")

        queries = mem_out[:, : self.num_instances]  # (B,I,ED)
        query_cls_logits = self.cls(queries)
        me = self.mask_embed(queries)
        pix_flat = pix_feat.flatten(2)  # (B,ED,HW)
        mask_flat = torch.bmm(me, pix_flat)
        mask_logits = mask_flat.view(b, int(self.num_instances), h4, w4)
        mask_logits = F.interpolate(mask_logits, size=(h, w), mode="nearest")

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes))

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "mem_tokens": mem_out,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "max_deeplab_panoptic_tiny": {"stem": 24, "c2": 24, "c3": 48, "c4": 64, "c5": 96, "depth": 1, "fpn": 64, "embed": 64, "enc": 2, "heads": 4, "mem": 16, "inst": 8},
    "max_deeplab_panoptic_small": {"stem": 24, "c2": 32, "c3": 64, "c4": 96, "c5": 128, "depth": 2, "fpn": 96, "embed": 96, "enc": 3, "heads": 4, "mem": 32, "inst": 16},
    "max_deeplab_panoptic_base": {"stem": 32, "c2": 40, "c3": 80, "c4": 128, "c5": 160, "depth": 2, "fpn": 128, "embed": 128, "enc": 4, "heads": 8, "mem": 64, "inst": 32},
}


def build_max_deeplab_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "max_deeplab_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MaX-DeepLab-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]

    def sc(v: int, *, min_ch: int = 16) -> int:
        return scale_channels(int(v), float(width_mult), min_ch=min_ch, divisor=8)

    stem = sc(spec["stem"])
    c2 = sc(spec["c2"])
    c3 = sc(spec["c3"])
    c4 = sc(spec["c4"])
    c5 = sc(spec["c5"])
    fpn = sc(spec["fpn"], min_ch=32)
    embed = sc(spec["embed"], min_ch=32)

    heads = int(spec["heads"])
    while heads > 1 and int(embed) % heads != 0:
        heads -= 1

    mem = max(4, int(round(int(spec["mem"]) * float(width_mult))))
    inst = max(1, min(mem, int(round(int(spec["inst"]) * float(width_mult)))))

    return MaXDeepLabPanoptic(
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
        embed_dim=int(embed),
        encoder_layers=int(spec["enc"]),
        num_heads=int(heads),
        num_mem_tokens=int(mem),
        num_instances=int(inst),
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_max_deeplab_panoptic_segmenter(
        in_channels=3, num_thing_classes=3, num_stuff_classes=2, variant="max_deeplab_panoptic_tiny", width_mult=0.5
    )
    out = m(x)
    print("max_deeplab_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    loss.backward()
    print("ok")


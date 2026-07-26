from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct

from ._common import (
    CoSegHead,
    GroupFusionBlock,
    TinyCoSegEncoder,
    check_btchw,
    flatten_group,
    logits_to_masks,
    unflatten_group,
)

_VARIANTS: dict[str, dict[str, int]] = {
    "co_attention_fpn_tiny": {"width": 16, "depth": 1},
    "co_attention_fpn_small": {"width": 24, "depth": 2},
    "co_attention_fpn_base": {"width": 32, "depth": 3},
}


class CoAttentionFPN(nn.Module):
    """Multi-scale co-attention fusion network."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TinyCoSegEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c1, c2, c3 = (int(x) for x in self.encoder.out_channels)
        hidden = max(32, c2)
        self.fuse1 = GroupFusionBlock(c1, mode="attention")
        self.fuse2 = GroupFusionBlock(c2, mode="attention")
        self.fuse3 = GroupFusionBlock(c3, mode="attention")
        self.merge = ConvBNAct(c1 + c2 + c3, hidden, kernel_size=3, stride=1, act="relu")
        self.head = CoSegHead(
            in_channels=hidden,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = check_btchw(images)
        b, t, _, h, w = images.shape
        flat = flatten_group(images)
        c1, c2, c3 = self.encoder(flat)
        g1, a1 = self.fuse1(unflatten_group(c1, batch=b, set_size=t))
        g2, a2 = self.fuse2(unflatten_group(c2, batch=b, set_size=t))
        g3, a3 = self.fuse3(unflatten_group(c3, batch=b, set_size=t))

        u2 = F.interpolate(
            flatten_group(g2), size=c1.shape[-2:], mode="bilinear", align_corners=False
        )
        u3 = F.interpolate(
            flatten_group(g3), size=c1.shape[-2:], mode="bilinear", align_corners=False
        )
        fused = self.merge(torch.cat([flatten_group(g1), u2, u3], dim=1))
        logits = self.head(unflatten_group(fused, batch=b, set_size=t), out_hw=(h, w))
        masks = logits_to_masks(logits)
        return {
            "logits": logits,
            "masks": masks,
            "co_attention": (
                a1["co_attention"],
                a2["co_attention"],
                a3["co_attention"],
            ),
        }


def build_co_attention_fpn_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "co_attention_fpn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Co-Attention-FPN variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CoAttentionFPN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_co_attention_fpn_co_segmentor(
        in_channels=3,
        num_classes=2,
        variant="co_attention_fpn_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("co_attention_fpn_tiny", tuple(out["logits"].shape), len(out["co_attention"]))
    loss = out["logits"].mean()
    for attn in out["co_attention"]:
        loss = loss + attn.mean()
    loss.backward()
    print("ok")

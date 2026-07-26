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
    "cosal_uformer_tiny": {"width": 16, "depth": 1},
    "cosal_uformer_small": {"width": 24, "depth": 2},
    "cosal_uformer_base": {"width": 32, "depth": 3},
}


class CoSalUFormer(nn.Module):
    """U-shaped decoder with group consensus injection."""

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
        self.fuse3 = GroupFusionBlock(c3, mode="consensus")
        self.fuse2 = GroupFusionBlock(hidden, mode="mean")
        self.decode2 = ConvBNAct(c2 + c3, hidden, kernel_size=3, stride=1, act="relu")
        self.decode1 = ConvBNAct(c1 + hidden, hidden, kernel_size=3, stride=1, act="relu")
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
        g3 = unflatten_group(c3, batch=b, set_size=t)
        g3, _ = self.fuse3(g3)

        up3 = F.interpolate(
            flatten_group(g3), size=c2.shape[-2:], mode="bilinear", align_corners=False
        )
        d2 = self.decode2(torch.cat([c2, up3], dim=1))
        gd2 = unflatten_group(d2, batch=b, set_size=t)
        gd2, _ = self.fuse2(gd2)

        up2 = F.interpolate(
            flatten_group(gd2), size=c1.shape[-2:], mode="bilinear", align_corners=False
        )
        d1 = self.decode1(torch.cat([c1, up2], dim=1))
        logits = self.head(unflatten_group(d1, batch=b, set_size=t), out_hw=(h, w))
        masks = logits_to_masks(logits)
        consensus_map = torch.softmax(logits, dim=2).mean(dim=1)
        return {
            "logits": logits,
            "masks": masks,
            "consensus_map": consensus_map,
        }


def build_cosal_uformer_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "cosal_uformer_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown CoSal-UFormer variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CoSalUFormer(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_cosal_uformer_co_segmentor(
        in_channels=3,
        num_classes=2,
        variant="cosal_uformer_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("cosal_uformer_tiny", tuple(out["logits"].shape), tuple(out["consensus_map"].shape))
    loss = out["logits"].mean() + out["consensus_map"].mean()
    loss.backward()
    print("ok")

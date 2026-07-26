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
    "siamese_coseg_tiny": {"width": 16, "depth": 1},
    "siamese_coseg_small": {"width": 24, "depth": 2},
    "siamese_coseg_base": {"width": 32, "depth": 3},
}


class SiameseCoSeg(nn.Module):
    """Shared encoder with group prototype matching."""

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
        c3 = int(self.encoder.out_channels[-1])
        self.group_fuse = GroupFusionBlock(c3, mode="mean")
        self.refine = ConvBNAct(c3 + 1, c3, kernel_size=3, stride=1, act="relu")
        self.head = CoSegHead(
            in_channels=c3,
            hidden_channels=max(32, c3),
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = check_btchw(images)
        b, t, _, h, w = images.shape
        flat = flatten_group(images)
        _, _, c3 = self.encoder(flat)
        g3 = unflatten_group(c3, batch=b, set_size=t)
        fused, _ = self.group_fuse(g3)

        proto = g3.mean(dim=1, keepdim=True).expand_as(g3)
        match = (F.normalize(g3, dim=2) * F.normalize(proto, dim=2)).sum(dim=2, keepdim=True)
        refined = self.refine(torch.cat([flatten_group(fused), flatten_group(match)], dim=1))
        refined = unflatten_group(refined, batch=b, set_size=t)

        logits = self.head(refined, out_hw=(h, w))
        masks = logits_to_masks(logits)
        return {
            "logits": logits,
            "masks": masks,
            "group_tokens": fused.mean(dim=(-1, -2)),
            "match_map": F.interpolate(
                flatten_group(match), size=(h, w), mode="bilinear", align_corners=False
            ).view(b, t, 1, h, w),
        }


def build_siamese_coseg_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "siamese_coseg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Siamese-CoSeg variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return SiameseCoSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_siamese_coseg_co_segmentor(
        in_channels=3,
        num_classes=2,
        variant="siamese_coseg_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("siamese_coseg_tiny", tuple(out["logits"].shape), tuple(out["masks"].shape))
    loss = out["logits"].mean() + out["match_map"].mean()
    loss.backward()
    print("ok")

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct

from ._common import CoSegHead, check_btchw, logits_to_masks

_VARIANTS: dict[str, dict[str, int]] = {
    "patchmatch_coseg_tiny": {"embed_dim": 48, "depth": 2, "heads": 4},
    "patchmatch_coseg_small": {"embed_dim": 64, "depth": 3, "heads": 4},
    "patchmatch_coseg_base": {"embed_dim": 96, "depth": 4, "heads": 8},
}


class PatchmatchCoSeg(nn.Module):
    """Token-level group transformer for co-segmentation."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(embed_dim)
        self.patch = nn.Sequential(
            nn.Conv2d(int(in_channels), dim, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.group_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(heads),
            dim_feedforward=max(64, dim * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth)))
        self.refine = ConvBNAct(dim * 2, dim, kernel_size=3, stride=1, act="relu")
        self.head = CoSegHead(
            in_channels=dim,
            hidden_channels=max(32, dim),
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = check_btchw(images)
        b, t, c, h, w = images.shape
        flat = images.reshape(b * t, c, h, w)
        feat = self.patch(flat)
        _, dim, hp, wp = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).reshape(b, t * hp * wp, dim)
        group_token = self.group_token.expand(b, -1, -1)
        encoded = self.encoder(torch.cat([group_token, tokens], dim=1))
        group_tokens = encoded[:, :1, :]
        patch_tokens = encoded[:, 1:, :].reshape(b * t, hp * wp, dim)
        patch_feat = patch_tokens.transpose(1, 2).reshape(b, t, dim, hp, wp)
        group_feat = group_tokens.reshape(b, 1, dim, 1, 1).expand(-1, t, -1, hp, wp)
        refined = self.refine(
            torch.cat(
                [
                    patch_feat.reshape(b * t, dim, hp, wp),
                    group_feat.reshape(b * t, dim, hp, wp),
                ],
                dim=1,
            )
        )
        refined = refined.reshape(b, t, dim, hp, wp)
        logits = self.head(refined, out_hw=(h, w))
        masks = logits_to_masks(logits)
        return {
            "logits": logits,
            "masks": masks,
            "group_tokens": group_tokens,
        }


def build_patchmatch_coseg_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "patchmatch_coseg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Transformer-CoSeg variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    dim = max(32, int(int(cfg["embed_dim"]) * float(width_mult)))
    heads = int(cfg["heads"])
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return PatchmatchCoSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=dim,
        depth=int(cfg["depth"]),
        heads=int(heads),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_patchmatch_coseg_co_segmentor(
        in_channels=3,
        num_classes=2,
        variant="patchmatch_coseg_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("patchmatch_coseg_tiny", tuple(out["logits"].shape), tuple(out["group_tokens"].shape))
    loss = out["logits"].mean() + out["group_tokens"].mean()
    loss.backward()
    print("ok")


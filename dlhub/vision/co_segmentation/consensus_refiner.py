from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct

from ._common import (
    CoSegHead,
    TinyCoSegEncoder,
    check_btchw,
    flatten_group,
    logits_to_masks,
    unflatten_group,
)

_VARIANTS: dict[str, dict[str, int]] = {
    "consensus_refiner_tiny": {"width": 16, "depth": 1},
    "consensus_refiner_small": {"width": 24, "depth": 2},
    "consensus_refiner_base": {"width": 32, "depth": 3},
}


class ConsensusRefiner(nn.Module):
    """Coarse mask prediction plus group consensus refinement."""

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
        self.num_classes = int(num_classes)
        self.encoder = TinyCoSegEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c3 = int(self.encoder.out_channels[-1])
        self.coarse_head = CoSegHead(
            in_channels=c3,
            hidden_channels=max(32, c3),
            num_classes=self.num_classes,
            dropout=float(dropout),
        )
        self.consensus_proj = nn.Sequential(
            nn.Conv2d(self.num_classes, c3, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.refine = ConvBNAct(c3 * 2, c3, kernel_size=3, stride=1, act="relu")
        self.refine_head = CoSegHead(
            in_channels=c3,
            hidden_channels=max(32, c3),
            num_classes=self.num_classes,
            dropout=float(dropout),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = check_btchw(images)
        b, t, _, h, w = images.shape
        flat = flatten_group(images)
        _, _, c3 = self.encoder(flat)
        g3 = unflatten_group(c3, batch=b, set_size=t)

        coarse_logits = self.coarse_head(g3, out_hw=(h, w))
        consensus = (
            torch.softmax(coarse_logits, dim=2).mean(dim=1, keepdim=True).expand(-1, t, -1, -1, -1)
        )
        consensus_low = F.interpolate(
            consensus.reshape(b * t, self.num_classes, h, w),
            size=c3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        consensus_feat = self.consensus_proj(consensus_low)
        refined = self.refine(torch.cat([c3, consensus_feat], dim=1))
        refined = unflatten_group(refined, batch=b, set_size=t)

        logits = coarse_logits + self.refine_head(refined, out_hw=(h, w))
        masks = logits_to_masks(logits)
        return {
            "logits": logits,
            "masks": masks,
            "coarse_logits": coarse_logits,
            "consensus_map": consensus[:, 0],
        }


def build_consensus_refiner_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "consensus_refiner_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Consensus-Refiner variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ConsensusRefiner(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_consensus_refiner_co_segmentor(
        in_channels=3,
        num_classes=2,
        variant="consensus_refiner_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("consensus_refiner_tiny", tuple(out["logits"].shape), tuple(out["consensus_map"].shape))
    loss = out["logits"].mean() + out["consensus_map"].mean()
    loss.backward()
    print("ok")

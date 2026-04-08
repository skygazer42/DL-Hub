from __future__ import annotations

import torch
from torch import nn

from ._common import (
    CoSegHead,
    GroupFusionBlock,
    TinyCoSegEncoder,
    check_btchw,
    flatten_group,
    logits_to_masks,
    unflatten_group,
)


class CrossViewCosegCoSegmentor(nn.Module):
    """Toy co-segmentation model for the cross_view_coseg family."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        fusion_mode: str,
        num_prototypes: int,
        use_prompt_tokens: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TinyCoSegEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        high_channels = int(self.encoder.out_channels[-1])
        self.prompt = (
            nn.Parameter(torch.randn(1, high_channels, 1, 1) * 0.02) if use_prompt_tokens else None
        )
        self.fusion = GroupFusionBlock(
            high_channels,
            mode=str(fusion_mode),
            num_prototypes=int(num_prototypes),
        )
        self.head = CoSegHead(
            in_channels=high_channels,
            hidden_channels=max(32, high_channels),
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = check_btchw(images)
        b, t, c, h, w = images.shape
        flat = flatten_group(images)
        _, _, feat = self.encoder(flat)
        grouped = unflatten_group(feat, batch=b, set_size=t)
        if self.prompt is not None:
            grouped = grouped + self.prompt.unsqueeze(0).expand(b, t, -1, grouped.shape[-2], grouped.shape[-1])
        fused, aux = self.fusion(grouped)
        logits = self.head(fused, out_hw=(h, w))
        masks = logits_to_masks(logits)
        out = {"logits": logits, "masks": masks}
        out.update(aux)
        return out


_VARIANTS: dict[str, dict[str, int | float]] = {
    "cross_view_coseg_tiny": {"width": 16, "depth": 1, "dropout": 0.0},
    "cross_view_coseg_small": {"width": 24, "depth": 2, "dropout": 0.0},
    "cross_view_coseg_base": {"width": 32, "depth": 3, "dropout": 0.1},
}


def build_cross_view_coseg_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "cross_view_coseg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown cross_view_coseg variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CrossViewCosegCoSegmentor(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        fusion_mode="consensus",
        num_prototypes=4,
        use_prompt_tokens=False,
        dropout=float(dropout if dropout > 0 else cfg["dropout"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_cross_view_coseg_co_segmentor(
        in_channels=3,
        num_classes=2,
        variant="cross_view_coseg_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("cross_view_coseg_tiny", {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})
    loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
    loss.backward()
    print("ok")

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

_VARIANTS: dict[str, dict[str, int]] = {
    "prototype_memory_coseg_tiny": {"width": 16, "depth": 1},
    "prototype_memory_coseg_small": {"width": 24, "depth": 2},
    "prototype_memory_coseg_base": {"width": 32, "depth": 3},
}


class PrototypeMemoryCoseg(nn.Module):
    def __init__(
        self, *, in_channels: int, num_classes: int, width: int, depth: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.encoder = TinyCoSegEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c3 = int(self.encoder.out_channels[-1])
        self.fuser = GroupFusionBlock(c3, mode="prototype", num_prototypes=8)
        self.prompt = None
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
        grouped = unflatten_group(c3, batch=b, set_size=t)
        fused, aux = self.fuser(grouped)

        logits = self.head(fused, out_hw=(h, w))
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "masks": logits_to_masks(logits),
        }
        for key, value in aux.items():
            if isinstance(value, torch.Tensor):
                out[key] = value
        return out


def build_prototype_memory_coseg_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "prototype_memory_coseg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    cfg = _VARIANTS[str(variant).lower().strip()]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return PrototypeMemoryCoseg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_prototype_memory_coseg_co_segmentor(
        in_channels=3, num_classes=2, variant="prototype_memory_coseg_tiny", width_mult=0.5
    )
    out = m(x)
    print(
        "prototype_memory_coseg_tiny",
        {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)},
    )
    loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
    loss.backward()
    print("ok")

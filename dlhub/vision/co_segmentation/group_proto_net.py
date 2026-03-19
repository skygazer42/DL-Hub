from __future__ import annotations

import torch
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
    "group_proto_net_tiny": {"width": 16, "depth": 1, "prototypes": 4},
    "group_proto_net_small": {"width": 24, "depth": 2, "prototypes": 6},
    "group_proto_net_base": {"width": 32, "depth": 3, "prototypes": 8},
}


class GroupProtoNet(nn.Module):
    """Group prototype mining and broadcast refinement."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        num_prototypes: int,
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
        self.proto = GroupFusionBlock(c3, mode="prototype", num_prototypes=int(num_prototypes))
        self.refine = ConvBNAct(c3 * 2, c3, kernel_size=3, stride=1, act="relu")
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
        fused, aux = self.proto(g3)

        proto = aux["group_tokens"]
        bridge = aux["prototype_assign"]
        proto_context = torch.einsum("btp,bpc->btc", bridge, proto)
        proto_context = proto_context.unsqueeze(-1).unsqueeze(-1).expand_as(g3)

        refined = self.refine(torch.cat([flatten_group(fused), flatten_group(proto_context)], dim=1))
        refined = unflatten_group(refined, batch=b, set_size=t)
        logits = self.head(refined, out_hw=(h, w))
        masks = logits_to_masks(logits)
        prototype_masks = logits.mean(dim=2)
        return {
            "logits": logits,
            "masks": masks,
            "group_tokens": proto,
            "prototype_masks": prototype_masks,
        }


def build_group_proto_net_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "group_proto_net_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Group-Proto-Net variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return GroupProtoNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        num_prototypes=int(cfg["prototypes"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_group_proto_net_co_segmentor(
        in_channels=3,
        num_classes=2,
        variant="group_proto_net_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("group_proto_net_tiny", tuple(out["logits"].shape), tuple(out["prototype_masks"].shape))
    loss = out["logits"].mean() + out["prototype_masks"].mean()
    loss.backward()
    print("ok")

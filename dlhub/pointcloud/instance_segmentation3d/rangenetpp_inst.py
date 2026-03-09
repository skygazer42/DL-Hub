
import torch
from torch import nn

from ._common import QueryMaskHead, RangeViewEncoder


_VARIANTS: dict[str, dict[str, object]] = {
    "rangenetpp_inst_tiny": {"width": 64, "h": 32, "w": 96, "queries": 16},
    "rangenetpp_inst_small": {"width": 96, "h": 48, "w": 128, "queries": 24},
    "rangenetpp_inst_base": {"width": 128, "h": 64, "w": 160, "queries": 32},
}


class RangeNetPPInst(nn.Module):
    """RangeNet++ instance segmentation (toy): range-view features + query masks."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        h: int,
        w: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = RangeViewEncoder(int(in_channels), int(width), h=int(h), w=int(w), dropout=float(dropout))
        self.head = QueryMaskHead(int(width), int(num_classes), num_queries=int(num_queries), dropout=float(dropout))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_rangenetpp_inst_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "rangenetpp_inst_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return RangeNetPPInst(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        h=int(cfg["h"]),
        w=int(cfg["w"]),
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_rangenetpp_inst_instance_segmenter3d(in_channels=3, num_classes=6, variant="rangenetpp_inst_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


import torch
from torch import nn

from ._common import QueryMaskHead, TransformerPointEncoder

_VARIANTS: dict[str, dict[str, object]] = {
    "groupformer3d_tiny": {"d_model": 64, "depth": 2, "queries": 16},
    "groupformer3d_small": {"d_model": 96, "depth": 3, "queries": 24},
    "groupformer3d_base": {"d_model": 128, "depth": 4, "queries": 32},
}


class Groupformer3D(nn.Module):
    """Groupformer3D (compact): transformer point encoder + query mask head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        depth: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.enc = TransformerPointEncoder(
            int(in_channels), int(d_model), depth=int(depth), dropout=float(dropout)
        )
        self.head = QueryMaskHead(
            int(d_model), int(num_classes), num_queries=int(num_queries), dropout=float(dropout)
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        return self.head(xyz, feat)


def build_groupformer3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "groupformer3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return Groupformer3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        depth=int(cfg["depth"]),
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_groupformer3d_instance_segmenter3d(
        in_channels=3, num_classes=6, variant="groupformer3d_tiny"
    )
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

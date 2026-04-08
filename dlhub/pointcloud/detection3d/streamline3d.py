import torch
from torch import nn

from ._common import PointQueryDetector3D, check_points

_VARIANTS: dict[str, dict[str, object]] = {
    "streamline3d_tiny": {"d_model": 64, "queries": 32},
    "streamline3d_small": {"d_model": 96, "queries": 48},
    "streamline3d_base": {"d_model": 128, "queries": 64},
}


class Streamline3D(nn.Module):
    """3DETR (toy): point transformer encoder + DETR-like query head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        num_queries: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.det = PointQueryDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            d_model=int(d_model),
            num_queries=int(num_queries),
            use_transformer=True,
            dropout=float(dropout),
            with_yaw=True,
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        return self.det(points)


def build_streamline3d_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "streamline3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return Streamline3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_streamline3d_detector3d(in_channels=3, num_classes=5, variant="streamline3d_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


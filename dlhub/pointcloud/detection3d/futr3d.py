import torch
from torch import nn

from ._common import PointQueryDetector3D

_VARIANTS: dict[str, dict[str, int]] = {
    "futr3d_tiny": {"d_model": 64, "queries": 32},
    "futr3d_small": {"d_model": 96, "queries": 48},
    "futr3d_base": {"d_model": 128, "queries": 64},
}


class Futr3dDetector3D(nn.Module):
    """Toy query-based 3D detector for the futr3d family."""

    def __init__(self, *, in_channels: int, num_classes: int, d_model: int, num_queries: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.det = PointQueryDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            d_model=int(d_model),
            num_queries=int(num_queries),
            use_transformer=True,
            with_yaw=True,
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.det(points)


def build_futr3d_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "futr3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return Futr3dDetector3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        num_queries=int(cfg["queries"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_futr3d_detector3d(in_channels=3, num_classes=3, variant="futr3d_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

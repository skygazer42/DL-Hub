
import torch
from torch import nn

from ._common import BEVBoxSpec, BEVAnchorFreeDetector3D, PointNetEncoder, check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "pointpainting_tiny": {"width": 64, "bev_h": 32, "bev_w": 32, "topk": 64},
    "pointpainting_small": {"width": 96, "bev_h": 40, "bev_w": 40, "topk": 96},
    "pointpainting_base": {"width": 128, "bev_h": 48, "bev_w": 48, "topk": 128},
}


class PointPainting(nn.Module):
    """PointPainting (toy): paint points with semantic logits, then run a BEV detector."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        bev_h: int,
        bev_w: int,
        topk: int,
        painted_classes: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.paint = nn.Sequential(
            nn.Linear(int(in_channels), int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(painted_classes)),
        )
        self.det = BEVAnchorFreeDetector3D(
            in_channels=int(in_channels) + int(painted_classes),
            num_classes=int(num_classes),
            width=int(width),
            bev=BEVBoxSpec(h=int(bev_h), w=int(bev_w)),
            topk=int(topk),
            with_yaw=True,
            dropout=float(dropout),
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        painted = self.paint(points).tanh()
        x = torch.cat([points, painted], dim=-1)
        return self.det(x)


def build_pointpainting_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointpainting_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointPainting(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        bev_h=int(cfg["bev_h"]),
        bev_w=int(cfg["bev_w"]),
        topk=int(cfg["topk"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pointpainting_detector3d(in_channels=3, num_classes=3, variant="pointpainting_tiny")
    x = torch.randn(2, 384, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})



import torch
from torch import nn

from dlhub.pointcloud.ops import farthest_point_sample, index_points

from ._common import PointQueryDetector3D, check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "groupfree3d_tiny": {"d_model": 64, "queries": 32, "groups": 32},
    "groupfree3d_small": {"d_model": 96, "queries": 48, "groups": 48},
    "groupfree3d_base": {"d_model": 128, "queries": 64, "groups": 64},
}


class GroupFree3D(nn.Module):
    """Group-Free 3D (toy): group points then run query detector on grouped tokens."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        num_queries: int,
        num_groups: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_groups = int(num_groups)
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
        xyz, _ = split_xyz_features(points)
        idx = farthest_point_sample(xyz, self.num_groups)  # (B,G)
        grouped = index_points(points, idx)  # (B,G,C)
        return self.det(grouped)


def build_groupfree3d_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "groupfree3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return GroupFree3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        num_queries=int(cfg["queries"]),
        num_groups=int(cfg["groups"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_groupfree3d_detector3d(in_channels=3, num_classes=5, variant="groupfree3d_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


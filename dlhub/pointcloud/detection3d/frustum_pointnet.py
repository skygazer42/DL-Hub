import torch
from torch import nn

from ._common import PointQueryDetector3D, check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "frustum_pointnet_tiny": {"d_model": 64, "queries": 32, "use_transformer": False},
    "frustum_pointnet_small": {"d_model": 96, "queries": 48, "use_transformer": True},
    "frustum_pointnet_base": {"d_model": 128, "queries": 64, "use_transformer": True},
}


class FrustumPointNet(nn.Module):
    """Frustum PointNet (toy): filter points by a random 'frustum' direction then detect."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        d_model: int,
        num_queries: int,
        use_transformer: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.det = PointQueryDetector3D(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            d_model=int(d_model),
            num_queries=int(num_queries),
            use_transformer=bool(use_transformer),
            dropout=float(dropout),
            with_yaw=True,
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)

        # "Frustum" selection by dot with a fixed direction.
        direction = torch.tensor([1.0, 0.3, 0.0], device=points.device, dtype=points.dtype)
        score = (xyz * direction).sum(dim=-1)  # (B,N)
        k = max(32, points.shape[1] // 2)
        idx = score.topk(k, dim=1).indices  # (B,k)
        b = points.shape[0]
        batch = torch.arange(b, device=points.device).unsqueeze(-1)
        sub = points[batch, idx]
        return self.det(sub)


def build_frustum_pointnet_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "frustum_pointnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    d_model = int(int(cfg["d_model"]) * float(width_mult))
    return FrustumPointNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        d_model=d_model,
        num_queries=int(cfg["queries"]),
        use_transformer=bool(cfg["use_transformer"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_frustum_pointnet_detector3d(
        in_channels=3, num_classes=3, variant="frustum_pointnet_tiny"
    )
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

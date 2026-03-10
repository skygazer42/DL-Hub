import torch
from torch import nn

from ._common import FeaturePropagation, PointMLP, SetAbstraction, check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "pointnext_tiny": {"width": 32},
    "pointnext_small": {"width": 48},
    "pointnext_base": {"width": 64},
}


class PointNeXtSeg(nn.Module):
    """PointNeXt semantic segmentation (toy): PointNet++ hierarchy with residual MLPs."""

    def __init__(
        self, *, in_channels: int, num_classes: int, width: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        w = int(width)
        self.stem = PointMLP(int(in_channels), w, depth=2, dropout=float(dropout))

        self.sa1 = SetAbstraction(w, w, npoint=64, k=16, dropout=float(dropout))
        self.sa2 = SetAbstraction(w, w * 2, npoint=32, k=16, dropout=float(dropout))
        self.sa3 = SetAbstraction(w * 2, w * 4, npoint=16, k=16, dropout=float(dropout))

        self.res1 = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, w))
        self.res2 = nn.Sequential(
            nn.Linear(w * 2, w * 2), nn.ReLU(inplace=True), nn.Linear(w * 2, w * 2)
        )
        self.res3 = nn.Sequential(
            nn.Linear(w * 4, w * 4), nn.ReLU(inplace=True), nn.Linear(w * 4, w * 4)
        )

        self.fp2 = FeaturePropagation(w * 4 + w * 2, w * 2, dropout=float(dropout))
        self.fp1 = FeaturePropagation(w * 2 + w, w, dropout=float(dropout))
        self.fp0 = FeaturePropagation(w + w, w, dropout=float(dropout))

        self.cls = nn.Sequential(
            nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, int(num_classes))
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        f0 = self.stem(x)

        xyz1, f1 = self.sa1(xyz, f0)
        f1 = f1 + 0.1 * self.res1(f1)
        xyz2, f2 = self.sa2(xyz1, f1)
        f2 = f2 + 0.1 * self.res2(f2)
        xyz3, f3 = self.sa3(xyz2, f2)
        f3 = f3 + 0.1 * self.res3(f3)

        f2u = self.fp2(xyz2, f2, xyz3, f3)
        f1u = self.fp1(xyz1, f1, xyz2, f2u)
        f0u = self.fp0(xyz, f0, xyz1, f1u)
        return self.cls(f0u)


def build_pointnext_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointnext_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointNeXtSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointnext_segmenter3d(in_channels=3, num_classes=6, variant="pointnext_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    (y.mean()).backward()
    print("logits:", tuple(y.shape))

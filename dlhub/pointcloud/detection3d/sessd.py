
import math

import torch
from torch import nn
from torch.nn import functional as F

from dlhub.pointcloud.ops import farthest_point_sample, index_points

from ._common import PointNetEncoder, check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "sessd_tiny": {"width": 64, "keypoints": 64},
    "sessd_small": {"width": 96, "keypoints": 96},
    "sessd_base": {"width": 128, "keypoints": 128},
}


class SESSD(nn.Module):
    """SE-SSD (toy): squeeze-excitation on keypoint features."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, keypoints: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.keypoints = int(keypoints)
        self.enc = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.se = nn.Sequential(nn.Linear(int(width), int(width) // 4), nn.ReLU(inplace=True), nn.Linear(int(width) // 4, int(width)), nn.Sigmoid())
        self.cls = nn.Linear(int(width), int(num_classes))
        self.box = nn.Linear(int(width), 7)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.enc(x)

        idx = farthest_point_sample(xyz, self.keypoints)
        kp_xyz = index_points(xyz, idx)
        kp_feat = index_points(p, idx)

        w = self.se(kp_feat.mean(dim=1)).unsqueeze(1)
        kp_feat = kp_feat * (0.5 + 0.5 * w)
        cls_logits = self.cls(kp_feat)

        raw = self.box(kp_feat)
        dims = F.softplus(raw[..., 3:6]) + 0.1
        yaw = raw[..., 6:7].tanh() * math.pi
        boxes = torch.cat([kp_xyz + raw[..., :3].tanh(), dims, yaw], dim=-1)
        return {"boxes": boxes, "cls_logits": cls_logits}


def build_sessd_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "sessd_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return SESSD(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        keypoints=int(cfg["keypoints"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_sessd_detector3d(in_channels=3, num_classes=3, variant="sessd_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from dlhub.pointcloud.ops import farthest_point_sample, index_points

from ._common import PointNetEncoder, TinyTransformerEncoder, check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "threedssd_tiny": {"width": 64, "keypoints": 64},
    "threedssd_small": {"width": 96, "keypoints": 96},
    "threedssd_base": {"width": 128, "keypoints": 128},
}


class ThreeDSSD(nn.Module):
    """3DSSD (toy): sample keypoints + single-stage box regression."""

    def __init__(self, *, in_channels: int, num_classes: int, width: int, keypoints: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.keypoints = int(keypoints)
        self.enc = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
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

        cls_logits = self.cls(kp_feat)
        raw = self.box(kp_feat)
        dims = F.softplus(raw[..., 3:6]) + 0.1
        yaw = raw[..., 6:7].tanh() * math.pi
        boxes = torch.cat([kp_xyz + raw[..., :3].tanh(), dims, yaw], dim=-1)
        return {"boxes": boxes, "cls_logits": cls_logits}


def build_threedssd_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "threedssd_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return ThreeDSSD(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        keypoints=int(cfg["keypoints"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_threedssd_detector3d(in_channels=3, num_classes=3, variant="threedssd_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


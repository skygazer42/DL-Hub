from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from dlhub.pointcloud.ops import farthest_point_sample, index_points

from ._common import PointNetEncoder, check_points, mlp, roi_pool_knn, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "h3dnet_tiny": {"width": 64, "seeds": 64, "proposals": 32, "vote_k": 16, "stages": 2},
    "h3dnet_small": {"width": 96, "seeds": 96, "proposals": 48, "vote_k": 24, "stages": 2},
    "h3dnet_base": {"width": 128, "seeds": 128, "proposals": 64, "vote_k": 32, "stages": 3},
}


class H3DNet(nn.Module):
    """H3DNet (toy): multi-stage voting refinement."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        num_seeds: int,
        num_proposals: int,
        vote_k: int,
        stages: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_seeds = int(num_seeds)
        self.num_proposals = int(num_proposals)
        self.vote_k = int(vote_k)
        self.num_classes = int(num_classes)
        self.stages = int(stages)

        self.enc = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.vote_stages = nn.ModuleList(
            [mlp(int(width), [int(width), int(width)], 3 + int(width), dropout=float(dropout)) for _ in range(self.stages)]
        )
        self.proj = nn.Linear(int(width), int(width))
        self.cls = nn.Linear(int(width), int(num_classes))
        self.box = nn.Linear(int(width), 7)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.enc(x)  # (B,N,D)

        seed_idx = farthest_point_sample(xyz, self.num_seeds)
        seed_xyz = index_points(xyz, seed_idx)
        seed_feat = index_points(p, seed_idx)

        vote_xyz = seed_xyz
        vote_feat = seed_feat
        for stage in self.vote_stages:
            out = stage(vote_feat)
            vote_xyz = vote_xyz + out[..., :3].tanh()
            vote_feat = vote_feat + self.proj(out[..., 3:])

        prop_idx = farthest_point_sample(vote_xyz, self.num_proposals)
        prop_xyz = index_points(vote_xyz, prop_idx)
        pooled = roi_pool_knn(vote_xyz, vote_feat, prop_xyz, k=self.vote_k)

        cls_logits = self.cls(pooled)
        raw = self.box(pooled)
        dims = F.softplus(raw[..., 3:6]) + 0.1
        yaw = raw[..., 6:7].tanh() * 3.14159265
        boxes = torch.cat([prop_xyz + raw[..., :3].tanh(), dims, yaw], dim=-1)
        return {"boxes": boxes, "cls_logits": cls_logits}


def build_h3dnet_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "h3dnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return H3DNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_seeds=int(cfg["seeds"]),
        num_proposals=int(cfg["proposals"]),
        vote_k=int(cfg["vote_k"]),
        stages=int(cfg["stages"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_h3dnet_detector3d(in_channels=3, num_classes=4, variant="h3dnet_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


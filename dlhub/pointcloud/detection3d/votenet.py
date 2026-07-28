import math

import torch
from torch import nn
from torch.nn import functional as F

from dlhub.pointcloud.ops import farthest_point_sample, index_points

from ._common import PointNetEncoder, check_points, mlp, roi_pool_knn, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "votenet_tiny": {"width": 64, "seeds": 64, "proposals": 32, "vote_k": 16},
    "votenet_small": {"width": 96, "seeds": 96, "proposals": 48, "vote_k": 24},
    "votenet_base": {"width": 128, "seeds": 128, "proposals": 64, "vote_k": 32},
}


class VoteNet(nn.Module):
    """VoteNet (compact): seed points vote for object centers -> proposal head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        num_seeds: int,
        num_proposals: int,
        vote_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_seeds = int(num_seeds)
        self.num_proposals = int(num_proposals)
        self.vote_k = int(vote_k)
        self.num_classes = int(num_classes)

        self.enc = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.vote = mlp(
            int(width), [int(width), int(width)], 3 + int(width), dropout=float(dropout)
        )
        self.proj = nn.Linear(int(width), int(width))

        self.cls = nn.Linear(int(width), int(num_classes))
        self.box = nn.Linear(int(width), 7)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.enc(x)  # (B,N,D)

        # Seed sampling
        seed_idx = farthest_point_sample(xyz, self.num_seeds)  # (B,S)
        seed_xyz = index_points(xyz, seed_idx)  # (B,S,3)
        seed_feat = index_points(p, seed_idx)  # (B,S,D)

        # Votes: offset + vote feature
        vote_out = self.vote(seed_feat)  # (B,S,3+D)
        offset = vote_out[..., :3].tanh()
        vote_xyz = seed_xyz + offset
        vote_feat = self.proj(vote_out[..., 3:])  # (B,S,D)

        # Cluster votes into proposals using FPS on vote centers
        prop_idx = farthest_point_sample(vote_xyz, self.num_proposals)  # (B,K)
        prop_xyz = index_points(vote_xyz, prop_idx)  # (B,K,3)

        pooled = roi_pool_knn(vote_xyz, vote_feat, prop_xyz, k=self.vote_k)
        cls_logits = self.cls(pooled)

        raw = self.box(pooled)
        delta = raw.tanh()
        dims = F.softplus(raw[..., 3:6]) + 0.1
        yaw = raw[..., 6:7].tanh() * math.pi
        boxes = torch.cat([prop_xyz + delta[..., :3], dims, yaw], dim=-1)
        return {"boxes": boxes, "cls_logits": cls_logits}


def build_votenet_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "votenet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return VoteNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_seeds=int(cfg["seeds"]),
        num_proposals=int(cfg["proposals"]),
        vote_k=int(cfg["vote_k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_votenet_detector3d(in_channels=3, num_classes=4, variant="votenet_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

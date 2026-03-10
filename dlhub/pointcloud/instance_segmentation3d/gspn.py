import math

import torch
from torch import nn

from dlhub.pointcloud.ops import knn_query

from ._common import MLPPointEncoder, l2_normalize, mlp

_VARIANTS: dict[str, dict[str, object]] = {
    "gspn_tiny": {"width": 64, "depth": 2, "instances": 16, "vote_k": 8},
    "gspn_small": {"width": 96, "depth": 3, "instances": 24, "vote_k": 16},
    "gspn_base": {"width": 128, "depth": 4, "instances": 32, "vote_k": 24},
}


class GSPN(nn.Module):
    """GSPN (toy): sample proposals and pool neighborhood features as 'shape proposals'."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        num_instances: int,
        vote_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = int(width)
        self.num_instances = int(num_instances)
        self.vote_k = int(vote_k)
        self.enc = MLPPointEncoder(int(in_channels), w, depth=int(depth), dropout=float(dropout))
        self.vote = mlp(w, [w, w], 3 + w, dropout=float(dropout))
        self.cls = nn.Linear(w, int(num_classes))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        b, n, w = feat.shape
        k = min(self.num_instances, n)

        from dlhub.pointcloud.ops import farthest_point_sample, index_points

        idx = farthest_point_sample(xyz, k)
        seed_xyz = index_points(xyz, idx)
        seed_feat = index_points(feat, idx)
        vote_out = self.vote(seed_feat)
        centers = seed_xyz + vote_out[..., :3].tanh()
        inst_feat = seed_feat + vote_out[..., 3:]

        neigh = knn_query(self.vote_k, xyz, centers)  # (B,K,vote_k)
        neigh_feat = index_points(feat, neigh).mean(dim=2)
        inst_feat = inst_feat + neigh_feat

        sim = torch.einsum("bkd,bnd->bkn", l2_normalize(inst_feat), l2_normalize(feat)) * math.sqrt(
            w
        )
        cls_logits = self.cls(inst_feat)
        return {"mask_logits": sim, "cls_logits": cls_logits}


def build_gspn_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "gspn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return GSPN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        num_instances=int(cfg["instances"]),
        vote_k=int(cfg["vote_k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_gspn_instance_segmenter3d(in_channels=3, num_classes=6, variant="gspn_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

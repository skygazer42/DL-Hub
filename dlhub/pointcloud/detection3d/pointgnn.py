
import math

import torch
from torch import nn
from torch.nn import functional as F

from ._common import EdgeConv, check_points, mlp, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "pointgnn_tiny": {"width": 64, "k": 8, "proposals": 32},
    "pointgnn_small": {"width": 96, "k": 16, "proposals": 48},
    "pointgnn_base": {"width": 128, "k": 24, "proposals": 64},
}


class PointGNN(nn.Module):
    """Point-GNN (toy): kNN graph message passing then query head."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        k: int,
        num_proposals: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_proposals = int(num_proposals)
        self.num_classes = int(num_classes)
        w = int(width)

        self.edge1 = EdgeConv(int(in_channels), w, k=int(k), dropout=float(dropout))
        self.edge2 = EdgeConv(w, w, k=int(k), dropout=float(dropout))
        self.head = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True))
        self.cls = nn.Linear(w, int(num_classes))
        self.box = nn.Linear(w, 7)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        f1 = self.edge1(x)
        f2 = self.edge2(f1)
        f = self.head(f2)

        # Proposal selection by point score (simple norm)
        score = f.norm(dim=-1)  # (B,N)
        top = score.topk(self.num_proposals, dim=1).indices
        b = points.shape[0]
        batch = torch.arange(b, device=points.device).unsqueeze(-1)
        centers = xyz[batch, top]  # (B,K,3)
        pooled = f[batch, top]  # (B,K,D)

        cls_logits = self.cls(pooled)
        raw = self.box(pooled)
        dims = F.softplus(raw[..., 3:6]) + 0.1
        yaw = raw[..., 6:7].tanh() * math.pi
        boxes = torch.cat([centers + raw[..., :3].tanh(), dims, yaw], dim=-1)
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": score.gather(1, top)}


def build_pointgnn_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointgnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointGNN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        k=int(cfg["k"]),
        num_proposals=int(cfg["proposals"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_pointgnn_detector3d(in_channels=3, num_classes=4, variant="pointgnn_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


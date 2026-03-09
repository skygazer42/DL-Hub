
import math

import torch
from torch import nn

from ._common import PointNet2Encoder, l2_normalize


_VARIANTS: dict[str, dict[str, object]] = {
    "condinst3d_tiny": {"width": 48, "instances": 16},
    "condinst3d_small": {"width": 64, "instances": 24},
    "condinst3d_base": {"width": 96, "instances": 32},
}


class CondInst3D(nn.Module):
    """CondInst3D (toy): proposal features generate per-instance mask kernels."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = int(width)
        self.num_instances = int(num_instances)
        self.enc = PointNet2Encoder(int(in_channels), w, dropout=float(dropout))
        self.kernel = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, w))
        self.cls = nn.Linear(w, int(num_classes))
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)  # (B,N,W)
        b, n, w = feat.shape
        k = min(self.num_instances, n)
        from dlhub.pointcloud.ops import farthest_point_sample, index_points

        idx = farthest_point_sample(xyz, k)
        inst_feat = index_points(feat, idx)
        inst_feat = self.drop(inst_feat)
        kernel = self.kernel(inst_feat)

        mask_logits = torch.einsum("bkd,bnd->bkn", l2_normalize(kernel), l2_normalize(feat)) * math.sqrt(w)
        cls_logits = self.cls(inst_feat)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


def build_condinst3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "condinst3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return CondInst3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_condinst3d_instance_segmenter3d(in_channels=3, num_classes=6, variant="condinst3d_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


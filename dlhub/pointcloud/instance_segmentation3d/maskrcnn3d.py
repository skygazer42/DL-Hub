
import math

import torch
from torch import nn

from ._common import CenterProposalHead, PointNet2Encoder, l2_normalize


_VARIANTS: dict[str, dict[str, object]] = {
    "maskrcnn3d_tiny": {"width": 48, "instances": 16},
    "maskrcnn3d_small": {"width": 64, "instances": 24},
    "maskrcnn3d_base": {"width": 96, "instances": 32},
}


class MaskRCNN3D(nn.Module):
    """Mask R-CNN 3D (toy): PointNet++ features + center proposals + ROI refinement."""

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
        self.enc = PointNet2Encoder(int(in_channels), w, dropout=float(dropout))
        self.stage1 = CenterProposalHead(w, int(num_classes), num_instances=int(num_instances), dropout=float(dropout))
        self.refine = nn.Sequential(nn.Linear(w, w), nn.ReLU(inplace=True), nn.Linear(w, w))
        self.cls = nn.Linear(w, int(num_classes))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        out1 = self.stage1(xyz, feat)
        m = out1["mask_logits"].sigmoid()
        w = m / m.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        inst_feat = torch.einsum("bkn,bnd->bkd", w, feat)
        inst_feat = inst_feat + 0.1 * self.refine(inst_feat).tanh()

        d = feat.shape[-1]
        mask_logits = torch.einsum("bkd,bnd->bkn", l2_normalize(inst_feat), l2_normalize(feat)) * math.sqrt(d)
        cls_logits = self.cls(inst_feat)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


def build_maskrcnn3d_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "maskrcnn3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return MaskRCNN3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_maskrcnn3d_instance_segmenter3d(in_channels=3, num_classes=6, variant="maskrcnn3d_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


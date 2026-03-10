import math

import torch
from torch import nn

from ._common import MLPPointEncoder, l2_normalize

_VARIANTS: dict[str, dict[str, object]] = {
    "bonet_tiny": {"width": 64, "depth": 2, "instances": 16},
    "bonet_small": {"width": 96, "depth": 3, "instances": 24},
    "bonet_base": {"width": 128, "depth": 4, "instances": 32},
}


class BoNet(nn.Module):
    """3D-BoNet (toy): fixed proposal embeddings predict masks + classes."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        num_instances: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = int(width)
        self.num_instances = int(num_instances)
        self.enc = MLPPointEncoder(int(in_channels), w, depth=int(depth), dropout=float(dropout))
        self.proposal = nn.Parameter(torch.randn(self.num_instances, w) / math.sqrt(w))
        self.center = nn.Linear(w, 3)
        self.cls = nn.Linear(w, int(num_classes))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        xyz, feat = self.enc(points)
        b, n, w = feat.shape
        prop = self.proposal.unsqueeze(0).expand(b, -1, -1)  # (B,K,W)
        centers = self.center(prop).tanh() * 2.0

        dist2 = torch.cdist(centers.to(torch.float32), xyz.to(torch.float32)) ** 2  # (B,K,N)
        sim = torch.einsum("bkd,bnd->bkn", l2_normalize(prop), l2_normalize(feat)) * math.sqrt(w)
        mask_logits = sim - 0.2 * dist2.to(sim.dtype)
        cls_logits = self.cls(prop)
        return {"mask_logits": mask_logits, "cls_logits": cls_logits}


def build_bonet_instance_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "bonet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return BoNet(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        num_instances=int(cfg["instances"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_bonet_instance_segmenter3d(in_channels=3, num_classes=6, variant="bonet_tiny")
    x = torch.randn(2, 128, 3)
    out = m(x)
    (out["mask_logits"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

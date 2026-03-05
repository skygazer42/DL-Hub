from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ._common import PointNetEncoder, check_points, split_xyz_features


_VARIANTS: dict[str, dict[str, object]] = {
    "fcaf3d_tiny": {"width": 32, "grid": (8, 16, 16), "topk": 48},
    "fcaf3d_small": {"width": 48, "grid": (8, 20, 20), "topk": 64},
    "fcaf3d_base": {"width": 64, "grid": (10, 24, 24), "topk": 96},
}


def _scatter_mean_3d(idx_dhw: torch.Tensor, values: torch.Tensor, *, d: int, h: int, w: int) -> torch.Tensor:
    # idx_dhw: (B,N,3) with (iz,iy,ix)
    b, n, _ = idx_dhw.shape
    _, _, c = values.shape
    device = values.device
    dtype = values.dtype
    idx_dhw = idx_dhw.to(torch.long)
    iz = idx_dhw[..., 0].clamp(0, d - 1)
    iy = idx_dhw[..., 1].clamp(0, h - 1)
    ix = idx_dhw[..., 2].clamp(0, w - 1)
    flat = (iz * (h * w) + iy * w + ix).view(b, n)
    out = torch.zeros(b, d * h * w, c, device=device, dtype=dtype)
    cnt = torch.zeros(b, d * h * w, 1, device=device, dtype=dtype)
    out.scatter_add_(1, flat.unsqueeze(-1).expand(b, n, c), values)
    cnt.scatter_add_(1, flat.unsqueeze(-1), torch.ones(b, n, 1, device=device, dtype=dtype))
    out = out / cnt.clamp_min(1.0)
    return out.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()  # (B,C,D,H,W)


def _topk_heatmap_3d(hm: torch.Tensor, *, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # hm: (B,C,D,H,W)
    b, c, d, h, w = hm.shape
    s = hm.sigmoid().view(b, c, d * h * w)
    top_s, top_i = s.topk(int(k), dim=-1)  # (B,C,K)
    score, cls = top_s.max(dim=1)  # (B,K)
    which = top_s.argmax(dim=1)  # (B,K)
    flat = top_i.gather(1, which.unsqueeze(1)).squeeze(1)  # (B,K)
    return score, cls.to(torch.long), flat.to(torch.long)


class FCAF3D(nn.Module):
    """FCAF3D (toy): dense 3D conv head on a coarse voxel grid."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        grid: tuple[int, int, int],
        topk: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.topk = int(topk)
        self.d, self.h, self.w = map(int, grid)

        self.point = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.backbone = nn.Sequential(
            nn.Conv3d(int(width), int(width), 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(int(width), int(width), 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.hm = nn.Conv3d(int(width), int(num_classes), 1)
        self.box = nn.Conv3d(int(width), 7, 1)

        # Metric range for decoding
        self.x_min, self.x_max = -10.0, 10.0
        self.y_min, self.y_max = -10.0, 10.0
        self.z_min, self.z_max = -2.0, 2.0

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.point(x)  # (B,N,C)

        # Quantize into a coarse voxel grid (D,H,W) based on xyz.
        xq = (xyz[..., 0] - self.x_min) / (self.x_max - self.x_min) * float(self.w)
        yq = (xyz[..., 1] - self.y_min) / (self.y_max - self.y_min) * float(self.h)
        zq = (xyz[..., 2] - self.z_min) / (self.z_max - self.z_min) * float(self.d)
        idx = torch.stack([zq.floor(), yq.floor(), xq.floor()], dim=-1)

        vox = _scatter_mean_3d(idx, p, d=self.d, h=self.h, w=self.w)  # (B,C,D,H,W)
        feat = self.backbone(vox)
        hm = self.hm(feat)
        box = self.box(feat)

        score, cls, flat = _topk_heatmap_3d(hm, k=self.topk)
        # Decode flat -> (iz,iy,ix)
        iz = flat // (self.h * self.w)
        rem = flat % (self.h * self.w)
        iy = rem // self.w
        ix = rem % self.w

        # Gather box params at locations
        b = points.shape[0]
        box_flat = box.view(b, 7, self.d * self.h * self.w).permute(0, 2, 1)  # (B,DHW,7)
        gathered = box_flat.gather(1, flat.unsqueeze(-1).expand(b, self.topk, 7))

        # Grid center coords
        xc = (ix.float() + 0.5) / float(self.w) * (self.x_max - self.x_min) + self.x_min
        yc = (iy.float() + 0.5) / float(self.h) * (self.y_max - self.y_min) + self.y_min
        zc = (iz.float() + 0.5) / float(self.d) * (self.z_max - self.z_min) + self.z_min

        delta = gathered[..., :3].tanh()
        dims = F.softplus(gathered[..., 3:6]) + 0.1
        yaw = gathered[..., 6:7].tanh() * math.pi
        boxes = torch.cat([torch.stack([xc, yc, zc], dim=-1) + delta, dims, yaw], dim=-1)

        cls_logits = torch.zeros(b, self.topk, self.num_classes, device=points.device, dtype=points.dtype)
        cls_logits.scatter_(-1, cls.unsqueeze(-1), score.unsqueeze(-1))
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": score}


def build_fcaf3d_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "fcaf3d_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return FCAF3D(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        grid=tuple(cfg["grid"]),
        topk=int(cfg["topk"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_fcaf3d_detector3d(in_channels=3, num_classes=3, variant="fcaf3d_tiny")
    x = torch.randn(2, 512, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})


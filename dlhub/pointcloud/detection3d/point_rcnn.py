import torch
from torch import nn
from torch.nn import functional as F

from ._common import PointNetEncoder, check_points, roi_pool_knn, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "point_rcnn_tiny": {"width": 64, "topk": 32, "roi_k": 8},
    "point_rcnn_small": {"width": 96, "topk": 48, "roi_k": 16},
    "point_rcnn_base": {"width": 128, "topk": 64, "roi_k": 24},
}


class PointRCNN(nn.Module):
    """PointRCNN (toy): stage1 point-wise objectness -> proposals -> ROI refinement."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        topk: int,
        roi_k: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.topk = int(topk)
        self.roi_k = int(roi_k)
        self.num_classes = int(num_classes)

        self.enc = PointNetEncoder(int(in_channels), width=int(width), dropout=float(dropout))
        self.obj = nn.Linear(int(width), 1)
        self.reg = nn.Linear(int(width), 7)

        self.refine = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.ReLU(inplace=True),
            nn.Linear(int(width), int(width)),
        )
        self.cls = nn.Linear(int(width), int(num_classes))
        self.box = nn.Linear(int(width), 7)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        p = self.enc(x)  # (B,N,D)
        obj = self.obj(p).squeeze(-1)  # (B,N)

        top = obj.topk(self.topk, dim=1).indices  # (B,K)
        b = points.shape[0]
        batch = torch.arange(b, device=points.device).unsqueeze(-1)
        centers = xyz[batch, top]  # (B,K,3)

        pooled = roi_pool_knn(xyz, p, centers, k=self.roi_k)
        r = self.refine(pooled)
        cls_logits = self.cls(r)

        raw = self.box(r)
        delta_xyz = raw[..., :3].tanh()
        dims = F.softplus(raw[..., 3:6]) + 0.1
        yaw = raw[..., 6:7].tanh() * 3.14159265
        boxes = torch.cat([centers + delta_xyz, dims, yaw], dim=-1)
        return {"boxes": boxes, "cls_logits": cls_logits, "scores": obj.gather(1, top).sigmoid()}


def build_point_rcnn_detector3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "point_rcnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointRCNN(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        topk=int(cfg["topk"]),
        roi_k=int(cfg["roi_k"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = build_point_rcnn_detector3d(in_channels=3, num_classes=4, variant="point_rcnn_tiny")
    x = torch.randn(2, 256, 3)
    out = m(x)
    (out["boxes"].mean() + out["cls_logits"].mean()).backward()
    print({k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)})

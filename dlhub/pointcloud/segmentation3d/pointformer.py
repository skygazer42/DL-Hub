import torch
from torch import nn

from ._common import EdgeConv, TinyTransformerEncoder, check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "pointformer_tiny": {"width": 64, "k": 8, "depth": 2},
    "pointformer_small": {"width": 96, "k": 16, "depth": 3},
    "pointformer_base": {"width": 128, "k": 24, "depth": 4},
}


class PointFormerSeg(nn.Module):
    """PointFormer semantic segmentation (toy): local EdgeConv pre-encoder + transformer."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        k: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = int(width)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.local = EdgeConv(w, w, k=int(k), dropout=float(dropout))
        self.enc = TinyTransformerEncoder(w, nhead=4, num_layers=int(depth), dropout=float(dropout))
        self.cls = nn.Sequential(
            nn.Linear(w, w),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(w, int(num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        check_points(points)
        xyz, feats = split_xyz_features(points)
        x = xyz if feats is None else torch.cat([xyz, feats], dim=-1)
        h = self.embed(x.to(torch.float32))
        h = h + self.local(h)
        h = self.enc(h)
        return self.cls(h)


def build_pointformer_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "pointformer_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return PointFormerSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        k=int(cfg["k"]),
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_pointformer_segmenter3d(in_channels=3, num_classes=6, variant="pointformer_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

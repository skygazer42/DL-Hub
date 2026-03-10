import torch
from torch import nn

from ._common import EdgeConv, TinyTransformerEncoder, check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "gdanet_tiny": {"width": 64, "k": 8, "depth": 2},
    "gdanet_small": {"width": 96, "k": 16, "depth": 3},
    "gdanet_base": {"width": 128, "k": 24, "depth": 4},
}


class GDANetSeg(nn.Module):
    """GDANet semantic segmentation (toy): local geometry stream + global attention stream."""

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
        self.local = nn.ModuleList(
            [EdgeConv(w, w, k=int(k), dropout=float(dropout)) for _ in range(int(depth))]
        )
        self.global_attn = TinyTransformerEncoder(
            w, nhead=4, num_layers=max(1, int(depth) // 2), dropout=float(dropout)
        )
        self.fuse = nn.Sequential(nn.Linear(w * 2, w), nn.ReLU(inplace=True))
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
        h0 = self.embed(x.to(torch.float32))

        hl = h0
        for blk in self.local:
            hl = hl + blk(hl)

        hg = self.global_attn(h0)
        y = self.fuse(torch.cat([hl, hg], dim=-1))
        return self.cls(y)


def build_gdanet_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "gdanet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return GDANetSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        k=int(cfg["k"]),
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_gdanet_segmenter3d(in_channels=3, num_classes=6, variant="gdanet_tiny")
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

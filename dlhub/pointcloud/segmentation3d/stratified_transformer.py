import torch
from torch import nn

from dlhub.pointcloud.ops import farthest_point_sample, index_points

from ._common import FeaturePropagation, TinyTransformerEncoder, check_points, split_xyz_features

_VARIANTS: dict[str, dict[str, object]] = {
    "stratified_transformer_tiny": {"width": 64, "depth": 2, "tokens": 48},
    "stratified_transformer_small": {"width": 96, "depth": 3, "tokens": 64},
    "stratified_transformer_base": {"width": 128, "depth": 4, "tokens": 96},
}


class StratifiedTransformerSeg(nn.Module):
    """Stratified Transformer semantic segmentation (compact): attend on sampled tokens then propagate."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        w = int(width)
        self.tokens = int(tokens)
        self.embed = nn.Sequential(nn.Linear(int(in_channels), w), nn.ReLU(inplace=True))
        self.enc = TinyTransformerEncoder(w, nhead=4, num_layers=int(depth), dropout=float(dropout))
        self.fp = FeaturePropagation(w + w, w, dropout=float(dropout))
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
        f0 = self.embed(x.to(torch.float32))

        n_tok = min(self.tokens, xyz.shape[1])
        idx = farthest_point_sample(xyz, n_tok)  # (B,T)
        t_xyz = index_points(xyz, idx)
        t_feat = index_points(f0, idx)
        t_feat = self.enc(t_feat)

        # Propagate sampled token features back to all points.
        f = self.fp(xyz, f0, t_xyz, t_feat, k=3)
        return self.cls(f)


def build_stratified_transformer_segmenter3d(
    in_channels: int,
    num_classes: int,
    *,
    variant: str = "stratified_transformer_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    cfg = _VARIANTS[str(variant)]
    width = int(int(cfg["width"]) * float(width_mult))
    return StratifiedTransformerSeg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        tokens=int(cfg["tokens"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    model = build_stratified_transformer_segmenter3d(
        in_channels=3, num_classes=6, variant="stratified_transformer_tiny"
    )
    x = torch.randn(2, 128, 3)
    y = model(x)
    y.mean().backward()
    print("logits:", tuple(y.shape))

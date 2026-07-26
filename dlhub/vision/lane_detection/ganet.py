import torch
from torch import nn

from ._common import GlobalContextHead, SegmentationDecoder, TinyLaneEncoder, scaled_channels


class GANetLaneDetector(nn.Module):
    """GANet-style detector with segmentation and global curve regression."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_lanes: int,
        num_points: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_lanes = int(num_lanes)
        self.num_points = int(num_points)
        self.encoder = TinyLaneEncoder(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            hidden_channels=int(hidden_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.decoder = SegmentationDecoder(
            low_channels=int(hidden_channels),
            high_channels=int(hidden_channels),
            out_channels=int(hidden_channels),
            dropout=float(dropout),
        )
        self.binary_head = nn.Conv2d(int(hidden_channels), 1, kernel_size=1)
        self.lane_head = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.num_lanes,
            dropout=float(dropout),
        )
        self.curve_head = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.num_lanes * self.num_points * 2,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        low, high = self.encoder(x)
        feats = self.decoder(low, high, output_size=tuple(x.shape[-2:]))
        lane_logits = self.lane_head(high)
        curve_points = torch.tanh(self.curve_head(high)).view(
            x.shape[0], self.num_lanes, self.num_points, 2
        )
        return {
            "binary_logits": self.binary_head(feats),
            "lane_logits": lane_logits,
            "curve_points": curve_points,
        }


_VARIANTS: dict[str, dict[str, int | float]] = {
    "ganet_tiny": {"stem": 16, "hidden": 32, "depth": 1, "dropout": 0.0},
    "ganet_small": {"stem": 24, "hidden": 48, "depth": 2, "dropout": 0.0},
    "ganet_base": {"stem": 32, "hidden": 64, "depth": 3, "dropout": 0.1},
}


def build_ganet_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "ganet_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del image_size, num_rows, grid_size, num_anchors, num_queries
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown GANet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return GANetLaneDetector(
        in_channels=int(in_channels),
        num_lanes=int(num_lanes),
        num_points=int(num_points),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_ganet_lane_detector(in_channels=3, num_lanes=4, num_points=16, variant="ganet_tiny")
    out = m(x)
    print("ganet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

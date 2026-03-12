import torch
from torch import nn

from ._common import SegmentationDecoder, SpatialMessagePassing, TinyLaneEncoder, scaled_channels


class SCNNLaneDetector(nn.Module):
    """SCNN-style lane detector with spatial message passing on decoded features."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        passes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
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
        self.propagation = nn.Sequential(
            *[SpatialMessagePassing(int(hidden_channels)) for _ in range(int(passes))]
        )
        self.binary_head = nn.Conv2d(int(hidden_channels), 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        low, high = self.encoder(x)
        feats = self.decoder(low, high, output_size=tuple(x.shape[-2:]))
        feats = self.propagation(feats)
        return {"binary_logits": self.binary_head(feats)}


_VARIANTS: dict[str, dict[str, int | float]] = {
    "scnn_tiny": {"stem": 16, "hidden": 32, "depth": 1, "passes": 1, "dropout": 0.0},
    "scnn_small": {"stem": 24, "hidden": 48, "depth": 2, "passes": 2, "dropout": 0.0},
    "scnn_base": {"stem": 32, "hidden": 64, "depth": 3, "passes": 3, "dropout": 0.1},
}


def build_scnn_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "scnn_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del num_lanes, image_size, num_points, num_rows, grid_size, num_anchors, num_queries
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SCNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return SCNNLaneDetector(
        in_channels=int(in_channels),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        passes=int(spec["passes"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_scnn_lane_detector(
        in_channels=3,
        num_lanes=4,
        variant="scnn_tiny",
        width_mult=1.0,
    )
    out = m(x)
    print("scnn_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

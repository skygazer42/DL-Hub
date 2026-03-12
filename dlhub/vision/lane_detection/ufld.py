import torch
import torch.nn.functional as F
from torch import nn

from ._common import GlobalContextHead, TinyLaneEncoder, scaled_channels


class UFLDLaneDetector(nn.Module):
    """Ultra-Fast style row-anchor lane detector."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_lanes: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        num_rows: int,
        grid_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_lanes = int(num_lanes)
        self.num_rows = int(num_rows)
        self.grid_size = int(grid_size)
        self.encoder = TinyLaneEncoder(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            hidden_channels=int(hidden_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.row_head = nn.Sequential(
            nn.Conv2d(int(hidden_channels), int(hidden_channels), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(hidden_channels)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_channels), self.num_lanes, kernel_size=1),
        )
        self.exist_head = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.num_lanes,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        _, high = self.encoder(x)
        pooled = F.adaptive_avg_pool2d(high, (self.num_rows, self.grid_size))
        row_logits = self.row_head(pooled)
        exist_logits = self.exist_head(high)
        return {"exist_logits": exist_logits, "row_logits": row_logits}


_VARIANTS: dict[str, dict[str, int | float]] = {
    "ufld_tiny": {"stem": 16, "hidden": 32, "depth": 1, "dropout": 0.0},
    "ufld_small": {"stem": 24, "hidden": 48, "depth": 2, "dropout": 0.0},
    "ufld_base": {"stem": 32, "hidden": 64, "depth": 3, "dropout": 0.1},
}


def build_ufld_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "ufld_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del image_size, num_points, num_anchors, num_queries
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown UFLD variant: {variant!r}. Supported: {sorted(_VARIANTS)}")

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return UFLDLaneDetector(
        in_channels=int(in_channels),
        num_lanes=int(num_lanes),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        num_rows=int(num_rows),
        grid_size=int(grid_size),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_ufld_lane_detector(
        in_channels=3,
        num_lanes=4,
        num_rows=12,
        grid_size=20,
        variant="ufld_tiny",
        width_mult=1.0,
    )
    out = m(x)
    print("ufld_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

import torch
from torch import nn

from ._common import GlobalContextHead, TinyLaneEncoder, scaled_channels


class BezierLaneNetLaneDetector(nn.Module):
    """BezierLaneNet-style detector that regresses lane control points."""

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
        self.lane_head = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.num_lanes,
            dropout=float(dropout),
        )
        self.point_head = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.num_lanes * self.num_points * 2,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        _, high = self.encoder(x)
        lane_logits = self.lane_head(high)
        control_points = torch.tanh(self.point_head(high)).view(
            x.shape[0], self.num_lanes, self.num_points, 2
        )
        return {"control_points": control_points, "lane_logits": lane_logits}


_VARIANTS: dict[str, dict[str, int | float]] = {
    "bezierlanenet_tiny": {"stem": 16, "hidden": 32, "depth": 1, "dropout": 0.0},
    "bezierlanenet_small": {"stem": 24, "hidden": 48, "depth": 2, "dropout": 0.0},
    "bezierlanenet_base": {"stem": 32, "hidden": 64, "depth": 3, "dropout": 0.1},
}


def build_bezierlanenet_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "bezierlanenet_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del image_size, num_rows, grid_size, num_anchors, num_queries
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown BezierLaneNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return BezierLaneNetLaneDetector(
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
    m = build_bezierlanenet_lane_detector(
        in_channels=3,
        num_lanes=4,
        num_points=5,
        variant="bezierlanenet_tiny",
    )
    out = m(x)
    print("bezierlanenet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

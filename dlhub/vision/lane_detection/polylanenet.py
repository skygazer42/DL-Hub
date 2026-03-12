import torch
from torch import nn

from ._common import GlobalContextHead, TinyLaneEncoder, scaled_channels


class PolyLaneNetLaneDetector(nn.Module):
    """PolyLaneNet-style detector that regresses polynomial lane coefficients."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_lanes: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        coeff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_lanes = int(num_lanes)
        self.coeff_dim = int(coeff_dim)

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
        self.coeff_head = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.num_lanes * self.coeff_dim,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        _, high = self.encoder(x)
        lane_logits = self.lane_head(high)
        poly_coeffs = self.coeff_head(high).view(x.shape[0], self.num_lanes, self.coeff_dim)
        return {"lane_logits": lane_logits, "poly_coeffs": poly_coeffs}


_VARIANTS: dict[str, dict[str, int | float]] = {
    "polylanenet_tiny": {"stem": 16, "hidden": 32, "depth": 1, "coeff_dim": 4, "dropout": 0.0},
    "polylanenet_small": {
        "stem": 24,
        "hidden": 48,
        "depth": 2,
        "coeff_dim": 4,
        "dropout": 0.0,
    },
    "polylanenet_base": {"stem": 32, "hidden": 64, "depth": 3, "coeff_dim": 4, "dropout": 0.1},
}


def build_polylanenet_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "polylanenet_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del image_size, num_points, num_rows, grid_size, num_anchors, num_queries
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown PolyLaneNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return PolyLaneNetLaneDetector(
        in_channels=int(in_channels),
        num_lanes=int(num_lanes),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        coeff_dim=int(spec["coeff_dim"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_polylanenet_lane_detector(
        in_channels=3,
        num_lanes=4,
        variant="polylanenet_tiny",
    )
    out = m(x)
    print("polylanenet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

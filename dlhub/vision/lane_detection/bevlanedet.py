import torch
from torch import nn

from ._common import TinyLaneEncoder, choose_attention_heads, scaled_channels


class BEVLaneDetDetector(nn.Module):
    """BEV-LaneDet-style detector with BEV features and lane queries."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_queries: int,
        num_points: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        model_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_points = int(num_points)
        self.model_dim = int(model_dim)
        self.encoder = TinyLaneEncoder(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            hidden_channels=int(hidden_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.bev_proj = nn.Sequential(
            nn.Conv2d(int(hidden_channels), self.model_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.model_dim),
            nn.ReLU(inplace=True),
        )
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.model_dim) * 0.02)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.model_dim,
                nhead=choose_attention_heads(self.model_dim),
                dim_feedforward=self.model_dim * 2,
                dropout=float(dropout),
                batch_first=True,
            ),
            num_layers=2,
        )
        self.lane_head = nn.Linear(self.model_dim, 1)
        self.curve_head = nn.Linear(self.model_dim, self.num_points * 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        _, high = self.encoder(x)
        bev_features = self.bev_proj(high)
        memory = bev_features.flatten(2).transpose(1, 2)
        b = x.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        decoded = self.decoder(tgt=queries, memory=memory)
        lane_logits = self.lane_head(decoded).squeeze(-1)
        curve_points = torch.tanh(self.curve_head(decoded)).view(
            b, self.num_queries, self.num_points, 2
        )
        return {
            "bev_features": bev_features,
            "curve_points": curve_points,
            "lane_logits": lane_logits,
        }


_VARIANTS: dict[str, dict[str, int | float]] = {
    "bevlanedet_tiny": {
        "stem": 16,
        "hidden": 32,
        "depth": 1,
        "model_dim": 32,
        "dropout": 0.0,
    },
    "bevlanedet_small": {
        "stem": 24,
        "hidden": 48,
        "depth": 2,
        "model_dim": 48,
        "dropout": 0.0,
    },
    "bevlanedet_base": {
        "stem": 32,
        "hidden": 64,
        "depth": 3,
        "model_dim": 64,
        "dropout": 0.1,
    },
}


def build_bevlanedet_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "bevlanedet_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del num_lanes, image_size, num_rows, grid_size, num_anchors
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown BEVLaneDet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    model_dim = scaled_channels(int(spec["model_dim"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return BEVLaneDetDetector(
        in_channels=int(in_channels),
        num_queries=int(num_queries),
        num_points=int(num_points),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        model_dim=int(model_dim),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_bevlanedet_lane_detector(
        in_channels=3, num_lanes=4, num_points=16, num_queries=6, variant="bevlanedet_tiny"
    )
    out = m(x)
    print("bevlanedet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

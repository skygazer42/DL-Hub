import torch
from torch import nn

from ._common import (
    GlobalContextHead,
    SpatialMessagePassing,
    TinyLaneEncoder,
    choose_attention_heads,
    scaled_channels,
)


class BevlaneformerLaneDetector(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_queries: int,
        num_points: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        token_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_points = int(num_points)
        self.token_dim = int(token_dim)
        self.encoder = TinyLaneEncoder(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            hidden_channels=int(hidden_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.spatial = SpatialMessagePassing(int(hidden_channels))
        self.context = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.token_dim,
            dropout=float(dropout),
        )
        self.memory_proj = nn.Conv2d(int(hidden_channels), self.token_dim, kernel_size=1)
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, self.token_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=choose_attention_heads(self.token_dim),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dim * 2, self.token_dim),
        )
        self.lane_head = nn.Linear(self.token_dim, 1)
        self.curve_head = nn.Linear(self.token_dim, self.num_points * 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        _, high = self.encoder(x)
        high = self.spatial(high)
        memory = self.memory_proj(high).flatten(2).transpose(1, 2)
        b = x.shape[0]
        queries = self.query_embed.unsqueeze(0).expand(b, -1, -1)
        queries = queries + self.context(high).unsqueeze(1)
        refined, _ = self.attn(queries, memory, memory, need_weights=False)
        lane_tokens = queries + refined
        lane_tokens = lane_tokens + self.ffn(lane_tokens)
        lane_logits = self.lane_head(lane_tokens).squeeze(-1)
        curve_points = torch.tanh(self.curve_head(lane_tokens)).view(
            b, self.num_queries, self.num_points, 2
        )
        return {
            "curve_points": curve_points,
            "lane_logits": lane_logits,
            "lane_tokens": lane_tokens,
        }


_VARIANTS: dict[str, dict[str, int | float]] = {
    "bevlaneformer_tiny": {"stem": 16, "hidden": 32, "depth": 1, "token_dim": 32, "dropout": 0.0},
    "bevlaneformer_small": {"stem": 24, "hidden": 48, "depth": 2, "token_dim": 48, "dropout": 0.0},
    "bevlaneformer_base": {"stem": 32, "hidden": 64, "depth": 3, "token_dim": 64, "dropout": 0.1},
}


def build_bevlaneformer_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 7,
    variant: str = "bevlaneformer_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del num_lanes, image_size, num_rows, grid_size, num_anchors
    spec = _VARIANTS[str(variant).lower().strip()]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    token_dim = scaled_channels(int(spec["token_dim"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return BevlaneformerLaneDetector(
        in_channels=int(in_channels),
        num_queries=int(num_queries),
        num_points=int(num_points),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        token_dim=int(token_dim),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_bevlaneformer_lane_detector(
        in_channels=3, num_lanes=4, num_points=16, variant="bevlaneformer_tiny"
    )
    out = m(x)
    print("bevlaneformer_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

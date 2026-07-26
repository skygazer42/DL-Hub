import torch
from torch import nn

from ._common import GlobalContextHead, TinyLaneEncoder, choose_attention_heads, scaled_channels


class LaneATTLaneDetector(nn.Module):
    """LaneATT-style detector with anchor tokens and curve regression."""

    def __init__(
        self,
        *,
        in_channels: int,
        stem_channels: int,
        hidden_channels: int,
        depth: int,
        num_anchors: int,
        num_points: int,
        token_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.num_points = int(num_points)
        self.token_dim = int(token_dim)

        self.encoder = TinyLaneEncoder(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            hidden_channels=int(hidden_channels),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.context = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=int(token_dim),
            dropout=float(dropout),
        )
        self.anchor_embed = nn.Parameter(torch.randn(self.num_anchors, self.token_dim) * 0.02)
        self.token_mixer = nn.MultiheadAttention(
            embed_dim=self.token_dim,
            num_heads=choose_attention_heads(self.token_dim),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.token_dim)
        self.cls_head = nn.Linear(self.token_dim, 1)
        self.curve_head = nn.Linear(self.token_dim, self.num_points * 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        _, high = self.encoder(x)
        b = x.shape[0]
        context = self.context(high).unsqueeze(1)
        anchors = self.anchor_embed.unsqueeze(0).expand(b, -1, -1)
        tokens = anchors + context
        mixed, _ = self.token_mixer(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(tokens + mixed)
        cls_logits = self.cls_head(tokens).squeeze(-1)
        curve_points = torch.tanh(self.curve_head(tokens)).view(
            b, self.num_anchors, self.num_points, 2
        )
        return {
            "anchor_embeddings": tokens,
            "cls_logits": cls_logits,
            "curve_points": curve_points,
        }


_VARIANTS: dict[str, dict[str, int | float]] = {
    "laneatt_tiny": {"stem": 16, "hidden": 32, "depth": 1, "token_dim": 32, "dropout": 0.0},
    "laneatt_small": {"stem": 24, "hidden": 48, "depth": 2, "token_dim": 48, "dropout": 0.0},
    "laneatt_base": {"stem": 32, "hidden": 64, "depth": 3, "token_dim": 64, "dropout": 0.1},
}


def build_laneatt_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "laneatt_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del num_lanes, image_size, num_rows, grid_size, num_queries
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown LaneATT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    token_dim = scaled_channels(int(spec["token_dim"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return LaneATTLaneDetector(
        in_channels=int(in_channels),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        num_anchors=int(num_anchors),
        num_points=int(num_points),
        token_dim=int(token_dim),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_laneatt_lane_detector(
        in_channels=3,
        num_lanes=4,
        num_points=18,
        num_anchors=28,
        variant="laneatt_tiny",
        width_mult=1.0,
    )
    out = m(x)
    print("laneatt_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

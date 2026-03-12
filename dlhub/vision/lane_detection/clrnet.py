import torch
from torch import nn

from ._common import (
    GlobalContextHead,
    TinyLaneEncoder,
    choose_attention_heads,
    scaled_channels,
)


class CLRNetLaneDetector(nn.Module):
    """CLRNet-style lane detector with curve proposals and refinement offsets."""

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
        self.context = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=self.token_dim,
            dropout=float(dropout),
        )
        self.lane_embed = nn.Parameter(torch.randn(self.num_queries, self.token_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
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
        self.refine_head = nn.Linear(self.token_dim, self.num_points * 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        _, high = self.encoder(x)
        b = x.shape[0]
        memory = high.flatten(2).transpose(1, 2)
        if memory.shape[-1] != self.token_dim:
            raise RuntimeError(
                f"Token dim mismatch: memory has {memory.shape[-1]}, expected {self.token_dim}"
            )

        queries = self.lane_embed.unsqueeze(0).expand(b, -1, -1)
        context = self.context(high).unsqueeze(1)
        queries = queries + context
        refined, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        tokens = queries + refined
        tokens = tokens + self.ffn(tokens)

        lane_logits = self.lane_head(tokens).squeeze(-1)
        curve_points = torch.tanh(self.curve_head(tokens)).view(
            b, self.num_queries, self.num_points, 2
        )
        refinement_offsets = torch.tanh(self.refine_head(tokens)).view(
            b, self.num_queries, self.num_points, 2
        )
        return {
            "curve_points": curve_points,
            "lane_logits": lane_logits,
            "refinement_offsets": refinement_offsets,
        }


_VARIANTS: dict[str, dict[str, int | float]] = {
    "clrnet_tiny": {"stem": 16, "hidden": 32, "depth": 1, "token_dim": 32, "dropout": 0.0},
    "clrnet_small": {"stem": 24, "hidden": 48, "depth": 2, "token_dim": 48, "dropout": 0.0},
    "clrnet_base": {"stem": 32, "hidden": 64, "depth": 3, "token_dim": 64, "dropout": 0.1},
}


def build_clrnet_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "clrnet_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del num_lanes, image_size, num_rows, grid_size, num_anchors
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CLRNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    token_dim = scaled_channels(int(spec["token_dim"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return CLRNetLaneDetector(
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
    m = build_clrnet_lane_detector(
        in_channels=3,
        num_lanes=4,
        num_points=16,
        variant="clrnet_tiny",
    )
    out = m(x)
    print("clrnet_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

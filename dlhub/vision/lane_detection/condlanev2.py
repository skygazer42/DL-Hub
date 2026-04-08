import torch
from torch import nn

from ._common import (
    GlobalContextHead,
    SegmentationDecoder,
    SpatialMessagePassing,
    TinyLaneEncoder,
    choose_attention_heads,
    scaled_channels,
)


class Condlanev2LaneDetector(nn.Module):
    """Toy lane detector for the condlanev2 family."""

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
        use_message_passing: bool,
        use_prompt_tokens: bool,
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
        self.seg = SegmentationDecoder(
            low_channels=int(hidden_channels),
            high_channels=int(hidden_channels),
            out_channels=int(hidden_channels),
            dropout=float(dropout),
        )
        self.message = SpatialMessagePassing(int(hidden_channels)) if use_message_passing else nn.Identity()
        self.context = GlobalContextHead(
            int(hidden_channels),
            hidden_dim=int(hidden_channels),
            out_dim=int(token_dim),
            dropout=float(dropout),
        )
        self.query = nn.Parameter(torch.randn(self.num_queries, self.token_dim) * 0.02)
        self.prompt = (
            nn.Parameter(torch.randn(2, self.token_dim) * 0.02) if use_prompt_tokens else None
        )
        self.memory_proj = nn.Conv2d(int(hidden_channels), int(token_dim), kernel_size=1, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=int(token_dim),
            num_heads=choose_attention_heads(int(token_dim)),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.LayerNorm(int(token_dim)),
            nn.Linear(int(token_dim), int(token_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(token_dim) * 2, int(token_dim)),
        )
        self.lane_head = nn.Linear(int(token_dim), 1)
        self.curve_head = nn.Linear(int(token_dim), self.num_points * 2)
        self.seg_head = nn.Conv2d(int(hidden_channels), 2, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        low, high = self.encoder(x)
        low = self.message(low)
        seg_feat = self.seg(low, high, output_size=tuple(x.shape[-2:]))
        seg_logits = self.seg_head(seg_feat)

        memory = self.memory_proj(high).flatten(2).transpose(1, 2)
        b = x.shape[0]
        queries = self.query.unsqueeze(0).expand(b, -1, -1)
        queries = queries + self.context(high).unsqueeze(1)
        if self.prompt is not None:
            queries = queries + self.prompt.mean(dim=0, keepdim=True).unsqueeze(0)
        refined, _ = self.cross_attn(queries, memory, memory, need_weights=False)
        tokens = queries + refined
        tokens = tokens + self.ffn(tokens)
        lane_logits = self.lane_head(tokens).squeeze(-1)
        curve_points = torch.tanh(self.curve_head(tokens)).view(b, self.num_queries, self.num_points, 2)
        return {
            "lane_logits": lane_logits,
            "curve_points": curve_points,
            "segmentation_logits": seg_logits,
        }


_VARIANTS: dict[str, dict[str, int | float]] = {
    "condlanev2_tiny": {"stem": 16, "hidden": 32, "depth": 1, "token_dim": 32, "dropout": 0.0},
    "condlanev2_small": {"stem": 24, "hidden": 48, "depth": 2, "token_dim": 48, "dropout": 0.0},
    "condlanev2_base": {"stem": 32, "hidden": 64, "depth": 3, "token_dim": 64, "dropout": 0.1},
}


def build_condlanev2_lane_detector(
    *,
    in_channels: int,
    num_lanes: int,
    image_size: int = 64,
    num_points: int = 16,
    num_rows: int = 16,
    grid_size: int = 32,
    num_anchors: int = 24,
    num_queries: int = 6,
    variant: str = "condlanev2_small",
    width_mult: float = 1.0,
    dropout: float | None = None,
) -> nn.Module:
    del num_lanes, image_size, num_rows, grid_size, num_anchors
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown condlanev2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")

    spec = _VARIANTS[name]
    stem = scaled_channels(int(spec["stem"]), float(width_mult))
    hidden = scaled_channels(int(spec["hidden"]), float(width_mult))
    token_dim = scaled_channels(int(spec["token_dim"]), float(width_mult))
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return Condlanev2LaneDetector(
        in_channels=int(in_channels),
        num_queries=int(num_queries),
        num_points=int(num_points),
        stem_channels=int(stem),
        hidden_channels=int(hidden),
        depth=int(spec["depth"]),
        token_dim=int(token_dim),
        use_message_passing=True,
        use_prompt_tokens=False,
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_condlanev2_lane_detector(
        in_channels=3,
        num_lanes=4,
        num_points=16,
        variant="condlanev2_tiny",
    )
    out = m(x)
    print("condlanev2_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = sum(v.mean() for v in out.values())
    loss.backward()
    print("ok")

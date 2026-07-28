from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "ca_sum_tiny": {"width": 24, "depth": 2},
    "ca_sum_small": {"width": 32, "depth": 3},
    "ca_sum_base": {"width": 48, "depth": 4},
}


class ContentAttentionSummarizer(nn.Module):
    """CA-SUM-style content-attention summarizer (compact)."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim, max(32, dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(32, dim // 2), 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        global_ctx = feat.mean(dim=1, keepdim=True)
        q = self.to_q(feat)
        k = self.to_k(global_ctx)
        v = self.to_v(global_ctx)
        attn = torch.softmax(
            (q * k).sum(dim=-1, keepdim=True) / max(1.0, float(q.shape[-1]) ** 0.5), dim=1
        )
        fused = feat + attn * v
        scores = torch.sigmoid(self.head(fused).squeeze(-1))
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask}


def build_ca_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "ca_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CA-SUM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ContentAttentionSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_ca_sum_video_summarizer(in_channels=3, variant="ca_sum_tiny", width_mult=0.5)
    out = m(x)
    print("ca_sum_tiny", tuple(out["scores"].shape), tuple(out["summary_mask"].shape))
    loss = out["scores"].mean()
    loss.backward()
    print("ok")

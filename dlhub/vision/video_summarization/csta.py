from __future__ import annotations

import math

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "csta_tiny": {"width": 24, "depth": 2},
    "csta_small": {"width": 32, "depth": 3},
    "csta_base": {"width": 48, "depth": 4},
}


class CSTAVideoSummarizer(nn.Module):
    """CSTA-style spatiotemporal attention summarizer (compact).

    The compact version keeps two pieces:
    - local temporal mixing with 1D conv
    - global temporal self-attention over frame features
    """

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        self.local = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
        )
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(dim, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)  # (B,T,D)
        local = self.local(feat.transpose(1, 2)).transpose(1, 2)
        q = self.to_q(feat)
        k = self.to_k(feat)
        v = self.to_v(feat)
        attn = torch.softmax(
            torch.matmul(q, k.transpose(1, 2)) / math.sqrt(max(1, int(q.shape[-1]))), dim=-1
        )
        global_feat = torch.matmul(attn, v)
        fused = torch.cat([local, global_feat], dim=-1)
        scores = torch.sigmoid(self.head(fused).squeeze(-1))
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask, "attention_map": attn}


def build_csta_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "csta_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CSTA variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CSTAVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_csta_video_summarizer(in_channels=3, variant="csta_tiny", width_mult=0.5)
    out = m(x)
    print("csta_tiny", tuple(out["scores"].shape), tuple(out["attention_map"].shape))
    loss = out["scores"].mean() + out["attention_map"].mean()
    loss.backward()
    print("ok")

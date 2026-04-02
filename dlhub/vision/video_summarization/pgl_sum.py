from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "pgl_sum_tiny": {"width": 24, "depth": 2},
    "pgl_sum_small": {"width": 32, "depth": 3},
    "pgl_sum_base": {"width": 48, "depth": 4},
}


class PGLSUMVideoSummarizer(nn.Module):
    """PGL-SUM-style summarizer (toy).

    The model keeps two complementary paths:
    - a local temporal conv path for neighborhood cues
    - a global self-attention path for long-range context
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
        self.local_path = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=max(64, dim * 2),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.global_path = nn.TransformerEncoder(layer, num_layers=max(1, int(depth) - 1))
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(dim, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)  # (B,T,D)
        local = self.local_path(feat.transpose(1, 2)).transpose(1, 2)
        global_feat = self.global_path(feat)
        fused = torch.cat([local, global_feat], dim=-1)
        scores = torch.sigmoid(self.head(fused).squeeze(-1))
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask, "features": fused}


def build_pgl_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "pgl_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown PGL-SUM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return PGLSUMVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_pgl_sum_video_summarizer(in_channels=3, variant="pgl_sum_tiny", width_mult=0.5)
    out = m(x)
    print("pgl_sum_tiny", tuple(out["scores"].shape), tuple(out["summary_mask"].shape))
    loss = out["scores"].mean() + out["features"].mean()
    loss.backward()
    print("ok")


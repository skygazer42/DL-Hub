from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "fulltransnet_tiny": {"width": 24, "depth": 2},
    "fulltransnet_small": {"width": 32, "depth": 3},
    "fulltransnet_base": {"width": 48, "depth": 4},
}


class FullTransNetVideoSummarizer(nn.Module):
    """FullTransNet-style full transformer summarizer (compact).

    Uses both a transformer encoder over frame tokens and a transformer decoder with learned
    summary queries, then projects the fused per-frame representation to importance scores.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        seq_len: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=max(64, dim * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=max(64, dim * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder_tf = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(depth)))
        self.decoder_tf = nn.TransformerDecoder(dec_layer, num_layers=max(1, int(depth) - 1))
        self.query = nn.Parameter(torch.randn(1, int(self.seq_len), dim) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(dim, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)  # (B,T,D)
        b, t, _ = feat.shape
        mem = self.encoder_tf(feat)
        query = self.query[:, : int(t)].expand(int(b), -1, -1)
        dec = self.decoder_tf(query, mem)
        fused = torch.cat([mem, dec], dim=-1)
        scores = torch.sigmoid(self.head(fused).squeeze(-1))
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask, "decoder_tokens": dec}


def build_fulltransnet_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "fulltransnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown FullTransNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return FullTransNetVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        seq_len=int(seq_len),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_fulltransnet_video_summarizer(
        in_channels=3, seq_len=8, variant="fulltransnet_tiny", width_mult=0.5
    )
    out = m(x)
    print("fulltransnet_tiny", tuple(out["scores"].shape), tuple(out["decoder_tokens"].shape))
    loss = out["scores"].mean() + out["decoder_tokens"].mean()
    loss.backward()
    print("ok")

from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "mhscnet_tiny": {"width": 24, "depth": 2},
    "mhscnet_small": {"width": 32, "depth": 3},
    "mhscnet_base": {"width": 48, "depth": 4},
}


class MHSCNetVideoSummarizer(nn.Module):
    """MHSCNet-style shot-aware summarizer (toy, unimodal adaptation).

    This keeps the key inductive bias:
    - multi-scale temporal convolutions
    - shot-boundary awareness from temporal frame differences
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
        self.ms3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.ms5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2)
        self.ms7 = nn.Conv1d(dim, dim, kernel_size=7, padding=3)
        self.fuse = nn.Sequential(
            nn.Conv1d(dim * 3, dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(dim + 1, max(32, dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(max(32, dim // 2), 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)  # (B,T,D)
        x = feat.transpose(1, 2)
        ms = torch.cat([self.ms3(x), self.ms5(x), self.ms7(x)], dim=1)
        fused = self.fuse(ms).transpose(1, 2)

        # Toy shot cues from temporal feature differences.
        delta = (feat[:, 1:] - feat[:, :-1]).pow(2).mean(dim=-1, keepdim=True).sqrt()
        boundary = torch.cat([torch.zeros_like(delta[:, :1]), delta], dim=1)
        boundary = torch.tanh(boundary)

        scores = torch.sigmoid(self.head(torch.cat([fused, boundary], dim=-1)).squeeze(-1))
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask, "boundary_scores": boundary.squeeze(-1)}


def build_mhscnet_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "mhscnet_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MHSCNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MHSCNetVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_mhscnet_video_summarizer(in_channels=3, variant="mhscnet_tiny", width_mult=0.5)
    out = m(x)
    print("mhscnet_tiny", tuple(out["scores"].shape), tuple(out["boundary_scores"].shape))
    loss = out["scores"].mean() + out["boundary_scores"].mean()
    loss.backward()
    print("ok")


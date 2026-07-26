from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "checkmate_tiny": {"width": 24, "depth": 2},
    "checkmate_small": {"width": 32, "depth": 3},
    "checkmate_base": {"width": 48, "depth": 4},
}


def _reverse_cumavg(x: torch.Tensor) -> torch.Tensor:
    rev = x.flip(1)
    counts = torch.arange(1, int(x.shape[1]) + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
    return (rev.cumsum(dim=1) / counts).flip(1)


class CheckMATEVideoSummarizer(nn.Module):
    """Temporal encapsulation summarizer with mutually averaged context checks."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        hidden = max(32, dim // 2)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        counts = torch.arange(1, int(t) + 1, device=feat.device, dtype=feat.dtype).view(
            1, int(t), 1
        )
        left_avg = feat.cumsum(dim=1) / counts
        right_avg = _reverse_cumavg(feat)
        center_avg = F.avg_pool1d(
            feat.transpose(1, 2), kernel_size=3, stride=1, padding=1
        ).transpose(1, 2)
        mate = (left_avg + right_avg + center_avg) / 3.0

        proj_feat = F.normalize(self.proj(feat), dim=-1)
        proj_mate = F.normalize(self.proj(mate), dim=-1)
        temporal_match = (proj_feat * proj_mate).sum(dim=-1)

        novelty = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        if int(t) > 1:
            novelty[:, 1:] = (feat[:, 1:] - feat[:, :-1]).pow(2).mean(dim=-1).sqrt()

        fused = torch.cat([feat, mate, feat - mate], dim=-1)
        raw_scores = self.head(fused).squeeze(-1)
        scores = torch.sigmoid(raw_scores + 0.30 * temporal_match + 0.15 * novelty)
        summary_mask = scores_to_mask(scores)
        summary_token = torch.einsum("bt,btd->bd", torch.softmax(scores, dim=-1), feat)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "temporal_match": temporal_match,
            "summary_token": summary_token,
        }


def build_checkmate_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "checkmate_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CheckMATE variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CheckMATEVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_checkmate_video_summarizer(in_channels=3, variant="checkmate_tiny", width_mult=0.5)
    out = m(x)
    print("checkmate_tiny", tuple(out["scores"].shape), tuple(out["summary_token"].shape))
    loss = out["scores"].mean() + out["summary_token"].mean()
    loss.backward()
    print("ok")

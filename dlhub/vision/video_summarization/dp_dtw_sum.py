from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "dp_dtw_sum_tiny": {"width": 24, "depth": 2},
    "dp_dtw_sum_small": {"width": 32, "depth": 3},
    "dp_dtw_sum_base": {"width": 48, "depth": 4},
}


class DPDTWSumVideoSummarizer(nn.Module):
    """Prototype-DTW-style action-based summarizer."""

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
        self.num_proto = max(3, int(depth) + 1)
        self.prototype_bank = nn.Parameter(torch.randn(self.num_proto, dim) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(dim * 2 + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        bank = self.prototype_bank.to(device=feat.device, dtype=feat.dtype)
        norm_feat = F.normalize(feat, dim=-1)
        norm_bank = F.normalize(bank, dim=-1)

        proto_affinity = torch.matmul(norm_feat, norm_bank.transpose(0, 1))  # (B,T,P)
        align_path = torch.cumsum(torch.softmax(proto_affinity, dim=-1), dim=1)
        align_path = align_path / align_path[:, -1:, :].clamp_min(1e-6)
        aligned_proto = torch.matmul(torch.softmax(proto_affinity, dim=-1), bank)

        temporal_cost = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        if int(t) > 1:
            temporal_cost[:, 1:] = (
                (aligned_proto[:, 1:] - aligned_proto[:, :-1]).pow(2).mean(dim=-1).sqrt()
            )

        best_proto_score = proto_affinity.amax(dim=-1, keepdim=True)
        raw_scores = self.head(torch.cat([feat, aligned_proto, best_proto_score], dim=-1)).squeeze(
            -1
        )
        scores = torch.sigmoid(
            raw_scores + 0.30 * proto_affinity.amax(dim=-1) - 0.15 * temporal_cost
        )
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "proto_affinity": proto_affinity,
            "align_path": align_path,
        }


def build_dp_dtw_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "dp_dtw_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DP-DTW-SUM variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return DPDTWSumVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_dp_dtw_sum_video_summarizer(
        in_channels=3,
        variant="dp_dtw_sum_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("dp_dtw_sum_tiny", tuple(out["scores"].shape), tuple(out["proto_affinity"].shape))
    loss = out["scores"].mean() + out["align_path"].mean()
    loss.backward()
    print("ok")

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "viewpoint_sum_tiny": {"width": 24, "depth": 2},
    "viewpoint_sum_small": {"width": 32, "depth": 3},
    "viewpoint_sum_base": {"width": 48, "depth": 4},
}


class ViewpointSumVideoSummarizer(nn.Module):
    """Viewpoint-aware summarizer with latent viewpoint prototypes."""

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
        num_viewpoints = max(3, int(depth) + 2)
        self.viewpoint_bank = nn.Parameter(torch.randn(num_viewpoints, dim) * 0.02)
        self.score_head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        bank = self.viewpoint_bank.to(device=feat.device, dtype=feat.dtype)
        norm_feat = F.normalize(feat, dim=-1)
        norm_bank = F.normalize(bank, dim=-1)

        viewpoint_probs = torch.softmax(torch.matmul(norm_feat, norm_bank.transpose(0, 1)), dim=-1)
        viewpoint_context = torch.matmul(viewpoint_probs, bank)
        consensus = viewpoint_probs.mean(dim=1)
        consensus_context = torch.matmul(consensus, bank).unsqueeze(1).expand(-1, int(t), -1)

        novelty = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        if int(t) > 1:
            novelty[:, 1:] = (norm_feat[:, 1:] - norm_feat[:, :-1]).pow(2).mean(dim=-1).sqrt()

        fused = torch.cat([feat, viewpoint_context, consensus_context], dim=-1)
        raw_scores = self.score_head(fused).squeeze(-1)
        viewpoint_alignment = (norm_feat * F.normalize(viewpoint_context, dim=-1)).sum(dim=-1)
        scores = torch.sigmoid(raw_scores + 0.30 * viewpoint_alignment + 0.15 * novelty)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "viewpoint_probs": viewpoint_probs,
            "consensus": consensus,
        }


def build_viewpoint_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "viewpoint_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Viewpoint-SUM variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ViewpointSumVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_viewpoint_sum_video_summarizer(
        in_channels=3,
        variant="viewpoint_sum_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("viewpoint_sum_tiny", tuple(out["scores"].shape), tuple(out["viewpoint_probs"].shape))
    loss = out["scores"].mean() + out["consensus"].mean()
    loss.backward()
    print("ok")

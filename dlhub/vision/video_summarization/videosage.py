from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "videosage_tiny": {"width": 24, "depth": 2},
    "videosage_small": {"width": 32, "depth": 3},
    "videosage_base": {"width": 48, "depth": 4},
}


class VideoSAGEVideoSummarizer(nn.Module):
    """Sparse GraphSAGE-style video summarizer."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.radius = max(2, int(depth) + 1)
        self.steps = max(1, int(depth) - 1)
        self.encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.encoder.out_dim)
        hidden = max(32, dim // 2)
        self.self_proj = nn.Linear(dim, dim)
        self.neigh_proj = nn.Linear(dim, dim)
        self.combine = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def _sparse_affinity(self, feat: torch.Tensor) -> torch.Tensor:
        _, t, _ = feat.shape
        sim = torch.matmul(F.normalize(feat, dim=-1), F.normalize(feat, dim=-1).transpose(1, 2))
        pos = torch.arange(int(t), device=feat.device)
        dist = (pos.view(1, int(t), 1) - pos.view(1, 1, int(t))).abs()
        mask = dist <= int(self.radius)
        return torch.softmax(sim.masked_fill(~mask, -1e4), dim=-1)

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        affinity = self._sparse_affinity(feat)
        h = feat
        for _ in range(int(self.steps)):
            neigh = torch.matmul(affinity, self.neigh_proj(h))
            h = self.combine(torch.cat([self.self_proj(h), neigh], dim=-1))

        node_margin = affinity.amax(dim=-1)
        raw_scores = self.head(h).squeeze(-1)
        scores = torch.sigmoid(raw_scores + 0.25 * node_margin)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "graph_affinity": affinity,
            "node_embeddings": h,
        }


def build_videosage_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "videosage_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown VideoSAGE variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return VideoSAGEVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_videosage_video_summarizer(in_channels=3, variant="videosage_tiny", width_mult=0.5)
    out = m(x)
    print("videosage_tiny", tuple(out["scores"].shape), tuple(out["graph_affinity"].shape))
    loss = out["scores"].mean() + out["node_embeddings"].mean()
    loss.backward()
    print("ok")

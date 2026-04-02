from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "hsa_rnn_tiny": {"width": 24, "depth": 2},
    "hsa_rnn_small": {"width": 32, "depth": 3},
    "hsa_rnn_base": {"width": 48, "depth": 4},
}


class HSARNNVideoSummarizer(nn.Module):
    """Hierarchical structure-adaptive RNN summarizer."""

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
        self.frame_gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.shot_rnn = nn.GRU(
            input_size=dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.shot_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        frame_gate = torch.sigmoid(self.frame_gate(feat).squeeze(-1))

        win = max(2, min(4, int(t)))
        shot_feat = []
        shot_ranges = []
        for start in range(0, int(t), int(win)):
            end = min(int(t), start + int(win))
            weight = frame_gate[:, start:end].unsqueeze(-1)
            pooled = (feat[:, start:end] * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
            shot_feat.append(pooled.unsqueeze(1))
            shot_ranges.append((start, end))
        shots = torch.cat(shot_feat, dim=1)

        shot_ctx, _ = self.shot_rnn(shots)
        shot_scores = torch.sigmoid(self.shot_head(shot_ctx).squeeze(-1))

        frame_scores = torch.zeros(int(b), int(t), device=feat.device, dtype=feat.dtype)
        for idx, (start, end) in enumerate(shot_ranges):
            local = frame_gate[:, start:end]
            frame_scores[:, start:end] = local * shot_scores[:, idx : idx + 1]

        scores = frame_scores.clamp(0.0, 1.0)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "shot_scores": shot_scores,
            "frame_gate": frame_gate,
        }


def build_hsa_rnn_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "hsa_rnn_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown HSA-RNN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return HSARNNVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_hsa_rnn_video_summarizer(in_channels=3, variant="hsa_rnn_tiny", width_mult=0.5)
    out = m(x)
    print("hsa_rnn_tiny", tuple(out["scores"].shape), tuple(out["shot_scores"].shape))
    loss = out["scores"].mean() + out["frame_gate"].mean()
    loss.backward()
    print("ok")

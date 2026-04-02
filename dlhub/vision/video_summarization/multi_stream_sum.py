from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "multi_stream_sum_tiny": {"width": 24, "depth": 2},
    "multi_stream_sum_small": {"width": 32, "depth": 3},
    "multi_stream_sum_base": {"width": 48, "depth": 4},
}


class MultiStreamVideoSummarizer(nn.Module):
    """Multi-stream summarizer with appearance, motion, and context branches."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.appearance_encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        self.motion_encoder = TinyFrameEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        dim = int(self.appearance_encoder.out_dim)
        hidden = max(32, dim // 2)
        self.stream_gate = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 3),
        )
        self.head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        appearance = self.appearance_encoder(video)

        motion_video = torch.zeros_like(video)
        motion_video[:, 1:] = video[:, 1:] - video[:, :-1]
        motion = self.motion_encoder(motion_video)

        context = 0.5 * appearance + 0.5 * torch.roll(appearance, shifts=1, dims=1)
        fused_streams = torch.cat([appearance, motion, context], dim=-1)
        stream_weights = torch.softmax(self.stream_gate(fused_streams), dim=-1)

        weighted = (
            stream_weights[..., 0:1] * appearance
            + stream_weights[..., 1:2] * motion
            + stream_weights[..., 2:3] * context
        )
        raw_scores = self.head(torch.cat([weighted, appearance, motion], dim=-1)).squeeze(-1)
        motion_energy = motion.pow(2).mean(dim=-1).sqrt()
        scores = torch.sigmoid(raw_scores + 0.20 * motion_energy)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "stream_weights": stream_weights,
            "motion_energy": motion_energy,
        }


def build_multi_stream_sum_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "multi_stream_sum_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Multi-Stream-SUM variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return MultiStreamVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_multi_stream_sum_video_summarizer(
        in_channels=3,
        variant="multi_stream_sum_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("multi_stream_sum_tiny", tuple(out["scores"].shape), tuple(out["stream_weights"].shape))
    loss = out["scores"].mean() + out["motion_energy"].mean()
    loss.backward()
    print("ok")

from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "summdiff_tiny": {"width": 24, "depth": 2},
    "summdiff_small": {"width": 32, "depth": 3},
    "summdiff_base": {"width": 48, "depth": 4},
}


class ScoreDiffusion(nn.Module):
    def __init__(self, *, feat_dim: int, hidden_dim: int, steps: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.steps = int(max(1, steps))
        self.time = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 1 + hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feat.ndim != 3:
            raise ValueError(f"feat must have shape (B, T, D), got {tuple(feat.shape)}")
        b, t, _ = feat.shape
        score = torch.randn(int(b), int(t), 1, device=feat.device, dtype=feat.dtype)
        noise0 = score.clone()
        for i in range(int(self.steps)):
            tau = 1.0 - float(i) / max(1, int(self.steps))
            t_code = self.time(
                torch.full((int(b), int(t), 1), tau, device=feat.device, dtype=feat.dtype)
            )
            eps = self.net(torch.cat([feat, score, t_code], dim=-1))
            step = 0.7 / float(self.steps)
            score = score - float(step) * torch.tanh(eps)
        return score.squeeze(-1), noise0.squeeze(-1)


class SummDiffVideoSummarizer(nn.Module):
    """SummDiff-style diffusion summarizer (toy).

    Generates frame importance scores by iterative denoising conditioned on frame features.
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
        hidden = max(32, dim // 2)
        self.diff = ScoreDiffusion(feat_dim=dim, hidden_dim=hidden, steps=max(2, int(depth)), dropout=float(dropout))

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        raw_scores, noise0 = self.diff(feat)
        scores = torch.sigmoid(raw_scores)
        summary_mask = scores_to_mask(scores)
        return {"scores": scores, "summary_mask": summary_mask, "initial_noise": noise0}


def build_summdiff_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "summdiff_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown SummDiff variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return SummDiffVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_summdiff_video_summarizer(in_channels=3, variant="summdiff_tiny", width_mult=0.5)
    out = m(x)
    print("summdiff_tiny", tuple(out["scores"].shape), tuple(out["initial_noise"].shape))
    loss = out["scores"].mean() + out["initial_noise"].mean()
    loss.backward()
    print("ok")


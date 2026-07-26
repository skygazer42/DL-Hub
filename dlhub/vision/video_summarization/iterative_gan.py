from __future__ import annotations

import torch
from torch import nn

from ._common import TinyFrameEncoder, TemporalGRUScorer, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "iterative_gan_tiny": {"width": 24, "depth": 2},
    "iterative_gan_small": {"width": 32, "depth": 3},
    "iterative_gan_base": {"width": 48, "depth": 4},
}


class IterativeGANVideoSummarizer(nn.Module):
    """Iterative simplified GAN summarizer."""

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
        self.selector = TemporalGRUScorer(
            dim=dim, hidden_dim=hidden, layers=1, dropout=float(dropout)
        )
        self.refiner = nn.Sequential(
            nn.Linear(dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self.reconstruct = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        init_scores = torch.sigmoid(self.selector(feat))
        refined_logits = self.refiner(torch.cat([feat, init_scores.unsqueeze(-1)], dim=-1)).squeeze(
            -1
        )
        scores = torch.sigmoid(
            0.6 * refined_logits + 0.4 * torch.logit(init_scores.clamp(1e-4, 1 - 1e-4))
        )
        summary_mask = scores_to_mask(scores)

        summary_vec = (feat * scores.unsqueeze(-1)).sum(dim=1) / scores.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        recon = self.reconstruct(summary_vec)
        target = feat.mean(dim=1)
        recon_gap = (recon - target).pow(2).mean(dim=-1, keepdim=True)
        disc_logits = self.discriminator(summary_vec)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "discriminator_logits": disc_logits,
            "reconstruction_gap": recon_gap,
        }


def build_iterative_gan_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "iterative_gan_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown Iterative-GAN variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return IterativeGANVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_iterative_gan_video_summarizer(
        in_channels=3,
        variant="iterative_gan_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print(
        "iterative_gan_tiny", tuple(out["scores"].shape), tuple(out["discriminator_logits"].shape)
    )
    loss = out["scores"].mean() + out["reconstruction_gap"].mean()
    loss.backward()
    print("ok")

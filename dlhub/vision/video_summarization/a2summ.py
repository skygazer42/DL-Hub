from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "a2summ_tiny": {"width": 24, "depth": 2},
    "a2summ_small": {"width": 32, "depth": 3},
    "a2summ_base": {"width": 48, "depth": 4},
}


class A2SummVideoSummarizer(nn.Module):
    """Align-and-attend summarizer with dual contrastive latent modalities."""

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
        prompt_len = max(4, int(depth) + 2)
        self.audio_prompt = nn.Parameter(torch.randn(prompt_len, dim) * 0.02)
        self.text_prompt = nn.Parameter(torch.randn(prompt_len, dim) * 0.02)
        self.frame_proj = nn.Linear(dim, dim)
        self.audio_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(self, video: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        audio = self.audio_prompt.to(device=feat.device, dtype=feat.dtype).unsqueeze(0).expand(int(b), -1, -1)
        text = self.text_prompt.to(device=feat.device, dtype=feat.dtype).unsqueeze(0).expand(int(b), -1, -1)

        frame_key = self.frame_proj(feat)
        audio_key = self.audio_proj(audio)
        text_key = self.text_proj(text)
        audio_attn = torch.softmax(
            torch.einsum("btd,bld->btl", frame_key, audio_key) / math.sqrt(max(1, int(d))),
            dim=-1,
        )
        text_attn = torch.softmax(
            torch.einsum("btd,bld->btl", frame_key, text_key) / math.sqrt(max(1, int(d))),
            dim=-1,
        )
        audio_ctx = torch.einsum("btl,bld->btd", audio_attn, audio)
        text_ctx = torch.einsum("btl,bld->btd", text_attn, text)

        audio_align = (F.normalize(frame_key, dim=-1) * F.normalize(audio_ctx, dim=-1)).sum(dim=-1)
        text_align = (F.normalize(frame_key, dim=-1) * F.normalize(text_ctx, dim=-1)).sum(dim=-1)
        dual_contrast = 0.5 * (audio_align + text_align)

        fused = torch.cat([feat, audio_ctx, text_ctx], dim=-1)
        raw_scores = self.head(fused).squeeze(-1)
        scores = torch.sigmoid(raw_scores + 0.25 * dual_contrast)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "audio_attention": audio_attn,
            "text_attention": text_attn,
        }


def build_a2summ_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "a2summ_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown A2Summ variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return A2SummVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_a2summ_video_summarizer(in_channels=3, variant="a2summ_tiny", width_mult=0.5)
    out = m(x)
    print("a2summ_tiny", tuple(out["scores"].shape), tuple(out["audio_attention"].shape))
    loss = out["scores"].mean() + out["text_attention"].mean()
    loss.backward()
    print("ok")

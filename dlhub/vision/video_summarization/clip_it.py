from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import TinyFrameEncoder, scores_to_mask

_VARIANTS: dict[str, dict[str, int]] = {
    "clip_it_tiny": {"width": 24, "depth": 2},
    "clip_it_small": {"width": 32, "depth": 3},
    "clip_it_base": {"width": 48, "depth": 4},
}


def _prepare_text(
    text: torch.Tensor | None,
    *,
    batch: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    prompt: torch.Tensor,
) -> torch.Tensor:
    if text is None:
        return prompt.unsqueeze(0).expand(int(batch), -1, -1)

    x = text.to(device=device, dtype=dtype)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    elif x.ndim != 3:
        raise ValueError(f"text must have shape (L,D) or (B,L,D), got {tuple(x.shape)}")

    if int(x.shape[0]) == 1 and int(batch) > 1:
        x = x.expand(int(batch), -1, -1)
    elif int(x.shape[0]) != int(batch):
        raise ValueError(f"text batch {int(x.shape[0])} does not match video batch {int(batch)}")

    cur_dim = int(x.shape[-1])
    if cur_dim < int(dim):
        pad = torch.zeros(
            int(batch), int(x.shape[1]), int(dim) - cur_dim, device=device, dtype=dtype
        )
        x = torch.cat([x, pad], dim=-1)
    elif cur_dim > int(dim):
        x = x[..., : int(dim)]
    return x


class CLIPItVideoSummarizer(nn.Module):
    """Language-guided multimodal transformer summarizer."""

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
        self.text_prompt = nn.Parameter(torch.randn(prompt_len, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            dim_feedforward=max(64, dim * 2),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.mm_encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth) - 1))
        self.frame_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.head = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        video: torch.Tensor,
        text: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        feat = self.encoder(video)
        b, t, d = feat.shape
        text_feat = _prepare_text(
            text,
            batch=int(b),
            dim=int(d),
            device=feat.device,
            dtype=feat.dtype,
            prompt=self.text_prompt.to(device=feat.device, dtype=feat.dtype),
        )
        fused = self.mm_encoder(torch.cat([feat, text_feat], dim=1))
        video_ctx = fused[:, : int(t)]
        text_ctx = fused[:, int(t) :]
        text_global = text_ctx.mean(dim=1, keepdim=True).expand(-1, int(t), -1)

        cross_attn = torch.softmax(
            torch.einsum("btd,bld->btl", self.frame_proj(video_ctx), self.text_proj(text_ctx))
            / math.sqrt(max(1, int(d))),
            dim=-1,
        )
        guided_text = torch.einsum("btl,bld->btd", cross_attn, text_ctx)
        alignment = (F.normalize(video_ctx, dim=-1) * F.normalize(guided_text, dim=-1)).sum(dim=-1)

        raw_scores = self.head(torch.cat([video_ctx, text_global + guided_text], dim=-1)).squeeze(
            -1
        )
        scores = torch.sigmoid(raw_scores + 0.30 * alignment)
        summary_mask = scores_to_mask(scores)
        return {
            "scores": scores,
            "summary_mask": summary_mask,
            "cross_attn": cross_attn,
            "text_tokens": text_ctx,
        }


def build_clip_it_video_summarizer(
    *,
    in_channels: int,
    seq_len: int = 8,
    image_size: int = 64,
    variant: str = "clip_it_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del seq_len, image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown CLIP-It variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return CLIPItVideoSummarizer(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 8, 3, 32, 32)
    m = build_clip_it_video_summarizer(in_channels=3, variant="clip_it_tiny", width_mult=0.5)
    out = m(x)
    print("clip_it_tiny", tuple(out["scores"].shape), tuple(out["cross_attn"].shape))
    loss = out["scores"].mean() + out["text_tokens"].mean()
    loss.backward()
    print("ok")

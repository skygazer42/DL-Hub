"""Fine-R1 (MLLM reasoning for fine-grained recognition) - compact-first FGVC classifier.

References:
- "Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning"
  (arXiv 2026): https://arxiv.org/abs/2602.07605

This repo keeps it offline and lightweight:
- no pretrained weights, no tokenizers, no downloads
- "chain-of-thought" is modeled as a small set of learnable latent reasoning tokens
- reasoning tokens iteratively cross-attend to visual tokens
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import (
    TinyPatchEncoder,
    build_fgvc_model,
    check_nchw,
    make_fgvc_variants,
    smoke_test_classifier,
)


class _ReasoningBlock(nn.Module):
    def __init__(self, *, embed_dim: int, heads: int) -> None:
        super().__init__()
        e = int(embed_dim)
        h = int(heads)
        if e <= 0:
            raise ValueError("embed_dim must be > 0")
        if h <= 0:
            raise ValueError("heads must be > 0")

        self.norm_self = nn.LayerNorm(e)
        self.self_attn = nn.MultiheadAttention(e, h, dropout=0.0, batch_first=True)

        self.norm_q = nn.LayerNorm(e)
        self.norm_kv = nn.LayerNorm(e)
        self.cross_attn = nn.MultiheadAttention(e, h, dropout=0.0, batch_first=True)

        self.norm_mlp = nn.LayerNorm(e)
        hidden = max(64, e * 2)
        self.mlp = nn.Sequential(
            nn.Linear(e, hidden),
            nn.GELU(),
            nn.Linear(hidden, e),
        )

    def forward(self, reason: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        # reason: (B, L, E), visual: (B, N, E)
        r = self.norm_self(reason)
        reason = reason + self.self_attn(r, r, r, need_weights=False)[0]

        q = self.norm_q(reason)
        kv = self.norm_kv(visual)
        reason = reason + self.cross_attn(q, kv, kv, need_weights=False)[0]

        reason = reason + self.mlp(self.norm_mlp(reason))
        return reason


class FineR1(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        spec: dict[str, int],
        in_channels: int,
        num_classes: int,
        image_size: int,
        width_mult: float,
        dropout: float,
    ) -> None:
        super().__init__()
        embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
        self.family = str(family)
        self.reason_len = int(spec["parts"])

        self.visual_encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        # Latent reasoning tokens ("CoT") updated by cross-attention against visual tokens.
        self.reason_tokens = nn.Parameter(torch.randn(1, int(self.reason_len), int(embed)) * 0.02)

        # Keep reasoning shallow; the goal is an offline-friendly architectural sketch.
        reason_depth = max(1, int(spec["depth"]) // 2)
        heads = max(1, min(int(spec["heads"]), 8))
        self.blocks = nn.ModuleList(
            [
                _ReasoningBlock(embed_dim=int(embed), heads=int(heads))
                for _ in range(int(reason_depth))
            ]
        )

        self.proj = nn.Linear(int(embed), int(embed))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(embed), int(num_classes))

        # A small scale makes logits stable for compact training loops.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        visual = self.visual_encoder(x)  # (B, 1+N, E)
        b = int(visual.shape[0])

        reason = self.reason_tokens.expand(b, -1, -1)  # (B, L, E)
        for blk in self.blocks:
            reason = blk(reason, visual)

        # Use the final reasoning token as the class embedding.
        emb = torch.tanh(self.proj(reason[:, -1]))
        emb = F.normalize(emb, dim=-1)

        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * self.classifier(self.dropout(emb))

        return {
            "logits": logits,
            "embedding": emb,
            "cot_tokens": reason,
            "visual_cls": visual[:, 0],
            "logit_scale": scale,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("fine_r1", group="transformer")


def build_fine_r1_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "fine_r1_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        FineR1,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="fine_r1",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_fine_r1_fgvc_classifier, "fine_r1_tiny")

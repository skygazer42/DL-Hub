
"""ReFine-RFT / Cost of Thinking (toy-first) for FGVC.

Reference:
- "Can Textual Reasoning Improve the Performance of MLLMs on Fine-grained Visual Classification?"
  (arXiv 2026): https://arxiv.org/abs/2601.06993

The paper studies how long Chain-of-Thought can *hurt* fine-grained perception ("Cost of Thinking"),
and proposes training strategies to constrain reasoning length.

Toy interpretation here (offline, no LLM):
- maintain a set of latent reasoning tokens
- learn a per-token gate to regulate effective reasoning length
- cross-attend reasoning tokens to visual tokens, then classify from the regulated tokens
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels

from ._common import TinyPatchEncoder, build_fgvc_model, check_nchw, make_fgvc_variants, smoke_test_classifier


class _ReasonBlock(nn.Module):
    def __init__(self, *, embed_dim: int, heads: int) -> None:
        super().__init__()
        e = int(embed_dim)
        h = int(heads)
        self.norm_q = nn.LayerNorm(e)
        self.norm_kv = nn.LayerNorm(e)
        self.cross_attn = nn.MultiheadAttention(e, h, dropout=0.0, batch_first=True)

        self.norm_mlp = nn.LayerNorm(e)
        hidden = max(64, e * 2)
        self.mlp = nn.Sequential(nn.Linear(e, hidden), nn.GELU(), nn.Linear(hidden, e))

    def forward(self, reason: torch.Tensor, visual: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(reason)
        kv = self.norm_kv(visual)
        reason = reason + self.cross_attn(q, kv, kv, need_weights=False)[0]
        reason = reason + self.mlp(self.norm_mlp(reason))
        return reason


class ReFineRFTFGVC(nn.Module):
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
        heads = max(1, min(int(spec["heads"]), 8))
        self.family = str(family)

        # Use a slightly longer token budget than typical part count.
        self.max_reason_len = max(4, int(spec["parts"]) + 4)
        self.reason_tokens = nn.Parameter(torch.randn(1, int(self.max_reason_len), int(embed)) * 0.02)
        self.gate_logits = nn.Parameter(torch.zeros(int(self.max_reason_len)))

        self.encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        depth = max(1, int(spec["depth"]) // 2)
        self.blocks = nn.ModuleList([_ReasonBlock(embed_dim=int(embed), heads=int(heads)) for _ in range(int(depth))])

        self.proj = nn.Linear(int(embed), int(embed))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(embed), int(num_classes))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        visual = self.encoder(x)  # (B, 1+N, E)
        b = int(visual.shape[0])

        reason = self.reason_tokens.expand(b, -1, -1)  # (B, L, E)
        gate = torch.sigmoid(self.gate_logits).view(1, -1, 1)  # (1, L, 1)

        for blk in self.blocks:
            reason = blk(reason, visual)
            # Regulate reasoning length: gated tokens contribute less.
            reason = reason * gate

        # A differentiable proxy for "reasoning length": higher means more tokens active.
        thinking_cost = gate.mean()

        pooled = reason.sum(dim=1) / gate.sum(dim=1).clamp_min(1e-6)
        emb = torch.tanh(self.proj(pooled))
        emb = F.normalize(emb, dim=-1)

        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * self.classifier(self.dropout(emb))

        return {
            "logits": logits,
            "embedding": emb,
            "cot_tokens": reason,
            "cot_gate": gate.squeeze(0),
            "thinking_cost": thinking_cost,
            "logit_scale": scale,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("refine_rft", group="transformer")


def build_refine_rft_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "refine_rft_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        ReFineRFTFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="refine_rft",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_refine_rft_fgvc_classifier, "refine_rft_tiny")


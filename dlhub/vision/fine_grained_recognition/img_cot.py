"""ImgCoT (compact-first) for FGVC.

Reference:
- "ImgCoT: Compressing Long Chain of Thought into Compact Visual Tokens for Efficient Reasoning of Large Language Model"
  (arXiv 2026): https://arxiv.org/abs/2601.22730

This repo keeps it offline and lightweight:
- no pretrained weights, no autoencoders trained on text
- interpret "visual CoT tokens" as a small set of latent tokens that compress patch tokens
- optionally add a few "key" patch tokens (a loose ImgCoT flavor)
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


class ImgCoTFGVC(nn.Module):
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
        self.num_latents = int(spec["parts"])
        self.num_keys = max(1, int(spec["parts"]) // 2)  # "loose ImgCoT": a few key patches

        self.encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        # Latent compact tokens ("visual CoT" surrogate): (1, L, E)
        self.latents = nn.Parameter(torch.randn(1, int(self.num_latents), int(embed)) * 0.02)

        self.q_norm = nn.LayerNorm(int(embed))
        self.kv_norm = nn.LayerNorm(int(embed))
        self.compress_attn = nn.MultiheadAttention(
            int(embed), int(heads), dropout=0.0, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(int(embed)),
            nn.Linear(int(embed), max(64, int(embed) * 2)),
            nn.GELU(),
            nn.Linear(max(64, int(embed) * 2), int(embed)),
        )

        # Decoder-like head: reconstruct patch embeddings from latents.
        self.recon_attn = nn.MultiheadAttention(
            int(embed), int(heads), dropout=0.0, batch_first=True
        )

        self.key_scorer = nn.Linear(int(embed), 1)
        self.proj = nn.Linear(int(embed), int(embed))
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(embed), int(num_classes))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.encoder(x)  # (B, 1+N, E)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]  # (B, N, E)

        b = int(patches.shape[0])
        lat = self.latents.expand(b, -1, -1)  # (B, L, E)

        q = self.q_norm(lat)
        kv = self.kv_norm(patches)
        lat = lat + self.compress_attn(q, kv, kv, need_weights=False)[0]
        lat = lat + self.mlp(lat)

        # "Loose ImgCoT": add a few high-score patch tokens as extra tokens.
        scores = self.key_scorer(patches).squeeze(-1)  # (B, N)
        k = min(int(self.num_keys), int(patches.shape[1]))
        key_vals, key_idx = torch.topk(scores, k=k, dim=1)
        gather_idx = key_idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        key_tokens = torch.gather(patches, 1, gather_idx)  # (B, k, E)

        # Reconstruction loss: can be used as an auxiliary objective in compact training.
        recon = self.recon_attn(
            self.q_norm(patches), self.kv_norm(lat), self.kv_norm(lat), need_weights=False
        )[0]
        recon_loss = F.mse_loss(recon, patches, reduction="mean")

        # Pool compact tokens + a few key tokens.
        pooled = torch.cat([lat, key_tokens], dim=1).mean(dim=1)
        emb = torch.tanh(self.proj(cls + pooled))
        emb = F.normalize(emb, dim=-1)

        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * self.classifier(self.dropout(emb))

        return {
            "logits": logits,
            "embedding": emb,
            "imgcot_tokens": lat,
            "key_indices": key_idx,
            "key_scores": key_vals,
            "recon_loss": recon_loss,
            "logit_scale": scale,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("img_cot", group="transformer")


def build_img_cot_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "img_cot_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        ImgCoTFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="img_cot",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_img_cot_fgvc_classifier, "img_cot_tiny")

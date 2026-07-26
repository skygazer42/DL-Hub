"""FG-CLIP (fine-grained CLIP-style) - toy-first FGVC classifier.

Reference:
- "FG-CLIP: Fine-grained Visual and Textual Alignment" (arXiv 2025)

This implementation is a small, offline-friendly CLIP-shaped model:
- image encoder: tiny ViT-like patch encoder + token selection
- text encoder: toy prompt+class-token transformer (no real text, no downloads)
- logits: scaled cosine similarity between image embeddings and per-class "text" embeddings
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


class ToyTextPromptEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        embed_dim: int,
        prompt_len: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        k = int(num_classes)
        e = int(embed_dim)
        p = int(prompt_len)
        if k <= 0:
            raise ValueError("num_classes must be > 0")
        if p <= 0:
            raise ValueError("prompt_len must be > 0")
        self.num_classes = k
        self.prompt_len = p

        # Treat "text" as: [PROMPT... , CLASS_TOKEN] and encode it.
        self.class_embed = nn.Embedding(int(k), int(e))
        self.prompt = nn.Parameter(torch.randn(1, int(p), int(e)) * 0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, int(p) + 1, int(e)))

        layer = nn.TransformerEncoderLayer(
            d_model=int(e),
            nhead=int(heads),
            dim_feedforward=max(int(e) * 2, 64),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))

    def forward(self) -> torch.Tensor:
        # Return normalized per-class embedding: (K, E)
        k = int(self.num_classes)
        device = self.prompt.device
        ids = torch.arange(k, device=device, dtype=torch.long)
        cls_tok = self.class_embed(ids).unsqueeze(1)  # (K, 1, E)
        prompts = self.prompt.expand(k, -1, -1)  # (K, P, E)
        seq = torch.cat([prompts, cls_tok], dim=1)  # (K, P+1, E)
        seq = seq + self.pos_embed[:, : seq.shape[1]]
        out = self.encoder(seq)
        text = out[:, -1]  # class token output
        return F.normalize(text, dim=-1)


class ProtoClipFG(nn.Module):
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
        del dropout
        super().__init__()
        embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=32, divisor=8)
        self.family = str(family)
        self.num_parts = int(spec["parts"])

        self.image_encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )
        self.token_scorer = nn.Linear(int(embed), 1)
        self.image_proj = nn.Linear(int(embed), int(embed))

        # Keep the text side shallow: enough to be "text-shaped" but cheap.
        text_depth = max(1, int(spec["depth"]) // 2)
        text_heads = max(1, min(int(spec["heads"]), 8))
        prompt_len = 6 if int(embed) >= 128 else 4
        self.text_encoder = ToyTextPromptEncoder(
            num_classes=int(num_classes),
            embed_dim=int(embed),
            prompt_len=int(prompt_len),
            depth=int(text_depth),
            heads=int(text_heads),
        )

        # CLIP-style logit scale; initialize to ~1/0.07.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.image_encoder(x)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        scores = self.token_scorer(patch_tokens).squeeze(-1)
        k = min(int(self.num_parts), patch_tokens.shape[1])
        values, indices = torch.topk(scores, k=k, dim=1)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        selected = torch.gather(patch_tokens, 1, gather_idx)
        pooled = selected.mean(dim=1)

        image = torch.tanh(self.image_proj(cls + pooled))
        image = F.normalize(image, dim=-1)

        text = self.text_encoder()
        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * (image @ text.t())

        return {
            "logits": logits,
            "image_embedding": image,
            "text_embedding": text,
            "logit_scale": scale,
            "selected_indices": indices,
            "selected_scores": values,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("proto_clip_fg", group="transformer")


def build_proto_clip_fg_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "proto_clip_fg_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        ProtoClipFG,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="proto_clip_fg",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_proto_clip_fg_fgvc_classifier, "proto_clip_fg_tiny")


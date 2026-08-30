"""VPT (Visual Prompt Tuning) - compact-first FGVC classifier.

Reference:
- "Visual Prompt Tuning" (ECCV 2022): https://arxiv.org/abs/2203.12119

This implementation is intentionally lightweight:
- shallow prompts (a single learnable prompt token set at the input)
- pure torch, no pretrained weights or downloads
"""

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import build_fgvc_model, check_nchw, make_fgvc_variants, smoke_test_classifier


class PromptedPatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
        prompt_len: int,
    ) -> None:
        super().__init__()
        img = int(image_size)
        patch = int(patch_size)
        if img % patch != 0:
            raise ValueError(f"image_size ({img}) must be divisible by patch_size ({patch})")
        self.patch_embed = nn.Conv2d(
            int(in_channels), int(embed_dim), kernel_size=patch, stride=patch
        )
        grid = img // patch
        num_patches = grid * grid
        self.prompt_len = int(prompt_len)
        if self.prompt_len < 0:
            raise ValueError("prompt_len must be >= 0")

        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, int(self.prompt_len), int(embed_dim)) * 0.02
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1 + int(self.prompt_len), int(embed_dim))
        )

        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(heads),
            dim_feedforward=max(int(embed_dim) * 2, 64),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=int(depth), enable_nested_tensor=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns token sequence: [CLS] + [PROMPTS...] + [PATCHES...]
        x = check_nchw(x)
        tok = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tok.shape[0], -1, -1)
        if self.prompt_len > 0:
            prompts = self.prompt_tokens.expand(tok.shape[0], -1, -1)
            tok = torch.cat([cls, prompts, tok], dim=1)
        else:
            tok = torch.cat([cls, tok], dim=1)
        tok = tok + self.pos_embed[:, : tok.shape[1]]
        return self.encoder(tok)


class VPT(nn.Module):
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
        self.num_parts = int(spec["parts"])
        # Keep prompt length small and stable across variants for fast tests.
        self.prompt_len = 6 if int(spec["depth"]) >= 4 else 4

        self.encoder = PromptedPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
            prompt_len=int(self.prompt_len),
        )
        self.token_scorer = nn.Linear(int(embed), 1)
        self.proj = nn.Linear(int(embed), int(embed))
        self.classifier = nn.Linear(int(embed), int(num_classes))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.encoder(x)

        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1 + int(self.prompt_len) :]
        scores = self.token_scorer(patch_tokens).squeeze(-1)
        k = min(int(self.num_parts), patch_tokens.shape[1])
        values, indices = torch.topk(scores, k=k, dim=1)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        selected = torch.gather(patch_tokens, 1, gather_idx)
        pooled = selected.mean(dim=1)

        embedding = torch.tanh(self.proj(cls + pooled))
        embedding = F.normalize(embedding, dim=-1)
        logits = self.classifier(self.dropout(embedding))

        return {
            "logits": logits,
            "embedding": embedding,
            "prompt_tokens": self.encoder.prompt_tokens,
            "selected_indices": indices,
            "selected_scores": values,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("vpt", group="transformer")


def build_vpt_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "vpt_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        VPT,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="vpt",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_vpt_fgvc_classifier, "vpt_tiny")

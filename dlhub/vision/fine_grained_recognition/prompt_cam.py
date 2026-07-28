"""Prompt-CAM (compact-first).

Reference:
- "Prompt-CAM: A Simpler Interpretable Transformer for Fine-Grained Analysis" (arXiv 2025)
  https://arxiv.org/abs/2501.09333

Compact interpretation:
- Use a ViT-style patch encoder to produce patch tokens.
- Learn class-specific prompt/query vectors.
- Produce a class-wise "CAM-like" attention map over patches via dot-product.
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


class PromptCAMFGVC(nn.Module):
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
        self.num_classes = int(num_classes)
        self.patch = int(spec["patch"])
        self.grid = int(image_size) // max(int(self.patch), 1)
        if int(image_size) % max(int(self.patch), 1) != 0:
            raise ValueError("image_size must be divisible by patch size for Prompt-CAM")

        self.encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(self.patch),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        # Class-specific query/prompts (K, E).
        self.class_prompts = nn.Parameter(torch.randn(int(self.num_classes), int(embed)) * 0.02)

        # A small projection keeps the head stable across variants.
        self.proj = nn.Linear(int(embed), int(embed))
        self.dropout = nn.Dropout(float(dropout))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.encoder(x)  # (B, 1+N, E)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]  # (B, N, E)

        # Normalize for stable dot products.
        patches = F.normalize(self.proj(patch_tokens), dim=-1)
        prompts = F.normalize(self.class_prompts, dim=-1)  # (K, E)

        # (B, N, K): patch relevance per class
        logits_map = torch.einsum("bne,ke->bnk", patches, prompts) / math.sqrt(
            max(int(patches.shape[-1]), 1)
        )
        weights = torch.softmax(logits_map, dim=1)  # soft CAM over patches

        pooled = torch.einsum("bnk,bne->bke", weights, patches)  # (B, K, E)
        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * torch.einsum("bke,ke->bk", pooled, prompts)  # (B, K)

        cam = weights.transpose(1, 2).contiguous()  # (B, K, N)
        cam = cam.view(x.shape[0], int(self.num_classes), int(self.grid), int(self.grid))

        embedding = F.normalize(torch.tanh(cls), dim=-1)
        return {
            "logits": logits,
            "embedding": embedding,
            "prompt_cam": cam,
            "class_prompts": self.class_prompts,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("prompt_cam", group="transformer")


def build_prompt_cam_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "prompt_cam_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        PromptCAMFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="prompt_cam",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_prompt_cam_fgvc_classifier, "prompt_cam_tiny")

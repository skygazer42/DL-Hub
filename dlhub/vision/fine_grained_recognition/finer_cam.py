
"""Finer-CAM (toy-first) for FGVC.

Reference:
- "Finer-CAM: Spotting Difference Reveals Finer Details for Visual Explanation" (arXiv 2025)
  https://arxiv.org/abs/2501.11309

Toy interpretation in this repo (offline, no pretrained weights):
- Use a tiny ViT-style patch encoder to obtain patch tokens.
- Learn class-specific prompt vectors.
- Build a contrast prompt per class by subtracting the mean of "other class" prompts.
- Use contrast prompts to produce a CAM-like attention map over patches.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels

from ._common import TinyPatchEncoder, build_fgvc_model, check_nchw, make_fgvc_variants, smoke_test_classifier


class FinerCAMFGVC(nn.Module):
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
        img = int(image_size)
        if img % max(int(self.patch), 1) != 0:
            raise ValueError("image_size must be divisible by patch size for Finer-CAM")
        self.grid = img // max(int(self.patch), 1)

        self.encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(self.patch),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        # Class prompts: (K, E)
        self.class_prompts = nn.Parameter(torch.randn(int(self.num_classes), int(embed)) * 0.02)
        self.proj = nn.Linear(int(embed), int(embed))
        self.dropout = nn.Dropout(float(dropout))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.encoder(x)  # (B, 1+N, E)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]  # (B, N, E)

        patches = F.normalize(self.proj(patch_tokens), dim=-1)  # (B, N, E)
        prompts = F.normalize(self.class_prompts, dim=-1)  # (K, E)

        # Build contrast prompts: prompt_k - mean_{j!=k}(prompt_j)
        k = int(self.num_classes)
        if k <= 1:
            contrast = prompts
        else:
            sum_prompts = prompts.sum(dim=0, keepdim=True)  # (1, E)
            neg = (sum_prompts - prompts) / float(k - 1)  # (K, E)
            contrast = F.normalize(prompts - neg, dim=-1)

        # CAM-like patch weights from contrast prompts.
        diff_map = torch.einsum("bne,ke->bnk", patches, contrast) / math.sqrt(max(int(patches.shape[-1]), 1))
        weights = torch.softmax(diff_map, dim=1)  # (B, N, K)

        pooled = torch.einsum("bnk,bne->bke", weights, patches)  # (B, K, E)
        pooled = F.normalize(pooled, dim=-1)

        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * torch.einsum("bke,ke->bk", pooled, prompts)  # (B, K)

        # Also expose the "vanilla" prompt CAM for comparison/debugging.
        base_map = torch.einsum("bne,ke->bnk", patches, prompts) / math.sqrt(max(int(patches.shape[-1]), 1))
        base_weights = torch.softmax(base_map, dim=1)

        prompt_cam = base_weights.transpose(1, 2).contiguous().view(x.shape[0], k, int(self.grid), int(self.grid))
        finer_cam = weights.transpose(1, 2).contiguous().view(x.shape[0], k, int(self.grid), int(self.grid))

        embedding = F.normalize(torch.tanh(cls), dim=-1)
        return {
            "logits": logits,
            "embedding": embedding,
            "prompt_cam": prompt_cam,
            "finer_cam": finer_cam,
            "class_prompts": self.class_prompts,
            "contrast_prompts": contrast,
            "logit_scale": scale,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("finer_cam", group="transformer")


def build_finer_cam_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "finer_cam_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        FinerCAMFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="finer_cam",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_finer_cam_fgvc_classifier, "finer_cam_tiny")


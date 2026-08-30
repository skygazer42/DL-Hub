"""XR-VLM (compact-first) for FGVC.

Reference:
- "XR-VLM: Cross-Relationship Modeling with Multi-part Prompts and Visual Features for Fine-Grained Recognition"
  (arXiv 2025): https://arxiv.org/abs/2503.07075

Compact interpretation (offline, no pretrained weights):
- A tiny ViT patch encoder produces patch tokens.
- Each class owns multiple learnable "part prompts" that attend over patches.
- Per-class features are refined by a small self-attention block over the class dimension (class interaction).
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


class XRVLMFGVC(nn.Module):
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
        self.num_parts = int(spec["parts"])
        self.patch = int(spec["patch"])
        img = int(image_size)
        if img % max(int(self.patch), 1) != 0:
            raise ValueError("image_size must be divisible by patch size for XR-VLM")
        self.grid = img // max(int(self.patch), 1)

        self.encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(self.patch),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        # (K, P, E): multi-part prompts per class.
        self.class_part_prompts = nn.Parameter(
            torch.randn(int(self.num_classes), int(self.num_parts), int(embed)) * 0.02
        )
        self.patch_proj = nn.Linear(int(embed), int(embed))

        # Class interaction: treat K class embeddings as a short sequence and run a tiny transformer encoder.
        rel_heads = max(1, min(int(spec["heads"]), 8))
        rel_depth = max(1, int(spec["depth"]) // 2)
        layer = nn.TransformerEncoderLayer(
            d_model=int(embed),
            nhead=int(rel_heads),
            dim_feedforward=max(int(embed) * 2, 64),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.class_rel = nn.TransformerEncoder(
            layer, num_layers=int(rel_depth), enable_nested_tensor=False
        )

        # CLIP-style class prototypes.
        self.class_proto = nn.Embedding(int(self.num_classes), int(embed))
        self.dropout = nn.Dropout(float(dropout))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.encoder(x)  # (B, 1+N, E)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]  # (B, N, E)

        patches = F.normalize(self.patch_proj(patch_tokens), dim=-1)  # (B, N, E)
        prompts = F.normalize(self.class_part_prompts, dim=-1)  # (K, P, E)

        # Patch-to-prompt alignment: (B, K, P, N)
        scores = torch.einsum("bne,kpe->bkpn", patches, prompts) / math.sqrt(
            max(int(patches.shape[-1]), 1)
        )
        attn = torch.softmax(scores, dim=-1)

        # Per-class part features: (B, K, P, E) then aggregate parts -> (B, K, E)
        part_feat = torch.einsum("bkpn,bne->bkpe", attn, patches)
        class_feat = part_feat.mean(dim=2)

        # Cross-relationship modeling across classes.
        class_ctx = self.class_rel(class_feat)
        class_ctx = F.normalize(torch.tanh(class_ctx), dim=-1)

        ids = torch.arange(int(self.num_classes), device=x.device, dtype=torch.long)
        proto = F.normalize(self.class_proto(ids), dim=-1)  # (K, E)
        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * torch.einsum("bke,ke->bk", self.dropout(class_ctx), proto)

        # A single image embedding for convenience.
        embedding = F.normalize(torch.tanh(cls), dim=-1)

        cam = attn.mean(dim=2)  # average across parts: (B, K, N)
        cam = cam.view(x.shape[0], int(self.num_classes), int(self.grid), int(self.grid))

        return {
            "logits": logits,
            "embedding": embedding,
            "class_embeddings": class_ctx,
            "part_embeddings": part_feat,
            "prompt_cam": cam,
            "class_part_prompts": self.class_part_prompts,
            "logit_scale": scale,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("xr_vlm", group="transformer")


def build_xr_vlm_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "xr_vlm_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        XRVLMFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="xr_vlm",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_xr_vlm_fgvc_classifier, "xr_vlm_tiny")

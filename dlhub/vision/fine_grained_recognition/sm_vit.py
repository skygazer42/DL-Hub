
"""SM-ViT (Salient Mask-Guided Vision Transformer) - toy-first FGVC classifier.

Reference (one example of the idea):
- "Salient Mask-Guided Vision Transformer for Fine-Grained Classification" (arXiv 2023)

This repo keeps things toy-first and offline:
- no external saliency generator
- the salient mask is predicted by a tiny conv head and used to reweight patch embeddings
"""

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels

from ._common import build_fgvc_model, check_nchw, make_fgvc_variants, smoke_test_classifier


class SalientMaskPatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        heads: int,
    ) -> None:
        super().__init__()
        img = int(image_size)
        patch = int(patch_size)
        if img % patch != 0:
            raise ValueError(f"image_size ({img}) must be divisible by patch_size ({patch})")

        self.patch_size = int(patch)
        self.patch_embed = nn.Conv2d(int(in_channels), int(embed_dim), kernel_size=patch, stride=patch)

        # Lightweight learnable saliency predictor (mask in [0,1]).
        hid = max(8, int(in_channels) * 2)
        self.saliency = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hid), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hid), 1, kernel_size=1),
        )

        grid = img // patch
        num_patches = grid * grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, int(embed_dim)))

        layer = nn.TransformerEncoderLayer(
            d_model=int(embed_dim),
            nhead=int(heads),
            dim_feedforward=max(int(embed_dim) * 2, 64),
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(depth))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Returns (tokens, salient_mask_patches)
        x = check_nchw(x)
        b, _, h, w = x.shape

        sal = torch.sigmoid(self.saliency(x))  # (B, 1, H, W)
        # Downsample to patch grid as a soft mask.
        mask = F.avg_pool2d(sal, kernel_size=int(self.patch_size), stride=int(self.patch_size))  # (B,1,gh,gw)
        mask_flat = mask.flatten(2).transpose(1, 2)  # (B, N, 1)

        tok = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, E)
        # Reweight token magnitudes. Use a centered scale so initial ~1.0.
        tok = tok * (0.5 + mask_flat)  # scale in [0.5, 1.5]

        cls = self.cls_token.expand(b, -1, -1)
        tok = torch.cat([cls, tok], dim=1)
        tok = tok + self.pos_embed[:, : tok.shape[1]]
        return self.encoder(tok), mask


class SMViT(nn.Module):
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

        self.encoder = SalientMaskPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )
        self.token_scorer = nn.Linear(int(embed), 1)
        self.proj = nn.Linear(int(embed), int(embed))
        self.classifier = nn.Linear(int(embed), int(num_classes))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens, mask = self.encoder(x)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

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
            "salient_mask": mask,
            "selected_indices": indices,
            "selected_scores": values,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("sm_vit", group="transformer")


def build_sm_vit_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "sm_vit_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        SMViT,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="sm_vit",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_sm_vit_fgvc_classifier, "sm_vit_tiny")


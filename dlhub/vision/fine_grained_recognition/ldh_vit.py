
"""LDH-ViT (Local concealment + feature selection) - toy-first FGVC classifier.

Reference:
- "LDH-ViT: Local Concealment and Feature Selection for Fine-Grained Visual Classification"
  (Pattern Recognition, 2024)

Toy interpretation here:
- "local concealment" = patch dropout (randomly zero a subset of patch embeddings)
- "feature selection" = top-k token selection (TransFG-style)
"""

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels

from ._common import TinyPatchEncoder, build_fgvc_model, check_nchw, make_fgvc_variants, smoke_test_classifier


class LDHViT(nn.Module):
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
        self.conceal_prob = 0.18  # a modest default; kept constant for stability

        self.encoder = TinyPatchEncoder(
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
        tokens = self.encoder(x)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        # Local concealment: random per-patch dropout.
        if self.training and self.conceal_prob > 0.0:
            keep = (torch.rand(patch_tokens.shape[:2], device=patch_tokens.device) > float(self.conceal_prob)).to(
                patch_tokens.dtype
            )
            patch_tokens = patch_tokens * keep.unsqueeze(-1)
        else:
            keep = torch.ones(patch_tokens.shape[:2], device=patch_tokens.device, dtype=patch_tokens.dtype)

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
            "conceal_keep": keep,
            "selected_indices": indices,
            "selected_scores": values,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("ldh_vit", group="transformer")


def build_ldh_vit_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "ldh_vit_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        LDHViT,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="ldh_vit",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_ldh_vit_fgvc_classifier, "ldh_vit_tiny")


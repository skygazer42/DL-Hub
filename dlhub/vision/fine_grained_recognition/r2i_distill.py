
"""Zooming without Zooming / Region-to-Image Distillation (toy-first) for FGVC.

Reference:
- "Zooming without Zooming: Region-to-Image Distillation for Fine-Grained Multimodal Perception"
  (arXiv 2026): https://arxiv.org/abs/2602.11858

Toy interpretation:
- Build a single-image classifier that also produces a "region" embedding by selecting top-k patch tokens.
- Expose a distillation loss term (region -> image) so training can mimic the paper's idea without
  requiring any external VLM or crop-based tool use.
"""

import torch
from torch import nn
import torch.nn.functional as F

from dlhub.vision.backbones._blocks import scale_channels

from ._common import TinyPatchEncoder, build_fgvc_model, check_nchw, make_fgvc_variants, smoke_test_classifier


class R2IDistillFGVC(nn.Module):
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
        self.num_regions = int(spec["parts"])

        self.encoder = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        self.token_scorer = nn.Linear(int(embed), 1)
        self.global_proj = nn.Linear(int(embed), int(embed))
        self.region_proj = nn.Linear(int(embed), int(embed))

        self.global_head = nn.Linear(int(embed), int(num_classes))
        self.region_head = nn.Linear(int(embed), int(num_classes))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.encoder(x)  # (B, 1+N, E)
        cls = tokens[:, 0]
        patch_tokens = tokens[:, 1:]

        scores = self.token_scorer(patch_tokens).squeeze(-1)  # (B, N)
        k = min(int(self.num_regions), int(patch_tokens.shape[1]))
        values, indices = torch.topk(scores, k=k, dim=1)
        gather_idx = indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        selected = torch.gather(patch_tokens, 1, gather_idx)  # (B, k, E)

        global_emb = F.normalize(torch.tanh(self.global_proj(cls)), dim=-1)
        region_emb = F.normalize(torch.tanh(self.region_proj(selected.mean(dim=1))), dim=-1)

        global_logits = self.global_head(self.dropout(global_emb))
        region_logits = self.region_head(self.dropout(region_emb))

        # Region-to-image distillation: treat region branch as teacher (detach) so gradients go to global branch.
        distill_loss = F.mse_loss(global_emb, region_emb.detach(), reduction="mean")

        # Primary logits: use global branch (single-image inference).
        logits = global_logits

        return {
            "logits": logits,
            "global_logits": global_logits,
            "region_logits": region_logits,
            "global_embedding": global_emb,
            "region_embedding": region_emb,
            "distill_loss": distill_loss,
            "selected_indices": indices,
            "selected_scores": values,
            "selected_tokens": selected,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("r2i_distill", group="transformer")


def build_r2i_distill_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "r2i_distill_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        R2IDistillFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="r2i_distill",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_r2i_distill_fgvc_classifier, "r2i_distill_tiny")


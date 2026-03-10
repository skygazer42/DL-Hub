"""IIR-VLM (toy-first) for FGVC / instance-level recognition.

Reference:
- "IIR-VLM: In-Context Instance-level Recognition for Large Vision-Language Models"
  (arXiv 2026): https://arxiv.org/abs/2601.14188

Toy interpretation (offline, no ILR expert checkpoints):
- use a tiny ViT patch encoder as the "general VLM" vision encoder
- add a lightweight CNN "expert" branch to mimic instance-level specialization
- fuse the two embeddings with a learned gate, then classify
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


class IIRVLMFGVC(nn.Module):
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

        self.vit = TinyPatchEncoder(
            in_channels=int(in_channels),
            image_size=int(image_size),
            patch_size=int(spec["patch"]),
            embed_dim=int(embed),
            depth=int(spec["depth"]),
            heads=int(spec["heads"]),
        )

        # Token selection (TransFG-like) for fine-grained regions.
        self.token_scorer = nn.Linear(int(embed), 1)
        self.global_proj = nn.Linear(int(embed), int(embed))

        # A tiny CNN "expert" branch.
        e1 = scale_channels(int(embed) // 2, float(width_mult), min_ch=16, divisor=8)
        e2 = scale_channels(int(embed), float(width_mult), min_ch=32, divisor=8)
        self.expert = nn.Sequential(
            nn.Conv2d(int(in_channels), int(e1), kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(e1), int(e2), kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.expert_proj = nn.Linear(int(e2), int(embed))

        # Gated fusion.
        self.fuse = nn.Sequential(
            nn.Linear(int(embed) * 2, int(embed)),
            nn.GELU(),
            nn.Linear(int(embed), 1),
        )

        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(embed), int(num_classes))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        tokens = self.vit(x)  # (B, 1+N, E)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]

        scores = self.token_scorer(patches).squeeze(-1)
        k = min(int(self.num_parts), int(patches.shape[1]))
        vals, idx = torch.topk(scores, k=k, dim=1)
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        selected = torch.gather(patches, 1, gather_idx)
        pooled = selected.mean(dim=1)

        global_emb = F.normalize(torch.tanh(self.global_proj(cls + pooled)), dim=-1)

        exp = self.expert(x).flatten(1)
        expert_emb = F.normalize(torch.tanh(self.expert_proj(exp)), dim=-1)

        gate = torch.sigmoid(self.fuse(torch.cat([global_emb, expert_emb], dim=-1)))  # (B, 1)
        fused = F.normalize(gate * expert_emb + (1.0 - gate) * global_emb, dim=-1)

        scale = self.logit_scale.exp().clamp(0.0, 100.0)
        logits = scale * self.classifier(self.dropout(fused))

        return {
            "logits": logits,
            "embedding": fused,
            "global_embedding": global_emb,
            "expert_embedding": expert_emb,
            "fuse_gate": gate,
            "selected_indices": idx,
            "selected_scores": vals,
            "logit_scale": scale,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("iir_vlm", group="transformer")


def build_iir_vlm_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "iir_vlm_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        IIRVLMFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="iir_vlm",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_iir_vlm_fgvc_classifier, "iir_vlm_tiny")

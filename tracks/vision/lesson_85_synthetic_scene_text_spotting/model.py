from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

import dlhub.vision.scene_text_spotting as text_spotting_zoo


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 1
    text_length: int = 4
    hidden_channels: int = 24
    vocab_size: int = 37
    family: str = "spotter_v1"
    variant: str = "spotter_v1_tiny"
    width_mult: float = 1.0


class SceneTextSpotter(nn.Module):
    """Tiny spotting wrapper: score-map detection plus short-sequence recognition."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        builder_name = f"build_{cfg.family}_text_spotter"
        if not hasattr(text_spotting_zoo, builder_name):
            raise ValueError(f"unknown scene text spotting family: {cfg.family}")
        builder = getattr(text_spotting_zoo, builder_name)
        self.backbone = builder(
            in_channels=int(cfg.in_channels),
            variant=str(cfg.variant),
            width_mult=float(cfg.width_mult),
            vocab_size=int(cfg.vocab_size),
        )
        self.seq_proj = nn.Sequential(
            nn.LazyConv2d(int(cfg.hidden_channels), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.LazyConv2d(int(cfg.hidden_channels), kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.seq_head = nn.LazyLinear(int(cfg.vocab_size))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        images = images.to(torch.float32)
        base_out = self.backbone(images)
        feat = self.backbone.enc(images)
        seq_feat = self.seq_proj(feat)
        pooled = torch.nn.functional.adaptive_avg_pool2d(
            seq_feat, output_size=(1, int(self.cfg.text_length))
        )
        seq_tokens = pooled.squeeze(2).transpose(1, 2)
        seq_logits = self.seq_head(seq_tokens)
        return {
            "score_map": torch.sigmoid(base_out["score_map"]),
            "seq_logits": seq_logits,
            "aux_char_logits": base_out["char_logits"],
        }


def scene_text_spotting_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    det_loss = torch.nn.functional.binary_cross_entropy(
        outputs["score_map"],
        targets["score_map"].to(torch.float32),
    )
    rec_loss = torch.nn.functional.cross_entropy(
        outputs["seq_logits"].reshape(-1, outputs["seq_logits"].shape[-1]),
        targets["text_tokens"].reshape(-1),
    )
    aux_loss = torch.nn.functional.cross_entropy(
        outputs["aux_char_logits"],
        targets["first_token"].reshape(-1),
    )
    total = det_loss + rec_loss + 0.2 * aux_loss
    return total, {
        "det_loss": float(det_loss.item()),
        "rec_loss": float(rec_loss.item()),
        "aux_loss": float(aux_loss.item()),
    }


def sequence_word_accuracy(seq_logits: torch.Tensor, text_tokens: torch.Tensor) -> float:
    with torch.no_grad():
        pred = seq_logits.argmax(dim=-1)
        return float((pred == text_tokens).all(dim=1).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "SceneTextSpotter",
    "scene_text_spotting_loss",
    "sequence_word_accuracy",
]

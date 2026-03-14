from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ViLTConfig:
    vocab_size: int
    image_size: int
    patch_size: int = 32
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.0


class ViLTModel(nn.Module):
    uses_convolution = False
    uses_region_supervision = False

    def __init__(self, config: ViLTConfig) -> None:
        super().__init__()
        self.config = config
        patch_dim = int(config.hidden_size)
        num_patches_per_side = int(config.image_size // config.patch_size)
        self.num_patches = num_patches_per_side * num_patches_per_side

        self.word_embeddings = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.patch_projection = nn.Conv2d(
            3,
            patch_dim,
            kernel_size=int(config.patch_size),
            stride=int(config.patch_size),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(config.hidden_size)))
        self.text_position_embeddings = nn.Embedding(512, int(config.hidden_size))
        self.image_position_embeddings = nn.Embedding(self.num_patches, int(config.hidden_size))
        self.modal_type_embeddings = nn.Embedding(2, int(config.hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(config.hidden_size),
            nhead=int(config.num_heads),
            dim_feedforward=int(config.intermediate_size),
            dropout=float(config.dropout),
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(config.num_layers),
        )
        self.word_patch_alignment_head = nn.Linear(int(config.hidden_size), self.num_patches)

    def _build_text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        text_embeddings = self.word_embeddings(input_ids.to(torch.long))
        text_embeddings = text_embeddings + self.text_position_embeddings(positions)
        text_embeddings = text_embeddings + self.modal_type_embeddings.weight[0].view(1, 1, -1)
        return text_embeddings

    def _build_image_embeddings(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_projection(image)
        patches = patches.flatten(2).transpose(1, 2)
        positions = torch.arange(self.num_patches, device=image.device).unsqueeze(0).expand(image.shape[0], -1)
        patches = patches + self.image_position_embeddings(positions)
        patches = patches + self.modal_type_embeddings.weight[1].view(1, 1, -1)
        return patches

    def forward(
        self,
        *,
        image: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        text_embeddings = self._build_text_embeddings(input_ids)
        image_embeddings = self._build_image_embeddings(image)
        cls_token = self.cls_token.expand(input_ids.shape[0], -1, -1)
        hidden = torch.cat([cls_token, text_embeddings, image_embeddings], dim=1)
        hidden = self.transformer(hidden)

        text_len = text_embeddings.shape[1]
        image_len = image_embeddings.shape[1]
        cls_embedding = hidden[:, 0, :]
        contextual_text = hidden[:, 1 : 1 + text_len, :]
        contextual_image = hidden[:, 1 + text_len : 1 + text_len + image_len, :]
        alignment_scores = self.word_patch_alignment_head(contextual_text)

        return {
            "cls_embedding": cls_embedding,
            "text_embeddings": contextual_text,
            "image_embeddings": contextual_image,
            "word_patch_alignment": alignment_scores,
        }


__all__ = ["ViLTConfig", "ViLTModel"]

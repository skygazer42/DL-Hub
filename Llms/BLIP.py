from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CapFiltPair:
    image_id: str
    caption: str
    score: float


class CapFiltPipeline:
    def __init__(self, score_threshold: float = 0.5) -> None:
        self.score_threshold = float(score_threshold)

    def filter_pairs(self, pairs: tuple[CapFiltPair, ...]) -> tuple[CapFiltPair, ...]:
        return tuple(pair for pair in pairs if float(pair.score) >= self.score_threshold)


class BLIPTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_heads: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(int(vocab_size), int(hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_size),
            nhead=int(num_heads),
            dim_feedforward=int(hidden_size) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(num_layers))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.token_embeddings(input_ids.to(torch.long)))


class BLIPMultimodalEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(int(hidden_size), int(num_heads), batch_first=True)
        self.norm = nn.LayerNorm(int(hidden_size))
        self.ff = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) * 4, int(hidden_size)),
        )

    def forward(self, text_tokens: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        attended, _ = self.cross_attn(text_tokens, image_tokens, image_tokens, need_weights=False)
        fused = self.norm(text_tokens + attended)
        return fused + self.ff(fused)


class BLIPTextDecoder(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.decoder = nn.GRU(int(hidden_size), int(hidden_size), batch_first=True)
        self.lm_head = nn.Linear(int(hidden_size), int(vocab_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        decoded, _ = self.decoder(hidden_states)
        return self.lm_head(decoded)


@dataclass(frozen=True)
class BLIPConfig:
    vocab_size: int
    max_seq_len: int
    image_feat_dim: int = 256
    hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 6
    dropout: float = 0.0


class BLIPModel(nn.Module):
    training_tasks = ("itc", "itm", "lm")

    def __init__(self, config: BLIPConfig) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = nn.Linear(int(config.image_feat_dim), int(config.hidden_size))
        self.text_encoder = BLIPTextEncoder(
            int(config.vocab_size),
            int(config.hidden_size),
            int(config.num_heads),
            int(config.num_layers),
            float(config.dropout),
        )
        self.multimodal_encoder = BLIPMultimodalEncoder(
            int(config.hidden_size),
            int(config.num_heads),
            float(config.dropout),
        )
        self.text_decoder = BLIPTextDecoder(int(config.hidden_size), int(config.vocab_size))
        self.itm_head = nn.Linear(int(config.hidden_size), 2)

    def forward(self, *, image_features: torch.Tensor, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        image_tokens = self.image_encoder(image_features)
        text_tokens = self.text_encoder(input_ids)
        multimodal_tokens = self.multimodal_encoder(text_tokens, image_tokens)
        contrastive = image_tokens.mean(dim=1)
        matching = self.itm_head(multimodal_tokens.mean(dim=1))
        logits = self.text_decoder(multimodal_tokens)
        return {
            "image_text_contrastive": contrastive,
            "image_text_matching": matching,
            "logits": logits,
        }


__all__ = [
    "BLIPConfig",
    "BLIPModel",
    "BLIPMultimodalEncoder",
    "BLIPTextDecoder",
    "BLIPTextEncoder",
    "CapFiltPair",
    "CapFiltPipeline",
]

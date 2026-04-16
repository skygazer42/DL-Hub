from dataclasses import dataclass

import torch
from torch import nn

from tracks.nlp.lesson_02_toy_text_classification_transformer.model import (
    TransformerEncoderBlock,
    _masked_mean_pool,
)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    prompt_length: int = 4
    embed_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.1
    num_classes: int = 2


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class PromptTunedTextClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        total_length = int(cfg.prompt_length) + int(cfg.max_length)
        if total_length <= 0:
            raise ValueError("prompt_length + max_length must be positive")

        self.token_embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.pos_embed = nn.Embedding(total_length, int(cfg.embed_dim))
        self.dropout = nn.Dropout(p=float(cfg.dropout))
        self.soft_prompt = nn.Parameter(torch.randn(int(cfg.prompt_length), int(cfg.embed_dim)) * 0.02)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=int(cfg.embed_dim),
                    num_heads=int(cfg.num_heads),
                    ff_dim=int(cfg.ff_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))
        self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        for module in (self.token_embed, self.pos_embed, self.blocks, self.ln):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")

        b, t = input_ids.shape
        if t != int(self.cfg.max_length):
            raise ValueError(
                f"Expected max_length={int(self.cfg.max_length)} tokens, got sequence length {int(t)}"
            )

        prompt_length = int(self.cfg.prompt_length)
        prompt_mask = torch.ones((b, prompt_length), dtype=attention_mask.dtype, device=input_ids.device)
        combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        token_embed = self.token_embed(input_ids)
        prompt_embed = self.soft_prompt.unsqueeze(0).expand(b, -1, -1)
        x = torch.cat([prompt_embed, token_embed], dim=1)

        pos = torch.arange(prompt_length + t, device=input_ids.device).unsqueeze(0).expand(b, -1)
        x = x + self.pos_embed(pos)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask=combined_mask)
        x = self.ln(x)

        pooled = _masked_mean_pool(x, combined_mask)
        logits = self.head(pooled)
        return logits

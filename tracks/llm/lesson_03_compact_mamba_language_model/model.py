from dataclasses import dataclass

import torch
from torch import nn


class SelectiveStateSpaceMixer(nn.Module):
    """A tiny recurrent selective scan that mixes tokens through a latent state."""

    def __init__(self, embed_dim: int, state_dim: int, expansion_factor: int, dropout: float) -> None:
        super().__init__()
        inner_dim = int(embed_dim) * int(expansion_factor)
        self.inner_dim = inner_dim
        self.state_dim = int(state_dim)

        self.in_proj = nn.Linear(int(embed_dim), inner_dim)
        self.gate_proj = nn.Linear(int(embed_dim), inner_dim)
        self.decay_proj = nn.Linear(inner_dim, self.state_dim)
        self.drive_proj = nn.Linear(inner_dim, self.state_dim)
        self.out_state = nn.Linear(self.state_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, int(embed_dim))
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        u = self.in_proj(x)
        gates = torch.sigmoid(self.gate_proj(x))

        state = x.new_zeros((b, self.state_dim))
        outputs: list[torch.Tensor] = []
        for idx in range(t):
            u_t = u[:, idx]
            gate_t = gates[:, idx]
            mask_t = attention_mask[:, idx : idx + 1]

            decay_t = torch.sigmoid(self.decay_proj(u_t))
            drive_t = torch.tanh(self.drive_proj(u_t))
            candidate_state = decay_t * state + (1.0 - decay_t) * drive_t
            state = mask_t * candidate_state + (1.0 - mask_t) * state

            y_t = self.out_state(state) * gate_t
            outputs.append(y_t * mask_t)

        y = torch.stack(outputs, dim=1)
        y = self.out_proj(self.dropout(y))
        return y * attention_mask.unsqueeze(-1)


class MambaBlock(nn.Module):
    def __init__(self, embed_dim: int, state_dim: int, expansion_factor: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(embed_dim))
        self.mixer = SelectiveStateSpaceMixer(
            embed_dim=int(embed_dim),
            state_dim=int(state_dim),
            expansion_factor=int(expansion_factor),
            dropout=float(dropout),
        )
        self.drop1 = nn.Dropout(p=float(dropout))

        ff_dim = int(embed_dim) * 2
        self.ff_norm = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), ff_dim),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(ff_dim, int(embed_dim)),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.mixer(self.norm(x), attention_mask=attention_mask))
        x = x + self.drop2(self.ff(self.ff_norm(x)))
        return x * attention_mask.unsqueeze(-1)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    embed_dim: int = 128
    state_dim: int = 64
    num_layers: int = 2
    expansion_factor: int = 2
    dropout: float = 0.1


class CompactMambaLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(
            int(cfg.vocab_size), int(cfg.embed_dim), padding_idx=int(cfg.pad_id)
        )
        self.dropout = nn.Dropout(p=float(cfg.dropout))
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    embed_dim=int(cfg.embed_dim),
                    state_dim=int(cfg.state_dim),
                    expansion_factor=int(cfg.expansion_factor),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.norm = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.vocab_size), bias=False)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs["attention_mask"].to(torch.float32)
        if input_ids.ndim != 2:
            raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")

        _, t = input_ids.shape
        if t != int(self.cfg.max_length):
            raise ValueError(
                f"Expected max_length={int(self.cfg.max_length)} tokens, got sequence length {int(t)}"
            )

        x = self.dropout(self.token_embed(input_ids))
        x = x * attention_mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = self.head(x)
        return logits * attention_mask.unsqueeze(-1)


__all__ = ["ModelConfig", "CompactMambaLM"]

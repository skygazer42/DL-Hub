from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        if int(config.rank) <= 0:
            raise ValueError("rank must be > 0")
        self.base = base
        self.config = config
        self.rank = int(config.rank)
        self.scaling = float(config.alpha) / float(config.rank)
        self.dropout = nn.Dropout(float(config.dropout))
        self.lora_a = nn.Parameter(torch.randn(self.rank, base.in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
        self.merged = False

        for param in self.base.parameters():
            param.requires_grad = False

    @classmethod
    def from_linear(cls, base: nn.Linear, config: LoRAConfig) -> "LoRALinear":
        return cls(base, config)

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_b @ self.lora_a) * self.scaling

    def merge(self) -> None:
        if not self.merged:
            with torch.no_grad():
                self.base.weight.add_(self.delta_weight().to(dtype=self.base.weight.dtype))
            self.merged = True

    def unmerge(self) -> None:
        if self.merged:
            with torch.no_grad():
                self.base.weight.sub_(self.delta_weight().to(dtype=self.base.weight.dtype))
            self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.merged:
            return out
        dropped = self.dropout(x)
        delta = torch.matmul(dropped, self.delta_weight().transpose(0, 1))
        return out + delta.to(dtype=out.dtype)


__all__ = ["LoRAConfig", "LoRALinear"]

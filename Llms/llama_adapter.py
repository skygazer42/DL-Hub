from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .llama import LLaMAModel


@dataclass(frozen=True)
class LLaMAAdapterConfig:
    prompt_length: int = 10
    adapter_layers: int = 30


class ZeroInitPromptGate(nn.Module):
    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(int(num_heads)))

    def forward(self) -> torch.Tensor:
        return self.gate


class LLaMAAdapterModel(nn.Module):
    def __init__(self, base_model: LLaMAModel, config: LLaMAAdapterConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config

        total_layers = len(self.base_model.layers)
        adapter_layers = min(int(config.adapter_layers), total_layers)
        self.target_layer_indices = list(range(total_layers - adapter_layers, total_layers))

        hidden_size = int(self.base_model.config.dim)
        num_heads = int(self.base_model.config.n_heads)
        prompt_length = int(config.prompt_length)

        for param in self.base_model.parameters():
            param.requires_grad = False

        self.prompts = nn.ParameterList(
            [
                nn.Parameter(torch.randn(prompt_length, hidden_size) * 0.02)
                for _ in self.target_layer_indices
            ]
        )
        self.gates = nn.ModuleList(
            [ZeroInitPromptGate(num_heads) for _ in self.target_layer_indices]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(input_ids.shape[0])
        prompts: dict[int, torch.Tensor] = {}
        gates: dict[int, torch.Tensor] = {}
        for idx, layer_idx in enumerate(self.target_layer_indices):
            prompts[layer_idx] = self.prompts[idx].unsqueeze(0).expand(batch_size, -1, -1)
            gates[layer_idx] = self.gates[idx]()
        hidden = self.base_model.forward_hidden(
            input_ids,
            attention_mask,
            prompts=prompts,
            prompt_gates=gates,
        )
        return self.base_model.lm_head(hidden)


__all__ = ["LLaMAAdapterConfig", "LLaMAAdapterModel", "ZeroInitPromptGate"]

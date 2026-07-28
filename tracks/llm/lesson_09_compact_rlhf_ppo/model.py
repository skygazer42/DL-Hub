import torch
from torch import nn

from tracks.llm.lesson_01_compact_causal_lm_transformer.model import CausalTransformerLM as CompactPolicyLM
from tracks.llm.lesson_01_compact_causal_lm_transformer.model import ModelConfig


class CompactTokenRewardModel(nn.Module):
    def __init__(self, *, pad_id: int, good_token_id: int, bad_token_id: int) -> None:
        super().__init__()
        self.pad_id = int(pad_id)
        self.good_token_id = int(good_token_id)
        self.bad_token_id = int(bad_token_id)

    def forward(self, input_ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        response_mask_bool = response_mask > 0
        response_ids = input_ids.masked_fill(~response_mask_bool, self.pad_id)
        good_hits = (response_ids == self.good_token_id).to(torch.float32).sum(dim=1)
        bad_hits = (response_ids == self.bad_token_id).to(torch.float32).sum(dim=1)
        lengths = response_mask_bool.to(torch.float32).sum(dim=1).clamp_min(1.0)
        return (good_hits - bad_hits) / lengths


__all__ = ["ModelConfig", "CompactPolicyLM", "CompactTokenRewardModel"]

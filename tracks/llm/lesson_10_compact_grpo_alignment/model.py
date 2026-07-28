import torch
from torch import nn

from tracks.llm.lesson_01_compact_causal_lm_transformer.model import CausalTransformerLM
from tracks.llm.lesson_01_compact_causal_lm_transformer.model import ModelConfig


class CompactGrpoPolicyLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = CausalTransformerLM(cfg)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs["attention_mask"].to(torch.float32)

        if input_ids.ndim == 2:
            return self.backbone({"input_ids": input_ids, "attention_mask": attention_mask})
        if input_ids.ndim != 3:
            raise ValueError(
                f"Expected input_ids shape (B, T) or (B, G, T), got {tuple(input_ids.shape)}"
            )

        bsz, group_size, seq_len = input_ids.shape
        flat_ids = input_ids.view(bsz * group_size, seq_len)
        flat_mask = attention_mask.view(bsz * group_size, seq_len)
        flat_logits = self.backbone({"input_ids": flat_ids, "attention_mask": flat_mask})
        return flat_logits.view(bsz, group_size, seq_len, -1)


__all__ = ["ModelConfig", "CompactGrpoPolicyLM"]

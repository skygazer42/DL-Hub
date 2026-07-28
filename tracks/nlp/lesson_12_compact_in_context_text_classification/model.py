from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    pad_id: int


class InContextTextClassifier(nn.Module):
    """A non-parametric classifier that predicts from support examples only."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.vocab_size = int(config.vocab_size)
        self.pad_id = int(config.pad_id)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        support_ids = batch["support_input_ids"]
        support_labels = batch["support_labels"]
        query_ids = batch["query_input_ids"]
        query_labels = batch["query_labels"]

        batch_size, _, _ = support_ids.shape
        num_classes = int(support_labels.max().item()) + 1
        logits = torch.zeros((batch_size, num_classes), dtype=torch.float32, device=query_ids.device)

        for row in range(batch_size):
            query_tokens = query_ids[row]
            query_mask = query_tokens != self.pad_id
            for class_id in range(num_classes):
                class_mask = support_labels[row] == class_id
                class_support_tokens = support_ids[row][class_mask]
                if class_support_tokens.numel() == 0:
                    continue
                support_non_pad = class_support_tokens[class_support_tokens != self.pad_id]
                if support_non_pad.numel() == 0:
                    continue
                overlap = (query_tokens.unsqueeze(1) == support_non_pad.unsqueeze(0)) & query_mask.unsqueeze(1)
                logits[row, class_id] = overlap.any(dim=1).to(torch.float32).sum()

        predictions = logits.argmax(dim=-1)
        return {"logits": logits, "predictions": predictions, "labels": query_labels}


def classification_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return float((predictions == labels).to(torch.float32).mean().item())


__all__ = ["InContextTextClassifier", "ModelConfig", "classification_accuracy"]

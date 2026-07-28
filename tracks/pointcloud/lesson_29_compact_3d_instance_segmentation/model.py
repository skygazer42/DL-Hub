from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int = 64
    embedding_dim: int = 16
    num_instances: int = 2
    dropout: float = 0.0


class CompactInstanceSegmentation3DNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_features)
        embedding_dim = int(cfg.embedding_dim)
        num_instances = int(cfg.num_instances)
        dropout = float(cfg.dropout)

        if hidden < 4:
            raise ValueError("hidden_features must be >= 4")
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be >= 2")
        if num_instances < 2:
            raise ValueError("num_instances must be >= 2")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")

        self.encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, embedding_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(embedding_dim, num_instances)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        if points.ndim != 3 or points.size(-1) != 3:
            raise ValueError("expected point clouds shaped [batch, num_points, 3]")
        batch_size, num_points, channels = points.shape
        flat = points.reshape(batch_size * num_points, channels)
        embeddings = self.encoder(flat).reshape(batch_size, num_points, -1)
        logits = self.classifier(embeddings.reshape(batch_size * num_points, -1)).reshape(
            batch_size, num_points, -1
        )
        return {"embeddings": embeddings, "logits": logits}


def instance_segmentation_loss(
    logits: torch.Tensor, target_instance_ids: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, num_points, num_instances]")
    if target_instance_ids.ndim != 2:
        raise ValueError("target_instance_ids must have shape [batch, num_points]")
    if logits.shape[:2] != target_instance_ids.shape:
        raise ValueError("logits and target_instance_ids batch/point dimensions must match")

    ce_loss = F.cross_entropy(logits.transpose(1, 2), target_instance_ids)
    pred_ids = logits.argmax(dim=-1)
    point_acc = (pred_ids == target_instance_ids).to(torch.float32).mean()
    return ce_loss, {
        "ce_loss": float(ce_loss.detach().item()),
        "point_acc": float(point_acc.detach().item()),
    }


__all__ = ["ModelConfig", "CompactInstanceSegmentation3DNet", "instance_segmentation_loss"]

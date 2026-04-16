from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int = 64


class ToySceneFlowEstimator(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_features)
        if hidden < 4:
            raise ValueError("hidden_features must be >= 4")
        self.regressor = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if source.shape != target.shape:
            raise ValueError("source and target must have the same shape")
        if source.ndim != 3 or source.size(-1) != 3:
            raise ValueError("expected point clouds shaped [batch, num_points, 3]")

        residual = target - source
        batch_size, num_points, channels = residual.shape
        pred = self.regressor(residual.reshape(batch_size * num_points, channels))
        return pred.reshape(batch_size, num_points, channels)


def scene_flow_loss(
    pred_flow: torch.Tensor, target_flow: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred_flow.shape != target_flow.shape:
        raise ValueError("pred_flow and target_flow must have the same shape")

    diff = pred_flow - target_flow
    flow_loss = diff.square().mean()
    endpoint_error = diff.norm(dim=-1).mean()
    total = flow_loss + 0.1 * endpoint_error
    return total, {
        "flow_loss": float(flow_loss.detach().item()),
        "endpoint_error": float(endpoint_error.detach().item()),
    }


__all__ = ["ModelConfig", "ToySceneFlowEstimator", "scene_flow_loss"]

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int = 64


class ToyObjectTracker(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_features)
        if hidden < 8:
            raise ValueError("hidden_features must be >= 8")

        self.point_encoder = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 6),
        )

    def forward(self, prev_cloud: torch.Tensor, curr_cloud: torch.Tensor) -> torch.Tensor:
        if prev_cloud.shape != curr_cloud.shape:
            raise ValueError("prev_cloud and curr_cloud must have the same shape")
        if prev_cloud.ndim != 3 or prev_cloud.size(-1) != 3:
            raise ValueError("expected point clouds shaped [batch, num_points, 3]")

        batch_size, num_points, channels = prev_cloud.shape
        prev_feat = self.point_encoder(prev_cloud.reshape(batch_size * num_points, channels)).reshape(
            batch_size, num_points, -1
        )
        curr_feat = self.point_encoder(curr_cloud.reshape(batch_size * num_points, channels)).reshape(
            batch_size, num_points, -1
        )

        prev_global = prev_feat.mean(dim=1)
        curr_global = curr_feat.mean(dim=1)
        joint = torch.cat([prev_global, curr_global, curr_global - prev_global], dim=-1)
        return self.head(joint)


def tracking_loss(
    pred_state: torch.Tensor, target_state: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if pred_state.shape != target_state.shape:
        raise ValueError("pred_state and target_state must have the same shape")
    if pred_state.ndim != 2 or pred_state.size(-1) != 6:
        raise ValueError("expected states shaped [batch, 6]")

    diff = pred_state - target_state
    state_mse = diff.square().mean()
    center_mae = (pred_state[:, :3] - target_state[:, :3]).abs().mean()
    velocity_mae = (pred_state[:, 3:] - target_state[:, 3:]).abs().mean()
    total = state_mse + 0.5 * (center_mae + velocity_mae)
    return total, {
        "state_mse": float(state_mse.detach().item()),
        "center_mae": float(center_mae.detach().item()),
        "velocity_mae": float(velocity_mae.detach().item()),
    }


__all__ = ["ModelConfig", "ToyObjectTracker", "tracking_loss"]

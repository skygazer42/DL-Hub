import torch
import torch.nn as nn


class LinearRegressor(nn.Module):
    def __init__(self, in_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

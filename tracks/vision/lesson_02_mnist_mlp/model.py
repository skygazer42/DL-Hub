from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTMLP(nn.Module):
    def __init__(self, hidden_size: int = 300) -> None:
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

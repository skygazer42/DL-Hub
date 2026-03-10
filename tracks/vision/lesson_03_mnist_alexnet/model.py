from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 224
    hidden_dim: int = 256
    dropout: float = 0.5


class AlexNetLite(nn.Module):
    """AlexNet-style CNN scaled down for teaching.

    - Keeps the "AlexNet feel" (large first kernel + aggressive pooling + dropout + FC).
    - Uses smaller channel sizes to make CPU runs practical.
    - Computes the FC input dimension automatically from `input_size`.
    """

    def __init__(self, config: ModelConfig, num_classes: int = 10) -> None:
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=float(config.dropout))

        feature_dim = self._infer_feature_dim(input_size=int(config.input_size))
        self.fc1 = nn.Linear(feature_dim, int(config.hidden_dim))
        self.fc2 = nn.Linear(int(config.hidden_dim), int(config.hidden_dim))
        self.fc3 = nn.Linear(int(config.hidden_dim), num_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=3, stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=3, stride=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size=3, stride=2)
        return x

    def _infer_feature_dim(self, input_size: int) -> int:
        with torch.no_grad():
            x = torch.zeros(1, 1, input_size, input_size)
            feats = self._forward_features(x)
            return int(feats.numel())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    hidden_features: int = 64
    num_classes: int = 4
    dropout: float = 0.1


class ToyPointNetSemanticSeg3D(nn.Module):
    """Minimal PointNet-style semantic segmentation network."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        c_in = int(cfg.in_channels)
        hidden = int(cfg.hidden_features)
        c_out = int(cfg.num_classes)
        if hidden < 8:
            raise ValueError("hidden_features must be >= 8")
        if c_out < 2:
            raise ValueError("num_classes must be >= 2")

        self.backbone = nn.Sequential(
            nn.Conv1d(c_in, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden * 4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden * 8, hidden * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Conv1d(hidden * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, c_out, kernel_size=1, bias=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.size(-1) != int(self.cfg.in_channels):
            raise ValueError(
                f"expected points with shape [B, N, {self.cfg.in_channels}], got {tuple(points.shape)}"
            )

        x = points.to(torch.float32).transpose(1, 2)  # (B, C, N)
        local = self.backbone(x)  # (B, F, N)
        global_feat = torch.max(local, dim=2, keepdim=True).values.expand_as(local)
        fused = torch.cat([local, global_feat], dim=1)
        logits = self.head(fused)  # (B, K, N)
        return logits.transpose(1, 2).contiguous()  # (B, N, K)


def segmentation3d_loss(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [B, N, K]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [B, N]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels shape mismatch on [B, N]")

    bsz, num_points, num_classes = logits.shape
    ce = F.cross_entropy(logits.reshape(bsz * num_points, num_classes), labels.reshape(bsz * num_points))
    preds = logits.argmax(dim=-1)
    acc = (preds == labels).to(torch.float32).mean()
    return ce, {"loss_ce": float(ce.detach().item()), "acc": float(acc.detach().item())}


__all__ = ["ModelConfig", "ToyPointNetSemanticSeg3D", "segmentation3d_loss"]

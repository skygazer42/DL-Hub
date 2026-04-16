from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    hidden_features: int = 64
    num_classes: int = 2


class ToyDetector3D(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_features)
        num_classes = int(cfg.num_classes)
        if hidden < 8:
            raise ValueError("hidden_features must be >= 8")
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.backbone = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.box_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 7))
        self.cls_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("expected points with shape [batch, num_points, 3]")
        feat = self.backbone(points)
        pooled = feat.max(dim=1).values
        return {"boxes": self.box_head(pooled), "class_logits": self.cls_head(pooled)}


def detection3d_loss(
    outputs: dict[str, torch.Tensor],
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred_boxes = outputs["boxes"]
    pred_logits = outputs["class_logits"]

    if pred_boxes.shape != target_boxes.shape:
        raise ValueError("predicted boxes and target boxes must have the same shape")
    if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 7:
        raise ValueError("expected boxes with shape [batch, 7]")
    if pred_logits.ndim != 2 or target_labels.ndim != 1:
        raise ValueError("expected class logits [batch, classes] and labels [batch]")

    center_l1 = F.l1_loss(pred_boxes[:, :3], target_boxes[:, :3])
    size_pred = F.softplus(pred_boxes[:, 3:6]) + 1e-3
    size_l1 = F.l1_loss(size_pred, target_boxes[:, 3:6])
    box_l1 = center_l1 + size_l1

    yaw_residual = pred_boxes[:, 6] - target_boxes[:, 6]
    box_yaw = torch.abs(torch.sin(yaw_residual)).mean()
    cls_ce = F.cross_entropy(pred_logits, target_labels)
    total = box_l1 + 0.25 * box_yaw + 0.5 * cls_ce
    return total, {
        "box_l1": float(box_l1.detach().item()),
        "box_yaw": float(box_yaw.detach().item()),
        "cls_ce": float(cls_ce.detach().item()),
    }


__all__ = ["ModelConfig", "ToyDetector3D", "detection3d_loss"]

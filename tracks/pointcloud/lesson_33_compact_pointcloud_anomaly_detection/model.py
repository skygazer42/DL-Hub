from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "recon_anomaly3d:recon_anomaly3d_small"
    variant: str = ""
    width_mult: float = 1.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.pointcloud_anomaly_detection.recon_anomaly3d import (
        _VARIANTS as recon_variants,
    )

    return [f"recon_anomaly3d:{name}" for name in sorted(recon_variants)] + ["recon_anomaly3d"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        prefix, name = arch_raw.split(":", 1)
        arch = prefix.strip().lower()
        variant = name.strip()

    if arch in {"recon_anomaly3d", "anomaly"}:
        from dlhub.pointcloud.pointcloud_anomaly_detection.recon_anomaly3d import (
            build_recon_anomaly3d_anomaly_detector,
        )

        return build_recon_anomaly3d_anomaly_detector(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "recon_anomaly3d_small",
            width_mult=float(cfg.width_mult),
        )

    raise ValueError(
        f"Unknown arch: {arch_raw!r}. Supported: recon_anomaly3d:<variant>"
    )


def _score_to_probability(score: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-torch.relu(score.to(torch.float32)))


def anomaly_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = torch.nn.functional.l1_loss(
        outputs["reconstruction"].to(torch.float32),
        targets["reconstruction"].to(torch.float32),
    )
    point_prob = _score_to_probability(outputs["point_scores"])
    global_prob = _score_to_probability(outputs["global_score"])
    point_bce = torch.nn.functional.binary_cross_entropy(
        point_prob.clamp(1e-4, 1.0 - 1e-4),
        targets["point_labels"].to(torch.float32),
    )
    global_bce = torch.nn.functional.binary_cross_entropy(
        global_prob.clamp(1e-4, 1.0 - 1e-4),
        targets["label"].to(torch.float32),
    )
    total = reconstruction_loss + 0.5 * point_bce + 0.5 * global_bce
    return total, {
        "reconstruction_loss": float(reconstruction_loss.detach().item()),
        "point_bce": float(point_bce.detach().item()),
        "global_bce": float(global_bce.detach().item()),
    }


def anomaly_accuracy(global_score: torch.Tensor, labels: torch.Tensor) -> float:
    pred = _score_to_probability(global_score) >= 0.5
    target = labels.to(torch.float32) >= 0.5
    return float((pred == target).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "anomaly_accuracy",
    "anomaly_loss",
    "build_model",
    "list_supported_arches",
]

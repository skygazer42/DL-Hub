from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.anomaly_detection.patchcore import (
    build_patchcore_anomaly_detector,
)


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "patchcore:patchcore_small"
    variant: str = ""
    width_mult: float = 1.0


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()
    if ":" in arch_raw:
        prefix, name = arch_raw.split(":", 1)
        arch = prefix.strip().lower()
        variant = name.strip()

    if arch != "patchcore":
        raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: patchcore:<variant>")

    selected_variant = str(variant) if variant else "patchcore_small"
    return build_patchcore_anomaly_detector(
        in_channels=int(cfg.in_channels),
        variant=selected_variant,
        width_mult=float(cfg.width_mult),
    )


def anomaly_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstruction_loss = torch.nn.functional.mse_loss(
        outputs["reconstruction"],
        targets["reconstruction"].to(torch.float32),
    )
    anomaly_map_l1 = torch.nn.functional.l1_loss(
        outputs["anomaly_map"],
        targets["anomaly_map"].to(torch.float32),
    )
    score_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["score"].to(torch.float32),
        targets["label"].to(torch.float32),
    )
    total = reconstruction_loss + 0.5 * anomaly_map_l1 + score_bce
    return total, {
        "reconstruction_loss": float(reconstruction_loss.item()),
        "anomaly_map_l1": float(anomaly_map_l1.item()),
        "score_bce": float(score_bce.item()),
    }


def anomaly_accuracy(score_logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = (torch.sigmoid(score_logits.to(torch.float32)) >= 0.5).to(torch.float32)
    target = labels.to(torch.float32)
    return float((pred == target).to(torch.float32).mean().item())


__all__ = ["ModelConfig", "anomaly_accuracy", "anomaly_loss", "build_model"]


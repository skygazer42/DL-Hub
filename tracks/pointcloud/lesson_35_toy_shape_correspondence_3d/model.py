from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "fmnet_corr3d:fmnet_corr3d_small"
    variant: str = ""
    width_mult: float = 1.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.shape_correspondence_3d.fmnet_corr3d import _VARIANTS as fmnet_variants

    return [f"fmnet_corr3d:{name}" for name in sorted(fmnet_variants)] + ["fmnet_corr3d"]


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()

    if ":" in arch_raw:
        prefix, name = arch_raw.split(":", 1)
        arch = prefix.strip().lower()
        variant = name.strip()

    if arch in {"fmnet_corr3d", "fmnet"}:
        from dlhub.pointcloud.shape_correspondence_3d.fmnet_corr3d import (
            build_fmnet_corr3d_shape_correspondence_model,
        )

        return build_fmnet_corr3d_shape_correspondence_model(
            in_channels=int(cfg.in_channels),
            variant=str(variant) if variant else "fmnet_corr3d_small",
            width_mult=float(cfg.width_mult),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: fmnet_corr3d:<variant>")


def correspondence_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    scores = outputs["scores"]
    if scores.ndim != 3:
        raise ValueError("scores must have shape [batch, source_points, target_points]")
    if targets.ndim != 2:
        raise ValueError("targets must have shape [batch, source_points]")
    if scores.shape[:2] != targets.shape:
        raise ValueError("scores and targets shape mismatch on batch/source axes")

    batch_size, src_points, tgt_points = scores.shape
    flat_scores = scores.reshape(batch_size * src_points, tgt_points).to(torch.float32)
    flat_targets = targets.reshape(batch_size * src_points).to(torch.long)
    cross_entropy = torch.nn.functional.cross_entropy(flat_scores, flat_targets)
    return cross_entropy, {"cross_entropy": float(cross_entropy.detach().item())}


def correspondence_accuracy(matches: torch.Tensor, targets: torch.Tensor) -> float:
    if matches.shape != targets.shape:
        raise ValueError("matches and targets must have the same shape")
    return float((matches.to(torch.long) == targets.to(torch.long)).to(torch.float32).mean().item())


__all__ = [
    "ModelConfig",
    "build_model",
    "correspondence_accuracy",
    "correspondence_loss",
    "list_supported_arches",
]

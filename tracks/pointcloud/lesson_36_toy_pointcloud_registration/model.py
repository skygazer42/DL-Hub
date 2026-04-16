from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int = 3
    arch: str = "pointnetlk:pointnetlk_small"
    variant: str = ""
    width_mult: float = 1.0


def list_supported_arches() -> list[str]:
    from dlhub.pointcloud.registration.dcp import _VARIANTS as dcp_variants
    from dlhub.pointcloud.registration.pointnetlk import _VARIANTS as pointnetlk_variants
    from dlhub.pointcloud.registration.regtr import _VARIANTS as regtr_variants
    from dlhub.pointcloud.registration.rpmnet import _VARIANTS as rpmnet_variants

    return (
        [f"dcp:{name}" for name in sorted(dcp_variants)]
        + [f"pointnetlk:{name}" for name in sorted(pointnetlk_variants)]
        + [f"regtr:{name}" for name in sorted(regtr_variants)]
        + [f"rpmnet:{name}" for name in sorted(rpmnet_variants)]
        + ["dcp", "pointnetlk", "regtr", "rpmnet"]
    )


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    variant = str(cfg.variant).strip()

    if ":" in arch_raw:
        prefix, name = arch_raw.split(":", 1)
        arch = prefix.strip().lower()
        variant = name.strip()

    if arch == "pointnetlk":
        from dlhub.pointcloud.registration.pointnetlk import build_pointnetlk_registrar

        return build_pointnetlk_registrar(
            variant=str(variant) if variant else "pointnetlk_small",
            width_mult=float(cfg.width_mult),
        )
    if arch == "dcp":
        from dlhub.pointcloud.registration.dcp import build_dcp_registrar

        return build_dcp_registrar(
            variant=str(variant) if variant else "dcp_small",
            width_mult=float(cfg.width_mult),
        )
    if arch == "regtr":
        from dlhub.pointcloud.registration.regtr import build_regtr_registrar

        return build_regtr_registrar(
            variant=str(variant) if variant else "regtr_small",
            width_mult=float(cfg.width_mult),
        )
    if arch == "rpmnet":
        from dlhub.pointcloud.registration.rpmnet import build_rpmnet_registrar

        return build_rpmnet_registrar(
            variant=str(variant) if variant else "rpmnet_small",
            width_mult=float(cfg.width_mult),
        )

    raise ValueError(f"Unknown arch: {arch_raw!r}. Supported: {', '.join(list_supported_arches())}")


def registration_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    pose = outputs["pose6d"].to(torch.float32)
    expected = targets.to(torch.float32)
    if pose.ndim != 2 or pose.shape[1] != 6:
        raise ValueError("pose6d must have shape [batch, 6]")
    if expected.shape != pose.shape:
        raise ValueError("targets must match pose6d shape [batch, 6]")

    translation_mse = torch.nn.functional.mse_loss(pose[:, :3], expected[:, :3])
    rotation_mse = torch.nn.functional.mse_loss(pose[:, 3:], expected[:, 3:])
    total = translation_mse + rotation_mse
    return total, {
        "translation_mse": float(translation_mse.detach().item()),
        "rotation_mse": float(rotation_mse.detach().item()),
    }


def pose_l1_error(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> float:
    if pred_pose.shape != target_pose.shape:
        raise ValueError("pred_pose and target_pose must have the same shape")
    return float((pred_pose.to(torch.float32) - target_pose.to(torch.float32)).abs().mean().item())


__all__ = [
    "ModelConfig",
    "build_model",
    "list_supported_arches",
    "pose_l1_error",
    "registration_loss",
]


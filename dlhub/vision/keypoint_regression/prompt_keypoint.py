from __future__ import annotations

from torch import nn

from ._common import build_baseline_keypoint_model, smoke_test_keypoint_model


_VARIANTS: dict[str, dict[str, int]] = {
    "prompt_keypoint_tiny": {"width": 24, "depth": 1, "num_points": 8},
    "prompt_keypoint_small": {"width": 36, "depth": 2, "num_points": 8},
    "prompt_keypoint_base": {"width": 48, "depth": 3, "num_points": 8},
}


def build_prompt_keypoint_keypoint_model(
    *,
    in_channels: int,
    variant: str = "prompt_keypoint_small",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_baseline_keypoint_model(
        family="prompt_keypoint",
        mode="prompt",
        variants=_VARIANTS,
        in_channels=int(in_channels),
        variant=str(variant),
        width_mult=float(width_mult),
    )


if __name__ == "__main__":
    smoke_test_keypoint_model(build_prompt_keypoint_keypoint_model, "prompt_keypoint_tiny")

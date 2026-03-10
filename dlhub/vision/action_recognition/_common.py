import torch


def check_video_input(x: torch.Tensor) -> torch.Tensor:
    """Ensure NCTHW float32 input for video models."""

    x = x.to(torch.float32)
    if x.ndim != 5:
        raise ValueError(f"Expected video input shape (B, C, T, H, W), got {tuple(x.shape)}")
    return x


def check_skeleton_input(x: torch.Tensor) -> torch.Tensor:
    """Ensure NCTV float32 input for skeleton models.

    Convention:
    - C: coordinate channels (2 for xy, 3 for xyz)
    - T: sequence length
    - V: number of joints
    """

    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected skeleton input shape (B, C, T, V), got {tuple(x.shape)}")
    return x

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class TinyCameraCalibrationBlock(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.GroupNorm(1, int(channels))
        self.conv1 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), 3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), 1)
        self.depthwise = nn.Conv2d(int(channels), int(channels), 5, padding=2, groups=int(channels))
        self.prompt = (
            nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        if self.prompt is not None:
            h = h + self.prompt
        local = self.conv2(F.relu(self.conv1(h), inplace=True))
        if self.mode in {"pinhole", "fisheye", "checkerboard", "line", "selfsup"}:
            local = local + self.depthwise(h)
        elif self.mode == "transformer":
            local = local * torch.sigmoid(self.mix(h)) + self.depthwise(h)
        elif self.mode == "coarse":
            local = local + self.mix(F.avg_pool2d(h, 3, 1, 1))
        elif self.mode == "uncertainty":
            local = local + torch.tanh(self.mix(h))
        elif self.mode == "mamba":
            local = local + torch.tanh(self.depthwise(torch.roll(h, shifts=1, dims=-1)))
        return x + 0.2 * local


class TinyCameraCalibrator(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.stem = nn.Conv2d(int(in_channels), int(width), 3, padding=1)
        self.blocks = nn.ModuleList(
            [
                TinyCameraCalibrationBlock(channels=int(width), mode=str(mode))
                for _ in range(max(1, int(depth)))
            ]
        )
        self.intrinsics_head = nn.Linear(int(width), 4)
        self.distortion_head = nn.Linear(int(width), 5)
        self.confidence_head = nn.Linear(int(width), 1)
        self.error_head = nn.Linear(int(width), 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        feat = F.relu(self.stem(image), inplace=True)
        for block in self.blocks:
            feat = block(feat)
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)

        params = self.intrinsics_head(pooled)
        b = image.shape[0]
        h = float(image.shape[-2])
        w = float(image.shape[-1])

        fx = (0.5 + torch.sigmoid(params[:, 0])) * w
        fy = (0.5 + torch.sigmoid(params[:, 1])) * h
        cx = torch.sigmoid(params[:, 2]) * w
        cy = torch.sigmoid(params[:, 3]) * h

        intrinsics = (
            torch.eye(3, device=image.device, dtype=image.dtype).unsqueeze(0).repeat(b, 1, 1)
        )
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        distortion = 0.1 * torch.tanh(self.distortion_head(pooled))
        confidence = torch.sigmoid(self.confidence_head(pooled))
        reprojection_error = F.softplus(self.error_head(pooled))

        return {
            "intrinsics": intrinsics,
            "distortion": distortion,
            "confidence": confidence,
            "reprojection_error": reprojection_error,
        }


def build_baseline_camera_calibrator(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(
            f"Unknown variant for {family}: {variant!r}. Available: {sorted(variants)}"
        )
    spec = dict(variants[name])
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    depth = int(spec["depth"])
    return TinyCameraCalibrator(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=depth,
    )


def smoke_test_camera_calibrator(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["intrinsics"].shape), tuple(out["distortion"].shape))
    print("ok")

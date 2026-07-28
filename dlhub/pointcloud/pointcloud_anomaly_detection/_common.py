from __future__ import annotations

import torch
from torch import nn


def check_points(points: torch.Tensor, in_channels: int) -> torch.Tensor:
    points = points.to(torch.float32)
    if points.ndim != 3:
        raise ValueError(f"Expected input shape (B, N, C), got {tuple(points.shape)}")
    if points.shape[-1] != int(in_channels):
        raise ValueError(f"Expected {int(in_channels)} channels, got {int(points.shape[-1])}")
    return points


class TinyAnomalyBlock(nn.Module):
    def __init__(self, *, width: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.norm = nn.LayerNorm(int(width))
        self.mlp = nn.Sequential(
            nn.Linear(int(width), int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )
        self.mix = nn.Linear(int(width), int(width))
        self.prompt = nn.Parameter(torch.zeros(1, 1, int(width))) if self.mode == "prompt" else None

    def forward(self, feat: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        h = self.norm(feat)
        if self.prompt is not None:
            h = h + self.prompt
        update = self.mlp(h)
        if self.mode == "recon":
            update = update + 0.25 * context
        elif self.mode == "patchcore":
            update = update + self.mix(context)
        elif self.mode == "student_teacher":
            update = update + torch.tanh(self.mix(h - context))
        elif self.mode == "memory":
            update = 0.5 * update + 0.5 * self.mix(context)
        elif self.mode == "density":
            update = update + self.mix(h.mean(dim=1, keepdim=True))
        elif self.mode == "transformer":
            update = update * torch.sigmoid(self.mix(context)) + self.mix(h)
        elif self.mode == "diffusion":
            update = 0.7 * update + 0.3 * torch.tanh(self.mix(context - h))
        elif self.mode == "prompt":
            update = update + self.mix(context)
        elif self.mode == "openvocab":
            update = update + torch.sign(self.mix(context)) * 0.1
        elif self.mode == "mamba":
            update = update + torch.tanh(torch.roll(self.mix(h), shifts=1, dims=1))
        return feat + 0.2 * update


class TinyAnomalyDetector(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.in_channels = int(in_channels)
        self.input_proj = nn.Linear(int(in_channels), int(width))
        self.blocks = nn.ModuleList(
            [TinyAnomalyBlock(width=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.reconstruction_head = nn.Linear(int(width), int(in_channels))
        self.score_head = nn.Linear(int(width), 1)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        pts = check_points(points, self.in_channels)
        feat = self.input_proj(pts)
        center = feat.mean(dim=1, keepdim=True)
        context = center.expand_as(feat)
        for block in self.blocks:
            feat = block(feat, context)
        reconstruction = self.reconstruction_head(feat)
        residual = torch.abs(reconstruction - pts)
        point_scores = residual.mean(dim=-1) + 0.25 * torch.sigmoid(self.score_head(feat)).squeeze(
            -1
        )
        if self.mode == "density":
            point_scores = point_scores + 0.1 * torch.norm(
                pts - pts.mean(dim=1, keepdim=True), dim=-1
            )
        elif self.mode == "openvocab":
            point_scores = point_scores + 0.05 * torch.relu(pts[..., 0])
        global_score = point_scores.mean(dim=1)
        return {
            "reconstruction": reconstruction,
            "point_scores": point_scores,
            "global_score": global_score,
        }


def build_baseline_anomaly_detector(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    return TinyAnomalyDetector(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
    )


def smoke_test_anomaly_detector(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 48, 3))
    print(variant, tuple(out["point_scores"].shape), tuple(out["global_score"].shape))

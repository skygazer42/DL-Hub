from __future__ import annotations

import torch
from torch import nn


def check_sequence(points: torch.Tensor, in_channels: int) -> torch.Tensor:
    points = points.to(torch.float32)
    if points.ndim != 4:
        raise ValueError(f"Expected input shape (B, T, N, C), got {tuple(points.shape)}")
    if points.shape[-1] != int(in_channels):
        raise ValueError(f"Expected {int(in_channels)} channels, got {int(points.shape[-1])}")
    return points


class TinyForecastBlock(nn.Module):
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
        self.prompt = (
            nn.Parameter(torch.zeros(1, 1, int(width))) if self.mode == "prompt" else None
        )

    def forward(self, feat: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
        h = self.norm(feat)
        if self.prompt is not None:
            h = h + self.prompt
        update = self.mlp(h)
        if self.mode == "pointlstm":
            update = update + 0.25 * temporal
        elif self.mode == "traj":
            update = update + self.mix(temporal)
        elif self.mode == "motion":
            update = update + torch.tanh(self.mix(h + temporal))
        elif self.mode == "memory":
            update = 0.5 * update + 0.5 * self.mix(temporal)
        elif self.mode == "graph":
            update = update + self.mix(h.mean(dim=1, keepdim=True))
        elif self.mode == "transformer":
            update = update * torch.sigmoid(self.mix(temporal)) + self.mix(h)
        elif self.mode == "diffusion":
            update = 0.7 * update + 0.3 * torch.tanh(self.mix(temporal - h))
        elif self.mode == "prompt":
            update = update + self.mix(temporal)
        elif self.mode == "occupancy":
            update = update * torch.sigmoid(self.mix(h))
        elif self.mode == "mamba":
            update = update + torch.tanh(torch.roll(self.mix(h), shifts=1, dims=1))
        return feat + 0.2 * update


class TinyForecastingModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.in_channels = int(in_channels)
        self.horizon = max(1, int(horizon))
        self.input_proj = nn.Linear(int(in_channels), int(width))
        self.temporal_gru = nn.GRU(int(width), int(width), batch_first=True)
        self.blocks = nn.ModuleList(
            [TinyForecastBlock(width=int(width), mode=str(mode)) for _ in range(max(1, int(depth)))]
        )
        self.delta_head = nn.Linear(int(width), int(in_channels))
        self.gate_head = nn.Linear(int(width), int(in_channels))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = check_sequence(points, self.in_channels)
        bsz, steps, num_points, _ = seq.shape
        last = seq[:, -1]
        prev = seq[:, -2] if steps > 1 else seq[:, -1]
        trend = last - prev

        point_feat = self.input_proj(last)
        temporal_tokens = self.input_proj(seq.mean(dim=2))
        temporal_out, _ = self.temporal_gru(temporal_tokens)
        temporal = temporal_out[:, -1].unsqueeze(1).expand(bsz, num_points, -1)

        feat = point_feat
        for block in self.blocks:
            feat = block(feat, temporal)

        delta = self.delta_head(feat)
        gate = torch.sigmoid(self.gate_head(feat))
        forecasts: list[torch.Tensor] = []
        current = last
        for step in range(1, self.horizon + 1):
            base = current + 0.15 * trend + (0.1 * step) * gate * delta
            if self.mode == "motion":
                base = base + 0.05 * step * trend
            elif self.mode == "diffusion":
                base = 0.8 * base + 0.2 * torch.tanh(base)
            elif self.mode == "occupancy":
                base = torch.clamp(base, -1.5, 1.5)
            current = base
            forecasts.append(current)
        return {"forecast": torch.stack(forecasts, dim=1)}


def build_toy_forecasting_model(
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
    return TinyForecastingModel(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        horizon=int(cfg.get("horizon", 2)),
    )


def smoke_test_forecasting_model(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 4, 32, 3))
    print(variant, tuple(out["forecast"].shape))

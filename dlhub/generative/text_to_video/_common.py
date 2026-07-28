from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import nn


def _normalize_prompts(prompt: str | Sequence[str] | None, *, batch_size: int) -> list[str]:
    if prompt is None:
        return [""] * int(batch_size)
    if isinstance(prompt, str):
        return [prompt] * int(batch_size)
    prompts = [str(item) for item in prompt]
    if len(prompts) != int(batch_size):
        raise ValueError(f"Expected {batch_size} prompts, got {len(prompts)}")
    return prompts


def _prompt_features(prompts: Sequence[str], *, device: torch.device) -> torch.Tensor:
    features = torch.zeros(len(prompts), 32, device=device, dtype=torch.float32)
    for row, text in enumerate(prompts):
        encoded = str(text).encode("utf-8")[:32]
        if encoded:
            values = torch.tensor(list(encoded), device=device, dtype=torch.float32)
            features[row, : values.numel()] = values / 255.0
    return features


class CompactTextToVideo(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        frames: int,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.in_channels = int(in_channels)
        self.frames = max(2, int(frames))
        self.prompt_proj = nn.Sequential(
            nn.Linear(32, int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )
        layers: list[nn.Module] = []
        for layer_idx in range(max(1, int(depth))):
            in_features = int(width) if layer_idx > 0 else int(width) * 2
            layers.extend([nn.Linear(in_features, int(width)), nn.GELU()])
        self.backbone = nn.Sequential(*layers)
        frame_dim = self.in_channels * 8 * 8
        self.seed_head = nn.Linear(int(width), frame_dim)
        self.motion_head = nn.Linear(int(width), frame_dim)

    def forward(
        self,
        prompt: str | Sequence[str] | None = None,
        *,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> dict[str, torch.Tensor]:
        dev = torch.device("cpu") if device is None else torch.device(device)
        batch = int(batch_size)
        prompts = _normalize_prompts(prompt, batch_size=batch)
        prompt_feat = _prompt_features(prompts, device=dev)
        time_feat = torch.linspace(
            0.0, 1.0, steps=batch, device=dev, dtype=torch.float32
        ).unsqueeze(1)
        time_feat = time_feat.expand(batch, self.prompt_proj[0].out_features)
        fused = self.backbone(torch.cat([self.prompt_proj(prompt_feat), time_feat], dim=1))
        seed = self.seed_head(fused).view(batch, self.in_channels, 8, 8)
        motion = torch.tanh(self.motion_head(fused)).view(batch, self.in_channels, 8, 8)
        frames: list[torch.Tensor] = []
        for step in range(self.frames):
            alpha = float(step) / max(1, self.frames - 1)
            frames.append(torch.clamp(seed + alpha * motion, -1.0, 1.0))
        video = torch.stack(frames, dim=1)
        return {"video": video, "prompt_features": prompt_feat, "motion": motion}


def build_baseline_text_to_video(
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
    return CompactTextToVideo(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        frames=int(cfg.get("frames", 4)),
    )


def smoke_test_text_to_video(builder: Callable[..., nn.Module], variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(prompt=["a small robot", "a paper airplane"], batch_size=2)
    print(variant, tuple(out["video"].shape))


__all__ = ["CompactTextToVideo", "build_baseline_text_to_video", "smoke_test_text_to_video"]

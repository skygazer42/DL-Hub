from __future__ import annotations

import torch
from torch import nn


class CompactPromptLearner(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int, prompt_len: int):
        super().__init__()
        self.family = str(family)
        c = int(width)
        layers: list[nn.Module] = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)])
        self.visual = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.prompt_head = nn.Linear(c, int(prompt_len) * c)
        self.prompt_len = int(prompt_len)
        self.width = c

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
        pooled = self.pool(self.visual(x)).flatten(1)
        prompts = self.prompt_head(pooled).view(x.shape[0], self.prompt_len, self.width)
        return {"prompts": prompts, "pooled": pooled}


def build_baseline_prompt_learner(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
    prompt_len: int = 8,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return CompactPromptLearner(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        prompt_len=int(prompt_len),
    )


def smoke_test_prompt_learner(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(variant, tuple(out["prompts"].shape), tuple(out["pooled"].shape))

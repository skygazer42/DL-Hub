from __future__ import annotations

import torch
from torch import nn


class ToyReasoner(nn.Module):
    def __init__(
        self, *, family: str, in_channels: int, width: int, depth: int, reasoning_steps: int
    ):
        super().__init__()
        self.family = str(family)
        c = int(width)
        layers: list[nn.Module] = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.GELU()]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Conv2d(c, c, 3, 1, 1), nn.GELU()])
        self.visual = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reason_proj = nn.Linear(c, c)
        self.answer_head = nn.Linear(c, 8)
        self.reasoning_steps = max(1, int(reasoning_steps))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = image.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
        state = self.pool(self.visual(x)).flatten(1)
        trace: list[torch.Tensor] = []
        for _ in range(self.reasoning_steps):
            state = state + torch.tanh(self.reason_proj(state))
            trace.append(state)
        reasoning_trace = torch.stack(trace, dim=1)
        return {
            "logits": self.answer_head(state),
            "reasoning_trace": reasoning_trace,
            "state": state,
        }


def build_toy_reasoner(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyReasoner(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        reasoning_steps=int(spec["steps"]),
    )


def smoke_test_reasoner(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 48, 48))
    print(variant, tuple(out["logits"].shape), tuple(out["reasoning_trace"].shape))

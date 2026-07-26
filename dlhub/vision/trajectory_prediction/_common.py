from __future__ import annotations
import torch
from torch import nn


def check_btnc(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,T,N,C), got {tuple(x.shape)}")
    return x


class ToyTrajectoryPredictor(nn.Module):
    def __init__(self, *, family: str, coord_dim: int, width: int, depth: int, pred_steps: int):
        super().__init__()
        self.family = str(family)
        self.pred_steps = int(pred_steps)
        self.proj = nn.Linear(int(coord_dim), int(width))
        self.rnn = nn.GRU(int(width), int(width), batch_first=True)
        self.head = nn.Linear(int(width), int(coord_dim) * int(pred_steps))

    def forward(self, traj):
        x = check_btnc(traj)
        b, t, n, c = x.shape
        seq = self.proj(x.mean(dim=2))
        out, _ = self.rnn(seq)
        pred = self.head(out[:, -1]).view(b, self.pred_steps, c)
        return {"trajectory": pred}


def build_toy_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    coord_dim: int,
    variant: str,
    width_mult: float = 1.0,
    pred_steps: int = 12,
    **kwargs,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyTrajectoryPredictor(
        family=str(family),
        coord_dim=int(coord_dim),
        width=width,
        depth=int(spec["depth"]),
        pred_steps=int(pred_steps),
    )


def smoke_test_model(builder, variant: str):
    out = builder(coord_dim=2, variant=variant, width_mult=0.5)(torch.randn(2, 8, 4, 2))
    print(variant, tuple(out["trajectory"].shape))

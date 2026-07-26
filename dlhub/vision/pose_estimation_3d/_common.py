from __future__ import annotations
import torch
from torch import nn


def check_btj2(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,T,J,2), got {tuple(x.shape)}")
    return x


class ToyPose3D(nn.Module):
    def __init__(self, *, family: str, num_joints: int, width: int, depth: int):
        super().__init__()
        self.family = str(family)
        j = int(num_joints)
        c = int(width)
        self.proj = nn.Linear(2, c)
        self.rnn = nn.GRU(c, c, batch_first=True)
        self.head = nn.Linear(c, j * 3)

    def forward(self, keypoints2d):
        x = check_btj2(keypoints2d)
        b, t, j, _ = x.shape
        tok = self.proj(x.view(b, t * j, 2))
        seq, _ = self.rnn(tok)
        out = self.head(seq[:, -j:]).view(b, j, 3)
        return {"pose3d": out}


def build_toy_pose3d(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    num_joints: int,
    variant: str,
    width_mult: float = 1.0,
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyPose3D(
        family=str(family), num_joints=int(num_joints), width=width, depth=int(spec["depth"])
    )


def smoke_test_pose3d(builder, variant: str):
    model = builder(num_joints=17, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 8, 17, 2))
    print(variant, tuple(out["pose3d"].shape))

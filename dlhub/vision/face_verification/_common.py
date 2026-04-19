from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_face_pair(pair: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]) -> torch.Tensor:
    if isinstance(pair, (tuple, list)):
        if len(pair) != 2:
            raise ValueError(f"Expected 2 face tensors, got {len(pair)}")
        pair = torch.stack([pair[0], pair[1]], dim=1)
    pair = pair.to(torch.float32)
    if pair.ndim != 5 or pair.shape[1] != 2:
        raise ValueError(f"Expected input shape (B, 2, C, H, W), got {tuple(pair.shape)}")
    return pair


class TinyVerificationEncoder(nn.Module):
    def __init__(self, *, channels: int, mode: str) -> None:
        super().__init__()
        self.mode = str(mode)
        self.stem = nn.Conv2d(3, int(channels), kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=1)
        self.mix = nn.Conv2d(int(channels), int(channels), kernel_size=1)
        self.depthwise = nn.Conv2d(int(channels), int(channels), kernel_size=5, padding=2, groups=int(channels))
        self.prompt = nn.Parameter(torch.zeros(1, int(channels), 1, 1)) if self.mode == "prompt" else None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.stem(image), inplace=True)
        h = self.conv2(F.relu(self.conv1(x), inplace=True))
        if self.prompt is not None:
            h = h + self.prompt
        if self.mode in {"siamese", "triplet", "arcface", "cosface"}:
            h = h + self.depthwise(x)
        elif self.mode == "relation":
            h = h + self.mix(F.avg_pool2d(x, 3, 1, 1))
        elif self.mode == "transformer":
            h = h * torch.sigmoid(self.mix(x)) + self.depthwise(x)
        elif self.mode == "occlusion":
            h = h + (x - F.avg_pool2d(x, 5, 1, 2))
        elif self.mode == "contrastive":
            h = h + self.mix(h)
        elif self.mode == "mamba":
            h = h + torch.tanh(self.depthwise(torch.roll(x, shifts=1, dims=-1)))
        return F.adaptive_avg_pool2d(h, 1).flatten(1)


class TinyFaceVerifier(nn.Module):
    def __init__(self, *, family: str, mode: str, width: int, embedding_dim: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.encoder = TinyVerificationEncoder(channels=int(width), mode=str(mode))
        self.proj = nn.Linear(int(width), int(embedding_dim))
        self.head = nn.Sequential(
            nn.Linear(int(embedding_dim) * 2, int(embedding_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(embedding_dim), 1),
        )

    def forward(self, pair: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor]) -> dict[str, torch.Tensor]:
        pair = check_face_pair(pair)
        left = self.proj(self.encoder(pair[:, 0]))
        right = self.proj(self.encoder(pair[:, 1]))
        left = F.normalize(left, dim=-1)
        right = F.normalize(right, dim=-1)
        fused = torch.cat([torch.abs(left - right), left * right], dim=-1)
        match_logit = self.head(fused)
        embeddings = torch.stack([left, right], dim=1)
        return {"embeddings": embeddings, "match_logit": match_logit, "similarity": (left * right).sum(dim=-1, keepdim=True)}


def build_toy_face_verifier(*, family: str, mode: str, variants: dict[str, dict[str, int]], in_channels: int, variant: str, width_mult: float = 1.0) -> nn.Module:
    if int(in_channels) != 3:
        raise ValueError(f"TinyFaceVerifier expects 3-channel inputs, got {in_channels}")
    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(f"Unknown variant for {family}: {variant!r}. Available: {sorted(variants)}")
    spec = dict(variants[name])
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    embedding_dim = int(spec.get("embedding_dim", 64))
    return TinyFaceVerifier(family=str(family), mode=str(mode), width=width, embedding_dim=embedding_dim)


def smoke_test_face_verifier(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 2, 3, 64, 64))
    print(variant, tuple(out["embeddings"].shape), tuple(out["match_logit"].shape))
    print("ok")

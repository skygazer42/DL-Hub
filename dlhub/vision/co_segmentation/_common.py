from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct


def check_btchw(images: torch.Tensor) -> torch.Tensor:
    images = images.to(torch.float32)
    if images.ndim != 5:
        raise ValueError(f"Expected input shape (B, T, C, H, W), got {tuple(images.shape)}")
    return images


def logits_to_masks(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError(f"logits must have shape (B, T, K, H, W), got {tuple(logits.shape)}")
    return logits.argmax(dim=2)


def flatten_group(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError(f"Expected grouped tensor with shape (B, T, ...), got {tuple(x.shape)}")
    b, t = int(x.shape[0]), int(x.shape[1])
    return x.contiguous().view(b * t, *x.shape[2:])


def unflatten_group(x: torch.Tensor, *, batch: int, set_size: int) -> torch.Tensor:
    batch = int(batch)
    set_size = int(set_size)
    if batch <= 0 or set_size <= 0:
        raise ValueError("batch and set_size must be positive")
    expected = batch * set_size
    if int(x.shape[0]) != expected:
        raise ValueError(
            f"Expected leading dimension {expected} for grouped tensor, got {int(x.shape[0])}"
        )
    return x.contiguous().view(batch, set_size, *x.shape[1:])


class TinyCoSegEncoder(nn.Module):
    """Small multi-scale encoder for co-segmentation toy models."""

    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(width)
        d = max(1, int(depth))
        self.stem = nn.Sequential(
            ConvBNAct(int(in_channels), c, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(c, c, kernel_size=3, stride=1, act="relu"),
        )
        self.stage2 = self._stage(c, c * 2, depth=d, dropout=float(dropout))
        self.stage3 = self._stage(c * 2, c * 4, depth=d, dropout=float(dropout))
        self.out_channels = (c, c * 2, c * 4)

    @staticmethod
    def _stage(in_ch: int, out_ch: int, *, depth: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = [ConvBNAct(int(in_ch), int(out_ch), kernel_size=3, stride=2, act="relu")]
        for _ in range(max(1, int(depth)) - 1):
            layers.append(ConvBNAct(int(out_ch), int(out_ch), kernel_size=3, stride=1, act="relu"))
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (N, C, H, W), got {tuple(x.shape)}")
        x = x.to(torch.float32)
        c1 = self.stem(x)  # /2
        c2 = self.stage2(c1)  # /4
        c3 = self.stage3(c2)  # /8
        return c1, c2, c3


class GroupFusionBlock(nn.Module):
    """Fuse per-image features using simple group-level consensus strategies."""

    def __init__(
        self,
        channels: int,
        *,
        mode: str = "mean",
        num_prototypes: int = 4,
    ) -> None:
        super().__init__()
        self.mode = str(mode).lower().strip()
        c = int(channels)
        self.num_prototypes = max(1, int(num_prototypes))
        if self.mode == "attention":
            self.query = nn.Linear(c, c)
            self.key = nn.Linear(c, c)
            self.value = nn.Linear(c, c)
        elif self.mode == "prototype":
            self.assign = nn.Linear(c, self.num_prototypes)
            self.proto_proj = nn.Linear(c, c)
        self.mix = ConvBNAct(c * 2, c, kernel_size=3, stride=1, act="relu")

    def forward(self, feat: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor | tuple[torch.Tensor, ...]]]:
        if feat.ndim != 5:
            raise ValueError(f"Expected grouped feature map (B, T, C, H, W), got {tuple(feat.shape)}")
        b, t, c, h, w = feat.shape
        desc = feat.mean(dim=(-1, -2))
        aux: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {}

        if self.mode == "mean":
            context = desc.mean(dim=1, keepdim=True).expand(-1, t, -1)
        elif self.mode == "max":
            context = desc.amax(dim=1, keepdim=True).expand(-1, t, -1)
        elif self.mode == "attention":
            q = self.query(desc)
            k = self.key(desc)
            v = self.value(desc)
            attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(max(1, c))
            attn = torch.softmax(attn, dim=-1)
            context = torch.matmul(attn, v)
            aux["co_attention"] = attn
        elif self.mode == "prototype":
            weights = torch.softmax(self.assign(desc), dim=1)
            proto = torch.einsum("btp,btc->bpc", weights, desc)
            denom = weights.sum(dim=1).transpose(1, 0).transpose(0, 1).unsqueeze(-1)
            proto = proto / denom.clamp_min(1e-6)
            bridge = torch.matmul(self.proto_proj(desc), proto.transpose(-1, -2)) / math.sqrt(max(1, c))
            bridge = torch.softmax(bridge, dim=-1)
            context = torch.matmul(bridge, proto)
            aux["group_tokens"] = proto
            aux["prototype_assign"] = bridge
        elif self.mode == "consensus":
            mean = desc.mean(dim=1, keepdim=True)
            peak = desc.amax(dim=1, keepdim=True)
            context = (0.75 * mean + 0.25 * peak).expand(-1, t, -1)
        else:
            raise ValueError(f"Unsupported group fusion mode: {self.mode!r}")

        context_map = context.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
        mixed = torch.cat([feat, context_map], dim=2)
        fused = self.mix(flatten_group(mixed))
        return unflatten_group(fused, batch=b, set_size=t), aux


class CoSegHead(nn.Module):
    def __init__(self, *, in_channels: int, hidden_channels: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(int(in_channels), int(hidden_channels), kernel_size=3, stride=1, act="relu"),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(int(hidden_channels), int(num_classes), kernel_size=1, bias=True),
        )

    def forward(self, feat: torch.Tensor, *, out_hw: tuple[int, int]) -> torch.Tensor:
        if feat.ndim == 5:
            b, t = int(feat.shape[0]), int(feat.shape[1])
            flat = flatten_group(feat)
            grouped = True
        elif feat.ndim == 4:
            flat = feat
            grouped = False
            b = t = 0
        else:
            raise ValueError(f"Expected feature shape (B, T, C, H, W) or (N, C, H, W), got {tuple(feat.shape)}")
        logits = self.net(flat)
        logits = F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
        if grouped:
            logits = unflatten_group(logits, batch=b, set_size=t)
        return logits

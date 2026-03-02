from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(int(out_ch)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "unet"  # unet | tvseg:<name>
    in_channels: int = 1
    base_channels: int = 32
    dropout: float = 0.0


class TinyUNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        base = int(cfg.base_channels)

        self.down1 = ConvBlock(int(cfg.in_channels), base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.drop = nn.Dropout2d(p=float(cfg.dropout))
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)

        e1 = self.down1(x)  # (B, C, H, W)
        e2 = self.down2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.drop(d1)

        return self.out(d1)  # logits (B, 1, H, W)

class _TorchvisionSegmentationAdapter(nn.Module):
    def __init__(self, model: nn.Module, *, in_channels: int) -> None:
        super().__init__()
        self.model = model
        self.in_channels = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                "torchvision segmentation models expect 3 channels (or 1 that can be repeated to 3). "
                f"Got input channels={x.shape[1]}."
            )

        out = self.model(x)
        if isinstance(out, torch.Tensor):
            logits = out
        elif isinstance(out, dict) and "out" in out and isinstance(out["out"], torch.Tensor):
            logits = out["out"]
        else:
            raise TypeError(f"Unsupported segmentation output type: {type(out).__name__}")

        if logits.ndim != 4:
            raise ValueError(f"Expected logits shape (B, C, H, W), got {tuple(logits.shape)}")

        if logits.shape[1] == 1:
            return logits
        raise ValueError(
            "This lesson expects binary segmentation logits with 1 channel. "
            f"Got channels={logits.shape[1]}."
        )


def list_supported_arches() -> list[str]:
    arches = ["unet"]
    try:
        from dlhub.vision.zoo import DependencyNotAvailable, list_torchvision_arches
    except Exception:
        return arches

    try:
        tv = list_torchvision_arches()
    except DependencyNotAvailable:
        return arches

    arches.extend([f"tvseg:{name}" for name in tv.segmentation])
    return arches


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()

    if arch in {"unet", "tinyunet"}:
        return TinyUNet(cfg)

    if arch.startswith("tvseg:"):
        name = arch.split(":", 1)[1].strip()
        if not name:
            raise ValueError("Empty torchvision model name. Example: --arch tvseg:fcn_resnet50")

        try:
            from dlhub.vision.zoo import DependencyNotAvailable, build_torchvision_model, list_torchvision_arches
        except Exception as exc:
            raise RuntimeError(f"torchvision is required for arch={arch_raw!r}: {exc}") from exc

        try:
            tv = list_torchvision_arches()
        except DependencyNotAvailable as exc:
            raise RuntimeError(f"torchvision is required for arch={arch_raw!r}: {exc}") from exc

        tv_seg = set(tv.segmentation)
        if name not in tv_seg:
            raise ValueError(
                f"Unknown torchvision segmentation model: {name!r}. "
                "Tip: see all models via `python scripts/vision_zoo.py --list --search tvseg:`."
            )

        model = build_torchvision_model(f"tvseg:{name}", num_classes=1)
        return _TorchvisionSegmentationAdapter(model, in_channels=int(cfg.in_channels))

    raise ValueError(f"Unknown arch: {cfg.arch}. Supported: unet, tvseg:<name>")


__all__ = ["ConvBlock", "ModelConfig", "TinyUNet", "build_model", "list_supported_arches"]

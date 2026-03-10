from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.segmentation.unet import UNetSegmenter


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
        # Reuse the library U-Net while keeping the lesson API stable.
        # The original TinyUNet had 2 downsamples -> levels=3 here.
        self.model = UNetSegmenter(
            in_channels=int(cfg.in_channels),
            num_classes=1,
            base_channels=int(cfg.base_channels),
            levels=3,
            dropout=float(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # logits (B, 1, H, W)


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
            from dlhub.vision.zoo import (
                DependencyNotAvailable,
                build_torchvision_model,
                list_torchvision_arches,
            )
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

from dataclasses import dataclass

import torch
from torch import nn

from dlhub.vision.backbones.cnn import RepVGGClassifier


@dataclass(frozen=True)
class ModelConfig:
    arch: str = "resnet18"  # local: resnet18 | vgg16 | ... ; zoo: tv:<name> | timm:<name>
    in_channels: int = 1
    num_classes: int = 4
    image_size: int = 64
    width_mult: float = 1.0
    dropout: float = 0.1


class _ImageClassifierAdapter(nn.Module):
    def __init__(self, model: nn.Module, *, in_channels: int) -> None:
        super().__init__()
        self.model = model
        self.in_channels = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        # torchvision/timm image classification models almost universally assume 3-channel inputs.
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                "torchvision/timm image models expect 3 channels (or 1 that can be repeated to 3). "
                f"Got input channels={x.shape[1]}."
            )

        out = self.model(x)
        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "logits"):
            logits = getattr(out, "logits")
            if isinstance(logits, torch.Tensor):
                return logits
        if isinstance(out, tuple | list) and out and isinstance(out[0], torch.Tensor):
            return out[0]
        raise TypeError(f"Unsupported classifier output type: {type(out).__name__}")


def list_supported_arches(*, include_timm: bool = False) -> list[str]:
    """List architecture ids supported by this lesson's `build_model()`.

    Notes:
    - Local models are backed by `dlhub.vision.local_zoo` (names shown both with and without `dl:` prefix).
    - Torchvision/timm models are listed as `tv:<name>` / `tvq:<name>` / `timm:<name>`.
    """

    from dlhub.vision.local_zoo import list_local_arches

    local = list_local_arches()
    arches: list[str] = [a.removeprefix("dl:") for a in local] + local

    try:
        from dlhub.vision.zoo import (
            DependencyNotAvailable,
            list_timm_arches,
            list_torchvision_arches,
        )
    except Exception:
        return arches

    try:
        tv = list_torchvision_arches()
    except DependencyNotAvailable:
        tv = None
    if tv is not None:
        arches.extend([f"tv:{name}" for name in tv.image_classification])
        for name in tv.quantization:
            arches.append(f"tvq:{name.removeprefix('quantized_')}")

    if include_timm:
        try:
            timm_names = list_timm_arches()
        except DependencyNotAvailable:
            timm_names = []
        arches.extend([f"timm:{name}" for name in timm_names])
    return arches


def build_model(cfg: ModelConfig) -> nn.Module:
    arch_raw = str(cfg.arch).strip()
    arch = arch_raw.lower()
    in_channels = int(cfg.in_channels)
    num_classes = int(cfg.num_classes)
    image_size = int(cfg.image_size)

    if arch.startswith("tv:"):
        name = arch.split(":", 1)[1].strip()
        if not name:
            raise ValueError("Empty torchvision model name. Example: --arch tv:resnet18")

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
        tv_clf = set(tv.image_classification)
        if name not in tv_clf:
            raise ValueError(
                f"Unknown torchvision image-classification model: {name!r}. "
                "Tip: see all models via `python scripts/vision_zoo.py --list`."
            )

        model = build_torchvision_model(f"tv:{name}", num_classes=num_classes)
        return _ImageClassifierAdapter(model, in_channels=in_channels)

    if arch.startswith("tvq:"):
        name = arch.split(":", 1)[1].strip()
        if not name:
            raise ValueError("Empty quantized torchvision model name. Example: --arch tvq:resnet18")

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
        tvq = {n.removeprefix("quantized_") for n in tv.quantization}
        if name not in tvq:
            raise ValueError(
                f"Unknown torchvision quantized model: {name!r}. "
                "Tip: see all models via `python scripts/vision_zoo.py --list --search tvq:`."
            )

        model = build_torchvision_model(f"tvq:{name}", num_classes=num_classes)
        return _ImageClassifierAdapter(model, in_channels=in_channels)

    if arch.startswith("timm:"):
        name = arch.split(":", 1)[1].strip()
        if not name:
            raise ValueError("Empty timm model name. Example: --arch timm:resnet50")

        try:
            from dlhub.vision.zoo import DependencyNotAvailable, build_timm_model
        except Exception as exc:
            raise RuntimeError(f"timm is required for arch={arch_raw!r}: {exc}") from exc

        try:
            model = build_timm_model(f"timm:{name}", num_classes=num_classes)
        except DependencyNotAvailable as exc:
            raise RuntimeError(f"timm is required for arch={arch_raw!r}: {exc}") from exc
        return _ImageClassifierAdapter(model, in_channels=in_channels)

    # Local zoo (backward-compatible: accept `resnet-18` / `efficientnet-b0` etc).
    local_arch = arch_raw.strip()
    if ":" in local_arch:
        pref, name = local_arch.split(":", 1)
        local_arch = f"{pref}:{name.replace('-', '')}"
    else:
        local_arch = local_arch.replace("-", "")

    from dlhub.vision.local_zoo import build_local_model

    return build_local_model(
        local_arch,
        in_channels=in_channels,
        num_classes=num_classes,
        image_size=image_size,
        width_mult=float(cfg.width_mult),
        dropout=float(cfg.dropout),
    )


__all__ = [
    "ModelConfig",
    "build_model",
    "list_supported_arches",
    "RepVGGClassifier",
]

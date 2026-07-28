import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels

from ._common import (
    TinyFGBackbone,
    build_fgvc_model,
    check_nchw,
    make_fgvc_variants,
    smoke_test_classifier,
)


class GeMPool2d(nn.Module):
    """Generalized mean pooling (GeM).

    This is a simple but strong pooling head for fine-grained recognition / retrieval.
    We keep it compact-first: 1 learnable scalar p shared across channels.
    """

    def __init__(self, *, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C)
        if x.ndim != 4:
            raise ValueError(f"Expected (B, C, H, W) tensor, got {tuple(x.shape)}")
        p = self.p.clamp_min(0.1)
        # Backbone features are ReLU'd, but clamp for numerical safety.
        x = x.clamp_min(self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, (1, 1)).pow(1.0 / p)
        return x.flatten(1)


class GeMPoolingFGVC(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        spec: dict[str, int],
        in_channels: int,
        num_classes: int,
        image_size: int,
        width_mult: float,
        dropout: float,
    ) -> None:
        del image_size
        super().__init__()
        self.family = str(family)

        stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
        c2 = scale_channels(int(spec["c2"]), float(width_mult), min_ch=16, divisor=8)
        c3 = scale_channels(int(spec["c3"]), float(width_mult), min_ch=16, divisor=8)
        c4 = scale_channels(int(spec["c4"]), float(width_mult), min_ch=16, divisor=8)
        embed = scale_channels(int(spec["embed"]), float(width_mult), min_ch=16, divisor=8)
        depth = int(spec["depth"])

        self.backbone = TinyFGBackbone(
            in_channels=int(in_channels),
            stem=stem,
            c2=c2,
            c3=c3,
            c4=c4,
            depth=depth,
        )
        self.pool = GeMPool2d(p=3.0)
        self.proj = nn.Linear(c4, embed)
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(embed, int(num_classes))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        _, _, c4 = self.backbone(x)
        pooled = self.pool(c4)
        embedding = F.normalize(self.proj(pooled), dim=-1)
        logits = self.classifier(self.dropout(embedding))
        return {
            "logits": logits,
            "embedding": embedding,
            "gem_p": self.pool.p,
        }


_VARIANTS: dict[str, dict[str, int]] = make_fgvc_variants("gem_pooling", group="bilinear")


def build_gem_pooling_fgvc_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "gem_pooling_small",
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    return build_fgvc_model(
        GeMPoolingFGVC,
        variants=_VARIANTS,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        image_size=image_size,
        width_mult=width_mult,
        dropout=dropout,
        family="gem_pooling",
    )


if __name__ == "__main__":
    smoke_test_classifier(build_gem_pooling_fgvc_classifier, "gem_pooling_tiny")

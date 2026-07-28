from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import (
    CoSegHead,
    GroupFusionBlock,
    TinyCoSegEncoder,
    check_btchw,
    flatten_group,
    logits_to_masks,
    unflatten_group,
)

_VARIANTS: dict[str, dict[str, int]] = {
    "clip_coseg_tiny": {"width": 16, "depth": 1},
    "clip_coseg_small": {"width": 24, "depth": 2},
    "clip_coseg_base": {"width": 32, "depth": 3},
}


class ClipCoseg(nn.Module):
    """CLIP-inspired co-segmentor using text/image embedding similarity."""

    def __init__(
        self, *, in_channels: int, num_classes: int, width: int, depth: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.encoder = TinyCoSegEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c3 = int(self.encoder.out_channels[-1])
        self.fuser = GroupFusionBlock(c3, mode="mean", num_prototypes=4)
        self.text_prompt = nn.Parameter(torch.randn(c3) * 0.02)
        self.text_proj = nn.Linear(c3, c3)
        self.image_proj = nn.Conv2d(c3, c3, kernel_size=1)
        self.head = CoSegHead(
            in_channels=c3,
            hidden_channels=max(32, c3),
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(
        self,
        images: torch.Tensor,
        text_feat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        images = check_btchw(images)
        b, t, _, h, w = images.shape
        flat = flatten_group(images)
        _, _, c3 = self.encoder(flat)
        grouped = unflatten_group(c3, batch=b, set_size=t)
        fused, aux = self.fuser(grouped)

        if text_feat is None:
            text_vec = self.text_prompt.unsqueeze(0).expand(int(b), -1)
        else:
            text_vec = text_feat.to(device=fused.device, dtype=fused.dtype)
            if text_vec.ndim == 1:
                text_vec = text_vec.unsqueeze(0)
            elif text_vec.ndim == 3:
                text_vec = text_vec.mean(dim=1)
            elif text_vec.ndim != 2:
                raise ValueError(
                    f"text_feat must have shape (D), (B,D) or (B,Q,D), got {tuple(text_vec.shape)}"
                )
            if int(text_vec.shape[0]) == 1 and int(b) > 1:
                text_vec = text_vec.expand(int(b), -1)
            elif int(text_vec.shape[0]) != int(b):
                raise ValueError(
                    f"text batch {int(text_vec.shape[0])} does not match image batch {int(b)}"
                )
            channels = int(fused.shape[2])
            if int(text_vec.shape[-1]) < channels:
                pad = torch.zeros(
                    int(b),
                    channels - int(text_vec.shape[-1]),
                    device=fused.device,
                    dtype=fused.dtype,
                )
                text_vec = torch.cat([text_vec, pad], dim=-1)
            elif int(text_vec.shape[-1]) > channels:
                text_vec = text_vec[..., :channels]

        text_embed = F.normalize(self.text_proj(text_vec), dim=-1)
        image_embed = self.image_proj(flatten_group(fused))
        image_embed = unflatten_group(image_embed, batch=b, set_size=t)
        image_embed = F.normalize(image_embed, dim=2)
        similarity = (image_embed * text_embed[:, None, :, None, None]).sum(dim=2, keepdim=True)
        conditioned = fused * (1.0 + torch.tanh(similarity))
        conditioned = conditioned + text_embed[:, None, :, None, None]
        logits = self.head(conditioned, out_hw=(h, w))
        out: dict[str, torch.Tensor] = {
            "logits": logits,
            "masks": logits_to_masks(logits),
            "text_similarity": similarity.squeeze(2),
        }
        for key, value in aux.items():
            if isinstance(value, torch.Tensor):
                out[key] = value
        return out


def build_clip_coseg_co_segmentor(
    *,
    in_channels: int,
    num_classes: int,
    set_size: int = 3,
    image_size: int = 64,
    variant: str = "clip_coseg_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del set_size, image_size
    cfg = _VARIANTS[str(variant).lower().strip()]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return ClipCoseg(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 3, 64, 64)
    m = build_clip_coseg_co_segmentor(
        in_channels=3, num_classes=2, variant="clip_coseg_tiny", width_mult=0.5
    )
    out = m(x, torch.randn(2, m.text_proj.in_features))
    print(
        "clip_coseg_tiny",
        {k: tuple(v.shape) for k, v in out.items() if isinstance(v, torch.Tensor)},
    )
    loss = sum(
        v.mean() for v in out.values() if isinstance(v, torch.Tensor) and v.is_floating_point()
    )
    loss.backward()
    print("ok")

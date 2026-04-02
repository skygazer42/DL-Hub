from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ._common import check_nchw, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "facexformer_parse_tiny": {"embed_dim": 48, "depth": 2, "heads": 4},
    "facexformer_parse_small": {"embed_dim": 64, "depth": 3, "heads": 4},
    "facexformer_parse_base": {"embed_dim": 96, "depth": 4, "heads": 8},
}


class FaceXFormerParse(nn.Module):
    """Lightweight transformer parser with learnable part queries."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.patch = nn.Sequential(
            nn.Conv2d(int(in_channels), dim, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=int(heads),
            dim_feedforward=max(64, dim * 4),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth)))
        self.part_queries = nn.Parameter(torch.randn(self.num_classes, dim) * 0.02)
        self.query_proj = nn.Linear(dim, dim)
        self.token_proj = nn.Linear(dim, dim)
        self.part_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, self.num_classes),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        image = check_nchw(image)
        inp_hw = tuple(image.shape[-2:])
        feat = self.patch(image)
        b, d, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # (B,N,D)
        encoded = self.encoder(tokens)

        queries = self.part_queries.to(device=encoded.device, dtype=encoded.dtype)
        query_logits = torch.einsum(
            "kd,bnd->bkn",
            self.query_proj(queries),
            self.token_proj(encoded),
        ) / max(1.0, float(d) ** 0.5)
        query_attention = torch.softmax(query_logits, dim=-1)
        class_tokens = torch.einsum("bkn,bnd->bkd", query_attention, encoded)
        class_bias = self.part_head(class_tokens).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)

        coarse_logits = query_logits.view(int(b), self.num_classes, int(h), int(w))
        coarse_logits = coarse_logits + class_bias
        logits = F.interpolate(coarse_logits, size=inp_hw, mode="bilinear", align_corners=False)
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "query_attention": query_attention,
            "class_tokens": class_tokens,
        }


def build_facexformer_parse_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "facexformer_parse_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown FaceXFormer-Parse variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    cfg = _VARIANTS[name]
    dim = max(32, int(int(cfg["embed_dim"]) * float(width_mult)))
    heads = int(cfg["heads"])
    while heads > 1 and dim % heads != 0:
        heads -= 1
    return FaceXFormerParse(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=dim,
        depth=int(cfg["depth"]),
        heads=int(heads),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_facexformer_parse_face_parser(
        in_channels=3,
        num_classes=11,
        variant="facexformer_parse_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("facexformer_parse_tiny", tuple(out["logits"].shape), tuple(out["class_tokens"].shape))
    loss = out["logits"].mean() + out["class_tokens"].mean()
    loss.backward()
    print("ok")

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ._common import ParsingHead, TinyFaceEncoder, logits_to_parsing

_VARIANTS: dict[str, dict[str, int]] = {
    "farl_parse_tiny": {"width": 16, "depth": 1, "tokens": 8},
    "farl_parse_small": {"width": 24, "depth": 2, "tokens": 12},
    "farl_parse_base": {"width": 32, "depth": 3, "tokens": 16},
}


class FaRLParseFaceParser(nn.Module):
    """Visual-linguistic style face parser with prompt-token alignment."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        num_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = TinyFaceEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
        )
        c2 = int(self.encoder.out_channels[1])
        c3 = int(self.encoder.out_channels[2])
        hidden = max(32, c3)
        self.prompt_tokens = nn.Parameter(torch.randn(int(num_tokens), c3) * 0.02)
        self.token_proj = nn.Linear(c3, c3)
        self.feat_proj = nn.Linear(c3, c3)
        self.mm_gate = nn.Sequential(
            nn.Conv2d(c3 * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c3, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.head = ParsingHead(
            in_channels=c3 * 2 + c2,
            hidden_channels=hidden,
            num_classes=int(num_classes),
            dropout=float(dropout),
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        inp_hw = tuple(image.shape[-2:])
        _, c2, c3 = self.encoder(image)
        b, c, h, w = c3.shape
        tokens = c3.flatten(2).transpose(1, 2)
        prompt = self.prompt_tokens.to(device=c3.device, dtype=c3.dtype).unsqueeze(0).expand(int(b), -1, -1)
        attn = torch.softmax(
            torch.einsum("bnd,bkd->bnk", self.feat_proj(tokens), self.token_proj(prompt))
            / math.sqrt(max(1, int(c))),
            dim=-1,
        )
        prompt_ctx = torch.einsum("bnk,bkd->bnd", attn, prompt).transpose(1, 2).reshape(int(b), int(c), int(h), int(w))
        gate = self.mm_gate(torch.cat([c3, prompt_ctx], dim=1))
        fused = torch.cat(
            [
                c3,
                prompt_ctx * gate,
                F.interpolate(c2, size=c3.shape[-2:], mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        logits = self.head(fused, out_hw=inp_hw)
        parsing_map = logits_to_parsing(logits)
        return {
            "logits": logits,
            "parsing_map": parsing_map,
            "prompt_attention": attn,
            "fusion_gate": gate,
        }


def build_farl_parse_face_parser(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "farl_parse_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    del image_size
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown FaRL-Parse variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return FaRLParseFaceParser(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(cfg["depth"]),
        num_tokens=int(cfg["tokens"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_farl_parse_face_parser(
        in_channels=3,
        num_classes=11,
        variant="farl_parse_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("farl_parse_tiny", tuple(out["logits"].shape), tuple(out["prompt_attention"].shape))
    loss = out["logits"].mean() + out["fusion_gate"].mean()
    loss.backward()
    print("ok")

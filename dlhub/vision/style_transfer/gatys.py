from __future__ import annotations

import torch
from torch import nn

from ._common import TinyEncoder, gram_matrix, total_variation

_VARIANTS: dict[str, dict[str, int]] = {
    "gatys_tiny": {"width": 24, "depth": 2},
    "gatys_small": {"width": 32, "depth": 3},
    "gatys_base": {"width": 48, "depth": 4},
}


class GatysNST(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        width: int,
        depth: int,
        steps: int = 8,
        lr: float = 0.03,
        content_weight: float = 1.0,
        style_weight: float = 5.0,
        tv_weight: float = 1e-3,
    ) -> None:
        super().__init__()
        self.encoder = TinyEncoder(
            in_channels=int(in_channels),
            width=int(width),
            depth=int(depth),
            dropout=0.0,
        )
        self.steps = int(max(1, steps))
        self.lr = float(lr)
        self.content_weight = float(content_weight)
        self.style_weight = float(style_weight)
        self.tv_weight = float(tv_weight)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict[str, torch.Tensor]:
        content = content.to(torch.float32)
        style = style.to(torch.float32)

        with torch.no_grad():
            f_c = self.encoder(content)
            f_s = self.encoder(style)
            g_s = gram_matrix(f_s)

        y = nn.Parameter(content.clone())
        opt = torch.optim.Adam([y], lr=float(self.lr))

        last_loss = None
        for _ in range(int(self.steps)):
            opt.zero_grad(set_to_none=True)
            f_y = self.encoder(y)
            content_loss = (f_y - f_c).pow(2).mean()
            style_loss = (gram_matrix(f_y) - g_s).pow(2).mean()
            tv = total_variation(y)
            loss = (
                float(self.content_weight) * content_loss
                + float(self.style_weight) * style_loss
                + float(self.tv_weight) * tv
            )
            loss.backward()
            opt.step()
            with torch.no_grad():
                y.clamp_(-1.0, 1.0)
            last_loss = loss

        loss_out = torch.tensor(0.0, device=y.device) if last_loss is None else last_loss.detach()
        return {"stylized": y.detach(), "loss": loss_out}


def build_gatys_style_transfer(
    *,
    in_channels: int,
    image_size: int = 64,
    variant: str = "gatys_small",
    width_mult: float = 1.0,
    dropout: float = 0.0,  # unused, kept for zoo signature consistency
    steps: int = 8,
    lr: float = 0.03,
    content_weight: float = 1.0,
    style_weight: float = 5.0,
    tv_weight: float = 1e-3,
) -> nn.Module:
    _ = int(image_size)
    _ = float(dropout)
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Gatys variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    cfg = _VARIANTS[name]
    width = max(8, int(int(cfg["width"]) * float(width_mult)))
    return GatysNST(
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        steps=int(steps),
        lr=float(lr),
        content_weight=float(content_weight),
        style_weight=float(style_weight),
        tv_weight=float(tv_weight),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(1, 3, 64, 64)
    s = torch.randn(1, 3, 64, 64)
    m = build_gatys_style_transfer(in_channels=3, variant="gatys_tiny", width_mult=0.5, steps=2)
    out = m(x, s)
    print("gatys_tiny", tuple(out["stylized"].shape), float(out["loss"].item()))
    print("ok")

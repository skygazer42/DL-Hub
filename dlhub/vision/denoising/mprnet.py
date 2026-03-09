
import torch
from torch import nn
import torch.nn.functional as F

from ._utils import pad_to_multiple, unpad


def _act() -> nn.Module:
    return nn.ReLU(inplace=True)


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        c = int(channels)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)
        self.act = _act()
        self.conv2 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        return x + y


class UNetStage(nn.Module):
    """A tiny U-Net stage that predicts a residual image."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        width: int,
        depth: int,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        c_out = int(out_channels)
        w0 = int(width)
        d = int(depth)
        if c_in <= 0 or c_out <= 0:
            raise ValueError("channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("depth must be > 0")

        def make_blocks(ch: int) -> nn.Sequential:
            return nn.Sequential(*[ResBlock(ch) for _ in range(d)])

        self.intro = nn.Conv2d(c_in, w0, kernel_size=3, padding=1, bias=True)
        self.enc1 = make_blocks(w0)
        self.down1 = nn.Conv2d(w0, w0 * 2, kernel_size=2, stride=2, bias=True)
        self.enc2 = make_blocks(w0 * 2)
        self.down2 = nn.Conv2d(w0 * 2, w0 * 4, kernel_size=2, stride=2, bias=True)
        self.bott = make_blocks(w0 * 4)

        self.up2 = nn.Conv2d(w0 * 4, w0 * 2, kernel_size=1, bias=True)
        self.reduce2 = nn.Conv2d(w0 * 4, w0 * 2, kernel_size=1, bias=True)
        self.dec2 = make_blocks(w0 * 2)

        self.up1 = nn.Conv2d(w0 * 2, w0, kernel_size=1, bias=True)
        self.reduce1 = nn.Conv2d(w0 * 2, w0, kernel_size=1, bias=True)
        self.dec1 = make_blocks(w0)

        self.outro = nn.Conv2d(w0, c_out, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x1 = self.enc1(self.intro(x))
        x2 = self.enc2(self.down1(x1))
        x3 = self.bott(self.down2(x2))

        u2 = F.interpolate(x3, scale_factor=2, mode="nearest")
        u2 = self.up2(u2)
        u2 = self.dec2(self.reduce2(torch.cat([u2, x2], dim=1)))

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")
        u1 = self.up1(u1)
        u1 = self.dec1(self.reduce1(torch.cat([u1, x1], dim=1)))

        return self.outro(u1)


class MPRNet(nn.Module):
    """MPRNet-style progressive multi-stage denoiser (toy-first, pure torch).

    This simplified version keeps the key idea: iterative refinement over multiple stages.
    Each stage predicts a residual; stages 2/3 are conditioned on (noisy, previous_output).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        width: int = 24,
        stage_depth: int = 1,
        stages: int = 3,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        w0 = int(width)
        d = int(stage_depth)
        s = int(stages)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if w0 < 8:
            raise ValueError("width must be >= 8")
        if d <= 0:
            raise ValueError("stage_depth must be > 0")
        if s < 2:
            raise ValueError("stages must be >= 2")

        self.stage1 = UNetStage(in_channels=c_in, out_channels=c_in, width=w0, depth=d)
        self.stage2 = UNetStage(in_channels=c_in * 2, out_channels=c_in, width=w0, depth=d)
        self.stage3 = UNetStage(in_channels=c_in * 2, out_channels=c_in, width=w0, depth=d)
        self.stages = s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")

        x_pad, pad_hw = pad_to_multiple(x, 4, mode="reflect")
        inp = x_pad

        y1 = inp + self.stage1(inp)
        y2 = inp + self.stage2(torch.cat([inp, y1], dim=1))

        if self.stages <= 2:
            return unpad(y2, pad_hw)

        y3 = inp + self.stage3(torch.cat([inp, y2], dim=1))
        return unpad(y3, pad_hw)


_VARIANTS: dict[str, dict] = {
    "mprnet_tiny": {"width": 16, "depth": 1, "stages": 3},
    "mprnet_small": {"width": 24, "depth": 1, "stages": 3},
    "mprnet_base": {"width": 32, "depth": 2, "stages": 3},
}


def build_mprnet_denoiser(
    *,
    in_channels: int,
    variant: str = "mprnet_small",
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MPRNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return MPRNet(
        in_channels=int(in_channels),
        width=int(spec["width"]),
        stage_depth=int(spec["depth"]),
        stages=int(spec["stages"]),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.rand(2, 1, 64, 64)
    noisy = (x + torch.randn_like(x) * 0.1).clamp(0.0, 1.0)
    m = build_mprnet_denoiser(in_channels=1, variant="mprnet_tiny")
    y = m(noisy)
    print("mprnet_tiny", tuple(y.shape))
    loss = (y - x).pow(2).mean()
    loss.backward()
    print("ok")


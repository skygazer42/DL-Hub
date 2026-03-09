
import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import PatchEmbed


class VMambaBlock(nn.Module):
    """A 2D-friendly Mamba-like block (simplified).

    - depthwise Conv2d for local mixing
    - depthwise Conv1d for sequence mixing
    - gated linear projection
    """

    def __init__(self, dim: int, *, seq_kernel: int = 7, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        k = int(seq_kernel)
        self.norm = nn.LayerNorm(d)
        self.dw2d = nn.Conv2d(d, d, kernel_size=3, padding=1, groups=d, bias=False)
        self.dw1d = nn.Conv1d(d, d, kernel_size=k, padding=k // 2, groups=d, bias=False)
        self.proj = nn.Linear(d, 2 * d, bias=True)
        self.out = nn.Linear(d, d, bias=True)
        self.dp = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor, *, hw: tuple[int, int]) -> torch.Tensor:
        identity = x
        x = self.norm(x)
        b, n, d = x.shape
        h, w = int(hw[0]), int(hw[1])
        x2d = x.view(b, h, w, d).permute(0, 3, 1, 2).contiguous()
        x2d = self.dw2d(x2d)
        x = x2d.flatten(2).transpose(1, 2)
        x = self.dw1d(x.transpose(1, 2)).transpose(1, 2)
        u, v = self.proj(x).chunk(2, dim=-1)
        y = u * torch.sigmoid(v)
        y = self.out(y)
        return identity + self.dp(y)


class VMambaClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 256,
        depth: int = 12,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.hw = (int(image_size) // self.patch_size, int(image_size) // self.patch_size)
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        self.pos = nn.Parameter(torch.zeros(1, self.hw[0] * self.hw[1], int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.ModuleList([VMambaBlock(int(dim), seq_kernel=7, drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x) + self.pos
        for b in self.blocks:
            x = b(x, hw=self.hw)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "vmamba_tiny": {"dim": 256, "depth": 12, "patch": 4},
    "vmamba_small": {"dim": 384, "depth": 16, "patch": 4},
}


def build_vmamba_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "vmamba_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown VMamba variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return VMambaClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_vmamba_classifier(in_channels=3, num_classes=10, variant="vmamba_tiny", image_size=64)
    y = m(x)
    print("vmamba_tiny", tuple(y.shape))


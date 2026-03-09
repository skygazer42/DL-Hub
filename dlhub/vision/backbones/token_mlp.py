
import torch
from torch import nn

from dlhub.vision.backbones._blocks import GlobalAvgPoolHead, LayerNorm2d


class TokenMix(nn.Module):
    """Token-mixing MLP: linear over token dimension (fixed N)."""

    def __init__(self, num_tokens: int) -> None:
        super().__init__()
        n = int(num_tokens)
        self.n = n
        self.fc = nn.Linear(n, n, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> treat tokens as HW
        b, c, h, w = x.shape
        n = h * w
        if n != self.n:
            raise ValueError(f"Expected num_tokens={self.n}, got {n}")
        t = x.flatten(2)  # (B, C, N)
        t = self.fc(t)
        return t.view(b, c, h, w)


class TokenMLPBlock(nn.Module):
    def __init__(self, dim: int, *, num_tokens: int) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = LayerNorm2d(d)
        self.mix = TokenMix(int(num_tokens))
        self.norm2 = LayerNorm2d(d)
        self.mlp = nn.Sequential(nn.Conv2d(d, 4 * d, kernel_size=1), nn.GELU(), nn.Conv2d(4 * d, d, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mix(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TokenMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 10,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d = int(dim)
        p = int(patch_size)
        h = int(image_size) // p
        w = int(image_size) // p
        num_tokens = h * w
        self.patch = nn.Sequential(nn.Conv2d(int(in_channels), d, kernel_size=p, stride=p), LayerNorm2d(d))
        self.blocks = nn.Sequential(*[TokenMLPBlock(d, num_tokens=num_tokens) for _ in range(int(depth))])
        self.head = GlobalAvgPoolHead(d, int(num_classes), dropout=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x)
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "token_mlp_tiny": {"dim": 192, "depth": 10, "patch": 4},
    "token_mlp_small": {"dim": 256, "depth": 12, "patch": 4},
}


def build_token_mlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "token_mlp_tiny",
    image_size: int = 64,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown Token-MLP variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return TokenMLPClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_token_mlp_classifier(in_channels=3, num_classes=10, variant="token_mlp_tiny", image_size=64)
    y = m(x)
    print("token_mlp_tiny", tuple(y.shape))


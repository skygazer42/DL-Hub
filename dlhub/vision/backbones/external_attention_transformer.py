
import torch
from torch import nn

from dlhub.vision.backbones._blocks import DropPath
from dlhub.vision.backbones._transformer import MLP, PatchEmbed


class ExternalAttention(nn.Module):
    """External Attention (EA) token mixer.

    attn = softmax(x W1)
    y = attn W2
    """

    def __init__(self, dim: int, *, mem: int = 64) -> None:
        super().__init__()
        d = int(dim)
        m = int(mem)
        self.w1 = nn.Linear(d, m, bias=False)
        self.w2 = nn.Linear(m, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.softmax(self.w1(x), dim=1)
        return self.w2(a)


class EATBlock(nn.Module):
    def __init__(self, dim: int, *, mem: int = 64, drop_path: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        self.norm1 = nn.LayerNorm(d)
        self.ea = ExternalAttention(d, mem=int(mem))
        self.dp1 = DropPath(float(drop_path))
        self.norm2 = nn.LayerNorm(d)
        self.mlp = MLP(d, 4 * d, dropout=0.0, act="gelu")
        self.dp2 = DropPath(float(drop_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dp1(self.ea(self.norm1(x)))
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class ExternalAttentionTransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 4,
        dim: int = 192,
        depth: int = 8,
        mem: int = 64,
        drop_path: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = PatchEmbed(int(in_channels), int(dim), patch_size=int(patch_size))
        h = int(image_size) // int(patch_size)
        w = int(image_size) // int(patch_size)
        self.pos = nn.Parameter(torch.zeros(1, h * w, int(dim)))
        dp_rates = torch.linspace(0.0, float(drop_path), steps=int(depth)).tolist()
        self.blocks = nn.Sequential(*[EATBlock(int(dim), mem=int(mem), drop_path=float(dp_rates[i])) for i in range(int(depth))])
        self.norm = nn.LayerNorm(int(dim))
        self.drop = nn.Dropout(p=float(dropout))
        self.head = nn.Linear(int(dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch(x)
        x = self.blocks(x + self.pos)
        x = self.norm(x)
        x = self.drop(x.mean(dim=1))
        return self.head(x)


_VARIANTS: dict[str, dict] = {
    "eat_tiny": {"dim": 192, "depth": 8, "mem": 64, "patch": 4},
    "eat_small": {"dim": 256, "depth": 10, "mem": 96, "patch": 4},
}


def build_external_attention_transformer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "eat_tiny",
    image_size: int = 64,
    drop_path: float = 0.1,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown EAT variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    return ExternalAttentionTransformerClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(spec["patch"]),
        dim=int(spec["dim"]),
        depth=int(spec["depth"]),
        mem=int(spec["mem"]),
        drop_path=float(drop_path),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_external_attention_transformer_classifier(in_channels=3, num_classes=10, variant="eat_tiny", image_size=64)
    y = m(x)
    print("eat_tiny", tuple(y.shape))


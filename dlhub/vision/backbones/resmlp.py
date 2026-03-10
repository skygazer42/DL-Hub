import torch
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels


def _parse_patch_variant(variant: str, *, default_patch_size: int) -> tuple[str, int]:
    name = str(variant).lower().strip()
    patch_size = int(default_patch_size)
    if "_p" in name:
        base, suffix = name.rsplit("_p", 1)
        if suffix.isdigit():
            name = base
            patch_size = int(suffix)
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    return name, patch_size


class ResMLPBlock(nn.Module):
    def __init__(self, *, num_tokens: int, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        d = int(dim)
        t = int(num_tokens)
        if t <= 0:
            raise ValueError("num_tokens must be > 0")

        self.ln1 = nn.LayerNorm(d)
        self.token_mixer = nn.Linear(t, t)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(d)
        hidden = max(8, int(round(d * float(mlp_ratio))))
        self.channel_mlp = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(hidden, d),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x).transpose(1, 2)  # (B, C, T)
        y = self.token_mixer(y).transpose(1, 2)  # (B, T, C)
        x = x + self.drop1(y)

        y = self.channel_mlp(self.ln2(x))
        x = x + self.drop2(y)
        return x


class ResMLPClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if int(image_size) % int(patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if int(depth) <= 0:
            raise ValueError("depth must be > 0")

        grid = int(image_size) // int(patch_size)
        num_tokens = grid * grid

        self.patch_embed = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=int(patch_size),
            stride=int(patch_size),
            padding=0,
        )
        self.blocks = nn.Sequential(
            *[
                ResMLPBlock(
                    num_tokens=int(num_tokens),
                    dim=int(embed_dim),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(int(embed_dim))
        self.head = nn.Linear(int(embed_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


_SPECS: dict[str, tuple[int, int]] = {
    "resmlp_tiny": (128, 6),
    "resmlp_small": (192, 8),
    "resmlp_base": (256, 12),
    "tiny": (128, 6),
    "small": (192, 8),
    "base": (256, 12),
    "resmlp": (128, 6),
}


def build_resmlp_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    variant: str = "resmlp_tiny",
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    name, patch_size = _parse_patch_variant(variant, default_patch_size=8)
    if name not in _SPECS:
        raise ValueError(
            "Unknown ResMLP variant. Supported: resmlp_tiny|resmlp_small|resmlp_base (+ _p*)"
        )
    base_dim, depth = _SPECS[name]
    embed_dim = scale_channels(int(base_dim), float(width_mult), min_ch=64, divisor=8)
    return ResMLPClassifier(
        image_size=int(image_size),
        patch_size=int(patch_size),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=int(embed_dim),
        depth=int(depth),
        mlp_ratio=4.0,
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["resmlp_tiny", "resmlp_small", "resmlp_base", "resmlp_tiny_p16"]:
        m = build_resmlp_classifier(in_channels=3, num_classes=10, image_size=64, variant=v)
        y = m(x)
        print(v, tuple(y.shape))

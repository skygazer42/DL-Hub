import torch
from torch import nn


class _MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, dropout: float) -> None:
        super().__init__(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        embed_dim: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(embed_dim))
        self.token_mlp = _MLP(
            int(num_tokens), int(token_mlp_dim), int(num_tokens), dropout=float(dropout)
        )
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(int(embed_dim))
        self.channel_mlp = _MLP(
            int(embed_dim), int(channel_mlp_dim), int(embed_dim), dropout=float(dropout)
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = self.ln1(x).transpose(1, 2)  # (B, C, T)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + self.drop1(y)

        y = self.channel_mlp(self.ln2(x))
        x = x + self.drop2(y)
        return x


class MLPMixerClassifier(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        in_channels: int,
        num_classes: int,
        embed_dim: int,
        num_layers: int,
        token_mlp_dim: int,
        channel_mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if int(image_size) % int(patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")

        grid = int(image_size) // int(patch_size)
        num_tokens = grid * grid

        self.patch_embed = nn.Conv2d(
            int(in_channels),
            int(embed_dim),
            kernel_size=int(patch_size),
            stride=int(patch_size),
        )
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_tokens=int(num_tokens),
                    embed_dim=int(embed_dim),
                    token_mlp_dim=int(token_mlp_dim),
                    channel_mlp_dim=int(channel_mlp_dim),
                    dropout=float(dropout),
                )
                for _ in range(int(num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(embed_dim))
        self.head = nn.Linear(int(embed_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)  # (B, C, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C)
        x = self.blocks(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        return self.head(x)


_SPECS: dict[str, dict] = {
    "tiny": {"embed_dim": 128, "num_layers": 4},
    "small": {"embed_dim": 256, "num_layers": 6},
    "base": {"embed_dim": 384, "num_layers": 8},
}


def build_mlp_mixer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    patch_size: int = 8,
    variant: str = "tiny",
    token_mlp_dim: int | None = None,
    channel_mlp_dim: int | None = None,
    dropout: float = 0.1,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _SPECS:
        raise ValueError(f"Unknown MLP-Mixer variant: {variant!r}. Supported: {sorted(_SPECS)}")
    embed_dim = int(_SPECS[name]["embed_dim"])
    num_layers = int(_SPECS[name]["num_layers"])

    grid = int(image_size) // int(patch_size)
    num_tokens = grid * grid
    token_mlp_dim = int(token_mlp_dim) if token_mlp_dim is not None else max(64, num_tokens * 2)
    channel_mlp_dim = int(channel_mlp_dim) if channel_mlp_dim is not None else embed_dim * 4

    return MLPMixerClassifier(
        image_size=int(image_size),
        patch_size=int(patch_size),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        embed_dim=int(embed_dim),
        num_layers=int(num_layers),
        token_mlp_dim=int(token_mlp_dim),
        channel_mlp_dim=int(channel_mlp_dim),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    for v in ["tiny", "small", "base"]:
        m = build_mlp_mixer_classifier(in_channels=3, num_classes=10, variant=v, patch_size=8)
        y = m(x)
        print(f"mlp_mixer_{v}", tuple(y.shape))

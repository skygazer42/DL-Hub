
from dataclasses import dataclass

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(embed_dim // num_heads)

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        self.out = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        scale = float(self.head_dim) ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(embed_dim))
        self.attn = MultiHeadSelfAttention(embed_dim=int(embed_dim), num_heads=int(num_heads), dropout=dropout)
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), int(ff_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(ff_dim), int(embed_dim)),
        )
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ff(self.ln2(x)))
        return x


@dataclass(frozen=True)
class ViTConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    num_heads: int
    num_layers: int
    ff_dim: int
    dropout: float
    num_classes: int


class ViTClassifier(nn.Module):
    def __init__(self, cfg: ViTConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.cfg = cfg
        grid = int(cfg.image_size) // int(cfg.patch_size)
        num_patches = grid * grid

        self.patch_embed = nn.Conv2d(
            int(cfg.in_channels),
            int(cfg.embed_dim),
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, int(cfg.embed_dim)))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, int(cfg.embed_dim)))
        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=int(cfg.embed_dim),
                    num_heads=int(cfg.num_heads),
                    ff_dim=int(cfg.ff_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b = int(x.shape[0])

        tokens = self.patch_embed(x)  # (B, C, Gh, Gw)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, C)

        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, C)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+N, C)
        tokens = tokens + self.pos_embed
        tokens = self.drop(tokens)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.ln(tokens)
        cls_out = tokens[:, 0]
        return self.head(cls_out)


def build_vit_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    ff_dim: int,
    dropout: float = 0.1,
) -> nn.Module:
    return ViTClassifier(
        ViTConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            num_layers=int(num_layers),
            ff_dim=int(ff_dim),
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


class _MLP(nn.Sequential):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__(
            nn.Linear(int(in_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(out_dim)),
        )


class MixerBlock(nn.Module):
    def __init__(self, num_tokens: int, embed_dim: int, token_mlp_dim: int, channel_mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(embed_dim))
        self.token_mlp = _MLP(int(num_tokens), int(token_mlp_dim), int(num_tokens), dropout=float(dropout))
        self.drop1 = nn.Dropout(p=float(dropout))

        self.ln2 = nn.LayerNorm(int(embed_dim))
        self.channel_mlp = _MLP(int(embed_dim), int(channel_mlp_dim), int(embed_dim), dropout=float(dropout))
        self.drop2 = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        y = self.ln1(x).transpose(1, 2)  # (B, C, T)
        y = self.token_mlp(y).transpose(1, 2)
        x = x + self.drop1(y)

        y = self.channel_mlp(self.ln2(x))
        x = x + self.drop2(y)
        return x


@dataclass(frozen=True)
class MLPMixerConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    num_layers: int
    token_mlp_dim: int
    channel_mlp_dim: int
    dropout: float
    num_classes: int


class MLPMixerClassifier(nn.Module):
    def __init__(self, cfg: MLPMixerConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        grid = int(cfg.image_size) // int(cfg.patch_size)
        num_tokens = grid * grid

        self.patch_embed = nn.Conv2d(
            int(cfg.in_channels),
            int(cfg.embed_dim),
            kernel_size=int(cfg.patch_size),
            stride=int(cfg.patch_size),
        )
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_tokens=int(num_tokens),
                    embed_dim=int(cfg.embed_dim),
                    token_mlp_dim=int(cfg.token_mlp_dim),
                    channel_mlp_dim=int(cfg.channel_mlp_dim),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.num_layers))
            ]
        )
        self.ln = nn.LayerNorm(int(cfg.embed_dim))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)  # (B, C, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)  # (B, T, C)
        x = self.blocks(x)
        x = self.ln(x)
        x = x.mean(dim=1)
        return self.head(x)


def build_mlp_mixer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    num_layers: int,
    token_mlp_dim: int,
    channel_mlp_dim: int,
    dropout: float = 0.1,
) -> nn.Module:
    return MLPMixerClassifier(
        MLPMixerConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            num_layers=int(num_layers),
            token_mlp_dim=int(token_mlp_dim),
            channel_mlp_dim=int(channel_mlp_dim),
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


@dataclass(frozen=True)
class ConvMixerConfig:
    image_size: int
    patch_size: int
    in_channels: int
    embed_dim: int
    depth: int
    kernel_size: int
    dropout: float
    num_classes: int


class ConvMixerBlock(nn.Module):
    def __init__(self, dim: int, *, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(int(dim), int(dim), kernel_size=int(kernel_size), padding=int(kernel_size) // 2, groups=int(dim)),
            nn.GELU(),
            nn.BatchNorm2d(int(dim)),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(int(dim), int(dim), kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(int(dim)),
        )
        self.drop = nn.Dropout2d(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.dw(x))
        x = self.pw(x)
        return x


class ConvMixerClassifier(nn.Module):
    def __init__(self, cfg: ConvMixerConfig) -> None:
        super().__init__()
        if int(cfg.image_size) % int(cfg.patch_size) != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                int(cfg.in_channels),
                int(cfg.embed_dim),
                kernel_size=int(cfg.patch_size),
                stride=int(cfg.patch_size),
            ),
            nn.GELU(),
            nn.BatchNorm2d(int(cfg.embed_dim)),
        )
        self.blocks = nn.Sequential(
            *[
                ConvMixerBlock(int(cfg.embed_dim), kernel_size=int(cfg.kernel_size), dropout=float(cfg.dropout))
                for _ in range(int(cfg.depth))
            ]
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(int(cfg.embed_dim), int(cfg.num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def build_convmixer_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int,
    patch_size: int,
    embed_dim: int,
    depth: int,
    kernel_size: int = 9,
    dropout: float = 0.1,
) -> nn.Module:
    return ConvMixerClassifier(
        ConvMixerConfig(
            image_size=int(image_size),
            patch_size=int(patch_size),
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            depth=int(depth),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            num_classes=int(num_classes),
        )
    )


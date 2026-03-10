import torch
from torch import nn

from dlhub.vision.backbones._blocks import (
    ConvBNAct,
    GlobalAvgPoolHead,
    InvertedResidual,
    make_divisible,
)
from dlhub.vision.backbones._transformer import TransformerEncoderBlock


class TokenInteraction(nn.Module):
    """A minimal MobileFormer-style token <-> conv interaction."""

    def __init__(self, token_dim: int, conv_dim: int) -> None:
        super().__init__()
        td = int(token_dim)
        cd = int(conv_dim)
        self.to_token = nn.Linear(cd, td)
        self.to_conv = nn.Linear(td, cd)
        self.gate = nn.Sequential(nn.Linear(td, cd), nn.Sigmoid())

    def forward(
        self, tokens: torch.Tensor, feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # tokens: (B, T, Td), feat: (B, C, H, W)
        b, c, _, _ = feat.shape
        pooled = feat.mean(dim=(2, 3))  # (B, C)
        tokens = tokens + self.to_token(pooled)[:, None, :]
        token_summary = tokens.mean(dim=1)
        gate = self.gate(token_summary).view(b, c, 1, 1)
        feat = feat * (1.0 + gate) + self.to_conv(token_summary).view(b, c, 1, 1)
        return tokens, feat


class MobileFormerV2Classifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        token_dim: int = 192,
        num_tokens: int = 6,
        width_mult: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        w = float(width_mult)

        def c(ch: int) -> int:
            return make_divisible(int(round(int(ch) * w)), 8)

        self.tokens = nn.Parameter(torch.randn(1, int(num_tokens), int(token_dim)) * 0.02)
        self.token_blocks = nn.Sequential(
            TransformerEncoderBlock(int(token_dim), 6, mlp_ratio=2.0, dropout=0.0, drop_path=0.0),
            TransformerEncoderBlock(int(token_dim), 6, mlp_ratio=2.0, dropout=0.0, drop_path=0.0),
        )

        self.stem = ConvBNAct(int(in_channels), c(32), kernel_size=3, stride=2, act="silu")
        self.stage1 = InvertedResidual(
            c(32), c(64), stride=2, expand_ratio=4.0, se_ratio=0.25, act="silu"
        )
        self.inter1 = TokenInteraction(int(token_dim), c(64))

        self.stage2 = InvertedResidual(
            c(64), c(96), stride=2, expand_ratio=6.0, se_ratio=0.25, act="silu"
        )
        self.inter2 = TokenInteraction(int(token_dim), c(96))

        self.stage3 = InvertedResidual(
            c(96), c(160), stride=2, expand_ratio=6.0, se_ratio=0.25, act="silu"
        )
        self.inter3 = TokenInteraction(int(token_dim), c(160))

        self.head = nn.Sequential(
            ConvBNAct(c(160), c(640), kernel_size=1, stride=1, padding=0, act="silu"),
            GlobalAvgPoolHead(c(640), int(num_classes), dropout=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float32)
        b = x.shape[0]
        tokens = self.tokens.expand(b, -1, -1)
        tokens = self.token_blocks(tokens)

        feat = self.stem(x)
        feat = self.stage1(feat)
        tokens, feat = self.inter1(tokens, feat)
        feat = self.stage2(feat)
        tokens, feat = self.inter2(tokens, feat)
        feat = self.stage3(feat)
        tokens, feat = self.inter3(tokens, feat)
        return self.head(feat)


_VARIANTS: dict[str, dict] = {
    "mobileformer_v2_s": {"w": 0.75, "td": 160, "t": 6},
    "mobileformer_v2_b": {"w": 1.0, "td": 192, "t": 6},
    "mobileformer_v2_l": {"w": 1.25, "td": 256, "t": 8},
}


def build_mobileformer_v2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "mobileformer_v2_b",
    dropout: float = 0.2,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown MobileFormer-v2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    return MobileFormerV2Classifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        token_dim=int(spec["td"]),
        num_tokens=int(spec["t"]),
        width_mult=float(spec["w"]),
        dropout=float(dropout),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_mobileformer_v2_classifier(in_channels=3, num_classes=10, variant="mobileformer_v2_s")
    y = m(x)
    print("mobileformer_v2_s", tuple(y.shape))

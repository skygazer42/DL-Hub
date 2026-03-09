
import torch
from torch import nn

from dlhub.vision.backbones._transformer import PatchEmbed, TransformerEncoderBlock


_VIT_SPECS: dict[str, dict] = {
    'tiny':  {'embed_dim': 192, 'depth': 6,  'heads': 3},
    'small': {'embed_dim': 384, 'depth': 8,  'heads': 6},
    'base':  {'embed_dim': 512, 'depth': 10, 'heads': 8},
}


class GfnetClassifier(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        image_size: int = 64,
        patch_size: int = 8,
        variant: str = 'tiny',
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        name = str(variant).lower().strip()
        if name not in _VIT_SPECS:
            raise ValueError(f'Unknown variant: {variant!r}. Supported: {sorted(_VIT_SPECS)}')

        spec = _VIT_SPECS[name]
        embed_dim = int(spec['embed_dim'])
        depth = int(spec['depth'])
        heads = int(spec['heads'])

        self.patch = PatchEmbed(int(in_channels), embed_dim, patch_size=int(patch_size))
        num_patches = (int(image_size) // int(patch_size)) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos = nn.Parameter(torch.zeros(1, int(num_patches) + 1, embed_dim))

        self.drop = nn.Dropout(p=float(dropout))
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(embed_dim, heads, dropout=float(dropout)) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, int(num_classes))

        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if hasattr(self, 'dist_token'):
            nn.init.trunc_normal_(self.dist_token, std=0.02)

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        tokens = self.patch(x)  # (B, N, D)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        if hasattr(self, 'dist_token'):
            dist = self.dist_token.expand(b, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        else:
            tokens = torch.cat([cls, tokens], dim=1)

        tokens = tokens + self.pos
        tokens = self.drop(tokens)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        cls = tokens[:, 0]
        return self.head(cls)


def build_gfnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    patch_size: int = 8,
    variant: str = 'tiny',
    dropout: float = 0.1,
) -> nn.Module:
    return GfnetClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        image_size=int(image_size),
        patch_size=int(patch_size),
        variant=str(variant),
        dropout=float(dropout),
    )


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_gfnet_classifier(in_channels=3, num_classes=10, variant='tiny')
    y = m(x)
    if isinstance(y, tuple):
        print('gfnet', tuple(y[0].shape), tuple(y[1].shape))
    else:
        print('gfnet', tuple(y.shape))

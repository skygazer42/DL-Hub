import torch
import torch.nn.functional as F
from torch import nn

from dlhub.vision.backbones._blocks import scale_channels
from dlhub.vision.panoptic_segmentation._common import check_nchw, fuse_panoptic


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        c_in = int(in_ch)
        c_out = int(out_ch)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(int(in_ch), int(out_ch), kernel_size=2, stride=2, bias=True)
        self.conv = _DoubleConv(int(out_ch) * 2, int(out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class TransUNetPanoptic(nn.Module):
    """TransUNet-style panoptic segmentation (compact-first).

    U-Net encoder-decoder with a Transformer encoder at the bottleneck.
    Semantic logits are produced at full resolution; instance masks are produced via learned queries
    dot-producted with the decoder feature map.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_thing_classes: int,
        num_stuff_classes: int,
        base_channels: int = 32,
        levels: int = 4,
        transformer_depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_instances: int = 32,
        query_dim: int | None = None,
    ) -> None:
        super().__init__()
        nt = int(num_thing_classes)
        ns = int(num_stuff_classes)
        if nt <= 0:
            raise ValueError("num_thing_classes must be > 0")
        if ns <= 0:
            raise ValueError("num_stuff_classes must be > 0")
        base = int(base_channels)
        lv = int(levels)
        if lv < 2:
            raise ValueError("levels must be >= 2")
        n = int(num_instances)
        if n <= 0:
            raise ValueError("num_instances must be > 0")

        self.inc = _DoubleConv(int(in_channels), base)

        downs: list[nn.Module] = []
        ch = base
        for _ in range(lv - 1):
            downs.append(_Down(ch, ch * 2))
            ch *= 2
        self.downs = nn.ModuleList(downs)

        d = int(transformer_depth)
        h = int(num_heads)
        if d <= 0:
            raise ValueError("transformer_depth must be > 0")
        if h <= 0 or ch % h != 0:
            raise ValueError("num_heads must be > 0 and divide bottleneck channels")
        ff = int(round(ch * float(mlp_ratio)))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=ch,
            nhead=h,
            dim_feedforward=ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=d)

        ups: list[nn.Module] = []
        for _ in range(lv - 1):
            ups.append(_Up(ch, ch // 2))
            ch //= 2
        self.ups = nn.ModuleList(ups)

        feat_ch = base
        self.semantic_head = nn.Conv2d(feat_ch, nt + ns, kernel_size=1, bias=True)

        qd = int(query_dim) if query_dim is not None else int(feat_ch)
        if qd <= 0:
            raise ValueError("query_dim must be > 0")
        self.query = nn.Parameter(torch.randn(n, qd) * 0.02)
        self.proj = nn.Sequential(nn.Linear(feat_ch, qd), nn.ReLU(inplace=True))
        self.cls = nn.Linear(qd, nt)
        self.mask_embed = nn.Linear(qd, feat_ch)

        self.num_instances = n
        self.num_thing_classes = nt
        self.num_stuff_classes = ns

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(x)
        b, _, h, w = x.shape

        skips: list[torch.Tensor] = []
        x = self.inc(x)
        skips.append(x)
        for down in self.downs:
            x = down(x)
            skips.append(x)

        b, c, hb, wb = x.shape
        tok = x.permute(0, 2, 3, 1).reshape(b, hb * wb, c)
        tok = self.transformer(tok)
        x = tok.view(b, hb, wb, c).permute(0, 3, 1, 2).contiguous()

        for up, skip in zip(self.ups, reversed(skips[:-1]), strict=True):
            x = up(x, skip)

        feat = x  # (B, base, H, W)
        semantic_logits = self.semantic_head(feat)

        pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)  # (B,base)
        base_q = self.proj(pooled).unsqueeze(1)  # (B,1,QD)
        q = self.query.unsqueeze(0).expand(b, -1, -1)
        hq = base_q + q
        query_cls_logits = self.cls(hq)
        me = self.mask_embed(hq)  # (B,N,base)
        feat_flat = feat.flatten(2)  # (B,base,HW)
        mask_flat = torch.bmm(me, feat_flat)
        mask_logits = mask_flat.view(b, int(self.num_instances), h, w)

        scores = query_cls_logits.softmax(dim=-1).max(dim=-1).values
        panoptic_map = fuse_panoptic(
            semantic_logits, mask_logits, scores, thing_offset=int(self.num_stuff_classes)
        )

        return {
            "semantic_logits": semantic_logits,
            "query_cls_logits": query_cls_logits,
            "mask_logits": mask_logits,
            "panoptic_map": panoptic_map,
        }


_VARIANTS: dict[str, dict] = {
    "transunet_panoptic_tiny": {"base": 16, "levels": 3, "tdepth": 1, "heads": 4, "instances": 16},
    "transunet_panoptic_small": {"base": 24, "levels": 4, "tdepth": 2, "heads": 4, "instances": 32},
    "transunet_panoptic_base": {"base": 32, "levels": 4, "tdepth": 3, "heads": 8, "instances": 64},
}


def build_transunet_panoptic_segmenter(
    *,
    in_channels: int,
    num_thing_classes: int,
    num_stuff_classes: int,
    variant: str = "transunet_panoptic_small",
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown TransUNet-panoptic variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    base = scale_channels(int(spec["base"]), float(width_mult), min_ch=16, divisor=8)
    levels = int(spec["levels"])

    heads = int(spec["heads"])
    bottleneck = int(base) * (2 ** (levels - 1))
    while heads > 1 and bottleneck % heads != 0:
        heads -= 1

    instances = max(4, int(round(int(spec["instances"]) * float(width_mult))))

    return TransUNetPanoptic(
        in_channels=int(in_channels),
        num_thing_classes=int(num_thing_classes),
        num_stuff_classes=int(num_stuff_classes),
        base_channels=int(base),
        levels=int(levels),
        transformer_depth=int(spec["tdepth"]),
        num_heads=int(heads),
        num_instances=int(instances),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 3, 64, 64)
    m = build_transunet_panoptic_segmenter(
        in_channels=3,
        num_thing_classes=3,
        num_stuff_classes=2,
        variant="transunet_panoptic_tiny",
        width_mult=0.5,
    )
    out = m(x)
    print("transunet_panoptic_tiny", {k: tuple(v.shape) for k, v in out.items()})
    loss = (
        out["semantic_logits"].mean() + out["query_cls_logits"].mean() + out["mask_logits"].mean()
    )
    loss.backward()
    print("ok")

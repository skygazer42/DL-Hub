from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class TinyOCREncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        c = int(width)
        layers: list[nn.Module] = [
            nn.Conv2d(int(in_channels), c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        ]
        cur = c
        for _ in range(max(1, int(depth))):
            layers += [
                nn.Conv2d(cur, cur, 3, 2, 1, bias=False),
                nn.BatchNorm2d(cur),
                nn.ReLU(inplace=True),
            ]
        self.net = nn.Sequential(*layers)
        self.out_channels = int(cur)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(check_nchw(x))
        feat = F.adaptive_avg_pool2d(feat, (1, feat.shape[-1]))
        return feat.squeeze(2).transpose(1, 2)


class ToyOCRModel(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        vocab_size: int,
        seq_len: int,
        width: int,
        depth: int,
        decoder_mode: str = "gru",
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.seq_len = int(seq_len)
        self.encoder = TinyOCREncoder(
            in_channels=int(in_channels), width=int(width), depth=int(depth)
        )
        d = int(self.encoder.out_channels)
        self.decoder_mode = str(decoder_mode)
        if self.decoder_mode == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=4 if d % 4 == 0 else 2,
                dim_feedforward=max(64, d * 2),
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth)))
        else:
            self.decoder = nn.GRU(d, d, num_layers=1, batch_first=True, bidirectional=False)
        self.head = nn.Linear(d, int(vocab_size))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = self.encoder(image)
        if isinstance(self.decoder, nn.GRU):
            seq, _ = self.decoder(seq)
        else:
            seq = self.decoder(seq)
        seq = seq[:, : self.seq_len]
        logits = self.head(seq)
        tokens = logits.argmax(dim=-1)
        return {"logits": logits, "tokens": tokens}


def build_toy_ocr_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    vocab_size: int,
    seq_len: int,
    variant: str,
    width_mult: float = 1.0,
    decoder_mode: str = "gru",
) -> nn.Module:
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyOCRModel(
        family=str(family),
        in_channels=int(in_channels),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        width=width,
        depth=int(spec["depth"]),
        decoder_mode=str(decoder_mode),
    )


def smoke_test_ocr(builder, variant: str) -> None:
    model = builder(in_channels=1, vocab_size=32, seq_len=16, variant=variant, width_mult=0.5)
    x = torch.randn(2, 1, 32, 128)
    out = model(x)
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
    print("ok")

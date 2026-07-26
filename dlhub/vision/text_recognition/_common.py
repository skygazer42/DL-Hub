from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def check_nchw(x):
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B,C,H,W), got {tuple(x.shape)}")
    return x


class TinyEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int):
        super().__init__()
        c = int(width)
        layers = [nn.Conv2d(int(in_channels), c, 3, 2, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        return self.net(check_nchw(x))


class ToyTextRecognizer(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        vocab_size: int,
        seq_len: int,
        width: int,
        depth: int,
        decoder: str = "gru",
    ):
        super().__init__()
        self.family = str(family)
        self.seq_len = int(seq_len)
        self.enc = TinyEncoder(in_channels, width, depth)
        c = self.enc.out_channels
        self.decoder_type = str(decoder)
        if self.decoder_type == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=c,
                nhead=4 if c % 4 == 0 else 2,
                dim_feedforward=max(64, c * 2),
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=max(1, int(depth)))
        else:
            self.decoder = nn.GRU(c, c, batch_first=True)
        self.head = nn.Linear(c, int(vocab_size))

    def forward(self, image):
        feat = self.enc(image)
        seq = F.adaptive_avg_pool2d(feat, (1, feat.shape[-1])).squeeze(2).transpose(1, 2)
        seq = seq[:, : self.seq_len]
        if isinstance(self.decoder, nn.GRU):
            seq, _ = self.decoder(seq)
        else:
            seq = self.decoder(seq)
        logits = self.head(seq)
        return {"logits": logits, "tokens": logits.argmax(dim=-1)}


def build_toy_text_recognizer(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    vocab_size: int,
    seq_len: int,
    variant: str,
    width_mult: float = 1.0,
    decoder: str = "gru",
):
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return ToyTextRecognizer(
        family=str(family),
        in_channels=int(in_channels),
        vocab_size=int(vocab_size),
        seq_len=int(seq_len),
        width=width,
        depth=int(spec["depth"]),
        decoder=str(decoder),
    )


def smoke_test_rec(builder, variant: str):
    model = builder(in_channels=1, vocab_size=32, seq_len=16, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 1, 32, 128))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})

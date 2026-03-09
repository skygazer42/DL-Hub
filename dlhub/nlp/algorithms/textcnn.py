
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d

_VARIANTS: tuple[str, ...] = (
    "textcnn_dilated",
    "textcnn_k1357_d1",
    "textcnn_k1357_d2",
    "textcnn_k23",
    "textcnn_k2345",
    "textcnn_k23456_d1",
    "textcnn_k23456_d2",
    "textcnn_k23456_d3",
    "textcnn_k2345_d1",
    "textcnn_k2345_d2",
    "textcnn_k234_d1",
    "textcnn_k234_d2",
    "textcnn_k245_d1",
    "textcnn_k245_d2",
    "textcnn_k2467_d1",
    "textcnn_k2467_d2",
    "textcnn_k2468_d1",
    "textcnn_k2468_d2",
    "textcnn_k246_d1",
    "textcnn_k246_d2",
    "textcnn_k25",
    "textcnn_k256_d1",
    "textcnn_k256_d2",
    "textcnn_k345",
    "textcnn_k34567_d1",
    "textcnn_k34567_d2",
    "textcnn_k34567_d3",
    "textcnn_k3456_d1",
    "textcnn_k3456_d2",
    "textcnn_k345_d1",
    "textcnn_k345_d2",
    "textcnn_k35",
    "textcnn_k357_d1",
    "textcnn_k357_d2",
    "textcnn_k456_d1",
    "textcnn_k456_d2",
)


@dataclass(frozen=True)
class TextCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    num_filters: int
    kernel_sizes: tuple[int, ...]
    dilation: int = 1


class TextCNNClassifier(nn.Module):
    def __init__(self, cfg: TextCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.num_filters), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=int(d),
                    out_channels=int(c),
                    kernel_size=int(k),
                    dilation=int(cfg.dilation),
                )
                for k in cfg.kernel_sizes
            ]
        )
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(int(c) * len(cfg.kernel_sizes), int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        emb = self.embedding(input_ids).to(torch.float32)  # (B, T, D)
        x = emb.transpose(1, 2).contiguous()  # (B, D, T)

        pooled: list[torch.Tensor] = []
        for conv in self.convs:
            h = torch.relu(conv(x))  # (B, C, T')
            pooled.append(h.amax(dim=-1))

        feat = torch.cat(pooled, dim=1)
        feat = self.drop(feat)
        return self.head(feat)


def build_textcnn_classifier(
    *,
    vocab_size: int,
    pad_id: int,
    max_length: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.2,
    variant: str,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name in {"textcnn_k345", "textcnn"}:
        kernels, dilation = (3, 4, 5), 1
    elif name in {"textcnn_k23"}:
        kernels, dilation = (2, 3), 1
    elif name in {"textcnn_k2345"}:
        kernels, dilation = (2, 3, 4, 5), 1
    elif name in {"textcnn_k25"}:
        kernels, dilation = (2, 5), 1
    elif name in {"textcnn_k35"}:
        kernels, dilation = (3, 5), 1
    elif name in {"textcnn_dilated"}:
        kernels, dilation = (3, 5), 2
    else:
        # Lab format: textcnn_k234_d2 (kernel sizes + dilation).
        if name.startswith("textcnn_k") and "_d" in name:
            prefix, d_str = name.rsplit("_d", 1)
            kernel_str = prefix.removeprefix("textcnn_k")
            if not kernel_str.isdigit() or not d_str.isdigit():
                raise ValueError("Invalid TextCNN lab variant; expected textcnn_k<digits>_d<int>")
            kernels = tuple(int(ch) for ch in kernel_str)
            dilation = int(d_str)
            if not kernels or dilation <= 0 or any(k <= 0 for k in kernels):
                raise ValueError("Invalid TextCNN lab variant; kernel sizes and dilation must be > 0")
        else:
            raise ValueError(
                "Unknown TextCNN variant. Supported: textcnn_k345|textcnn_k2345|textcnn_dilated|textcnn_k<ks>_d<d>"
            )

    return TextCNNClassifier(
        TextCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            num_filters=96,
            kernel_sizes=tuple(int(k) for k in kernels),
            dilation=int(dilation),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Family alias
    r["textcnn"] = make_builder(build_textcnn_classifier, variant="textcnn_k345")

    for name in _VARIANTS:
        r[name] = make_builder(build_textcnn_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_textcnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=1.0,
        dropout=0.1,
        variant="textcnn_k345",
    )
    model.eval()

    x = torch.randint(0, vocab_size, (2, max_length), dtype=torch.long)
    attention_mask = torch.ones((2, max_length), dtype=torch.float32)
    with torch.no_grad():
        y = model({"input_ids": x, "attention_mask": attention_mask})

    n_params = sum(int(p.numel()) for p in model.parameters())
    print(f"smoke_ok: y.shape={tuple(y.shape)} params={n_params}")


if __name__ == "__main__":
    _smoke()


__all__ = ["TextCNNClassifier", "TextCNNConfig", "build_textcnn_classifier", "registry"]

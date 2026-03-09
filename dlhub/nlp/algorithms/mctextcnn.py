
from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d


@dataclass(frozen=True)
class MultiChannelTextCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    num_filters: int
    kernel_sizes: tuple[int, ...]
    freeze_static: bool = True


class MultiChannelTextCNNClassifier(nn.Module):
    """Multi-channel TextCNN (static + non-static embeddings), simplified."""

    def __init__(self, cfg: MultiChannelTextCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.num_filters), float(cfg.width_mult), min_dim=32, divisor=8)

        self.embed_static = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.embed_train = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))

        if bool(cfg.freeze_static):
            for p in self.embed_static.parameters():
                p.requires_grad_(False)

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=2,
                    out_channels=int(c),
                    kernel_size=(int(k), int(d)),
                )
                for k in cfg.kernel_sizes
            ]
        )
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(int(c) * len(cfg.kernel_sizes), int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)  # (B, T)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        e_static = self.embed_static(input_ids).to(torch.float32)  # (B, T, D)
        e_train = self.embed_train(input_ids).to(torch.float32)  # (B, T, D)
        x = torch.stack([e_static, e_train], dim=1)  # (B, 2, T, D)

        feats: list[torch.Tensor] = []
        for conv in self.convs:
            h = torch.relu(conv(x))  # (B, C, T', 1)
            h = h.squeeze(-1)  # (B, C, T')
            feats.append(h.amax(dim=-1))  # (B, C)

        feat = torch.cat(feats, dim=1)
        feat = self.drop(feat)
        return self.head(feat)


def build_mctextcnn_classifier(
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
    if name in {"mctextcnn_tiny", "mctextcnn"}:
        embed_dim, num_filters, kernels = 96, 96, (3, 4, 5)
    elif name in {"mctextcnn_small"}:
        embed_dim, num_filters, kernels = 128, 128, (3, 4, 5)
    elif name in {"mctextcnn_base"}:
        embed_dim, num_filters, kernels = 160, 160, (3, 4, 5)
    else:
        raise ValueError(
            "Unknown MCTextCNN variant. Supported: mctextcnn_tiny|mctextcnn_small|mctextcnn_base"
        )

    return MultiChannelTextCNNClassifier(
        MultiChannelTextCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=int(embed_dim),
            num_filters=int(num_filters),
            kernel_sizes=tuple(int(k) for k in kernels),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}
    r["mctextcnn"] = make_builder(build_mctextcnn_classifier, variant="mctextcnn_tiny")
    for name in ("mctextcnn_tiny", "mctextcnn_small", "mctextcnn_base"):
        r[name] = make_builder(build_mctextcnn_classifier, variant=name)
    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_mctextcnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="mctextcnn_tiny",
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


__all__ = [
    "MultiChannelTextCNNClassifier",
    "MultiChannelTextCNNConfig",
    "build_mctextcnn_classifier",
    "registry",
]


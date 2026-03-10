from dataclasses import dataclass

import torch
from torch import nn

from dlhub.nlp.algorithms._helpers import make_builder
from dlhub.nlp.types import Builder
from dlhub.nlp.utils import _d


@dataclass(frozen=True)
class KMaxTextCNNConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float
    dropout: float
    embed_dim: int
    num_filters: int
    kernel_sizes: tuple[int, ...]
    kmax: int


class KMaxTextCNNClassifier(nn.Module):
    """TextCNN with k-max pooling (Kalchbrenner-style), simplified."""

    def __init__(self, cfg: KMaxTextCNNConfig) -> None:
        super().__init__()
        self.cfg = cfg

        d = _d(int(cfg.embed_dim), float(cfg.width_mult), min_dim=32, divisor=8)
        c = _d(int(cfg.num_filters), float(cfg.width_mult), min_dim=32, divisor=8)
        kmax = int(cfg.kmax)
        if kmax <= 0:
            raise ValueError("kmax must be > 0")
        self.kmax = kmax

        self.embedding = nn.Embedding(int(cfg.vocab_size), int(d), padding_idx=int(cfg.pad_id))
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=int(d),
                    out_channels=int(c),
                    kernel_size=int(k),
                    padding=int(k) // 2,
                )
                for k in cfg.kernel_sizes
            ]
        )
        self.drop = nn.Dropout(p=float(cfg.dropout))
        self.head = nn.Linear(int(c) * len(cfg.kernel_sizes) * int(kmax), int(cfg.num_classes))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = inputs["input_ids"].to(torch.long)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = (input_ids != int(self.cfg.pad_id)).to(torch.float32)
        attention_mask = attention_mask.to(torch.float32)

        emb = self.embedding(input_ids).to(torch.float32)  # (B, T, D)
        x = emb.transpose(1, 2).contiguous()  # (B, D, T)

        feats: list[torch.Tensor] = []
        key_mask = attention_mask.to(torch.bool).unsqueeze(1)  # (B, 1, T)
        for conv in self.convs:
            h = torch.relu(conv(x))  # (B, C, T)
            h = h.masked_fill(~key_mask, float("-inf"))
            k = min(int(self.kmax), int(h.shape[-1]))
            topk = h.topk(k, dim=-1).values  # (B, C, k)
            feats.append(topk.reshape(input_ids.shape[0], -1))

        feat = torch.cat(feats, dim=1)
        feat = torch.where(torch.isfinite(feat), feat, torch.zeros_like(feat))
        feat = self.drop(feat)
        return self.head(feat)


def build_kmax_textcnn_classifier(
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
    if name in {"kmaxcnn_k8", "kmax_textcnn_k8", "kmaxcnn"}:
        kernels, kmax = (3, 4, 5), 8
    elif name in {"kmaxcnn_k4", "kmax_textcnn_k4"}:
        kernels, kmax = (3, 4, 5), 4
    elif name in {"kmaxcnn_k2"}:
        kernels, kmax = (3, 4, 5), 2
    elif name in {"kmaxcnn_k8_k2345"}:
        kernels, kmax = (2, 3, 4, 5), 8
    else:
        raise ValueError(
            "Unknown KMaxCNN variant. Supported: kmaxcnn_k8|kmaxcnn_k4|kmaxcnn_k8_k2345|..."
        )

    return KMaxTextCNNClassifier(
        KMaxTextCNNConfig(
            vocab_size=int(vocab_size),
            pad_id=int(pad_id),
            max_length=int(max_length),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            embed_dim=96,
            num_filters=96,
            kernel_sizes=tuple(int(k) for k in kernels),
            kmax=int(kmax),
        )
    )


def registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # Family alias (historically `nl:kmaxcnn`)
    r["kmaxcnn"] = make_builder(build_kmax_textcnn_classifier, variant="kmaxcnn_k8")

    for name in ("kmaxcnn_k2", "kmaxcnn_k4", "kmaxcnn_k8", "kmaxcnn_k8_k2345"):
        r[name] = make_builder(build_kmax_textcnn_classifier, variant=name)

    return r


def _smoke() -> None:
    vocab_size = 128
    pad_id = 0
    max_length = 32
    num_classes = 4

    model = build_kmax_textcnn_classifier(
        vocab_size=vocab_size,
        pad_id=pad_id,
        max_length=max_length,
        num_classes=num_classes,
        width_mult=0.5,
        dropout=0.1,
        variant="kmaxcnn_k4",
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
    "KMaxTextCNNClassifier",
    "KMaxTextCNNConfig",
    "build_kmax_textcnn_classifier",
    "registry",
]

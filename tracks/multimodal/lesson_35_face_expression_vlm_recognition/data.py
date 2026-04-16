from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

EMOTIONS: tuple[str, ...] = ("happy", "sad", "angry", "neutral")
EMOTION_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(EMOTIONS)}


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"

    @property
    def pad_id(self) -> int:
        return int(self.token_to_id[self.pad_token])

    @property
    def bos_id(self) -> int:
        return int(self.token_to_id[self.bos_token])

    @property
    def eos_id(self) -> int:
        return int(self.token_to_id[self.eos_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_prompt(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Prompt exceeds max_length={int(max_length)}.")
        pad_count = int(max_length) - len(seq)
        mask = [1.0] * len(seq) + [0.0] * pad_count
        seq.extend([self.pad_id] * pad_count)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    feature_dim: int = 16
    max_text_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "classify",
        "facial",
        "expression",
        "happy",
        "sad",
        "angry",
        "neutral",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _expression_prototype(emotion: str, feature_dim: int) -> torch.Tensor:
    proto = torch.zeros(int(feature_dim), dtype=torch.float32)
    if emotion == "happy":
        proto[0] = 1.4
        proto[1] = 1.0
    elif emotion == "sad":
        proto[2] = 1.3
        proto[3] = 0.9
    elif emotion == "angry":
        proto[4] = 1.5
        proto[5] = 1.1
    elif emotion == "neutral":
        proto[6] = 1.2
        proto[7] = 0.8
    else:
        raise ValueError(f"Unsupported emotion: {emotion}")
    return proto


def _build_record(*, cfg: DataConfig, vocab: Vocab, generator: torch.Generator) -> dict[str, object]:
    emotion = EMOTIONS[int(torch.randint(0, len(EMOTIONS), (1,), generator=generator).item())]
    label = EMOTION_TO_ID[emotion]

    feature_dim = int(cfg.feature_dim)
    face_features = 0.08 * torch.randn((feature_dim,), generator=generator, dtype=torch.float32)
    face_features = face_features + _expression_prototype(emotion, feature_dim)

    prompt_tokens = ["classify", "facial", "expression", emotion]
    prompt_ids, prompt_mask = vocab.encode_prompt(prompt_tokens, max_length=int(cfg.max_text_length))

    return {
        "face_features": face_features,
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "label": torch.tensor(label, dtype=torch.long),
        "emotion_label": emotion,
        "prompt_text": " ".join(prompt_tokens),
    }


class ToyFacialExpressionDataset(Dataset[dict[str, object]]):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.vocab = _build_vocab()
        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.records = [
            _build_record(cfg=cfg, vocab=self.vocab, generator=generator)
            for _ in range(int(cfg.num_samples))
        ]

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[int(index)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    dataset = ToyFacialExpressionDataset(cfg)
    train_indices, val_indices = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(cfg.val_fraction),
        seed=int(cfg.seed),
    )
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader, dataset.vocab


__all__ = [
    "EMOTION_TO_ID",
    "EMOTIONS",
    "DataConfig",
    "ToyFacialExpressionDataset",
    "Vocab",
    "get_dataloaders",
]

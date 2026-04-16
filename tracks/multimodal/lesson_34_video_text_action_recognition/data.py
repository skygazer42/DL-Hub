from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

ACTION_TYPES: tuple[str, ...] = ("jump", "wave", "sit")
ACTION_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(ACTION_TYPES)}


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

    def encode_query(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Query exceeds max_length={int(max_length)}.")
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
    num_frames: int = 10
    feature_dim: int = 24
    max_text_length: int = 10
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "recognize",
        "action",
        "in",
        "clip",
        "jump",
        "wave",
        "sit",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _sample_segment(cfg: DataConfig, generator: torch.Generator) -> tuple[int, int]:
    num_frames = int(cfg.num_frames)
    max_start = num_frames - 3
    start = int(torch.randint(1, max_start + 1, (1,), generator=generator).item())
    max_len = min(4, num_frames - start)
    seg_len = int(torch.randint(2, max_len + 1, (1,), generator=generator).item())
    end = int(start + seg_len - 1)
    return start, end


def _action_template(action_type: str, feature_dim: int) -> torch.Tensor:
    base = torch.zeros(int(feature_dim), dtype=torch.float32)
    if action_type == "jump":
        base[0] = 1.3
        base[1] = 0.9
    elif action_type == "wave":
        base[2] = 1.2
        base[3] = 1.0
    elif action_type == "sit":
        base[4] = 1.1
        base[5] = 1.1
    else:
        raise ValueError(f"Unsupported action_type: {action_type}")
    return base


def _build_record(
    *,
    cfg: DataConfig,
    vocab: Vocab,
    idx: int,
    generator: torch.Generator,
) -> dict[str, object]:
    del idx
    action_type = ACTION_TYPES[int(torch.randint(0, len(ACTION_TYPES), (1,), generator=generator).item())]
    label = ACTION_TO_ID[action_type]
    seg_start, seg_end = _sample_segment(cfg, generator)

    num_frames = int(cfg.num_frames)
    feature_dim = int(cfg.feature_dim)
    video_features = 0.05 * torch.randn((num_frames, feature_dim), generator=generator, dtype=torch.float32)

    action_vec = _action_template(action_type, feature_dim)
    video_features[seg_start : seg_end + 1] = video_features[seg_start : seg_end + 1] + action_vec.unsqueeze(0)

    query_tokens = ["recognize", "action", action_type, "in", "clip"]
    query_ids, attention_mask = vocab.encode_query(query_tokens, max_length=int(cfg.max_text_length))

    return {
        "video_features": video_features,
        "query_ids": query_ids,
        "attention_mask": attention_mask,
        "label": torch.tensor(label, dtype=torch.long),
        "query_text": " ".join(query_tokens),
        "action_type": action_type,
    }


class ToyVideoTextActionRecognitionDataset(Dataset[dict[str, object]]):
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.vocab = _build_vocab()
        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.records = [
            _build_record(cfg=cfg, vocab=self.vocab, idx=idx, generator=generator)
            for idx in range(int(cfg.num_samples))
        ]

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.records[int(index)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader[dict[str, object]], DataLoader[dict[str, object]], Vocab]:
    dataset = ToyVideoTextActionRecognitionDataset(cfg)
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
    "ACTION_TO_ID",
    "ACTION_TYPES",
    "DataConfig",
    "ToyVideoTextActionRecognitionDataset",
    "Vocab",
    "get_dataloaders",
]

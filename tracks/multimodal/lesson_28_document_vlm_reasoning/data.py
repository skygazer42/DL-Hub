from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_CITIES: tuple[str, ...] = ("rome", "oslo", "lima", "bern")
_TOTALS: tuple[str, ...] = ("12", "27", "39", "54")
_PRIORITIES: tuple[str, ...] = ("low", "high")
_FONT: dict[str, tuple[str, str, str, str, str]] = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "a": ("111", "101", "111", "101", "101"),
    "b": ("110", "101", "110", "101", "110"),
    "c": ("111", "100", "100", "100", "111"),
    "e": ("111", "100", "110", "100", "111"),
    "g": ("111", "100", "101", "101", "111"),
    "h": ("101", "101", "111", "101", "101"),
    "i": ("111", "010", "010", "010", "111"),
    "l": ("100", "100", "100", "100", "111"),
    "m": ("111", "111", "101", "101", "101"),
    "n": ("110", "101", "101", "101", "101"),
    "o": ("111", "101", "101", "101", "111"),
    "p": ("111", "101", "111", "100", "100"),
    "r": ("110", "101", "110", "101", "101"),
    "s": ("111", "100", "111", "001", "111"),
    "t": ("111", "010", "010", "010", "010"),
    "u": ("101", "101", "101", "101", "111"),
    "v": ("101", "101", "101", "101", "010"),
    "w": ("101", "101", "111", "111", "101"),
    "y": ("101", "101", "111", "001", "111"),
}


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

    def encode_tokens(self, tokens: list[str], *, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.eos_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Sequence exceeds max_length={int(max_length)}")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count
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
    batch_size: int = 16
    image_size: int = 32
    max_doc_length: int = 20
    max_query_length: int = 8
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab() -> Vocab:
    tokens = [
        "<pad>",
        "<bos>",
        "<eos>",
        "invoice",
        "city",
        "total",
        "priority",
        "is",
        "what",
        "the",
        "rome",
        "oslo",
        "lima",
        "bern",
        "12",
        "27",
        "39",
        "54",
        "low",
        "high",
    ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    return Vocab(token_to_id=token_to_id, id_to_token=list(tokens))


def _draw_text(
    image: torch.Tensor,
    *,
    text: str,
    top: int,
    left: int,
    ink: tuple[float, float, float],
) -> None:
    max_h = int(image.shape[1])
    max_w = int(image.shape[2])
    color = torch.tensor(ink, dtype=torch.float32).view(3, 1, 1)
    for char_idx, char in enumerate(text.lower()):
        glyph = _FONT.get(char)
        if glyph is None:
            continue
        char_left = int(left) + int(char_idx) * 4
        for row_idx, row in enumerate(glyph):
            y = int(top) + int(row_idx)
            if y < 0 or y >= max_h:
                continue
            for col_idx, value in enumerate(row):
                x = char_left + int(col_idx)
                if x < 0 or x >= max_w:
                    continue
                if value == "1":
                    image[:, y, x] = color.squeeze(-1).squeeze(-1)


def _render_document(
    *,
    city: str,
    total: str,
    priority: str,
    image_size: int,
) -> torch.Tensor:
    image = torch.full((3, int(image_size), int(image_size)), 0.96, dtype=torch.float32)
    image[:, 1:-1, 1] = 0.82
    image[:, 1:-1, -2] = 0.82
    image[:, 1, 1:-1] = 0.82
    image[:, -2, 1:-1] = 0.82

    rows = [
        ("invoice", ""),
        ("city", city),
        ("total", total),
        ("priority", priority),
    ]
    for row_idx, (key, value) in enumerate(rows):
        top = 3 + int(row_idx) * 7
        if top + 4 >= int(image_size):
            break
        _draw_text(image, text=key, top=top, left=2, ink=(0.14, 0.2, 0.35))
        if value:
            _draw_text(image, text=value, top=top, left=18, ink=(0.1, 0.1, 0.1))
        if top + 6 < int(image_size) - 2:
            image[:, top + 6, 2:-2] = 0.9

    return image.clamp(0.0, 1.0)


class ToyDocumentVlmReasoningDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        self.generator = torch.Generator().manual_seed(int(cfg.seed))
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 24:
            raise ValueError("image_size must be >= 24")
        if int(cfg.max_doc_length) < 10:
            raise ValueError("max_doc_length must be >= 10")
        if int(cfg.max_query_length) < 6:
            raise ValueError("max_query_length must be >= 6")

    def __len__(self) -> int:
        return int(self.cfg.num_samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        del idx
        city = _CITIES[int(torch.randint(0, len(_CITIES), (1,), generator=self.generator).item())]
        total = _TOTALS[int(torch.randint(0, len(_TOTALS), (1,), generator=self.generator).item())]
        priority = _PRIORITIES[
            int(torch.randint(0, len(_PRIORITIES), (1,), generator=self.generator).item())
        ]

        ask_total = bool(torch.randint(2, (1,), generator=self.generator).item() == 0)
        if ask_total:
            answer = "high" if int(total) >= 30 else "low"
            query_tokens = ["what", "is", "the", "total", "priority"]
        else:
            answer = "high" if city in {"rome", "bern"} else "low"
            query_tokens = ["what", "is", "the", "city", "priority"]
        label = 1 if answer == "high" else 0

        doc_tokens = ["invoice", "city", city, "total", total, "priority", priority]
        doc_ids, doc_mask = self.vocab.encode_tokens(doc_tokens, max_length=int(self.cfg.max_doc_length))
        query_ids, query_mask = self.vocab.encode_tokens(
            query_tokens, max_length=int(self.cfg.max_query_length)
        )

        return {
            "image": _render_document(
                city=city,
                total=total,
                priority=priority,
                image_size=int(self.cfg.image_size),
            ),
            "doc_input_ids": doc_ids,
            "doc_attention_mask": doc_mask,
            "query_input_ids": query_ids,
            "query_attention_mask": query_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "query_text": " ".join(query_tokens),
            "answer_text": answer,
        }


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyDocumentVlmReasoningDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "doc_input_ids": torch.stack([sample["doc_input_ids"] for sample in batch], dim=0),
            "doc_attention_mask": torch.stack(
                [sample["doc_attention_mask"] for sample in batch], dim=0
            ),
            "query_input_ids": torch.stack([sample["query_input_ids"] for sample in batch], dim=0),
            "query_attention_mask": torch.stack(
                [sample["query_attention_mask"] for sample in batch], dim=0
            ),
            "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
            "query_text": [str(sample["query_text"]) for sample in batch],
            "answer_text": [str(sample["answer_text"]) for sample in batch],
        }

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab

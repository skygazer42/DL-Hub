from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dlhub.data.splits import train_val_split_indices

_FIELDS: tuple[str, ...] = ("name", "id", "date", "total", "city", "status")
_VALUES: dict[str, tuple[str, ...]] = {
    "name": ("amy", "bob", "eve", "leo"),
    "id": ("1024", "2048", "3141", "4096"),
    "date": ("0314", "0402", "0519", "0608"),
    "total": ("12", "27", "39", "54"),
    "city": ("rome", "oslo", "lima", "bern"),
    "status": ("paid", "sent", "done", "hold"),
}
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
    "d": ("110", "101", "101", "101", "110"),
    "e": ("111", "100", "110", "100", "111"),
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
    "y": ("101", "101", "111", "001", "111"),
}


@dataclass(frozen=True)
class Vocab:
    token_to_id: dict[str, int]
    id_to_token: list[str]
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    sep_token: str = "<sep>"
    none_token: str = "none"

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
    def sep_id(self) -> int:
        return int(self.token_to_id[self.sep_token])

    @property
    def none_id(self) -> int:
        return int(self.token_to_id[self.none_token])

    @property
    def size(self) -> int:
        return int(len(self.id_to_token))

    def encode_prompt(self, tokens: list[str], *, max_length: int) -> torch.Tensor:
        seq = [self.bos_id, *[int(self.token_to_id[token]) for token in tokens], self.sep_id]
        if len(seq) > int(max_length):
            raise ValueError(f"Prompt exceeds max_length={int(max_length)}.")
        pad_count = int(max_length) - len(seq)
        seq.extend([self.pad_id] * pad_count)
        return torch.tensor(seq, dtype=torch.long)

    def encode_example(
        self, prompt_tokens: list[str], answer_tokens: list[str], *, max_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_ids = [self.bos_id, *[int(self.token_to_id[token]) for token in prompt_tokens], self.sep_id]
        answer_ids = [int(self.token_to_id[token]) for token in answer_tokens]
        full_ids = [*prompt_ids, *answer_ids]
        if len(full_ids) > int(max_length):
            raise ValueError(f"Example exceeds max_length={int(max_length)}.")

        labels = [-100] * (len(prompt_ids) - 1) + [*answer_ids, self.eos_id]
        pad_count = int(max_length) - len(full_ids)
        full_ids.extend([self.pad_id] * pad_count)
        labels.extend([-100] * pad_count)
        mask = [1.0] * (int(max_length) - pad_count) + [0.0] * pad_count

        return (
            torch.tensor(full_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float32),
        )

    def decode_ids(self, ids: list[int]) -> list[str]:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token[int(idx)]
            if token in {self.pad_token, self.bos_token, self.sep_token}:
                continue
            if token == self.eos_token:
                break
            tokens.append(token)
        return tokens

    def to_dict(self) -> dict[str, object]:
        return {
            "pad_id": int(self.pad_id),
            "bos_id": int(self.bos_id),
            "eos_id": int(self.eos_id),
            "sep_id": int(self.sep_id),
            "none_id": int(self.none_id),
            "token_to_id": {k: int(v) for k, v in self.token_to_id.items()},
        }


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 512
    batch_size: int = 32
    image_size: int = 32
    max_text_length: int = 20
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0
    min_fields: int = 3
    max_fields: int = 5


def _build_vocab() -> Vocab:
    special = ["<pad>", "<bos>", "<eos>", "<sep>", "read", "none"]
    value_tokens = sorted({value for values in _VALUES.values() for value in values})
    tokens = [*special, *_FIELDS, *value_tokens]
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
        char_left = int(left) + int(char_idx) * 3
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


def _render_document(*, fields: dict[str, str], image_size: int) -> torch.Tensor:
    image = torch.full((3, int(image_size), int(image_size)), 0.96, dtype=torch.float32)
    image[:, 1:-1, 1] = 0.82
    image[:, 1:-1, -2] = 0.82
    image[:, 1, 1:-1] = 0.82
    image[:, -2, 1:-1] = 0.82

    sorted_fields = sorted(fields.items(), key=lambda item: _FIELDS.index(item[0]))
    for row_idx, (key, value) in enumerate(sorted_fields):
        top = 3 + int(row_idx) * 6
        if top + 4 >= int(image_size):
            break
        _draw_text(image, text=key, top=top, left=1, ink=(0.18, 0.24, 0.42))
        image[:, top + 1, 19] = torch.tensor([0.22, 0.22, 0.22], dtype=torch.float32)
        image[:, top + 3, 19] = torch.tensor([0.22, 0.22, 0.22], dtype=torch.float32)
        _draw_text(image, text=value, top=top, left=20, ink=(0.12, 0.12, 0.12))
        if top + 5 < int(image_size) - 2:
            image[:, top + 5, 2:-2] = 0.9

    return image.clamp(0.0, 1.0)


def _sample_fields(
    *,
    cfg: DataConfig,
    generator: torch.Generator,
) -> dict[str, str]:
    num_fields = int(
        torch.randint(
            int(cfg.min_fields),
            int(cfg.max_fields) + 1,
            (1,),
            generator=generator,
        ).item()
    )
    perm = torch.randperm(len(_FIELDS), generator=generator).tolist()
    chosen = [_FIELDS[idx] for idx in perm[:num_fields]]
    fields: dict[str, str] = {}
    for key in chosen:
        value_choices = _VALUES[key]
        value_idx = int(torch.randint(0, len(value_choices), (1,), generator=generator).item())
        fields[key] = value_choices[value_idx]
    return fields


def _build_record(
    *,
    cfg: DataConfig,
    idx: int,
    vocab: Vocab,
    generator: torch.Generator,
) -> dict[str, object]:
    fields = _sample_fields(cfg=cfg, generator=generator)
    positive = int(idx) % 2 == 0

    if positive:
        key_list = list(fields.keys())
        key = key_list[int(torch.randint(0, len(key_list), (1,), generator=generator).item())]
        answer_token = fields[key]
        present = 1.0
    else:
        missing = [field for field in _FIELDS if field not in fields]
        key = missing[int(torch.randint(0, len(missing), (1,), generator=generator).item())]
        answer_token = "none"
        present = 0.0

    prompt_tokens = ["read", key]
    answer_tokens = [answer_token]

    prompt_ids = vocab.encode_prompt(prompt_tokens, max_length=int(cfg.max_text_length))
    input_ids, labels, attention_mask = vocab.encode_example(
        prompt_tokens,
        answer_tokens,
        max_length=int(cfg.max_text_length),
    )

    image = _render_document(fields=fields, image_size=int(cfg.image_size))
    return {
        "image": image,
        "prompt_ids": prompt_ids,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "present": torch.tensor(float(present), dtype=torch.float32),
        "query_text": " ".join(prompt_tokens),
        "answer_text": " ".join(answer_tokens),
    }


class ToyKeyValueOcrDataset(Dataset):
    def __init__(self, cfg: DataConfig, *, vocab: Vocab) -> None:
        self.cfg = cfg
        self.vocab = vocab
        if int(cfg.num_samples) <= 0:
            raise ValueError("num_samples must be positive")
        if int(cfg.image_size) < 32:
            raise ValueError("image_size must be >= 32")
        if int(cfg.max_text_length) < 8:
            raise ValueError("max_text_length must be >= 8")
        if int(cfg.max_fields) < int(cfg.min_fields):
            raise ValueError("max_fields must be >= min_fields")

        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.records = [
            _build_record(cfg=cfg, idx=idx, vocab=vocab, generator=generator)
            for idx in range(int(cfg.num_samples))
        ]

    def __len__(self) -> int:
        return int(len(self.records))

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.records[int(idx)]


def get_dataloaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader, Vocab]:
    vocab = _build_vocab()
    dataset = ToyKeyValueOcrDataset(cfg, vocab=vocab)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset), val_fraction=float(cfg.val_fraction), seed=int(cfg.seed)
    )

    def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
        return {
            "image": torch.stack([sample["image"] for sample in batch], dim=0),
            "prompt_ids": torch.stack([sample["prompt_ids"] for sample in batch], dim=0),
            "input_ids": torch.stack([sample["input_ids"] for sample in batch], dim=0),
            "labels": torch.stack([sample["labels"] for sample in batch], dim=0),
            "attention_mask": torch.stack([sample["attention_mask"] for sample in batch], dim=0),
            "present": torch.stack([sample["present"] for sample in batch], dim=0),
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


__all__ = ["DataConfig", "ToyKeyValueOcrDataset", "Vocab", "get_dataloaders"]

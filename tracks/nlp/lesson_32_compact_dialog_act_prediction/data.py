from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import Vocab, simple_tokenize

DIALOG_ACTS = ("greet", "inform", "request", "confirm", "deny", "goodbye")
_CUISINES = ("italian", "sushi", "indian")
_AREAS = ("downtown", "uptown", "riverside")
_SLOTS = ("price", "time", "party")


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 320
    batch_size: int = 16
    max_length: int = 24
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _build_vocab(texts: list[str]) -> Vocab:
    token_to_id = {"<pad>": 0, "<unk>": 1}
    id_to_token = ["<pad>", "<unk>"]
    for text in texts:
        for token in simple_tokenize(text):
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def _encode(vocab: Vocab, text: str, *, max_length: int) -> tuple[list[int], list[int]]:
    ids = [vocab.token_to_id.get(tok, vocab.unk_id) for tok in simple_tokenize(text)[: int(max_length)]]
    attention = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        attention.append(0)
    return ids, attention


def _render_turn(act: str, cuisine: str, area: str, slot: str) -> str:
    if act == "greet":
        return "system greet user hello welcome to restaurant assistant"
    if act == "inform":
        return f"system inform restaurant cuisine {cuisine} area {area} available"
    if act == "request":
        return f"user request restaurant {slot} in {area}"
    if act == "confirm":
        return f"system confirm booking party two at restaurant in {area}"
    if act == "deny":
        return f"user deny that choice need different {slot} for {cuisine} restaurant"
    return "system goodbye thanks for using restaurant assistant"


def _make_example(rng: np.random.Generator) -> tuple[str, int]:
    act = str(rng.choice(DIALOG_ACTS))
    cuisine = str(rng.choice(_CUISINES))
    area = str(rng.choice(_AREAS))
    slot = str(rng.choice(_SLOTS))
    text = _render_turn(act, cuisine, area, slot)
    return text, int(DIALOG_ACTS.index(act))


def _make_examples(config: DataConfig) -> list[tuple[str, int]]:
    rng = np.random.default_rng(int(config.seed))
    examples = [_make_example(rng) for _ in range(int(config.num_samples))]
    rng.shuffle(examples)
    return examples


class DialogActDataset:
    def __init__(self, *, examples: list[tuple[str, int]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, label = self.examples[int(idx)]
        input_ids, attention_mask = _encode(self.vocab, text, max_length=int(self.cfg.max_length))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "labels": torch.tensor(int(label), dtype=torch.long),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab = _build_vocab([text for text, _ in examples])
    dataset = DialogActDataset(examples=examples, vocab=vocab, cfg=config)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "labels": torch.stack([item["labels"] for item in batch], dim=0),
        }

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=_collate,
    )
    return train_loader, val_loader, vocab


__all__ = ["DIALOG_ACTS", "DataConfig", "DialogActDataset", "get_dataloaders"]

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab

INTENT_TO_ID = {
    "book_flight": 0,
    "cancel_flight": 1,
    "reschedule_flight": 2,
    "flight_status": 3,
}

SLOT_TO_ID = {
    "O": 0,
    "B-from_city": 1,
    "I-from_city": 2,
    "B-to_city": 3,
    "I-to_city": 4,
    "B-date": 5,
    "I-date": 6,
}

_FROM_CITY_VALUES = (
    ("boston",),
    ("new", "york"),
    ("san", "francisco"),
    ("seattle",),
)
_TO_CITY_VALUES = (
    ("denver",),
    ("los", "angeles"),
    ("chicago",),
    ("salt", "lake"),
)
_DATE_VALUES = (
    ("monday",),
    ("next", "friday"),
    ("july", "fifth"),
    ("tomorrow",),
)


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 320
    batch_size: int = 16
    max_length: int = 12
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


def _append_span(
    tokens: list[str],
    slot_ids: list[int],
    span_tokens: tuple[str, ...],
    begin_id: int,
    inside_id: int,
) -> None:
    for idx, token in enumerate(span_tokens):
        tokens.append(str(token))
        slot_ids.append(int(begin_id if idx == 0 else inside_id))


def _generate_example(intent_name: str, rng: np.random.Generator) -> tuple[list[str], int, list[int]]:
    from_city = _FROM_CITY_VALUES[int(rng.integers(0, len(_FROM_CITY_VALUES)))]
    to_city = _TO_CITY_VALUES[int(rng.integers(0, len(_TO_CITY_VALUES)))]
    date_value = _DATE_VALUES[int(rng.integers(0, len(_DATE_VALUES)))]

    tokens: list[str] = []
    slots: list[int] = []

    if intent_name == "book_flight":
        tokens.extend(["book", "flight", "from", "city"])
        slots.extend([0, 0, 0, 0])
        _append_span(tokens, slots, from_city, SLOT_TO_ID["B-from_city"], SLOT_TO_ID["I-from_city"])
        tokens.extend(["to", "city"])
        slots.extend([0, 0])
        _append_span(tokens, slots, to_city, SLOT_TO_ID["B-to_city"], SLOT_TO_ID["I-to_city"])
        tokens.extend(["on", "date"])
        slots.extend([0, 0])
        _append_span(tokens, slots, date_value, SLOT_TO_ID["B-date"], SLOT_TO_ID["I-date"])
    elif intent_name == "cancel_flight":
        tokens.extend(["cancel", "flight", "from", "city"])
        slots.extend([0, 0, 0, 0])
        _append_span(tokens, slots, from_city, SLOT_TO_ID["B-from_city"], SLOT_TO_ID["I-from_city"])
        tokens.extend(["to", "city"])
        slots.extend([0, 0])
        _append_span(tokens, slots, to_city, SLOT_TO_ID["B-to_city"], SLOT_TO_ID["I-to_city"])
    elif intent_name == "reschedule_flight":
        tokens.extend(["reschedule", "flight", "to", "city"])
        slots.extend([0, 0, 0, 0])
        _append_span(tokens, slots, to_city, SLOT_TO_ID["B-to_city"], SLOT_TO_ID["I-to_city"])
        tokens.extend(["from", "city"])
        slots.extend([0, 0])
        _append_span(tokens, slots, from_city, SLOT_TO_ID["B-from_city"], SLOT_TO_ID["I-from_city"])
        tokens.extend(["on", "date"])
        slots.extend([0, 0])
        _append_span(tokens, slots, date_value, SLOT_TO_ID["B-date"], SLOT_TO_ID["I-date"])
    else:
        tokens.extend(["status", "for", "flight", "from", "city"])
        slots.extend([0, 0, 0, 0, 0])
        _append_span(tokens, slots, from_city, SLOT_TO_ID["B-from_city"], SLOT_TO_ID["I-from_city"])
        tokens.extend(["to", "city"])
        slots.extend([0, 0])
        _append_span(tokens, slots, to_city, SLOT_TO_ID["B-to_city"], SLOT_TO_ID["I-to_city"])

    return tokens, int(INTENT_TO_ID[intent_name]), slots


def _make_examples(config: DataConfig) -> list[tuple[list[str], int, list[int]]]:
    rng = np.random.default_rng(int(config.seed))
    intents = list(INTENT_TO_ID.keys())
    examples: list[tuple[list[str], int, list[int]]] = []
    for _ in range(int(config.num_samples)):
        intent_name = str(rng.choice(intents))
        examples.append(_generate_example(intent_name, rng))
    rng.shuffle(examples)
    return examples


def _build_vocab(token_sequences: list[list[str]]) -> Vocab:
    token_to_id = {"<pad>": 0, "<unk>": 1}
    id_to_token = ["<pad>", "<unk>"]
    for sequence in token_sequences:
        for token in sequence:
            if token in token_to_id:
                continue
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
    return Vocab(token_to_id=token_to_id, id_to_token=id_to_token, pad_id=0, unk_id=1)


def _encode_tokens(vocab: Vocab, tokens: list[str], max_length: int) -> tuple[list[int], list[int]]:
    ids = [vocab.token_to_id.get(token, vocab.unk_id) for token in tokens[: int(max_length)]]
    attention = [1] * len(ids)
    while len(ids) < int(max_length):
        ids.append(vocab.pad_id)
        attention.append(0)
    return ids, attention


class JointIntentSlotDataset:
    def __init__(self, *, examples: list[tuple[list[str], int, list[int]]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        tokens, intent_id, slot_ids = self.examples[int(idx)]
        input_ids, attention_mask = _encode_tokens(self.vocab, tokens, max_length=int(self.cfg.max_length))
        padded_slots = slot_ids[: int(self.cfg.max_length)]
        while len(padded_slots) < int(self.cfg.max_length):
            padded_slots.append(SLOT_TO_ID["O"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "intent_labels": torch.tensor(int(intent_id), dtype=torch.long),
            "slot_labels": torch.tensor(padded_slots, dtype=torch.long),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab = _build_vocab([tokens for tokens, _, _ in examples])
    dataset = JointIntentSlotDataset(examples=examples, vocab=vocab, cfg=config)
    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "intent_labels": torch.stack([item["intent_labels"] for item in batch], dim=0),
            "slot_labels": torch.stack([item["slot_labels"] for item in batch], dim=0),
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


__all__ = [
    "DataConfig",
    "INTENT_TO_ID",
    "SLOT_TO_ID",
    "JointIntentSlotDataset",
    "get_dataloaders",
]

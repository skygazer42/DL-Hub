from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import Vocab, simple_tokenize

_CUISINES = ("italian", "sushi", "indian")
_AREAS = ("downtown", "uptown", "riverside")
_PARTIES = ("two", "four", "six")


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


def _sample_history_value(rng: np.random.Generator, values: tuple[str, ...]) -> str:
    if rng.random() < 0.25:
        return "none"
    return str(rng.choice(values))


def _carryover_decision(
    *,
    slot_name: str,
    history_value: str,
    candidates: tuple[str, ...],
    rng: np.random.Generator,
) -> tuple[list[str], int]:
    if history_value == "none":
        if rng.random() < 0.55:
            new_value = str(rng.choice(candidates))
            return [f"followup set {slot_name} {new_value}"], 0
        return [f"followup skip {slot_name}"], 0

    action = str(rng.choice(("keep", "replace", "clear"), p=np.array([0.45, 0.35, 0.20])))
    if action == "keep":
        return [f"followup carry {slot_name} keep"], 1
    if action == "replace":
        replacement = str(rng.choice([value for value in candidates if value != history_value]))
        return [f"followup replace {slot_name} {replacement}"], 0
    return [f"followup clear {slot_name}"], 0


def _generate_example(rng: np.random.Generator) -> tuple[str, tuple[int, int, int]]:
    history_cuisine = _sample_history_value(rng, _CUISINES)
    history_area = _sample_history_value(rng, _AREAS)
    history_party = _sample_history_value(rng, _PARTIES)

    turns = [
        "system greet",
        f"history cuisine {history_cuisine}",
        f"history area {history_area}",
        f"history party {history_party}",
    ]

    slot_order = ["cuisine", "area", "party"]
    rng.shuffle(slot_order)

    labels = {"cuisine": 0, "area": 0, "party": 0}
    for slot_name in slot_order:
        if slot_name == "cuisine":
            slot_turns, label = _carryover_decision(
                slot_name=slot_name,
                history_value=history_cuisine,
                candidates=_CUISINES,
                rng=rng,
            )
        elif slot_name == "area":
            slot_turns, label = _carryover_decision(
                slot_name=slot_name,
                history_value=history_area,
                candidates=_AREAS,
                rng=rng,
            )
        else:
            slot_turns, label = _carryover_decision(
                slot_name=slot_name,
                history_value=history_party,
                candidates=_PARTIES,
                rng=rng,
            )
        labels[slot_name] = label
        turns.extend(slot_turns)

    turns.append("system summarize carry history followup")
    text = " ".join(turns)
    return text, (int(labels["cuisine"]), int(labels["area"]), int(labels["party"]))


def _make_examples(config: DataConfig) -> list[tuple[str, tuple[int, int, int]]]:
    rng = np.random.default_rng(int(config.seed))
    examples = [_generate_example(rng) for _ in range(int(config.num_samples))]
    rng.shuffle(examples)
    return examples


class SlotCarryoverDataset:
    def __init__(self, *, examples: list[tuple[str, tuple[int, int, int]]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        text, labels = self.examples[int(idx)]
        input_ids, attention_mask = _encode(self.vocab, text, max_length=int(self.cfg.max_length))
        cuisine_label, area_label, party_label = labels
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
            "cuisine_labels": torch.tensor(int(cuisine_label), dtype=torch.long),
            "area_labels": torch.tensor(int(area_label), dtype=torch.long),
            "party_labels": torch.tensor(int(party_label), dtype=torch.long),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab = _build_vocab([text for text, _ in examples])
    dataset = SlotCarryoverDataset(examples=examples, vocab=vocab, cfg=config)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )

    def _collate(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
            "cuisine_labels": torch.stack([item["cuisine_labels"] for item in batch], dim=0),
            "area_labels": torch.stack([item["area_labels"] for item in batch], dim=0),
            "party_labels": torch.stack([item["party_labels"] for item in batch], dim=0),
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


__all__ = ["DataConfig", "SlotCarryoverDataset", "get_dataloaders"]

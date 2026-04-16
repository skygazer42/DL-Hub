from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.toy_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 320
    batch_size: int = 16
    max_length: int = 20
    num_candidates: int = 4
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_INTENTS = ("greet", "hours", "location", "booking", "price")
_AREAS = ("downtown", "uptown", "riverside")
_PARTY_SIZES = ("two", "four", "six")


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


def _build_context(intent: str, area: str, party_size: str) -> str:
    if intent == "greet":
        return "system greet user hello user ask restaurant help"
    if intent == "hours":
        return f"system greet user ask restaurant hours area {area}"
    if intent == "location":
        return f"user ask restaurant location near {area}"
    if intent == "booking":
        return f"user want restaurant booking party {party_size} area {area}"
    return f"user ask restaurant price range area {area}"


def _response_for_intent(intent: str, area: str, party_size: str) -> str:
    if intent == "greet":
        return "system respond hello i can help with restaurant requests"
    if intent == "hours":
        return "system respond our restaurant is open from ten to ten"
    if intent == "location":
        return f"system respond the restaurant is in {area}"
    if intent == "booking":
        return f"system respond i booked a table for {party_size}"
    return "system respond our restaurant prices are moderate"


def _hard_negative(intent: str, area: str, party_size: str, rng: np.random.Generator) -> str:
    wrong_intents = [x for x in _INTENTS if x != intent]
    sampled_intent = str(rng.choice(wrong_intents))
    return _response_for_intent(sampled_intent, area, party_size)


def _make_example(config: DataConfig, rng: np.random.Generator) -> tuple[str, list[str], int]:
    intent = str(rng.choice(_INTENTS))
    area = str(rng.choice(_AREAS))
    party_size = str(rng.choice(_PARTY_SIZES))
    context = _build_context(intent, area, party_size)

    positive = _response_for_intent(intent, area, party_size)
    candidates = [positive]
    while len(candidates) < int(config.num_candidates):
        candidates.append(_hard_negative(intent, area, party_size, rng))

    rng.shuffle(candidates)
    label = int(candidates.index(positive))
    return context, candidates, label


def _make_examples(config: DataConfig) -> list[tuple[str, list[str], int]]:
    rng = np.random.default_rng(int(config.seed))
    examples = [_make_example(config, rng) for _ in range(int(config.num_samples))]
    rng.shuffle(examples)
    return examples


class DialogResponseSelectionDataset:
    def __init__(self, *, examples: list[tuple[str, list[str], int]], vocab: Vocab, cfg: DataConfig) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        import torch

        context, candidates, label = self.examples[int(idx)]
        context_ids, context_mask = _encode(self.vocab, context, max_length=int(self.cfg.max_length))
        candidate_ids: list[list[int]] = []
        candidate_masks: list[list[int]] = []
        for candidate in candidates:
            ids, mask = _encode(self.vocab, candidate, max_length=int(self.cfg.max_length))
            candidate_ids.append(ids)
            candidate_masks.append(mask)
        return {
            "context_ids": torch.tensor(context_ids, dtype=torch.long),
            "context_attention_mask": torch.tensor(context_mask, dtype=torch.float32),
            "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long),
            "candidate_attention_mask": torch.tensor(candidate_masks, dtype=torch.float32),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def get_dataloaders(config: DataConfig):
    import torch
    from torch.utils.data import DataLoader, Subset

    examples = _make_examples(config)
    vocab_texts: list[str] = []
    for context, candidates, _ in examples:
        vocab_texts.append(context)
        vocab_texts.extend(candidates)
    vocab = _build_vocab(vocab_texts)
    dataset = DialogResponseSelectionDataset(examples=examples, vocab=vocab, cfg=config)

    train_idx, val_idx = train_val_split_indices(
        n=len(dataset),
        val_fraction=float(config.val_fraction),
        seed=int(config.seed),
    )

    def _collate(batch):
        return {
            "context_ids": torch.stack([item["context_ids"] for item in batch], dim=0),
            "context_attention_mask": torch.stack([item["context_attention_mask"] for item in batch], dim=0),
            "candidate_ids": torch.stack([item["candidate_ids"] for item in batch], dim=0),
            "candidate_attention_mask": torch.stack([item["candidate_attention_mask"] for item in batch], dim=0),
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


__all__ = ["DataConfig", "DialogResponseSelectionDataset", "get_dataloaders"]

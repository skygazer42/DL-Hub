from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlhub.data.splits import train_val_split_indices
from tracks.nlp.synthetic_text import Vocab, simple_tokenize


@dataclass(frozen=True)
class DataConfig:
    num_samples: int = 256
    batch_size: int = 16
    max_length: int = 12
    num_classes: int = 4
    shots_per_class: int = 3
    val_fraction: float = 0.2
    seed: int = 0
    num_workers: int = 0


_INTENTS = (
    {"name": "weather", "keyword": "rain", "context": "forecast", "action": "umbrella"},
    {"name": "travel", "keyword": "ticket", "context": "airport", "action": "booking"},
    {"name": "music", "keyword": "playlist", "context": "speaker", "action": "play"},
    {"name": "calendar", "keyword": "meeting", "context": "schedule", "action": "remind"},
    {"name": "finance", "keyword": "budget", "context": "expense", "action": "report"},
    {"name": "lighting", "keyword": "lamp", "context": "room", "action": "switch"},
)

_SUPPORT_TEMPLATES = (
    "fewshot intent {name} uses {keyword} with {context} and {action}",
    "lowshot intent {name} keeps {keyword} near {context} for {action}",
    "intent {name} example pairs {keyword} and {action} in {context}",
)

_QUERY_TEMPLATES = (
    "fewshot user asks {action} after {keyword} in {context}",
    "intent query needs {keyword} and {action} around {context}",
    "classify intent {name} from {keyword} plus {action}",
)


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


def _canonical_supports(
    selected_intents: list[dict[str, str]],
    shots_per_class: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    supports: list[list[str]] = []
    for intent in selected_intents:
        class_supports = []
        for shot_idx in range(int(shots_per_class)):
            template = str(rng.choice(_SUPPORT_TEMPLATES))
            class_supports.append(f"{template.format(**intent)} exemplar {shot_idx}")
        supports.append(class_supports)
    return supports


def _make_examples(config: DataConfig) -> list[tuple[str, int]]:
    if int(config.num_classes) < 2 or int(config.num_classes) > len(_INTENTS):
        raise ValueError("num_classes must be in [2, len(_INTENTS)]")
    if int(config.shots_per_class) < 1:
        raise ValueError("shots_per_class must be >= 1")

    rng = np.random.default_rng(int(config.seed))
    selected = [dict(_INTENTS[idx]) for idx in range(int(config.num_classes))]
    supports = _canonical_supports(selected, int(config.shots_per_class), rng)
    paraphrase_tags = ["fewshot", "lowshot", "adapt", "intent", "label"]

    examples: list[tuple[str, int]] = []
    for _ in range(int(config.num_samples)):
        label = int(rng.integers(0, int(config.num_classes)))
        intent = selected[label]
        base_text = str(rng.choice(supports[label]))
        query = str(rng.choice(_QUERY_TEMPLATES)).format(**intent)
        tag = str(rng.choice(paraphrase_tags))
        text = f"{base_text} {query} {tag}"
        examples.append((text, label))
    rng.shuffle(examples)
    return examples


class LowShotIntentDataset:
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
    dataset = LowShotIntentDataset(examples=examples, vocab=vocab, cfg=config)

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


__all__ = ["DataConfig", "LowShotIntentDataset", "get_dataloaders"]
